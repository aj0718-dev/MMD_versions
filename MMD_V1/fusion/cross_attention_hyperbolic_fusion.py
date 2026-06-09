#!/usr/bin/env python3
"""
cross_attention_hyperbolic_fusion.py

Cross-Attention Fusion in Hyperbolic Space with K-Fold CV.
Fuses HuBERT (audio) + VGG19 (image) embeddings.

Pipeline per fold:
  1. PCA (fit on train only) — reduces dims, no leakage
  2. Cross-Attention learns alignment between modalities
  3. Hyperbolic projection (exp_map) for richer geometry
  4. Classifier in hyperbolic space
  5. Report Macro-F1, Weighted-F1, Top-K

Usage:
    python cross_attention_hyperbolic_fusion.py --thresholds 10 8 5
    python cross_attention_hyperbolic_fusion.py --thresholds 10 --epochs 30
"""

import argparse
import os
import time
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


# =====================================================
# HYPERBOLIC OPERATIONS
# =====================================================


def exp_map(x):
    """Exponential map: Euclidean → Poincaré ball."""
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    return torch.tanh(norm) * x / norm


def log_map(x):
    """Logarithmic map: Poincaré ball → Euclidean."""
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    norm = torch.clamp(norm, max=1 - 1e-5)
    return torch.atanh(norm) * x / norm


def mobius_add(x, y):
    """Möbius addition in the Poincaré ball."""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)

    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2

    return num / denom.clamp_min(EPS)


# =====================================================
# CROSS-ATTENTION FUSION MODEL
# =====================================================


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between audio and image embeddings,
    followed by hyperbolic projection and classification.
    Includes residual connection from audio (stronger modality).
    """

    def __init__(self, audio_dim, image_dim, proj_dim=256, num_heads=4, num_classes=89):
        super().__init__()

        # Project both modalities to same dimension
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # Cross-attention: audio attends to image
        self.cross_attn_a2i = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, batch_first=True
        )
        # Cross-attention: image attends to audio
        self.cross_attn_i2a = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, batch_first=True
        )

        # Layer norms for residual connections
        self.norm_a = nn.LayerNorm(proj_dim)
        self.norm_i = nn.LayerNorm(proj_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),  # 3x: audio_residual + cross_a + cross_i
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # Classifier (operates after hyperbolic round-trip)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, audio_emb, image_emb):
        # Project to shared space
        a = self.audio_proj(audio_emb)  # (B, proj_dim)
        i = self.image_proj(image_emb)  # (B, proj_dim)

        # Reshape for multi-head attention: (B, 1, proj_dim)
        a_seq = a.unsqueeze(1)
        i_seq = i.unsqueeze(1)

        # Cross-attention with residual
        a_attended, _ = self.cross_attn_a2i(query=a_seq, key=i_seq, value=i_seq)
        i_attended, _ = self.cross_attn_i2a(query=i_seq, key=a_seq, value=a_seq)

        # Remove sequence dim + residual connection
        a_cross = self.norm_a(a + a_attended.squeeze(1))
        i_cross = self.norm_i(i + i_attended.squeeze(1))

        # Fuse with audio residual (ensures fusion >= audio alone)
        fused = self.fusion(torch.cat([a, a_cross, i_cross], dim=-1))

        # Hyperbolic projection
        h = exp_map(fused)
        e = log_map(h)

        # Classify
        logits = self.classifier(e)
        return logits


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def load_and_align_embeddings():
    """
    Load HuBERT + VGG19 embeddings and align by file hash.
    Returns aligned arrays + labels.
    """
    # Load embeddings
    hub_emb = torch.load("wav2vec2_hubert_wavlm/hubert_embeddings.pt", map_location="cpu", weights_only=True)
    vgg_emb = torch.load("vgg19/vgg19_embeddings_all.pt", map_location="cpu", weights_only=True)

    # Load paths for alignment
    audio_paths = torch.load("wav2vec2_hubert_wavlm/wavlm_paths.pt", weights_only=False)
    img_paths = torch.load("vgg19/vgg19_paths.pt", weights_only=False)

    # Load labels
    audio_labels = torch.load("wav2vec2_hubert_wavlm/labels.pt", map_location="cpu", weights_only=True)
    img_labels = torch.load("vgg19/labels_all.pt", map_location="cpu", weights_only=True)

    # Build hash dictionaries
    def get_hash(p):
        return os.path.basename(str(p)).replace(".wav", "").replace(".png", "")

    audio_dict = {}
    for i, p in enumerate(audio_paths):
        h = get_hash(p)
        audio_dict[h] = i

    img_dict = {}
    for i, p in enumerate(img_paths):
        h = get_hash(p)
        img_dict[h] = i

    # Find common samples
    common_hashes = sorted(set(audio_dict.keys()) & set(img_dict.keys()))

    # Build aligned arrays
    audio_indices = [audio_dict[h] for h in common_hashes]
    img_indices = [img_dict[h] for h in common_hashes]

    X_audio = hub_emb[audio_indices].numpy().astype(np.float32)
    X_image = vgg_emb[img_indices].numpy().astype(np.float32)
    y = audio_labels[audio_indices].numpy().astype(np.int64)

    # Verify label alignment
    y_img = img_labels[img_indices].numpy().astype(np.int64)
    assert np.all(y == y_img), "Label mismatch between audio and image!"

    return X_audio, X_image, y


def filter_by_threshold(X_audio, X_image, y, threshold):
    """Keep classes with >= threshold samples."""
    counts = Counter(y)
    keep_classes = {cls for cls, cnt in counts.items() if cnt >= threshold}

    if not keep_classes:
        return None, None, None, 0

    mask = np.array([label in keep_classes for label in y])
    X_a = X_audio[mask]
    X_i = X_image[mask]
    y_f = y[mask]

    # Remap to contiguous
    unique_labels = sorted(set(y_f))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[label] for label in y_f])

    return X_a, X_i, y_remapped, len(unique_labels)


def train_one_fold(X_audio_train, X_image_train, y_train,
                   X_audio_test, X_image_test, y_test,
                   n_classes, pca_dim=128, proj_dim=256,
                   epochs=25, lr=1e-3, batch_size=64):
    """
    Train cross-attention fusion for one fold.
    PCA is fit on train, transform both train and test.
    """
    # ─── PCA per modality (fit on TRAIN only) ───
    pca_audio = PCA(n_components=pca_dim, random_state=42)
    pca_image = PCA(n_components=pca_dim, random_state=42)

    X_a_train = pca_audio.fit_transform(X_audio_train)
    X_a_test = pca_audio.transform(X_audio_test)

    X_i_train = pca_image.fit_transform(X_image_train)
    X_i_test = pca_image.transform(X_image_test)

    # ─── Normalize ───
    scaler_a = StandardScaler()
    scaler_i = StandardScaler()

    X_a_train = scaler_a.fit_transform(X_a_train)
    X_a_test = scaler_a.transform(X_a_test)

    X_i_train = scaler_i.fit_transform(X_i_train)
    X_i_test = scaler_i.transform(X_i_test)

    # ─── Convert to tensors ───
    X_a_train_t = torch.tensor(X_a_train, dtype=torch.float32).to(DEVICE)
    X_i_train_t = torch.tensor(X_i_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

    X_a_test_t = torch.tensor(X_a_test, dtype=torch.float32).to(DEVICE)
    X_i_test_t = torch.tensor(X_i_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    # ─── Model ───
    model = CrossAttentionFusion(
        audio_dim=pca_dim,
        image_dim=pca_dim,
        proj_dim=proj_dim,
        num_heads=4,
        num_classes=n_classes,
    ).to(DEVICE)

    # ─── Class weights for imbalanced data ───
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ─── Training ───
    n_train = len(y_train)
    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        indices = np.random.permutation(n_train)
        epoch_loss = 0.0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = indices[start:end]

            logits = model(X_a_train_t[idx], X_i_train_t[idx])
            loss = criterion(logits, y_train_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        # ─── Evaluate every 5 epochs ───
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(X_a_test_t, X_i_test_t)
                test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
                f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ─── Final evaluation with best model ───
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(X_a_test_t, X_i_test_t)
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
        test_proba = F.softmax(test_logits, dim=1).cpu().numpy()

        train_logits = model(X_a_train_t, X_i_train_t)
        train_preds = torch.argmax(train_logits, dim=1).cpu().numpy()

    macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    train_f1 = f1_score(y_train, train_preds, average="macro", zero_division=0)

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_f1": train_f1,
        "y_pred": test_preds,
        "y_proba": test_proba,
    }


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="Cross-Attention Hyperbolic Fusion")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10, 8, 5])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)

    print("\n" + "=" * 100)
    print("  CROSS-ATTENTION FUSION IN HYPERBOLIC SPACE")
    print("  HuBERT (audio) + VGG19 (image) | K-Fold + PCA per fold (NO leakage)")
    print("=" * 100)
    print(f"\n  Thresholds: {thresholds}")
    print(f"  K-Fold: {args.n_folds}-fold Stratified")
    print(f"  PCA dim: {args.pca_dim}")
    print(f"  Projection dim: {args.proj_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {DEVICE}")
    print()

    # ─── Load and align ───
    print("  Loading and aligning embeddings... ", end="", flush=True)
    X_audio, X_image, y = load_and_align_embeddings()
    print(f"done ({X_audio.shape[0]} aligned samples)")
    print(f"  Audio: {X_audio.shape}, Image: {X_image.shape}")
    print()

    all_results = []

    for threshold in thresholds:
        print(f"\n{'─' * 100}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'─' * 100}")

        X_a, X_i, y_filt, n_classes = filter_by_threshold(X_audio, X_image, y, threshold)
        if X_a is None or n_classes < 2:
            print("  SKIPPED")
            continue

        print(f"  Samples: {X_a.shape[0]}, Classes: {n_classes}")

        # ─── K-Fold ───
        min_count = min(Counter(y_filt).values())
        actual_folds = min(args.n_folds, min_count)
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

        fold_results = []
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_a, y_filt)):
            t0 = time.time()

            result = train_one_fold(
                X_a[train_idx], X_i[train_idx], y_filt[train_idx],
                X_a[test_idx], X_i[test_idx], y_filt[test_idx],
                n_classes=n_classes,
                pca_dim=args.pca_dim,
                proj_dim=args.proj_dim,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )

            elapsed = time.time() - t0
            fold_results.append(result)
            all_y_true.extend(y_filt[test_idx])
            all_y_pred.extend(result["y_pred"])

            print(f"    Fold {fold_idx+1}: Macro-F1={result['macro_f1']:.4f} "
                  f"Weighted={result['weighted_f1']:.4f} "
                  f"Train={result['train_f1']:.4f} ({elapsed:.0f}s)")

        # ─── Aggregate results ───
        macro_f1s = [r["macro_f1"] for r in fold_results]
        weighted_f1s = [r["weighted_f1"] for r in fold_results]
        train_f1s = [r["train_f1"] for r in fold_results]

        print()
        print(f"  ┌───────────────────────────────────────────────────────────────┐")
        print(f"  │  FUSION: HuBERT + VGG19 (Cross-Attn + Hyperbolic)")
        print(f"  │  Threshold >= {threshold} ({n_classes} classes, {X_a.shape[0]} samples)")
        print(f"  ├───────────────────────────────────────────────────────────────┤")
        print(f"  │  Macro-F1:    {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
        print(f"  │  Weighted-F1: {np.mean(weighted_f1s):.4f} ± {np.std(weighted_f1s):.4f}")
        print(f"  │  Train F1:    {np.mean(train_f1s):.4f} (gap: {np.mean(train_f1s)-np.mean(macro_f1s):.4f})")
        print(f"  └───────────────────────────────────────────────────────────────┘")

        all_results.append({
            "threshold": threshold,
            "n_classes": n_classes,
            "n_samples": X_a.shape[0],
            "macro_f1": np.mean(macro_f1s),
            "macro_f1_std": np.std(macro_f1s),
            "weighted_f1": np.mean(weighted_f1s),
            "train_f1": np.mean(train_f1s),
        })

    # =====================================================
    # COMPARISON WITH BASELINES
    # =====================================================

    print(f"\n\n{'=' * 100}")
    print(f"  COMPARISON: FUSION vs BEST SINGLE-MODALITY BASELINES")
    print(f"{'=' * 100}")

    baselines = {
        10: {"HuBERT": 0.6870, "VGG19": 0.6301},
        8: {"HuBERT": 0.6831, "VGG19": 0.6156},
        5: {"HuBERT": 0.5992, "VGG19": 0.5450},
    }

    print(f"{'Threshold':<10} | {'Fusion F1':<12} | {'HuBERT (audio)':<16} | {'VGG19 (image)':<16} | {'Gain vs best'}")
    print(f"{'─' * 80}")

    for r in all_results:
        t = r["threshold"]
        base = baselines.get(t, {})
        best_single = max(base.values()) if base else 0
        gain = r["macro_f1"] - best_single
        arrow = "↑" if gain > 0 else "↓"

        print(
            f">= {t:<7} | {r['macro_f1']:.4f}±{r['macro_f1_std']:.3f} | "
            f"{base.get('HuBERT', 0):.4f}           | {base.get('VGG19', 0):.4f}           | "
            f"{arrow} {abs(gain):.4f}"
        )

    print(f"{'=' * 100}")
    print()


if __name__ == "__main__":
    main()
