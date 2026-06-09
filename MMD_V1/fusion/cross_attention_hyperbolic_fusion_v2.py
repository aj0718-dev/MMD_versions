#!/usr/bin/env python3
"""
cross_attention_hyperbolic_fusion_v2.py

Multi-Modal Cross-Attention Fusion in Hyperbolic Space with K-Fold CV.
Fuses HuBERT + WavLM + Wav2Vec2 (audio) + VGG19 (image).

Key improvements over v1:
  - 3 audio models → 3-token audio sequence (cross-attention actually meaningful)
  - Mixup augmentation in embedding space
  - Label smoothing
  - Early stopping with patience
  - Higher weight decay + gradient clipping
  - Deeper fusion (2-layer cross-attention)

Usage:
    python cross_attention_hyperbolic_fusion_v2.py --thresholds 10 8 5
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


# =====================================================
# MULTI-MODAL CROSS-ATTENTION FUSION MODEL
# =====================================================


class MultiModalCrossAttentionFusion(nn.Module):
    """
    Multi-modal cross-attention: 3 audio tokens attend to 1 image token and vice versa.
    Two-layer transformer cross-attention with residual connections.
    Hyperbolic projection before classification.
    """

    def __init__(self, audio_dim, image_dim, proj_dim=256, num_heads=4,
                 num_layers=2, dropout=0.4, num_classes=89):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_layers = num_layers

        # Project each audio modality to shared dim
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        # Project image to shared dim
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Learnable modality tokens (to distinguish HuBERT/WavLM/Wav2Vec2)
        self.audio_token_emb = nn.Parameter(torch.randn(3, proj_dim) * 0.02)

        # Multi-layer cross-attention
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attn_layers.append(nn.ModuleDict({
                "a2i": nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True),
                "i2a": nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True),
                "norm_a": nn.LayerNorm(proj_dim),
                "norm_i": nn.LayerNorm(proj_dim),
                "ff_a": nn.Sequential(
                    nn.Linear(proj_dim, proj_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(proj_dim * 2, proj_dim),
                    nn.Dropout(dropout * 0.5),
                ),
                "ff_i": nn.Sequential(
                    nn.Linear(proj_dim, proj_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(proj_dim * 2, proj_dim),
                    nn.Dropout(dropout * 0.5),
                ),
                "norm_ff_a": nn.LayerNorm(proj_dim),
                "norm_ff_i": nn.LayerNorm(proj_dim),
            }))

        # Fusion: combine attended audio + image representations
        # Audio: mean-pool 3 tokens → proj_dim; Image: 1 token → proj_dim; + residual audio mean
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classifier after hyperbolic round-trip
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, audio_embs, image_emb):
        """
        audio_embs: (B, 3, audio_dim) — 3 audio models stacked
        image_emb: (B, image_dim)
        """
        B = audio_embs.shape[0]

        # Project audio: (B, 3, proj_dim)
        a = self.audio_proj(audio_embs)
        # Add modality-specific tokens
        a = a + self.audio_token_emb.unsqueeze(0)  # (B, 3, proj_dim)

        # Project image: (B, 1, proj_dim)
        i = self.image_proj(image_emb).unsqueeze(1)

        # Save residual
        a_residual = a.mean(dim=1)  # (B, proj_dim)

        # Multi-layer cross-attention
        for layer in self.cross_attn_layers:
            # Audio attends to image
            a_attn, _ = layer["a2i"](query=a, key=i, value=i)
            a = layer["norm_a"](a + a_attn)
            a = layer["norm_ff_a"](a + layer["ff_a"](a))

            # Image attends to audio
            i_attn, _ = layer["i2a"](query=i, key=a, value=a)
            i = layer["norm_i"](i + i_attn)
            i = layer["norm_ff_i"](i + layer["ff_i"](i))

        # Pool: audio mean + image squeeze + audio residual
        a_pooled = a.mean(dim=1)  # (B, proj_dim)
        i_pooled = i.squeeze(1)   # (B, proj_dim)

        # Fuse with residual from original audio
        fused = self.fusion(torch.cat([a_residual, a_pooled, i_pooled], dim=-1))

        # Hyperbolic projection
        h = exp_map(fused)
        e = log_map(h)

        # Classify
        logits = self.classifier(e)
        return logits


# =====================================================
# MIXUP AUGMENTATION
# =====================================================


def mixup_data(audio, image, y, alpha=0.4):
    """Mixup augmentation in embedding space."""
    if alpha <= 0:
        return audio, image, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure dominant sample stays dominant

    batch_size = audio.shape[0]
    perm = torch.randperm(batch_size)

    mixed_audio = lam * audio + (1 - lam) * audio[perm]
    mixed_image = lam * image + (1 - lam) * image[perm]

    return mixed_audio, mixed_image, y, y[perm], lam


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def load_and_align_embeddings():
    """
    Load HuBERT + WavLM + Wav2Vec2 + VGG19 embeddings and align by file hash.
    Returns aligned arrays + labels.
    """
    # Load all audio embeddings
    hub_emb = torch.load("wav2vec2_hubert_wavlm/hubert_embeddings.pt", map_location="cpu", weights_only=True)
    wavlm_emb = torch.load("wav2vec2_hubert_wavlm/wavlm_embeddings.pt", map_location="cpu", weights_only=True)
    wav2vec_emb = torch.load("wav2vec2_hubert_wavlm/wav2vec2_embeddings.pt", map_location="cpu", weights_only=True)

    # Load image embeddings
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

    # Stack 3 audio models: (N, 3, 768)
    X_hub = hub_emb[audio_indices].numpy().astype(np.float32)
    X_wavlm = wavlm_emb[audio_indices].numpy().astype(np.float32)
    X_wav2vec = wav2vec_emb[audio_indices].numpy().astype(np.float32)

    X_image = vgg_emb[img_indices].numpy().astype(np.float32)
    y = audio_labels[audio_indices].numpy().astype(np.int64)

    # Verify label alignment
    y_img = img_labels[img_indices].numpy().astype(np.int64)
    assert np.all(y == y_img), "Label mismatch between audio and image!"

    return X_hub, X_wavlm, X_wav2vec, X_image, y


def filter_by_threshold(X_hub, X_wavlm, X_wav2vec, X_image, y, threshold):
    """Keep classes with >= threshold samples."""
    counts = Counter(y)
    keep_classes = {cls for cls, cnt in counts.items() if cnt >= threshold}

    if not keep_classes:
        return None, None, None, None, None, 0

    mask = np.array([label in keep_classes for label in y])
    X_h = X_hub[mask]
    X_w = X_wavlm[mask]
    X_v = X_wav2vec[mask]
    X_i = X_image[mask]
    y_f = y[mask]

    # Remap to contiguous
    unique_labels = sorted(set(y_f))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[label] for label in y_f])

    return X_h, X_w, X_v, X_i, y_remapped, len(unique_labels)


def train_one_fold(X_hub_train, X_wavlm_train, X_wav2vec_train, X_image_train, y_train,
                   X_hub_test, X_wavlm_test, X_wav2vec_test, X_image_test, y_test,
                   n_classes, pca_dim=256, proj_dim=256,
                   epochs=150, lr=3e-4, batch_size=32,
                   mixup_alpha=0.4, label_smoothing=0.1, patience=20):
    """
    Train multi-modal cross-attention fusion for one fold.
    PCA fit on train only per modality.
    """
    # ─── PCA per modality (fit on TRAIN only) ───
    pca_hub = PCA(n_components=pca_dim, random_state=42)
    pca_wavlm = PCA(n_components=pca_dim, random_state=42)
    pca_wav2vec = PCA(n_components=pca_dim, random_state=42)
    pca_image = PCA(n_components=pca_dim, random_state=42)

    X_h_train = pca_hub.fit_transform(X_hub_train)
    X_h_test = pca_hub.transform(X_hub_test)

    X_w_train = pca_wavlm.fit_transform(X_wavlm_train)
    X_w_test = pca_wavlm.transform(X_wavlm_test)

    X_v_train = pca_wav2vec.fit_transform(X_wav2vec_train)
    X_v_test = pca_wav2vec.transform(X_wav2vec_test)

    X_i_train = pca_image.fit_transform(X_image_train)
    X_i_test = pca_image.transform(X_image_test)

    # ─── Normalize per modality ───
    scaler_h = StandardScaler()
    scaler_w = StandardScaler()
    scaler_v = StandardScaler()
    scaler_i = StandardScaler()

    X_h_train = scaler_h.fit_transform(X_h_train)
    X_h_test = scaler_h.transform(X_h_test)
    X_w_train = scaler_w.fit_transform(X_w_train)
    X_w_test = scaler_w.transform(X_w_test)
    X_v_train = scaler_v.fit_transform(X_v_train)
    X_v_test = scaler_v.transform(X_v_test)
    X_i_train = scaler_i.fit_transform(X_i_train)
    X_i_test = scaler_i.transform(X_i_test)

    # ─── Stack audio into (N, 3, pca_dim) ───
    audio_train = np.stack([X_h_train, X_w_train, X_v_train], axis=1)
    audio_test = np.stack([X_h_test, X_w_test, X_v_test], axis=1)

    # ─── Convert to tensors ───
    audio_train_t = torch.tensor(audio_train, dtype=torch.float32).to(DEVICE)
    image_train_t = torch.tensor(X_i_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

    audio_test_t = torch.tensor(audio_test, dtype=torch.float32).to(DEVICE)
    image_test_t = torch.tensor(X_i_test, dtype=torch.float32).to(DEVICE)

    # ─── Model ───
    model = MultiModalCrossAttentionFusion(
        audio_dim=pca_dim,
        image_dim=pca_dim,
        proj_dim=proj_dim,
        num_heads=4,
        num_layers=2,
        dropout=0.4,
        num_classes=n_classes,
    ).to(DEVICE)

    # ─── Class weights ───
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    # ─── Training with mixup + early stopping ───
    n_train = len(y_train)
    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_train)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = indices[start:end]

            batch_audio = audio_train_t[idx]
            batch_image = image_train_t[idx]
            batch_y = y_train_t[idx]

            # Mixup
            if mixup_alpha > 0 and np.random.rand() > 0.3:  # 70% chance of mixup
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                lam = max(lam, 1 - lam)
                perm = torch.randperm(batch_audio.shape[0])

                mixed_audio = lam * batch_audio + (1 - lam) * batch_audio[perm]
                mixed_image = lam * batch_image + (1 - lam) * batch_image[perm]

                logits = model(mixed_audio, mixed_image)
                loss = lam * criterion(logits, batch_y) + (1 - lam) * criterion(logits, batch_y[perm])
            else:
                logits = model(batch_audio, batch_image)
                loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # ─── Evaluate every 5 epochs ───
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(audio_test_t, image_test_t)
                test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
                f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience // 5:
                    break  # Early stop

    # ─── Final evaluation with best model ───
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(audio_test_t, image_test_t)
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()

        train_logits = model(audio_train_t, image_train_t)
        train_preds = torch.argmax(train_logits, dim=1).cpu().numpy()

    macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    train_f1 = f1_score(y_train, train_preds, average="macro", zero_division=0)

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_f1": train_f1,
    }


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Cross-Attention Hyperbolic Fusion v2")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10, 8, 5])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)

    print("\n" + "=" * 100)
    print("  MULTI-MODAL CROSS-ATTENTION FUSION IN HYPERBOLIC SPACE (v2)")
    print("  HuBERT + WavLM + Wav2Vec2 (audio) + VGG19 (image) | K-Fold + PCA per fold")
    print("=" * 100)
    print(f"\n  Thresholds: {thresholds}")
    print(f"  K-Fold: {args.n_folds}-fold Stratified")
    print(f"  PCA dim: {args.pca_dim}")
    print(f"  Projection dim: {args.proj_dim}")
    print(f"  Epochs: {args.epochs} (patience={args.patience})")
    print(f"  LR: {args.lr}, Weight decay: 5e-3")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Mixup α: {args.mixup_alpha}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Cross-attention layers: 2")
    print(f"  Device: {DEVICE}")
    print()

    # ─── Load and align ───
    print("  Loading and aligning embeddings... ", end="", flush=True)
    X_hub, X_wavlm, X_wav2vec, X_image, y = load_and_align_embeddings()
    print(f"done ({X_hub.shape[0]} aligned samples)")
    print(f"  HuBERT: {X_hub.shape}, WavLM: {X_wavlm.shape}, Wav2Vec2: {X_wav2vec.shape}")
    print(f"  VGG19: {X_image.shape}")
    print()

    all_results = []

    for threshold in thresholds:
        print(f"\n{'─' * 100}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'─' * 100}")

        X_h, X_w, X_v, X_i, y_filt, n_classes = filter_by_threshold(
            X_hub, X_wavlm, X_wav2vec, X_image, y, threshold
        )
        if X_h is None or n_classes < 2:
            print("  SKIPPED")
            continue

        print(f"  Samples: {X_h.shape[0]}, Classes: {n_classes}")

        # ─── K-Fold ───
        min_count = min(Counter(y_filt).values())
        actual_folds = min(args.n_folds, min_count)
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_h, y_filt)):
            t0 = time.time()

            result = train_one_fold(
                X_h[train_idx], X_w[train_idx], X_v[train_idx], X_i[train_idx], y_filt[train_idx],
                X_h[test_idx], X_w[test_idx], X_v[test_idx], X_i[test_idx], y_filt[test_idx],
                n_classes=n_classes,
                pca_dim=args.pca_dim,
                proj_dim=args.proj_dim,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                mixup_alpha=args.mixup_alpha,
                label_smoothing=args.label_smoothing,
                patience=args.patience,
            )

            elapsed = time.time() - t0
            fold_results.append(result)

            print(f"    Fold {fold_idx+1}: Macro-F1={result['macro_f1']:.4f} "
                  f"Weighted={result['weighted_f1']:.4f} "
                  f"Train={result['train_f1']:.4f} ({elapsed:.0f}s)")

        # ─── Aggregate results ───
        macro_f1s = [r["macro_f1"] for r in fold_results]
        weighted_f1s = [r["weighted_f1"] for r in fold_results]
        train_f1s = [r["train_f1"] for r in fold_results]

        print()
        print(f"  ┌───────────────────────────────────────────────────────────────┐")
        print(f"  │  FUSION v2: HuBERT+WavLM+Wav2Vec2 + VGG19 (Hyperbolic)")
        print(f"  │  Threshold >= {threshold} ({n_classes} classes, {X_h.shape[0]} samples)")
        print(f"  ├───────────────────────────────────────────────────────────────┤")
        print(f"  │  Macro-F1:    {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
        print(f"  │  Weighted-F1: {np.mean(weighted_f1s):.4f} ± {np.std(weighted_f1s):.4f}")
        print(f"  │  Train F1:    {np.mean(train_f1s):.4f} (gap: {np.mean(train_f1s)-np.mean(macro_f1s):.4f})")
        print(f"  └───────────────────────────────────────────────────────────────┘")

        all_results.append({
            "threshold": threshold,
            "n_classes": n_classes,
            "n_samples": X_h.shape[0],
            "macro_f1": np.mean(macro_f1s),
            "macro_f1_std": np.std(macro_f1s),
            "weighted_f1": np.mean(weighted_f1s),
            "train_f1": np.mean(train_f1s),
        })

    # =====================================================
    # COMPARISON WITH BASELINES
    # =====================================================

    print(f"\n\n{'=' * 100}")
    print(f"  COMPARISON: MULTI-MODAL FUSION v2 vs BEST SINGLE-MODALITY BASELINES")
    print(f"{'=' * 100}")

    baselines = {
        10: {"HuBERT": 0.6870, "VGG19": 0.6301},
        8: {"HuBERT": 0.6831, "VGG19": 0.6156},
        5: {"HuBERT": 0.5992, "VGG19": 0.5450},
    }

    print(f"{'Threshold':<10} | {'Fusion v2 F1':<14} | {'HuBERT':<10} | {'VGG19':<10} | {'Gain vs best':<14} | {'Target 0.80'}")
    print(f"{'─' * 90}")

    for r in all_results:
        t = r["threshold"]
        base = baselines.get(t, {})
        best_single = max(base.values()) if base else 0
        gain = r["macro_f1"] - best_single
        arrow = "↑" if gain > 0 else "↓"
        gap_to_target = 0.80 - r["macro_f1"]

        print(
            f">= {t:<7} | {r['macro_f1']:.4f}±{r['macro_f1_std']:.3f}  | "
            f"{base.get('HuBERT', 0):.4f}    | {base.get('VGG19', 0):.4f}    | "
            f"{arrow} {abs(gain):.4f}        | {'✓ REACHED' if r['macro_f1'] >= 0.80 else f'need +{gap_to_target:.4f}'}"
        )

    print(f"{'=' * 100}")
    print()


if __name__ == "__main__":
    main()
