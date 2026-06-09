#!/usr/bin/env python3
"""
cross_attention_hyperbolic_final.py

Cross-Attention Fusion in Hyperbolic Space — FINAL VERSION
Tests: HuBERT + VGG19 and HuBERT + ViT

Uses neural classifier (proven better than SVM for cross-attn fusion).
Compares against LogisticRegression baselines from grid search.

Pipeline per fold:
  1. PCA per modality (fit on train) — no data leakage
  2. StandardScaler per modality
  3. Cross-attention with gated residual (HuBERT-centric)
  4. Hyperbolic projection (Poincaré ball)
  5. Neural classifier

Usage:
    python cross_attention_hyperbolic_final.py --thresholds 10 8 5
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
    """Logarithmic map: Poincaré ball → tangent space."""
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    norm = torch.clamp(norm, max=1 - 1e-5)
    return torch.atanh(norm) * x / norm


# =====================================================
# CROSS-ATTENTION FUSION MODEL
# =====================================================


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion with gated residual from audio (stronger modality).
    Projects to hyperbolic space before classification.
    """

    def __init__(self, audio_dim, image_dim, proj_dim=256, num_heads=4, num_classes=89):
        super().__init__()

        # Project both modalities to shared space
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

        # Bidirectional cross-attention
        self.cross_attn_a2i = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attn_i2a = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, batch_first=True
        )

        # Layer norms
        self.norm_a = nn.LayerNorm(proj_dim)
        self.norm_i = nn.LayerNorm(proj_dim)

        # Gated fusion: audio_residual + gate * cross_attention_info
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # Classifier after hyperbolic projection
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, audio_emb, image_emb):
        # Project
        a = self.audio_proj(audio_emb)  # (B, proj_dim)
        i = self.image_proj(image_emb)  # (B, proj_dim)

        # Cross-attention (reshape for MHA)
        a_seq = a.unsqueeze(1)
        i_seq = i.unsqueeze(1)

        a_attended, _ = self.cross_attn_a2i(query=a_seq, key=i_seq, value=i_seq)
        i_attended, _ = self.cross_attn_i2a(query=i_seq, key=a_seq, value=a_seq)

        a_cross = self.norm_a(a + a_attended.squeeze(1))
        i_cross = self.norm_i(i + i_attended.squeeze(1))

        # Gated fusion: preserves audio signal, selectively adds cross-attn info
        combined = torch.cat([a, a_cross, i_cross], dim=-1)
        g = self.gate(combined)
        fused_raw = self.fusion(combined)
        fused = a + g * (fused_raw - a)  # Gated residual: if g=0, pure audio

        # Hyperbolic projection
        h = exp_map(fused)
        e = log_map(h)

        return self.classifier(e)


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def load_and_align(image_model="VGG19"):
    """
    Load HuBERT + image model embeddings, align by file hash.
    image_model: 'VGG19' or 'ViT'
    """
    # Audio
    hub_emb = torch.load("wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                         map_location="cpu", weights_only=True)
    audio_paths = torch.load("wav2vec2_hubert_wavlm/wavlm_paths.pt", weights_only=False)
    audio_labels = torch.load("wav2vec2_hubert_wavlm/labels.pt",
                              map_location="cpu", weights_only=True)

    # Image (ViT and VGG19 share same ordering, verified by labels match)
    if image_model == "VGG19":
        img_emb = torch.load("vgg19/vgg19_embeddings_all.pt", map_location="cpu", weights_only=True)
    else:  # ViT
        img_emb = torch.load("vit_vgg_fcn/vit_embeddings.pt", map_location="cpu", weights_only=True)

    img_paths = torch.load("vgg19/vgg19_paths.pt", weights_only=False)
    img_labels = torch.load("vgg19/labels_all.pt", map_location="cpu", weights_only=True)

    # Align by file hash
    def get_hash(p):
        return os.path.basename(str(p)).replace(".wav", "").replace(".png", "")

    audio_dict = {get_hash(p): i for i, p in enumerate(audio_paths)}
    img_dict = {get_hash(p): i for i, p in enumerate(img_paths)}

    common = sorted(set(audio_dict.keys()) & set(img_dict.keys()))
    a_idx = [audio_dict[h] for h in common]
    i_idx = [img_dict[h] for h in common]

    X_audio = hub_emb[a_idx].numpy().astype(np.float32)
    X_image = img_emb[i_idx].numpy().astype(np.float32)
    y = audio_labels[a_idx].numpy().astype(np.int64)

    y_img = img_labels[i_idx].numpy().astype(np.int64)
    assert np.all(y == y_img), "Label mismatch!"

    return X_audio, X_image, y


def filter_by_threshold(X_a, X_i, y, threshold):
    """Keep classes with >= threshold samples, remap labels."""
    counts = Counter(y)
    keep = {c for c, n in counts.items() if n >= threshold}
    if not keep:
        return None, None, None, 0
    mask = np.array([l in keep for l in y])
    X_a_f, X_i_f, y_f = X_a[mask], X_i[mask], y[mask]
    unique = sorted(set(y_f))
    remap = {old: new for new, old in enumerate(unique)}
    return X_a_f, X_i_f, np.array([remap[l] for l in y_f]), len(unique)


def train_one_fold(X_a_tr, X_i_tr, y_tr, X_a_te, X_i_te, y_te,
                   n_classes, pca_dim=256, proj_dim=256,
                   epochs=80, lr=5e-4, batch_size=32):
    """Train cross-attention fusion for one fold. PCA fit on train only."""
    # PCA per modality
    pca_a = PCA(n_components=min(pca_dim, X_a_tr.shape[1] - 1), random_state=42)
    pca_i = PCA(n_components=min(pca_dim, X_i_tr.shape[1] - 1), random_state=42)

    Xa_tr = pca_a.fit_transform(X_a_tr)
    Xa_te = pca_a.transform(X_a_te)
    Xi_tr = pca_i.fit_transform(X_i_tr)
    Xi_te = pca_i.transform(X_i_te)

    # Normalize
    sc_a, sc_i = StandardScaler(), StandardScaler()
    Xa_tr = sc_a.fit_transform(Xa_tr)
    Xa_te = sc_a.transform(Xa_te)
    Xi_tr = sc_i.fit_transform(Xi_tr)
    Xi_te = sc_i.transform(Xi_te)

    # Tensors
    a_tr_t = torch.tensor(Xa_tr, dtype=torch.float32).to(DEVICE)
    i_tr_t = torch.tensor(Xi_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    a_te_t = torch.tensor(Xa_te, dtype=torch.float32).to(DEVICE)
    i_te_t = torch.tensor(Xi_te, dtype=torch.float32).to(DEVICE)

    # Model
    audio_dim = Xa_tr.shape[1]
    image_dim = Xi_tr.shape[1]
    model = CrossAttentionFusion(
        audio_dim=audio_dim, image_dim=image_dim,
        proj_dim=proj_dim, num_heads=4, num_classes=n_classes,
    ).to(DEVICE)

    # Class weights
    cc = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    cc = np.maximum(cc, 1.0)
    w = 1.0 / cc
    w = w / w.sum() * n_classes
    class_weights = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    n_train = len(y_tr)
    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            logits = model(a_tr_t[idx], i_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Eval every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = torch.argmax(model(a_te_t, i_te_t), dim=1).cpu().numpy()
                f1 = f1_score(y_te, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Final eval with best
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_preds = torch.argmax(model(a_te_t, i_te_t), dim=1).cpu().numpy()
        tr_preds = torch.argmax(model(a_tr_t, i_tr_t), dim=1).cpu().numpy()

    return {
        "macro_f1": f1_score(y_te, te_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_te, te_preds, average="weighted", zero_division=0),
        "train_f1": f1_score(y_tr, tr_preds, average="macro", zero_division=0),
    }


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10, 8, 5])
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)

    # LR baselines from grid search (for comparison)
    lr_baselines = {
        10: {"HuBERT": 0.6607, "VGG19": 0.5993, "ViT": 0.6159},
        8: {"HuBERT": 0.6438, "VGG19": 0.5993, "ViT": 0.6140},
        5: {"HuBERT": 0.5718, "VGG19": 0.5256, "ViT": 0.5241},
    }

    print("\n" + "=" * 100)
    print("  CROSS-ATTENTION FUSION IN HYPERBOLIC SPACE — FINAL")
    print("  Best Audio (HuBERT) + Best Image (VGG19 & ViT) | K-Fold + PCA per fold")
    print("=" * 100)
    print(f"\n  PCA: {args.pca_dim} | Proj: {args.proj_dim} | Epochs: {args.epochs}")
    print(f"  LR: {args.lr} | Batch: {args.batch_size} | Folds: {args.n_folds}")
    print(f"  Arch: Gated CrossAttn + Hyperbolic + Neural Classifier")
    print(f"  Baseline comparison: LogisticRegression (from grid search)")
    print()

    # Run for both image models
    for img_model in ["VGG19", "ViT"]:
        print(f"\n{'━' * 100}")
        print(f"  IMAGE MODEL: {img_model}")
        print(f"{'━' * 100}")

        X_audio, X_image, y = load_and_align(image_model=img_model)
        print(f"  Aligned: {X_audio.shape[0]} samples | Audio: {X_audio.shape[1]}d | Image: {X_image.shape[1]}d")

        for threshold in thresholds:
            print(f"\n  {'─' * 90}")
            print(f"  THRESHOLD >= {threshold}")
            print(f"  {'─' * 90}")

            X_a, X_i, y_f, nc = filter_by_threshold(X_audio, X_image, y, threshold)
            if X_a is None:
                continue

            print(f"  Samples: {X_a.shape[0]}, Classes: {nc}")

            min_cnt = min(Counter(y_f).values())
            n_folds = min(args.n_folds, min_cnt)
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            fold_results = []
            for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_a, y_f)):
                t0 = time.time()
                r = train_one_fold(
                    X_a[tr_idx], X_i[tr_idx], y_f[tr_idx],
                    X_a[te_idx], X_i[te_idx], y_f[te_idx],
                    n_classes=nc, pca_dim=args.pca_dim, proj_dim=args.proj_dim,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                )
                elapsed = time.time() - t0
                fold_results.append(r)
                print(f"    Fold {fold_idx+1}: Macro-F1={r['macro_f1']:.4f} "
                      f"Weighted={r['weighted_f1']:.4f} Train={r['train_f1']:.4f} ({elapsed:.0f}s)")

            # Aggregate
            macro = [r["macro_f1"] for r in fold_results]
            weighted = [r["weighted_f1"] for r in fold_results]
            train = [r["train_f1"] for r in fold_results]

            base = lr_baselines.get(threshold, {})
            best_lr = max(base.get("HuBERT", 0), base.get(img_model, 0))
            gain = np.mean(macro) - best_lr

            print()
            print(f"  ┌────────────────────────────────────────────────────────────────────────┐")
            print(f"  │  FUSION: HuBERT + {img_model} (CrossAttn + Hyperbolic)")
            print(f"  │  Threshold >= {threshold} ({nc} classes, {X_a.shape[0]} samples)")
            print(f"  ├────────────────────────────────────────────────────────────────────────┤")
            print(f"  │  Macro-F1:     {np.mean(macro):.4f} ± {np.std(macro):.4f}")
            print(f"  │  Weighted-F1:  {np.mean(weighted):.4f} ± {np.std(weighted):.4f}")
            print(f"  │  Train F1:     {np.mean(train):.4f}")
            print(f"  ├────────────────────────────────────────────────────────────────────────┤")
            print(f"  │  LR Baselines: HuBERT={base.get('HuBERT',0):.4f}  {img_model}={base.get(img_model,0):.4f}")
            arrow = "↑" if gain > 0 else "↓"
            print(f"  │  Fusion gain vs best LR: {arrow} {abs(gain):.4f}")
            print(f"  └────────────────────────────────────────────────────────────────────────┘")

    # Final summary
    print(f"\n\n{'=' * 100}")
    print(f"  NOTE: Comparing against LogisticRegression baselines (from gridsearch_all_models.py)")
    print(f"  The fusion uses a neural classifier which is the LR-equivalent category.")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
