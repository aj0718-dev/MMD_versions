#!/usr/bin/env python3
"""
cross_attn_euclidean.py

Cross-Attention Fusion in Euclidean Space (no hyperbolic mapping).
Top 3 pairs: VGG19+HuBERT, ViT+HuBERT, VGG19+WavLM

For each fold:
  1. PCA each modality (fit on train only)
  2. Project → Cross-Attention (bidirectional) → FFN → Gated Fusion → Classify
  3. Euclidean CE loss (regular and balanced)

Tests: heads=4 and heads=8

Usage:
    python cross_attn_euclidean.py
    python cross_attn_euclidean.py --heads 4 8 --epochs 80
"""

import argparse
import os
import time
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

PAIRS = {
    "VGG19+HuBERT": {
        "modA": {"name": "VGG19", "embeddings": "vgg19/vgg19_embeddings_all.pt",
                 "labels": "vgg19/labels_all.pt", "paths": "vgg19/vgg19_paths.pt"},
        "modB": {"name": "HuBERT", "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "ViT+HuBERT": {
        "modA": {"name": "ViT", "embeddings": "vit_vgg_fcn/vit_embeddings.pt",
                 "labels": "vit_vgg_fcn/labels.pt", "paths": "vgg19/vgg19_paths.pt"},
        "modB": {"name": "HuBERT", "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "VGG19+WavLM": {
        "modA": {"name": "VGG19", "embeddings": "vgg19/vgg19_embeddings_all.pt",
                 "labels": "vgg19/labels_all.pt", "paths": "vgg19/vgg19_paths.pt"},
        "modB": {"name": "WavLM", "embeddings": "wav2vec2_hubert_wavlm/wavlm_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
}

PCA_DIM = 256
PROJ_DIM = 256
FF_DIM = 512
DROPOUT = 0.25
EPOCHS = 80
LR = 1e-3
PATIENCE = 10
BATCH_SIZE = 256
N_FOLDS = 5
RANDOM_STATE = 42
DEVICE = torch.device("cpu")


# =====================================================
# CROSS-ATTENTION MODEL (EUCLIDEAN)
# =====================================================


class CrossAttentionEuclidean(nn.Module):
    """
    Bidirectional cross-attention fusion in Euclidean space.
    Architecture:
      - Project each modality to proj_dim
      - Bidirectional cross-attention (A→B, B→A)
      - FFN blocks (transformer-style)
      - Gated fusion
      - FC classifier
    """

    def __init__(self, dim_a, dim_b, proj_dim=256, num_heads=4,
                 ff_dim=512, dropout=0.25, num_classes=89):
        super().__init__()

        # Modality projections
        self.proj_a = nn.Sequential(
            nn.Linear(dim_a, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.proj_b = nn.Sequential(
            nn.Linear(dim_b, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Bidirectional cross-attention
        self.cross_attn_a2b = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.cross_attn_b2a = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        # Post-attention layer norms
        self.norm_a = nn.LayerNorm(proj_dim)
        self.norm_b = nn.LayerNorm(proj_dim)

        # FFN blocks
        self.ffn_a = nn.Sequential(
            nn.Linear(proj_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, proj_dim),
            nn.Dropout(dropout * 0.5),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(proj_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, proj_dim),
            nn.Dropout(dropout * 0.5),
        )
        self.norm_ffn_a = nn.LayerNorm(proj_dim)
        self.norm_ffn_b = nn.LayerNorm(proj_dim)

        # Gated fusion: concat(a_res, a_cross, b_cross) → fused
        self.gate_net = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.Sigmoid(),
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classifier (Euclidean — no exp_map/log_map)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, emb_a, emb_b):
        # Project
        a = self.proj_a(emb_a)  # (B, proj_dim)
        b = self.proj_b(emb_b)

        # Save residual
        a_res = a

        # Reshape for MHA: (B, 1, proj_dim)
        a_seq = a.unsqueeze(1)
        b_seq = b.unsqueeze(1)

        # A→B: modality A attends to modality B
        a_attn, _ = self.cross_attn_a2b(query=a_seq, key=b_seq, value=b_seq)
        a_cross = self.norm_a(a + a_attn.squeeze(1))

        # B→A: modality B attends to modality A
        b_attn, _ = self.cross_attn_b2a(query=b_seq, key=a_seq, value=a_seq)
        b_cross = self.norm_b(b + b_attn.squeeze(1))

        # FFN blocks
        a_cross = self.norm_ffn_a(a_cross + self.ffn_a(a_cross))
        b_cross = self.norm_ffn_b(b_cross + self.ffn_b(b_cross))

        # Gated fusion
        combined = torch.cat([a_res, a_cross, b_cross], dim=-1)
        gate = self.gate_net(combined)
        fused_candidate = self.fusion_net(combined)
        fused = a_res + gate * (fused_candidate - a_res)

        # Classify directly in Euclidean space
        logits = self.classifier(fused)
        return logits


# =====================================================
# DATA LOADING & ALIGNMENT (same as fusion_all_pairs.py)
# =====================================================


def load_embeddings(config):
    if not os.path.exists(config["embeddings"]):
        return None, None, None
    emb = torch.load(config["embeddings"], map_location="cpu", weights_only=True)
    labels = torch.load(config["labels"], map_location="cpu", weights_only=True)
    paths = None
    if "paths" in config and os.path.exists(config["paths"]):
        paths = torch.load(config["paths"], map_location="cpu", weights_only=False)
    return emb.numpy().astype(np.float32), labels.numpy().astype(np.int64), paths


def get_sample_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def align_pairs(X_a, y_a, paths_a, X_b, y_b, paths_b, is_aligned):
    if is_aligned:
        n = min(len(X_a), len(X_b))
        return X_a[:n], X_b[:n], y_a[:n]

    if paths_a is None or paths_b is None:
        n = min(len(X_a), len(X_b))
        if np.array_equal(y_a[:n], y_b[:n]):
            return X_a[:n], X_b[:n], y_a[:n]
        print("WARNING: No paths and labels don't match!")
        return None, None, None

    ids_a = {get_sample_id(p): i for i, p in enumerate(paths_a)}
    ids_b = {get_sample_id(p): i for i, p in enumerate(paths_b)}
    common_ids = [sid for sid in ids_a if sid in ids_b]

    if len(common_ids) == 0:
        print("WARNING: No common sample IDs!")
        return None, None, None

    idx_a = np.array([ids_a[sid] for sid in common_ids])
    idx_b = np.array([ids_b[sid] for sid in common_ids])
    return X_a[idx_a], X_b[idx_b], y_a[idx_a]


def filter_by_threshold(X_a, X_b, y, threshold):
    counts = Counter(y)
    keep_classes = {cls for cls, cnt in counts.items() if cnt >= threshold}
    if not keep_classes:
        return None, None, None, 0

    mask = np.array([label in keep_classes for label in y])
    X_a_f, X_b_f, y_f = X_a[mask], X_b[mask], y[mask]

    unique_labels = sorted(set(y_f))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[l] for l in y_f])
    return X_a_f, X_b_f, y_remapped, len(unique_labels)


def compute_class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# =====================================================
# TRAINING
# =====================================================


def train_cross_attn_fold(X_a_train, X_b_train, y_train, X_a_val, X_b_val, y_val,
                          n_classes, pca_dim, proj_dim, num_heads, balanced=False,
                          epochs=80, patience=10, fold_idx=0):
    """Train cross-attention model on one fold with PCA inside."""
    # PCA per modality (fit on train only)
    pca_a = PCA(n_components=min(pca_dim, X_a_train.shape[1], X_a_train.shape[0] - 1),
                random_state=RANDOM_STATE)
    pca_b = PCA(n_components=min(pca_dim, X_b_train.shape[1], X_b_train.shape[0] - 1),
                random_state=RANDOM_STATE)

    X_a_tr_pca = pca_a.fit_transform(X_a_train)
    X_a_val_pca = pca_a.transform(X_a_val)
    X_b_tr_pca = pca_b.fit_transform(X_b_train)
    X_b_val_pca = pca_b.transform(X_b_val)

    # StandardScaler per modality
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    X_a_tr_pca = scaler_a.fit_transform(X_a_tr_pca)
    X_a_val_pca = scaler_a.transform(X_a_val_pca)
    X_b_tr_pca = scaler_b.fit_transform(X_b_tr_pca)
    X_b_val_pca = scaler_b.transform(X_b_val_pca)

    dim_a = X_a_tr_pca.shape[1]
    dim_b = X_b_tr_pca.shape[1]

    # Tensors
    X_a_tr_t = torch.tensor(X_a_tr_pca, dtype=torch.float32).to(DEVICE)
    X_b_tr_t = torch.tensor(X_b_tr_pca, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_a_val_t = torch.tensor(X_a_val_pca, dtype=torch.float32).to(DEVICE)
    X_b_val_t = torch.tensor(X_b_val_pca, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    # Model
    torch.manual_seed(RANDOM_STATE + fold_idx)
    model = CrossAttentionEuclidean(
        dim_a=dim_a, dim_b=dim_b, proj_dim=proj_dim,
        num_heads=num_heads, ff_dim=FF_DIM, dropout=DROPOUT,
        num_classes=n_classes,
    ).to(DEVICE)

    # Loss
    if balanced:
        weights = compute_class_weights(y_train, n_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training with early stopping
    best_f1 = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_a_tr_t))

        for i in range(0, len(X_a_tr_t), BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_a_tr_t[batch_idx], X_b_tr_t[batch_idx])
            loss = criterion(logits, y_tr_t[batch_idx])
            loss.backward()
            optimizer.step()

        # Val
        model.eval()
        with torch.no_grad():
            preds = model(X_a_val_t, X_b_val_t).argmax(dim=1).cpu().numpy()
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_a_val_t, X_b_val_t).argmax(dim=1).cpu().numpy()
    return f1_score(y_val, preds, average="macro", zero_division=0)


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="Cross-Attention Euclidean Fusion")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10])
    parser.add_argument("--pca_dim", type=int, default=PCA_DIM)
    parser.add_argument("--proj_dim", type=int, default=PROJ_DIM)
    parser.add_argument("--heads", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)

    print("\n" + "=" * 110)
    print("  CROSS-ATTENTION FUSION (EUCLIDEAN) — K=5, PCA inside each fold")
    print("=" * 110)
    print(f"\n  Pairs: {list(PAIRS.keys())}")
    print(f"  Thresholds: {thresholds}")
    print(f"  K-Fold: {N_FOLDS}-fold Stratified")
    print(f"  PCA dim per modality: {args.pca_dim}")
    print(f"  Proj dim: {args.proj_dim}")
    print(f"  Heads: {args.heads}")
    print(f"  FF dim: {FF_DIM}")
    print(f"  Epochs: {args.epochs} (patience={args.patience})")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Modes: CrossAttn (CE), CrossAttn_Balanced (weighted CE)")
    print()

    all_results = []
    total_start = time.time()

    for threshold in thresholds:
        print(f"\n{'━' * 110}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'━' * 110}")

        for pair_name, pair_cfg in PAIRS.items():
            print(f"\n  [{pair_name}] Loading... ", end="", flush=True)

            X_a, y_a, paths_a = load_embeddings(pair_cfg["modA"])
            X_b, y_b, paths_b = load_embeddings(pair_cfg["modB"])

            if X_a is None or X_b is None:
                print("FILE NOT FOUND")
                continue

            is_aligned = pair_cfg.get("aligned", False)
            X_a, X_b, y = align_pairs(X_a, y_a, paths_a, X_b, y_b, paths_b, is_aligned)
            if y is None:
                print("ALIGNMENT FAILED")
                continue

            print(f"{X_a.shape[0]} samples ({pair_cfg['modA']['name']}:{X_a.shape[1]}d + "
                  f"{pair_cfg['modB']['name']}:{X_b.shape[1]}d)")

            X_a_f, X_b_f, y_f, n_classes = filter_by_threshold(X_a, X_b, y, threshold)
            if X_a_f is None or n_classes < 2:
                print(f"    SKIPPED")
                continue

            print(f"    After filtering: {len(y_f)} samples, {n_classes} classes")

            # Run each config: heads × balanced
            for num_heads in args.heads:
                for mode_name, balanced in [("CrossAttn", False), ("CrossAttn_Bal", True)]:
                    label = f"{mode_name}_h{num_heads}"
                    print(f"    [{label}] ", end="", flush=True)
                    t0 = time.time()

                    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                    fold_scores = []

                    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_a_f, y_f)):
                        f1 = train_cross_attn_fold(
                            X_a_f[train_idx], X_b_f[train_idx], y_f[train_idx],
                            X_a_f[val_idx], X_b_f[val_idx], y_f[val_idx],
                            n_classes=n_classes, pca_dim=args.pca_dim,
                            proj_dim=args.proj_dim, num_heads=num_heads,
                            balanced=balanced, epochs=args.epochs,
                            patience=args.patience, fold_idx=fold_idx,
                        )
                        fold_scores.append(f1)
                        print(f"f{fold_idx+1}={f1:.4f} ", end="", flush=True)

                    elapsed = time.time() - t0
                    mean_f1 = np.mean(fold_scores)
                    std_f1 = np.std(fold_scores)
                    print(f"→ {mean_f1:.4f} ±{std_f1:.4f} ({elapsed:.0f}s)")

                    all_results.append({
                        "pair": pair_name, "method": label, "heads": num_heads,
                        "balanced": balanced, "threshold": threshold,
                        "n_classes": n_classes, "n_samples": len(y_f),
                        "mean_f1": mean_f1, "std_f1": std_f1,
                        "fold_scores": fold_scores,
                    })

    total_elapsed = time.time() - total_start

    # =====================================================
    # RESULTS
    # =====================================================

    print(f"\n\n{'=' * 110}")
    print(f"  FINAL RESULTS — CROSS-ATTENTION EUCLIDEAN (total: {total_elapsed:.0f}s)")
    print(f"{'=' * 110}")
    print(f"{'Pair':<16} | {'Method':<18} | {'Thresh':<6} | {'Classes':<7} | "
          f"{'Macro-F1':<9} | {'Std':<6} | {'Fold Scores'}")
    print(f"{'─' * 110}")

    for r in sorted(all_results, key=lambda x: (-x["threshold"], -x["mean_f1"])):
        folds_str = " ".join(f"{s:.4f}" for s in r["fold_scores"])
        print(
            f"{r['pair']:<16} | {r['method']:<18} | {r['threshold']:<6} | "
            f"{r['n_classes']:<7} | {r['mean_f1']:<9.4f} | {r['std_f1']:<6.4f} | "
            f"{folds_str}"
        )

    print(f"{'=' * 110}")

    # Best per pair
    print(f"\n  BEST CONFIG PER PAIR:")
    print(f"  {'─' * 80}")
    for pair_name in PAIRS:
        subset = [r for r in all_results if r["pair"] == pair_name]
        if subset:
            best = max(subset, key=lambda x: x["mean_f1"])
            print(f"  {pair_name:<16}: {best['method']} → {best['mean_f1']:.4f} ±{best['std_f1']:.4f}")

    # Compare with FCN_Balanced (best from fusion_all_pairs.py)
    print(f"\n  COMPARISON WITH FCN_Balanced (K=5):")
    print(f"  {'─' * 80}")
    fcn_baselines = {
        "VGG19+HuBERT": 0.7067,
        "ViT+HuBERT": 0.7066,
        "VGG19+WavLM": 0.6713,
    }
    for pair_name in PAIRS:
        subset = [r for r in all_results if r["pair"] == pair_name]
        if subset:
            best = max(subset, key=lambda x: x["mean_f1"])
            fcn_score = fcn_baselines.get(pair_name, 0)
            diff = best["mean_f1"] - fcn_score
            arrow = "↑" if diff > 0 else "↓"
            print(f"  {pair_name:<16}: CrossAttn={best['mean_f1']:.4f} vs FCN_Bal={fcn_score:.4f} "
                  f"({arrow}{abs(diff):.4f}) [{best['method']}]")

    # Compare with single-modality SVM baseline
    print(f"\n  COMPARISON WITH BEST SINGLE-MODALITY (HuBERT SVM=0.6870):")
    print(f"  {'─' * 80}")
    for pair_name in PAIRS:
        subset = [r for r in all_results if r["pair"] == pair_name]
        if subset:
            best = max(subset, key=lambda x: x["mean_f1"])
            diff = best["mean_f1"] - 0.6870
            arrow = "↑" if diff > 0 else "↓"
            print(f"  {pair_name:<16}: {best['mean_f1']:.4f} vs 0.6870 "
                  f"({arrow}{abs(diff):.4f}) [{best['method']}]")

    print()


if __name__ == "__main__":
    main()
