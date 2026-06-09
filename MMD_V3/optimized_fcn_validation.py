#!/usr/bin/env python3
"""
Experiment A+B: Multi-seed validation & multi-threshold optimized FCN
=====================================================================
Validates the best DINOv2+HuBERT FCN_Balanced result across multiple seeds
and thresholds to confirm stability.

Usage:
    # Experiment A: Multi-seed validation at threshold 10
    python3 optimized_fcn_validation.py --threshold 10 --pca 256 --seeds 42 123 777 2025 3407

    # Experiment B: Optimized FCN at other thresholds
    python3 optimized_fcn_validation.py --threshold 8 --pca 384 --seeds 42 123 777 2025 3407
    python3 optimized_fcn_validation.py --threshold 5 --pca 512 --seeds 42 123 777 2025 3407
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

DEVICE = "cpu"
K_FOLDS = 5
HIDDEN_DIM = 1024
DROPOUT = 0.3
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 256
LR = 1e-3

DINO_CONFIG = {
    "embeddings": "image_embeddings/dinov2_embeddings.pt",
    "labels": "image_embeddings/dinov2_labels.pt",
    "paths": "image_embeddings/dinov2_paths.pt",
}
HUBERT_CONFIG = {
    "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
    "labels": "wav2vec2_hubert_wavlm/labels.pt",
    "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt",
}


# =====================================================
# MODEL
# =====================================================

class FusionFCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# DATA LOADING
# =====================================================

def load_embeddings(config):
    emb = torch.load(config["embeddings"], map_location="cpu", weights_only=True)
    labels = torch.load(config["labels"], map_location="cpu", weights_only=True)
    paths = torch.load(config["paths"], map_location="cpu", weights_only=False)
    return emb.numpy().astype(np.float32), labels.numpy().astype(np.int64), paths


def get_sample_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def align_pair(X_a, y_a, paths_a, X_b, y_b, paths_b):
    ids_a = {get_sample_id(p): i for i, p in enumerate(paths_a)}
    ids_b = {get_sample_id(p): i for i, p in enumerate(paths_b)}
    common_ids = sorted(set(ids_a.keys()) & set(ids_b.keys()))
    idx_a = np.array([ids_a[sid] for sid in common_ids])
    idx_b = np.array([ids_b[sid] for sid in common_ids])
    return X_a[idx_a], X_b[idx_b], y_a[idx_a]


def filter_by_threshold(labels, threshold):
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= threshold]
    mask = np.isin(labels, valid_classes)
    return mask


# =====================================================
# TRAINING
# =====================================================

def train_fcn_fold(X_train, y_train, X_val, y_val, n_classes, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    model = FusionFCN(input_dim, n_classes)

    # Class-balanced weights
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    class_weights = torch.tensor(weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_f1 = -1
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            logits_val = model(torch.tensor(X_val))
            preds = logits_val.argmax(dim=1).numpy()
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_f1


def run_single_seed(X_a, X_b, y, n_classes, pca_dim, seed):
    """Run 5-fold CV with a single seed."""
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_a, y)):
        X_a_tr, X_a_val = X_a[train_idx], X_a[val_idx]
        X_b_tr, X_b_val = X_b[train_idx], X_b[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # PCA per modality
        pca_a = PCA(n_components=min(pca_dim, X_a_tr.shape[1], X_a_tr.shape[0] - 1),
                    random_state=seed)
        pca_b = PCA(n_components=min(pca_dim, X_b_tr.shape[1], X_b_tr.shape[0] - 1),
                    random_state=seed)

        X_a_tr_pca = pca_a.fit_transform(X_a_tr)
        X_a_val_pca = pca_a.transform(X_a_val)
        X_b_tr_pca = pca_b.fit_transform(X_b_tr)
        X_b_val_pca = pca_b.transform(X_b_val)

        # Concat + scale
        X_tr = np.concatenate([X_a_tr_pca, X_b_tr_pca], axis=1)
        X_val = np.concatenate([X_a_val_pca, X_b_val_pca], axis=1)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # Train
        f1 = train_fcn_fold(X_tr, y_tr, X_val, y_val, n_classes, seed + fold_idx)
        fold_scores.append(f1)

    return np.array(fold_scores)


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--pca", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777, 2025, 3407])
    args = parser.parse_args()

    print("=" * 70)
    print("  MULTI-SEED FCN_Balanced VALIDATION (DINOv2 + HuBERT)")
    print("=" * 70)
    print(f"  PCA: {args.pca}, Threshold: ≥{args.threshold}")
    print(f"  Hidden: {HIDDEN_DIM}, Dropout: {DROPOUT}")
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {LR}")
    print(f"  Seeds: {args.seeds}")
    print("=" * 70)

    # Load
    print("\n[1] Loading DINOv2 + HuBERT...")
    X_dino, y_dino, paths_dino = load_embeddings(DINO_CONFIG)
    X_hub, y_hub, paths_hub = load_embeddings(HUBERT_CONFIG)

    # Align
    X_a, X_b, y = align_pair(X_dino, y_dino, paths_dino, X_hub, y_hub, paths_hub)
    print(f"    Aligned: {len(y)} samples")

    # Threshold filter
    mask = filter_by_threshold(y, args.threshold)
    X_a, X_b, y = X_a[mask], X_b[mask], y[mask]

    # Remap labels to contiguous
    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y])
    n_classes = len(unique_labels)
    print(f"    After threshold ≥{args.threshold}: {len(y)} samples, {n_classes} classes")

    # Run each seed
    print(f"\n[2] Running {len(args.seeds)} seeds × {K_FOLDS} folds = {len(args.seeds) * K_FOLDS} runs\n")

    all_seed_means = []
    all_fold_scores = []

    for seed in args.seeds:
        print(f"  {'─' * 60}")
        print(f"  Seed {seed}")
        print(f"  {'─' * 60}")

        fold_scores = run_single_seed(X_a, X_b, y, n_classes, args.pca, seed)
        mean_f1 = fold_scores.mean()
        std_f1 = fold_scores.std()
        all_seed_means.append(mean_f1)
        all_fold_scores.extend(fold_scores.tolist())

        for i, score in enumerate(fold_scores):
            print(f"    Fold {i+1}: {score:.4f}")
        print(f"    → Mean: {mean_f1:.4f} ± {std_f1:.4f}\n")

    # Summary
    all_seed_means = np.array(all_seed_means)
    all_fold_scores = np.array(all_fold_scores)

    print("\n" + "=" * 70)
    print("  MULTI-SEED SUMMARY")
    print("=" * 70)
    print(f"  {'─' * 60}")
    print(f"  {'Seed':<10} | {'Mean F1':<10} | {'Std':<10}")
    print(f"  {'─' * 60}")
    for seed, mean_f1 in zip(args.seeds, all_seed_means):
        print(f"  {seed:<10} | {mean_f1:.4f}     | —")
    print(f"  {'─' * 60}")
    print(f"  {'GRAND MEAN':<10} | {all_seed_means.mean():.4f}     | {all_seed_means.std():.4f}")
    print(f"  {'ALL FOLDS':<10} | {all_fold_scores.mean():.4f}     | {all_fold_scores.std():.4f}")
    print(f"  {'─' * 60}")
    print(f"\n  Min seed:  {all_seed_means.min():.4f}")
    print(f"  Max seed:  {all_seed_means.max():.4f}")
    print(f"  Range:     {all_seed_means.max() - all_seed_means.min():.4f}")
    print(f"\n  Config: PCA={args.pca}, Threshold≥{args.threshold}, "
          f"{n_classes} classes, {len(y)} samples")
    print("=" * 70)


if __name__ == "__main__":
    main()
