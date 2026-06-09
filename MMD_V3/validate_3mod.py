#!/usr/bin/env python3
"""
Multi-seed validation for best 3-modality combo: DINOv2+HuBERT+ConvNeXt
PCA=256, Dropout=0.4, FCN_Balanced
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

DEVICE = "cpu"
K_FOLDS = 5
HIDDEN_DIM = 1024
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 256
LR = 1e-3

MODALITIES = {
    "DINOv2": {
        "embeddings": "image_embeddings/dinov2_embeddings.pt",
        "labels": "image_embeddings/dinov2_labels.pt",
        "paths": "image_embeddings/dinov2_paths.pt",
    },
    "HuBERT": {
        "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
        "labels": "wav2vec2_hubert_wavlm/labels.pt",
        "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt",
    },
    "ConvNeXt": {
        "embeddings": "image_embeddings/convnext_base_embeddings.pt",
        "labels": "image_embeddings/convnext_base_labels.pt",
        "paths": "image_embeddings/convnext_base_paths.pt",
    },
}


class FusionFCN(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_embeddings(config):
    emb = torch.load(config["embeddings"], map_location="cpu", weights_only=True)
    labels = torch.load(config["labels"], map_location="cpu", weights_only=True)
    paths = torch.load(config["paths"], map_location="cpu", weights_only=False)
    return emb.numpy().astype(np.float32), labels.numpy().astype(np.int64), paths


def get_sample_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def align_multi(modality_data):
    id_maps = []
    for name, (X, y, paths) in modality_data:
        id_map = {get_sample_id(p): i for i, p in enumerate(paths)}
        id_maps.append(id_map)

    common_ids = set(id_maps[0].keys())
    for id_map in id_maps[1:]:
        common_ids &= set(id_map.keys())
    common_ids = sorted(common_ids)

    X_list = []
    for i, (name, (X, y, paths)) in enumerate(modality_data):
        idx = np.array([id_maps[i][sid] for sid in common_ids])
        X_list.append(X[idx])

    first_idx = np.array([id_maps[0][sid] for sid in common_ids])
    y_aligned = modality_data[0][1][1][first_idx]
    return X_list, y_aligned


def filter_by_threshold(labels, threshold):
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= threshold]
    return np.isin(labels, valid_classes)


def train_fcn_fold(X_train, y_train, X_val, y_val, n_classes, dropout, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FusionFCN(X_train.shape[1], n_classes, dropout=dropout)

    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_f1 = -1
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_val)).argmax(dim=1).numpy()
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_f1


def run_seed(X_list, y, n_classes, pca_dim, dropout, seed):
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_list[0], y)):
        X_tr_parts, X_val_parts = [], []
        for X_mod in X_list:
            pca = PCA(n_components=min(pca_dim, X_mod.shape[1], len(train_idx) - 1),
                      random_state=seed)
            X_tr_parts.append(pca.fit_transform(X_mod[train_idx]))
            X_val_parts.append(pca.transform(X_mod[val_idx]))

        X_tr = np.concatenate(X_tr_parts, axis=1)
        X_val = np.concatenate(X_val_parts, axis=1)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        f1 = train_fcn_fold(X_tr, y[train_idx], X_val, y[val_idx],
                            n_classes, dropout, seed + fold_idx)
        fold_scores.append(f1)

    return np.array(fold_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--pca", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777, 2025, 3407])
    args = parser.parse_args()

    print("=" * 70)
    print("  MULTI-SEED: DINOv2 + HuBERT + ConvNeXt FCN_Balanced")
    print("=" * 70)
    print(f"  PCA: {args.pca}, Dropout: {args.dropout}, Threshold: ≥{args.threshold}")
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {LR}")
    print(f"  Seeds: {args.seeds}")
    print("=" * 70)

    # Load
    print("\n[1] Loading modalities...")
    loaded = {}
    for name, config in MODALITIES.items():
        X, y, paths = load_embeddings(config)
        loaded[name] = (X, y, paths)
        print(f"    {name}: {X.shape}")

    # Align
    modality_data = [(name, loaded[name]) for name in ["DINOv2", "HuBERT", "ConvNeXt"]]
    X_list, y = align_multi(modality_data)
    print(f"    Aligned: {len(y)} samples")

    # Threshold
    mask = filter_by_threshold(y, args.threshold)
    X_list = [X[mask] for X in X_list]
    y = y[mask]

    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y])
    n_classes = len(unique_labels)
    print(f"    After threshold ≥{args.threshold}: {len(y)} samples, {n_classes} classes")

    # Run seeds
    print(f"\n[2] Running {len(args.seeds)} seeds × {K_FOLDS} folds = "
          f"{len(args.seeds) * K_FOLDS} runs\n")

    all_seed_means = []
    all_fold_scores = []

    for seed in args.seeds:
        print(f"  {'─' * 60}")
        print(f"  Seed {seed}")
        print(f"  {'─' * 60}")

        fold_scores = run_seed(X_list, y, n_classes, args.pca, args.dropout, seed)
        mean_f1 = fold_scores.mean()
        std_f1 = fold_scores.std()
        all_seed_means.append(mean_f1)
        all_fold_scores.extend(fold_scores.tolist())

        for i, s in enumerate(fold_scores):
            print(f"    Fold {i+1}: {s:.4f}")
        print(f"    → Mean: {mean_f1:.4f} ± {std_f1:.4f}\n")

    all_seed_means = np.array(all_seed_means)
    all_fold_scores = np.array(all_fold_scores)

    print("\n" + "=" * 70)
    print("  MULTI-SEED SUMMARY: DINOv2+HuBERT+ConvNeXt")
    print("=" * 70)
    print(f"  {'─' * 60}")
    print(f"  {'Seed':<10} | {'Mean F1':<10}")
    print(f"  {'─' * 60}")
    for seed, mean_f1 in zip(args.seeds, all_seed_means):
        print(f"  {seed:<10} | {mean_f1:.4f}")
    print(f"  {'─' * 60}")
    print(f"  {'GRAND MEAN':<10} | {all_seed_means.mean():.4f} ± {all_seed_means.std():.4f}")
    print(f"  {'ALL FOLDS':<10} | {all_fold_scores.mean():.4f} ± {all_fold_scores.std():.4f}")
    print(f"  {'─' * 60}")
    print(f"\n  Min seed:  {all_seed_means.min():.4f}")
    print(f"  Max seed:  {all_seed_means.max():.4f}")
    print(f"  Range:     {all_seed_means.max() - all_seed_means.min():.4f}")
    print(f"\n  REFERENCE: DINOv2+HuBERT (2-mod, multi-seed) = 0.7231")
    delta = all_seed_means.mean() - 0.7231
    sign = "+" if delta >= 0 else ""
    print(f"  THIS:      DINOv2+HuBERT+ConvNeXt (3-mod) = {all_seed_means.mean():.4f} "
          f"({sign}{delta:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
