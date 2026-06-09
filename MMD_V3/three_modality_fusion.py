#!/usr/bin/env python3
"""
Experiment C: Three-modality feature-level fusion
=================================================
Test whether adding a third modality (Swin / ConvNeXt) to DINOv2+HuBERT
improves over the best two-modality result (0.7361).

Combos:
  - DINOv2 + HuBERT + Swin
  - DINOv2 + HuBERT + ConvNeXt
  - DINOv2 + HuBERT + Swin + ConvNeXt

Architecture: Optimized FCN_Balanced (same as best result)
PCA per modality → concat → StandardScaler → FCN

Usage:
    python3 three_modality_fusion.py --threshold 10 --pca 128
    python3 three_modality_fusion.py --threshold 10 --pca 256
    python3 three_modality_fusion.py --threshold 10 --pca 128 256
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
RANDOM_STATE = 42
HIDDEN_DIM = 1024
DROPOUT = 0.3
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 256
LR = 1e-3

# Also try dropout 0.4 for higher-dim concat
DROPOUT_VARIANTS = [0.3, 0.4]

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
    "Swin": {
        "embeddings": "image_embeddings/swin_base_embeddings.pt",
        "labels": "image_embeddings/swin_base_labels.pt",
        "paths": "image_embeddings/swin_base_paths.pt",
    },
    "ConvNeXt": {
        "embeddings": "image_embeddings/convnext_base_embeddings.pt",
        "labels": "image_embeddings/convnext_base_labels.pt",
        "paths": "image_embeddings/convnext_base_paths.pt",
    },
}

COMBOS = [
    ("DINOv2", "HuBERT", "Swin"),
    ("DINOv2", "HuBERT", "ConvNeXt"),
    ("DINOv2", "HuBERT", "Swin", "ConvNeXt"),
]


# =====================================================
# MODEL
# =====================================================

class FusionFCN(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# DATA LOADING & ALIGNMENT
# =====================================================

def load_embeddings(config):
    emb = torch.load(config["embeddings"], map_location="cpu", weights_only=True)
    labels = torch.load(config["labels"], map_location="cpu", weights_only=True)
    paths = torch.load(config["paths"], map_location="cpu", weights_only=False)
    return emb.numpy().astype(np.float32), labels.numpy().astype(np.int64), paths


def get_sample_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def align_multi(modality_data):
    """Align N modalities by common sample IDs."""
    # Build ID→index maps for each modality
    id_maps = []
    for name, (X, y, paths) in modality_data:
        id_map = {get_sample_id(p): i for i, p in enumerate(paths)}
        id_maps.append(id_map)

    # Find common IDs across all
    common_ids = set(id_maps[0].keys())
    for id_map in id_maps[1:]:
        common_ids &= set(id_map.keys())
    common_ids = sorted(common_ids)

    if len(common_ids) == 0:
        return None, None, None

    # Extract aligned arrays
    X_list = []
    for i, (name, (X, y, paths)) in enumerate(modality_data):
        idx = np.array([id_maps[i][sid] for sid in common_ids])
        X_list.append(X[idx])

    # Labels from first modality
    first_idx = np.array([id_maps[0][sid] for sid in common_ids])
    y_aligned = modality_data[0][1][1][first_idx]

    return X_list, y_aligned, common_ids


def filter_by_threshold(labels, threshold):
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= threshold]
    mask = np.isin(labels, valid_classes)
    return mask


# =====================================================
# TRAINING
# =====================================================

def train_fcn_fold(X_train, y_train, X_val, y_val, n_classes, dropout, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    model = FusionFCN(input_dim, n_classes, dropout=dropout)

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
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_f1


def run_combo(X_list, y, n_classes, pca_dim, dropout, seed=RANDOM_STATE):
    """Run 5-fold CV for a multi-modality combo."""
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_list[0], y)):
        # PCA each modality separately
        X_tr_parts = []
        X_val_parts = []

        for X_mod in X_list:
            X_mod_tr = X_mod[train_idx]
            X_mod_val = X_mod[val_idx]

            n_components = min(pca_dim, X_mod_tr.shape[1], X_mod_tr.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=seed)
            X_mod_tr_pca = pca.fit_transform(X_mod_tr)
            X_mod_val_pca = pca.transform(X_mod_val)

            X_tr_parts.append(X_mod_tr_pca)
            X_val_parts.append(X_mod_val_pca)

        # Concat all modalities
        X_tr = np.concatenate(X_tr_parts, axis=1)
        X_val = np.concatenate(X_val_parts, axis=1)

        # Scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        y_tr, y_val = y[train_idx], y[val_idx]

        # Train FCN
        f1 = train_fcn_fold(X_tr, y_tr, X_val, y_val, n_classes, dropout,
                            seed=seed + fold_idx)
        fold_scores.append(f1)

    return np.array(fold_scores)


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--pca", type=int, nargs="+", default=[128, 256])
    args = parser.parse_args()

    print("=" * 70)
    print("  THREE-MODALITY FEATURE FUSION (FCN_Balanced)")
    print("=" * 70)
    print(f"  Threshold: ≥{args.threshold}")
    print(f"  PCA dims: {args.pca}")
    print(f"  Dropout variants: {DROPOUT_VARIANTS}")
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {LR}")
    print("=" * 70)

    # Load all modalities
    print("\n[1] Loading modalities...")
    loaded = {}
    for name, config in MODALITIES.items():
        if os.path.exists(config["embeddings"]):
            X, y, paths = load_embeddings(config)
            loaded[name] = (X, y, paths)
            print(f"    {name}: {X.shape}")
        else:
            print(f"    {name}: NOT FOUND — skipping")

    # Process each combo
    results = []
    total_runs = 0
    for combo in COMBOS:
        if all(m in loaded for m in combo):
            total_runs += len(args.pca) * len(DROPOUT_VARIANTS) * K_FOLDS

    print(f"\n[2] Running {len(COMBOS)} combos × {len(args.pca)} PCA × "
          f"{len(DROPOUT_VARIANTS)} dropout × {K_FOLDS} folds = {total_runs} training runs\n")

    for combo in COMBOS:
        combo_name = "+".join(combo)
        print(f"\n  {'═' * 60}")
        print(f"  {combo_name}")
        print(f"  {'═' * 60}")

        # Check all modalities exist
        if not all(m in loaded for m in combo):
            missing = [m for m in combo if m not in loaded]
            print(f"    SKIP: Missing modalities {missing}")
            continue

        # Align
        modality_data = [(name, loaded[name]) for name in combo]
        X_list, y, common_ids = align_multi(modality_data)
        if X_list is None:
            print("    SKIP: No common samples")
            continue
        print(f"    Aligned: {len(y)} samples")

        # Threshold
        mask = filter_by_threshold(y, args.threshold)
        X_list = [X[mask] for X in X_list]
        y = y[mask]

        # Remap labels
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[l] for l in y])
        n_classes = len(unique_labels)
        print(f"    After threshold ≥{args.threshold}: {len(y)} samples, {n_classes} classes")

        for pca_dim in args.pca:
            for dropout in DROPOUT_VARIANTS:
                label = f"PCA={pca_dim}, Drop={dropout}"
                print(f"\n    {label}")

                fold_scores = run_combo(X_list, y, n_classes, pca_dim, dropout)
                mean_f1 = fold_scores.mean()
                std_f1 = fold_scores.std()

                for i, s in enumerate(fold_scores):
                    print(f"      Fold {i+1}: {s:.4f}")
                print(f"      → Mean: {mean_f1:.4f} ± {std_f1:.4f}")

                results.append({
                    "combo": combo_name,
                    "pca": pca_dim,
                    "dropout": dropout,
                    "mean": mean_f1,
                    "std": std_f1,
                    "folds": fold_scores.tolist(),
                })

    # Final summary
    results.sort(key=lambda r: r["mean"], reverse=True)
    print("\n\n" + "=" * 70)
    print("  FINAL RESULTS (Threshold ≥{}, {} classes)".format(args.threshold, n_classes))
    print("=" * 70)
    print(f"  {'─' * 65}")
    print(f"  {'Rank':<5}| {'Combo':<30}| {'PCA':<5}| {'Drop':<5}| {'F1':<8}| {'Std':<6}")
    print(f"  {'─' * 65}")
    for i, r in enumerate(results):
        star = " ★" if i == 0 else ""
        print(f"  {i+1:<5}| {r['combo']:<30}| {r['pca']:<5}| {r['dropout']:<5}| "
              f"{r['mean']:.4f} | {r['std']:.4f}{star}")
    print(f"  {'─' * 65}")
    print(f"\n  REFERENCE: DINOv2+HuBERT FCN_Balanced (2-modality) = 0.7361")
    if results:
        best = results[0]
        delta = best["mean"] - 0.7361
        sign = "+" if delta >= 0 else ""
        print(f"  BEST:      {best['combo']} (PCA={best['pca']}, Drop={best['dropout']}) "
              f"= {best['mean']:.4f} ({sign}{delta:.4f} vs 0.7361)")
    print("=" * 70)


if __name__ == "__main__":
    main()
