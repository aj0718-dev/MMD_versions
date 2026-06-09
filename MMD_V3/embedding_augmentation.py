#!/usr/bin/env python3
"""
Embedding-Space Augmentation Experiment
========================================
Tests three augmentation strategies on frozen PCA embeddings:

A) Gaussian noise:       x_aug = x + N(0, σ²I)
B) Intra-class Mixup:    x_aug = λ·x_i + (1-λ)·x_j, same class
C) Gaussian + Mixup:     Both applied together

Tested on:
  - 2-encoder: DINOv2 + HuBERT
  - 3-encoder: DINOv2 + HuBERT + ConvNeXt

Augmentation applied ONLY to training data during training epochs.
Validation folds are never augmented.

Usage:
    python3 embedding_augmentation.py --encoders 2
    python3 embedding_augmentation.py --encoders 3
    python3 embedding_augmentation.py --encoders 2 3
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 256
LR = 1e-3
SEED = 42  # Screening seed

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

# Augmentation hyperparameter grids
GAUSSIAN_SIGMAS = [0.005, 0.01, 0.02, 0.03]
MIXUP_ALPHAS = [0.2, 0.4]
COMBINED_CONFIGS = [
    {"sigma": 0.01, "alpha": 0.2},
    {"sigma": 0.01, "alpha": 0.4},
    {"sigma": 0.02, "alpha": 0.2},
]


# =====================================================
# MODEL
# =====================================================

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


# =====================================================
# AUGMENTED DATASET
# =====================================================

class AugmentedDataset(Dataset):
    """Dataset that applies online augmentation to training embeddings."""

    def __init__(self, X, y, aug_type="none", sigma=0.0, alpha=0.2):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.aug_type = aug_type
        self.sigma = sigma
        self.alpha = alpha

        # Pre-compute class indices for mixup
        if aug_type in ("mixup", "combined"):
            self.class_indices = {}
            for idx in range(len(y)):
                c = int(y[idx])
                if c not in self.class_indices:
                    self.class_indices[c] = []
                self.class_indices[c].append(idx)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        label = self.y[idx]

        if self.aug_type == "gaussian":
            noise = torch.randn_like(x) * self.sigma
            x = x + noise

        elif self.aug_type == "mixup":
            x = self._mixup(x, int(label))

        elif self.aug_type == "combined":
            # Mixup first, then add noise
            x = self._mixup(x, int(label))
            noise = torch.randn_like(x) * self.sigma
            x = x + noise

        return x, label

    def _mixup(self, x, label):
        """Intra-class mixup: interpolate with a random same-class sample."""
        same_class = self.class_indices[label]
        if len(same_class) < 2:
            return x
        partner_idx = same_class[torch.randint(len(same_class), (1,)).item()]
        partner = self.X[partner_idx]
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 (closer to original)
        return lam * x + (1 - lam) * partner


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


def align_multi(modality_data):
    """Align multiple modalities by sample ID."""
    id_maps = []
    for name, (X, y, paths) in modality_data:
        id_map = {get_sample_id(p): i for i, p in enumerate(paths)}
        id_maps.append(id_map)

    common_ids = set(id_maps[0].keys())
    for id_map in id_maps[1:]:
        common_ids &= set(id_map.keys())
    common_ids = sorted(common_ids)

    aligned_X = []
    for id_map, (name, (X, y, paths)) in zip(id_maps, modality_data):
        indices = [id_map[sid] for sid in common_ids]
        aligned_X.append(X[indices])

    ref_labels = modality_data[0][1][1]
    ref_indices = [id_maps[0][sid] for sid in common_ids]
    y_aligned = ref_labels[ref_indices]

    return aligned_X, y_aligned


def filter_by_threshold(labels, threshold):
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= threshold]
    mask = np.isin(labels, valid_classes)
    return mask


# =====================================================
# TRAINING
# =====================================================

def train_fcn_fold(X_train, y_train, X_val, y_val, n_classes, seed,
                   dropout, aug_type="none", sigma=0.0, alpha=0.2):
    """Train one fold with augmentation applied to training data."""
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

    # Training dataset with augmentation
    train_ds = AugmentedDataset(X_train, y_train, aug_type=aug_type,
                                sigma=sigma, alpha=alpha)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Validation (no augmentation)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

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

        # Validate (no augmentation)
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
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


def run_cv(aligned_Xs, y, n_classes, pca_dim, dropout, seed,
           aug_type="none", sigma=0.0, alpha=0.2):
    """Run 5-fold CV with specified augmentation."""
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(aligned_Xs[0], y)):
        # Split each modality
        Xs_train = [X[train_idx] for X in aligned_Xs]
        Xs_val = [X[val_idx] for X in aligned_Xs]
        y_tr, y_val = y[train_idx], y[val_idx]

        # PCA per modality (fitted on train only)
        Xs_train_pca = []
        Xs_val_pca = []
        for X_tr, X_v in zip(Xs_train, Xs_val):
            n_comp = min(pca_dim, X_tr.shape[1], X_tr.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=seed)
            Xs_train_pca.append(pca.fit_transform(X_tr))
            Xs_val_pca.append(pca.transform(X_v))

        # Concatenate
        X_tr_cat = np.concatenate(Xs_train_pca, axis=1)
        X_val_cat = np.concatenate(Xs_val_pca, axis=1)

        # StandardScaler (fitted on train only)
        scaler = StandardScaler()
        X_tr_cat = scaler.fit_transform(X_tr_cat)
        X_val_cat = scaler.transform(X_val_cat)

        # Train with augmentation
        f1 = train_fcn_fold(X_tr_cat, y_tr, X_val_cat, y_val, n_classes,
                            seed + fold_idx, dropout, aug_type, sigma, alpha)
        fold_scores.append(f1)

    return np.array(fold_scores)


# =====================================================
# EXPERIMENT RUNNER
# =====================================================

def run_experiment(aligned_Xs, y, n_classes, pca_dim, dropout, encoder_label):
    """Run all augmentation configs and print results."""
    results = []

    # Baseline: no augmentation
    print(f"\n  [Baseline] No augmentation (dropout={dropout})")
    scores = run_cv(aligned_Xs, y, n_classes, pca_dim, dropout, SEED,
                    aug_type="none")
    mean_f1, std_f1 = scores.mean(), scores.std()
    print(f"    Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    results.append(("No augmentation", mean_f1, std_f1))

    # A) Gaussian noise
    print(f"\n  --- (A) Gaussian Noise ---")
    for sigma in GAUSSIAN_SIGMAS:
        print(f"  [Gaussian] σ={sigma}")
        scores = run_cv(aligned_Xs, y, n_classes, pca_dim, dropout, SEED,
                        aug_type="gaussian", sigma=sigma)
        mean_f1, std_f1 = scores.mean(), scores.std()
        print(f"    Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}")
        results.append((f"Gaussian(σ={sigma})", mean_f1, std_f1))

    # B) Intra-class Mixup
    print(f"\n  --- (B) Intra-class Mixup ---")
    for alpha in MIXUP_ALPHAS:
        print(f"  [Mixup] α={alpha}")
        scores = run_cv(aligned_Xs, y, n_classes, pca_dim, dropout, SEED,
                        aug_type="mixup", alpha=alpha)
        mean_f1, std_f1 = scores.mean(), scores.std()
        print(f"    Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}")
        results.append((f"Mixup(α={alpha})", mean_f1, std_f1))

    # C) Combined: Gaussian + Mixup
    print(f"\n  --- (C) Combined: Mixup + Gaussian ---")
    for cfg in COMBINED_CONFIGS:
        sigma, alpha = cfg["sigma"], cfg["alpha"]
        print(f"  [Combined] σ={sigma}, α={alpha}")
        scores = run_cv(aligned_Xs, y, n_classes, pca_dim, dropout, SEED,
                        aug_type="combined", sigma=sigma, alpha=alpha)
        mean_f1, std_f1 = scores.mean(), scores.std()
        print(f"    Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}")
        results.append((f"Combined(σ={sigma},α={alpha})", mean_f1, std_f1))

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {encoder_label} | PCA={pca_dim} | Dropout={dropout}")
    print(f"{'=' * 70}")
    print(f"  {'Method':<35} {'Macro-F1':>10} {'Std':>8} {'Δ vs Base':>10}")
    print(f"  {'─' * 65}")

    baseline_f1 = results[0][1]
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    for method, f1, std in results_sorted:
        delta = f1 - baseline_f1
        marker = " ★" if f1 > baseline_f1 + 0.001 else ""
        print(f"  {method:<35} {f1:>10.4f} {std:>8.4f} {delta:>+10.4f}{marker}")
    print(f"  {'─' * 65}")

    return results


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", type=int, nargs="+", default=[2, 3],
                        choices=[2, 3], help="2=DINOv2+HuBERT, 3=+ConvNeXt")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--pca", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    args = parser.parse_args()

    print("=" * 70)
    print("  EMBEDDING-SPACE AUGMENTATION EXPERIMENT")
    print("=" * 70)
    print(f"  Threshold: ≥{args.threshold} | PCA: {args.pca} | Dropout: {args.dropout}")
    print(f"  Epochs: {EPOCHS} | Patience: {PATIENCE} | Seed: {SEED}")
    print(f"  Augmentations:")
    print(f"    (A) Gaussian noise: σ ∈ {GAUSSIAN_SIGMAS}")
    print(f"    (B) Intra-class Mixup: α ∈ {MIXUP_ALPHAS}")
    print(f"    (C) Combined configs: {len(COMBINED_CONFIGS)}")
    print(f"  Encoder configs to test: {args.encoders}")
    print("=" * 70)

    # Load all needed modalities
    print("\n[1] Loading embeddings...")
    modality_data = []
    for name in ["DINOv2", "HuBERT", "ConvNeXt"]:
        if name == "ConvNeXt" and 3 not in args.encoders:
            continue
        X, y, paths = load_embeddings(MODALITIES[name])
        modality_data.append((name, (X, y, paths)))
        print(f"    {name}: {X.shape}")

    # Align all modalities
    print("\n[2] Aligning modalities...")
    aligned_Xs, y = align_multi(modality_data)
    print(f"    Aligned: {len(y)} samples")

    # Threshold filter
    mask = filter_by_threshold(y, args.threshold)
    aligned_Xs = [X[mask] for X in aligned_Xs]
    y = y[mask]

    # Remap labels
    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y])
    n_classes = len(unique_labels)
    print(f"    After threshold ≥{args.threshold}: {len(y)} samples, {n_classes} classes")

    # Run experiments
    all_results = {}

    if 2 in args.encoders:
        print("\n" + "=" * 70)
        print("  EXPERIMENT: 2-Encoder (DINOv2 + HuBERT)")
        print("=" * 70)
        # Use only first 2 modalities (DINOv2, HuBERT)
        Xs_2enc = aligned_Xs[:2]
        results_2 = run_experiment(Xs_2enc, y, n_classes, args.pca,
                                   args.dropout, "DINOv2+HuBERT")
        all_results["2-encoder"] = results_2

    if 3 in args.encoders:
        print("\n" + "=" * 70)
        print("  EXPERIMENT: 3-Encoder (DINOv2 + HuBERT + ConvNeXt)")
        print("=" * 70)
        Xs_3enc = aligned_Xs[:3]
        results_3 = run_experiment(Xs_3enc, y, n_classes, args.pca,
                                   args.dropout, "DINOv2+HuBERT+ConvNeXt")
        all_results["3-encoder"] = results_3

    # Final combined summary
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON ACROSS ALL CONFIGS")
    print("=" * 70)
    for enc_label, results in all_results.items():
        baseline_f1 = results[0][1]
        best_method, best_f1, best_std = max(results, key=lambda x: x[1])
        delta = best_f1 - baseline_f1
        print(f"\n  {enc_label}:")
        print(f"    Baseline:  {baseline_f1:.4f}")
        print(f"    Best:      {best_f1:.4f} ± {best_std:.4f} ({best_method})")
        print(f"    Δ:         {delta:+.4f}")
        if delta > 0.002:
            print(f"    → Augmentation HELPS (+{delta:.4f})")
        elif delta < -0.002:
            print(f"    → Augmentation HURTS ({delta:.4f})")
        else:
            print(f"    → Augmentation has negligible effect")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
