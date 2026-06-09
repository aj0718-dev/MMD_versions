#!/usr/bin/env python3
"""
Multi-Seed Validation: Same-Class Mixup (α=0.4)
=================================================
Validates the best augmentation finding from embedding_augmentation.py:
  - Intra-class Mixup with α=0.4

Tests:
  - 2-encoder (DINOv2+HuBERT), dropout=0.3, Mixup α=0.4
  - 3-encoder (DINOv2+HuBERT+ConvNeXt), dropout=0.4, Mixup α=0.4

Each config is run with 5 seeds × 5 folds = 25 runs.
Reports both "no augmentation" and "mixup" per seed to isolate the
relative gain independent of seed luck.

Usage:
    python3 -u validate_mixup.py --encoders 2 3 --seeds 42 123 777 2025 3407
    python3 -u validate_mixup.py --encoders 2 --seeds 42 123 777 2025 3407
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
PCA_DIM = 256
THRESHOLD = 10
MIXUP_ALPHA = 0.4

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

# Encoder configs: (encoder_count, dropout)
ENCODER_CONFIGS = {
    2: {"modalities": ["DINOv2", "HuBERT"], "dropout": 0.3},
    3: {"modalities": ["DINOv2", "HuBERT", "ConvNeXt"], "dropout": 0.4},
}


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
# DATASET WITH MIXUP
# =====================================================

class MixupDataset(Dataset):
    """Dataset with optional intra-class mixup augmentation."""

    def __init__(self, X, y, use_mixup=False, alpha=0.4):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.use_mixup = use_mixup
        self.alpha = alpha

        if use_mixup:
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

        if self.use_mixup:
            same_class = self.class_indices[int(label)]
            if len(same_class) >= 2:
                partner_idx = same_class[torch.randint(len(same_class), (1,)).item()]
                partner = self.X[partner_idx]
                lam = np.random.beta(self.alpha, self.alpha)
                lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
                x = lam * x + (1 - lam) * partner

        return x, label


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
                   dropout, use_mixup=False):
    """Train one fold with optional mixup."""
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

    # Training dataset
    train_ds = MixupDataset(X_train, y_train, use_mixup=use_mixup,
                            alpha=MIXUP_ALPHA)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Validation (never augmented)
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


def run_cv_seed(aligned_Xs, y, n_classes, dropout, seed, use_mixup=False):
    """Run 5-fold CV for one seed."""
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(aligned_Xs[0], y)):
        Xs_train = [X[train_idx] for X in aligned_Xs]
        Xs_val = [X[val_idx] for X in aligned_Xs]
        y_tr, y_val = y[train_idx], y[val_idx]

        # PCA per modality (fitted on train only)
        Xs_train_pca = []
        Xs_val_pca = []
        for X_tr, X_v in zip(Xs_train, Xs_val):
            n_comp = min(PCA_DIM, X_tr.shape[1], X_tr.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=seed)
            Xs_train_pca.append(pca.fit_transform(X_tr))
            Xs_val_pca.append(pca.transform(X_v))

        # Concatenate and scale
        X_tr_cat = np.concatenate(Xs_train_pca, axis=1)
        X_val_cat = np.concatenate(Xs_val_pca, axis=1)

        scaler = StandardScaler()
        X_tr_cat = scaler.fit_transform(X_tr_cat)
        X_val_cat = scaler.transform(X_val_cat)

        # Train
        f1 = train_fcn_fold(X_tr_cat, y_tr, X_val_cat, y_val, n_classes,
                            seed + fold_idx, dropout, use_mixup=use_mixup)
        fold_scores.append(f1)

    return np.array(fold_scores)


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", type=int, nargs="+", default=[2, 3],
                        choices=[2, 3])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 777, 2025, 3407])
    args = parser.parse_args()

    print("=" * 70)
    print("  MULTI-SEED VALIDATION: Same-Class Mixup (α=0.4)")
    print("=" * 70)
    print(f"  Threshold: ≥{THRESHOLD} | PCA: {PCA_DIM}")
    print(f"  Mixup α: {MIXUP_ALPHA}")
    print(f"  Epochs: {EPOCHS} | Patience: {PATIENCE}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Encoder configs: {args.encoders}")
    print("=" * 70)

    # Load modalities
    print("\n[1] Loading embeddings...")
    all_modality_data = []
    for name in ["DINOv2", "HuBERT", "ConvNeXt"]:
        if name == "ConvNeXt" and 3 not in args.encoders:
            continue
        X, y, paths = load_embeddings(MODALITIES[name])
        all_modality_data.append((name, (X, y, paths)))
        print(f"    {name}: {X.shape}")

    # Align
    print("\n[2] Aligning modalities...")
    aligned_Xs, y = align_multi(all_modality_data)
    print(f"    Aligned: {len(y)} samples")

    # Threshold filter
    mask = filter_by_threshold(y, THRESHOLD)
    aligned_Xs = [X[mask] for X in aligned_Xs]
    y = y[mask]

    # Remap labels
    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y])
    n_classes = len(unique_labels)
    print(f"    After threshold ≥{THRESHOLD}: {len(y)} samples, {n_classes} classes")

    # Run validation for each encoder config
    for n_enc in args.encoders:
        cfg = ENCODER_CONFIGS[n_enc]
        dropout = cfg["dropout"]
        Xs = aligned_Xs[:n_enc]
        label = "+".join(cfg["modalities"])

        print(f"\n{'=' * 70}")
        print(f"  VALIDATING: {label} | Dropout={dropout}")
        print(f"{'=' * 70}")

        seed_results_base = []
        seed_results_mixup = []

        for seed in args.seeds:
            print(f"\n  --- Seed {seed} ---")

            # Baseline (no augmentation)
            scores_base = run_cv_seed(Xs, y, n_classes, dropout, seed,
                                      use_mixup=False)
            mean_base = scores_base.mean()
            seed_results_base.append(mean_base)
            print(f"    No augmentation: {mean_base:.4f} "
                  f"(folds: {', '.join(f'{s:.4f}' for s in scores_base)})")

            # Mixup α=0.4
            scores_mixup = run_cv_seed(Xs, y, n_classes, dropout, seed,
                                       use_mixup=True)
            mean_mixup = scores_mixup.mean()
            seed_results_mixup.append(mean_mixup)
            delta = mean_mixup - mean_base
            print(f"    Mixup α=0.4:     {mean_mixup:.4f} "
                  f"(folds: {', '.join(f'{s:.4f}' for s in scores_mixup)})")
            print(f"    Δ (mixup - base): {delta:+.4f}")

        # Summary
        seed_results_base = np.array(seed_results_base)
        seed_results_mixup = np.array(seed_results_mixup)
        deltas = seed_results_mixup - seed_results_base

        print(f"\n  {'=' * 60}")
        print(f"  SUMMARY: {label}")
        print(f"  {'=' * 60}")
        print(f"  {'Seed':<8} {'Baseline':>10} {'Mixup α=0.4':>12} {'Δ':>10}")
        print(f"  {'─' * 42}")
        for i, seed in enumerate(args.seeds):
            print(f"  {seed:<8} {seed_results_base[i]:>10.4f} "
                  f"{seed_results_mixup[i]:>12.4f} "
                  f"{deltas[i]:>+10.4f}")
        print(f"  {'─' * 42}")
        print(f"  {'Mean':<8} {seed_results_base.mean():>10.4f} "
              f"{seed_results_mixup.mean():>12.4f} "
              f"{deltas.mean():>+10.4f}")
        print(f"  {'Std':<8} {seed_results_base.std():>10.4f} "
              f"{seed_results_mixup.std():>12.4f} "
              f"{deltas.std():>10.4f}")
        print(f"  {'─' * 42}")

        # Statistical interpretation
        grand_mean_base = seed_results_base.mean()
        grand_std_base = seed_results_base.std()
        grand_mean_mixup = seed_results_mixup.mean()
        grand_std_mixup = seed_results_mixup.std()
        mean_delta = deltas.mean()
        std_delta = deltas.std()

        print(f"\n  Grand Mean (Baseline): {grand_mean_base:.4f} ± {grand_std_base:.4f}")
        print(f"  Grand Mean (Mixup):    {grand_mean_mixup:.4f} ± {grand_std_mixup:.4f}")
        print(f"  Mean Δ:                {mean_delta:+.4f} ± {std_delta:.4f}")

        # Check if gain is consistent
        n_positive = (deltas > 0).sum()
        print(f"  Positive Δ:            {n_positive}/{len(deltas)} seeds")

        if mean_delta > 0.002 and n_positive >= 4:
            print(f"  → Mixup CONSISTENTLY HELPS ({mean_delta:+.4f})")
        elif mean_delta > 0.001:
            print(f"  → Mixup provides MARGINAL benefit ({mean_delta:+.4f})")
        else:
            print(f"  → Mixup benefit NOT validated across seeds")

    print(f"\n{'=' * 70}")
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
