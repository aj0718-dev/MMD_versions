#!/usr/bin/env python3
"""
fusion_all_pairs.py

Comprehensive multimodal fusion comparison:
  - Concat + SVM-RBF
  - FCN (learnable, regular CE)
  - FCN Balanced (learnable, class-weighted CE)

All modality pairs, multiple PCA dims, k=5, PCA fitted inside each fold.

Usage:
    python fusion_all_pairs.py --thresholds 10
    python fusion_all_pairs.py --thresholds 10 8 5
    python fusion_all_pairs.py --thresholds 10 --pairs DINOv2+HuBERT ConvNeXt+HuBERT
    python fusion_all_pairs.py --thresholds 10 --pca_dims 128 256
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

PAIRS = {
    # --- Old pairs (image+image) ---
    "ViT+VGG19": {
        "modA": {"name": "ViT", "embeddings": "vit_vgg_fcn/vit_embeddings.pt",
                 "labels": "vit_vgg_fcn/labels.pt", "paths": "vgg19/vgg19_paths.pt"},
        "modB": {"name": "VGG19", "embeddings": "vit_vgg_fcn/vgg_embeddings.pt",
                 "labels": "vit_vgg_fcn/labels.pt", "paths": "vgg19/vgg19_paths.pt"},
        "aligned": True,
    },
    # --- Old pairs (image+audio) ---
    "ViT+HuBERT": {
        "modA": {"name": "ViT", "embeddings": "vit_vgg_fcn/vit_embeddings.pt",
                 "labels": "vit_vgg_fcn/labels.pt", "paths": "vgg19/vgg19_paths.pt"},
        "modB": {"name": "HuBERT", "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "ViT+WavLM": {
        "modA": {"name": "ViT", "embeddings": "vit_vgg_fcn/vit_embeddings.pt",
                 "labels": "vit_vgg_fcn/labels.pt", "paths": "vgg19/vgg19_paths.pt"},
        "modB": {"name": "WavLM", "embeddings": "wav2vec2_hubert_wavlm/wavlm_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "ViT+Wav2Vec2": {
        "modA": {"name": "ViT", "embeddings": "vit_vgg_fcn/vit_embeddings.pt",
                 "labels": "vit_vgg_fcn/labels.pt", "paths": "vgg19/vgg19_paths.pt"},
        "modB": {"name": "Wav2Vec2", "embeddings": "wav2vec2_hubert_wavlm/wav2vec2_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "VGG19+HuBERT": {
        "modA": {"name": "VGG19", "embeddings": "vgg19/vgg19_embeddings_all.pt",
                 "labels": "vgg19/labels_all.pt", "paths": "vgg19/vgg19_paths.pt"},
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
    "VGG19+Wav2Vec2": {
        "modA": {"name": "VGG19", "embeddings": "vgg19/vgg19_embeddings_all.pt",
                 "labels": "vgg19/labels_all.pt", "paths": "vgg19/vgg19_paths.pt"},
        "modB": {"name": "Wav2Vec2", "embeddings": "wav2vec2_hubert_wavlm/wav2vec2_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    # --- New pairs: top image PTMs + HuBERT ---
    "DINOv2+HuBERT": {
        "modA": {"name": "DINOv2", "embeddings": "image_embeddings/dinov2_embeddings.pt",
                 "labels": "image_embeddings/dinov2_labels.pt", "paths": "image_embeddings/dinov2_paths.pt"},
        "modB": {"name": "HuBERT", "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "ConvNeXt+HuBERT": {
        "modA": {"name": "ConvNeXt", "embeddings": "image_embeddings/convnext_base_embeddings.pt",
                 "labels": "image_embeddings/convnext_base_labels.pt", "paths": "image_embeddings/convnext_base_paths.pt"},
        "modB": {"name": "HuBERT", "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "EfficientNet+HuBERT": {
        "modA": {"name": "EfficientNet", "embeddings": "image_embeddings/efficientnet_b0_embeddings.pt",
                 "labels": "image_embeddings/efficientnet_b0_labels.pt", "paths": "image_embeddings/efficientnet_b0_paths.pt"},
        "modB": {"name": "HuBERT", "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "Swin+HuBERT": {
        "modA": {"name": "Swin", "embeddings": "image_embeddings/swin_base_embeddings.pt",
                 "labels": "image_embeddings/swin_base_labels.pt", "paths": "image_embeddings/swin_base_paths.pt"},
        "modB": {"name": "HuBERT", "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "MobileNet+HuBERT": {
        "modA": {"name": "MobileNet", "embeddings": "image_embeddings/mobilenetv3_large_embeddings.pt",
                 "labels": "image_embeddings/mobilenetv3_large_labels.pt", "paths": "image_embeddings/mobilenetv3_large_paths.pt"},
        "modB": {"name": "HuBERT", "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    # --- New pairs: top image PTMs + WavLM ---
    "DINOv2+WavLM": {
        "modA": {"name": "DINOv2", "embeddings": "image_embeddings/dinov2_embeddings.pt",
                 "labels": "image_embeddings/dinov2_labels.pt", "paths": "image_embeddings/dinov2_paths.pt"},
        "modB": {"name": "WavLM", "embeddings": "wav2vec2_hubert_wavlm/wavlm_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "ConvNeXt+WavLM": {
        "modA": {"name": "ConvNeXt", "embeddings": "image_embeddings/convnext_base_embeddings.pt",
                 "labels": "image_embeddings/convnext_base_labels.pt", "paths": "image_embeddings/convnext_base_paths.pt"},
        "modB": {"name": "WavLM", "embeddings": "wav2vec2_hubert_wavlm/wavlm_embeddings.pt",
                 "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
}

# --- FCN config ---
PCA_DIMS = [128, 256, 384, 512]
HIDDEN_DIM = 1024
DROPOUT = 0.3
EPOCHS = 80
FCN_LR = 1e-3
PATIENCE = 10
BATCH_SIZE = 256

K_FOLDS = 5
RANDOM_STATE = 42
DEVICE = torch.device("cpu")


# =====================================================
# MODEL
# =====================================================


class FusionFCN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# DATA LOADING & ALIGNMENT
# =====================================================


def load_embeddings(config):
    """Load embedding and label tensors + optional paths."""
    if not os.path.exists(config["embeddings"]):
        return None, None, None
    emb = torch.load(config["embeddings"], map_location="cpu", weights_only=True)
    labels = torch.load(config["labels"], map_location="cpu", weights_only=True)
    paths = None
    if "paths" in config and os.path.exists(config["paths"]):
        paths = torch.load(config["paths"], map_location="cpu", weights_only=False)
    return emb.numpy().astype(np.float32), labels.numpy().astype(np.int64), paths


def get_sample_id(path):
    """Extract sample ID from file path (basename without extension)."""
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def align_pairs(X_a, y_a, paths_a, X_b, y_b, paths_b, is_aligned):
    """Align two modality arrays by sample ID matching."""
    if is_aligned:
        n = min(len(X_a), len(X_b))
        return X_a[:n], X_b[:n], y_a[:n]

    if paths_a is None or paths_b is None:
        # Fallback: truncate to min and hope ordering is consistent
        n = min(len(X_a), len(X_b))
        if np.array_equal(y_a[:n], y_b[:n]):
            return X_a[:n], X_b[:n], y_a[:n]
        print("WARNING: No paths and labels don't match!")
        return None, None, None

    # Build ID → index maps
    ids_a = {get_sample_id(p): i for i, p in enumerate(paths_a)}
    ids_b = {get_sample_id(p): i for i, p in enumerate(paths_b)}

    # Find common IDs
    common_ids = [sid for sid in ids_a if sid in ids_b]

    if len(common_ids) == 0:
        print("WARNING: No common sample IDs found!")
        return None, None, None

    idx_a = np.array([ids_a[sid] for sid in common_ids])
    idx_b = np.array([ids_b[sid] for sid in common_ids])

    X_a_aligned = X_a[idx_a]
    X_b_aligned = X_b[idx_b]
    y_aligned = y_a[idx_a]

    return X_a_aligned, X_b_aligned, y_aligned


def filter_by_threshold(X_a, X_b, y, threshold):
    """Keep classes with >= threshold samples, remap to contiguous."""
    counts = Counter(y)
    keep_classes = {cls for cls, cnt in counts.items() if cnt >= threshold}

    if not keep_classes:
        return None, None, None, 0

    mask = np.array([label in keep_classes for label in y])
    X_a_f = X_a[mask]
    X_b_f = X_b[mask]
    y_f = y[mask]

    unique_labels = sorted(set(y_f))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[l] for l in y_f])

    return X_a_f, X_b_f, y_remapped, len(unique_labels)


# =====================================================
# CONCAT + SKLEARN CLASSIFIERS
# =====================================================


def run_concat_gridsearch(X_a, X_b, y, n_classes, pca_dim, n_folds, classifier="svm"):
    """
    Concat fusion with manual k-fold and per-modality PCA.
    For each fold:
      1. PCA modA on train → transform
      2. PCA modB on train → transform
      3. Normalize: L2 for LR, StandardScaler for SVM
      4. Fit classifier on concat train, predict concat val
    """
    min_count = min(Counter(y).values())
    actual_folds = min(n_folds, min_count)
    if actual_folds < 2:
        return None, None, None

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for train_idx, val_idx in cv.split(X_a, y):
        X_a_tr, X_a_val = X_a[train_idx], X_a[val_idx]
        X_b_tr, X_b_val = X_b[train_idx], X_b[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # PCA per modality (fit on train only)
        pca_a = PCA(n_components=min(pca_dim, X_a_tr.shape[1], X_a_tr.shape[0] - 1),
                    random_state=RANDOM_STATE)
        pca_b = PCA(n_components=min(pca_dim, X_b_tr.shape[1], X_b_tr.shape[0] - 1),
                    random_state=RANDOM_STATE)

        X_a_tr_pca = pca_a.fit_transform(X_a_tr)
        X_a_val_pca = pca_a.transform(X_a_val)
        X_b_tr_pca = pca_b.fit_transform(X_b_tr)
        X_b_val_pca = pca_b.transform(X_b_val)

        # Concat
        X_tr_cat = np.concatenate([X_a_tr_pca, X_b_tr_pca], axis=1)
        X_val_cat = np.concatenate([X_a_val_pca, X_b_val_pca], axis=1)

        # Normalize: L2 for LR (angle-based), StandardScaler for SVM (distance-based)
        if classifier == "lr":
            from sklearn.preprocessing import Normalizer
            norm = Normalizer(norm="l2")
            X_tr_cat = norm.fit_transform(X_tr_cat)
            X_val_cat = norm.transform(X_val_cat)
        else:
            scaler = StandardScaler()
            X_tr_cat = scaler.fit_transform(X_tr_cat)
            X_val_cat = scaler.transform(X_val_cat)

        # Fit classifier
        if classifier == "svm":
            clf = SVC(kernel="rbf", C=100, gamma="scale", class_weight="balanced",
                      random_state=RANDOM_STATE)
        elif classifier == "lr":
            clf = LogisticRegression(C=1000, solver="lbfgs", class_weight="balanced",
                                     max_iter=3000, random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

        clf.fit(X_tr_cat, y_tr)
        preds = clf.predict(X_val_cat)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_scores.append(f1)

    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    return mean_f1, std_f1, fold_scores


# =====================================================
# FCN TRAINING
# =====================================================


def compute_class_weights(y, n_classes):
    """Compute balanced class weights."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_fcn_fold(X_a_train, X_b_train, y_train, X_a_val, X_b_val, y_val,
                   n_classes, pca_dim, balanced=False, epochs=80, patience=10, fold_idx=0):
    """Train FCN on one fold with PCA inside."""
    # PCA per modality
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

    # Concat
    X_tr_cat = np.concatenate([X_a_tr_pca, X_b_tr_pca], axis=1)
    X_val_cat = np.concatenate([X_a_val_pca, X_b_val_pca], axis=1)
    input_dim = X_tr_cat.shape[1]

    # Tensors
    X_tr = torch.tensor(X_tr_cat, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_v = torch.tensor(X_val_cat, dtype=torch.float32).to(DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    # Model
    torch.manual_seed(RANDOM_STATE + fold_idx)
    model = FusionFCN(input_dim, n_classes, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)

    # Loss
    if balanced:
        weights = compute_class_weights(y_train, n_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=FCN_LR)

    # Training
    best_f1 = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_tr))

        for i in range(0, len(X_tr), BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_tr[batch_idx])
            loss = criterion(logits, y_tr[batch_idx])
            loss.backward()
            optimizer.step()

        # Val
        model.eval()
        with torch.no_grad():
            preds = model(X_v).argmax(dim=1).cpu().numpy()
        macro_f1 = f1_score(y_v.cpu().numpy(), preds, average="macro", zero_division=0)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_v).argmax(dim=1).cpu().numpy()
    return f1_score(y_v.cpu().numpy(), preds, average="macro", zero_division=0)


def run_fcn(X_a, X_b, y, n_classes, pca_dim, n_folds, balanced=False):
    """Run FCN across k-folds."""
    min_count = min(Counter(y).values())
    actual_folds = min(n_folds, min_count)
    if actual_folds < 2:
        return None, None, None

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_a, y)):
        f1 = train_fcn_fold(
            X_a[train_idx], X_b[train_idx], y[train_idx],
            X_a[val_idx], X_b[val_idx], y[val_idx],
            n_classes=n_classes, pca_dim=pca_dim,
            balanced=balanced, epochs=EPOCHS, patience=PATIENCE,
            fold_idx=fold_idx,
        )
        fold_scores.append(f1)

    return np.mean(fold_scores), np.std(fold_scores), fold_scores


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Fusion Comparison")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10],
                        help="Family size thresholds (default: 10)")
    parser.add_argument("--pca_dims", nargs="+", type=int, default=PCA_DIMS,
                        help=f"PCA dims to sweep (default: {PCA_DIMS})")
    parser.add_argument("--pairs", nargs="+", type=str, default=None,
                        help="Specific pairs to run (default: all)")
    parser.add_argument("--methods", nargs="+", type=str,
                        default=["concat_svm", "concat_lr", "fcn", "fcn_balanced"],
                        help="Methods to run")
    parser.add_argument("--k_folds", type=int, default=K_FOLDS,
                        help=f"Number of CV folds (default: {K_FOLDS})")
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)
    pca_dims = sorted(args.pca_dims)
    k = args.k_folds
    pairs_to_run = args.pairs if args.pairs else list(PAIRS.keys())

    print("\n" + "=" * 120)
    print("  COMPREHENSIVE FUSION COMPARISON — All Pairs × All Methods × PCA Sweep")
    print("  PCA fitted inside each fold (NO data leakage)")
    print("=" * 120)
    print(f"\n  Pairs ({len(pairs_to_run)}): {pairs_to_run}")
    print(f"  Thresholds: {thresholds}")
    print(f"  K-Folds: {k}")
    print(f"  PCA dims: {pca_dims}")
    print(f"  Methods: {args.methods}")
    print(f"  Concat+SVM: C=100, RBF, gamma=scale, balanced, StandardScaler")
    print(f"  Concat+LR:  C=1000, balanced, lbfgs, L2 Normalizer")
    print(f"  FCN: concat→{HIDDEN_DIM}→ReLU→Drop({DROPOUT})→n_classes, epochs={EPOCHS}, patience={PATIENCE}")
    print()

    all_results = []
    total_start = time.time()

    for threshold in thresholds:
        print(f"\n{'━' * 120}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'━' * 120}")

        for pair_name in pairs_to_run:
            if pair_name not in PAIRS:
                print(f"\n  [{pair_name}] NOT FOUND — skipping")
                continue

            pair_cfg = PAIRS[pair_name]
            print(f"\n  [{pair_name}] Loading... ", end="", flush=True)

            X_a, y_a, paths_a = load_embeddings(pair_cfg["modA"])
            X_b, y_b, paths_b = load_embeddings(pair_cfg["modB"])

            if X_a is None or X_b is None:
                print("FILE NOT FOUND")
                continue

            # Align
            is_aligned = pair_cfg.get("aligned", False)
            X_a, X_b, y = align_pairs(X_a, y_a, paths_a, X_b, y_b, paths_b, is_aligned)
            if y is None:
                print("ALIGNMENT FAILED")
                continue

            print(f"{X_a.shape[0]} samples ({pair_cfg['modA']['name']}:{X_a.shape[1]}d + "
                  f"{pair_cfg['modB']['name']}:{X_b.shape[1]}d)")

            # Filter
            X_a_f, X_b_f, y_f, n_classes = filter_by_threshold(X_a, X_b, y, threshold)
            if X_a_f is None or n_classes < 2:
                print(f"    SKIPPED — not enough classes")
                continue

            print(f"    After filtering: {len(y_f)} samples, {n_classes} classes")

            # Sweep PCA dims × methods
            for pca_dim in pca_dims:
                # Skip if PCA dim exceeds modality dimensions
                if pca_dim >= X_a_f.shape[1] and pca_dim >= X_b_f.shape[1]:
                    print(f"\n    PCA={pca_dim}: skipped (exceeds embedding dims)")
                    continue

                print(f"\n    PCA={pca_dim}:")

                # --- Concat + SVM ---
                if "concat_svm" in args.methods:
                    print(f"      [Concat+SVM] ", end="", flush=True)
                    t0 = time.time()
                    mean_f1, std_f1, folds = run_concat_gridsearch(
                        X_a_f, X_b_f, y_f, n_classes, pca_dim, n_folds=k, classifier="svm")
                    elapsed = time.time() - t0
                    if mean_f1 is not None:
                        folds_str = " ".join(f"{s:.4f}" for s in folds)
                        print(f"{mean_f1:.4f} ±{std_f1:.4f} ({elapsed:.0f}s) [{folds_str}]")
                        all_results.append({
                            "pair": pair_name, "method": "Concat+SVM", "pca_dim": pca_dim,
                            "k": k, "threshold": threshold, "n_classes": n_classes,
                            "n_samples": len(y_f), "mean_f1": mean_f1,
                            "std_f1": std_f1, "fold_scores": folds,
                        })
                    else:
                        print("SKIPPED")

                # --- Concat + LR ---
                if "concat_lr" in args.methods:
                    print(f"      [Concat+LR]  ", end="", flush=True)
                    t0 = time.time()
                    mean_f1, std_f1, folds = run_concat_gridsearch(
                        X_a_f, X_b_f, y_f, n_classes, pca_dim, n_folds=k, classifier="lr")
                    elapsed = time.time() - t0
                    if mean_f1 is not None:
                        folds_str = " ".join(f"{s:.4f}" for s in folds)
                        print(f"{mean_f1:.4f} ±{std_f1:.4f} ({elapsed:.0f}s) [{folds_str}]")
                        all_results.append({
                            "pair": pair_name, "method": "Concat+LR", "pca_dim": pca_dim,
                            "k": k, "threshold": threshold, "n_classes": n_classes,
                            "n_samples": len(y_f), "mean_f1": mean_f1,
                            "std_f1": std_f1, "fold_scores": folds,
                        })
                    else:
                        print("SKIPPED")

                # --- FCN ---
                if "fcn" in args.methods:
                    print(f"      [FCN]        ", end="", flush=True)
                    t0 = time.time()
                    mean_f1, std_f1, folds = run_fcn(
                        X_a_f, X_b_f, y_f, n_classes, pca_dim, n_folds=k, balanced=False)
                    elapsed = time.time() - t0
                    if mean_f1 is not None:
                        folds_str = " ".join(f"{s:.4f}" for s in folds)
                        print(f"{mean_f1:.4f} ±{std_f1:.4f} ({elapsed:.0f}s) [{folds_str}]")
                        all_results.append({
                            "pair": pair_name, "method": "FCN", "pca_dim": pca_dim,
                            "k": k, "threshold": threshold, "n_classes": n_classes,
                            "n_samples": len(y_f), "mean_f1": mean_f1,
                            "std_f1": std_f1, "fold_scores": folds,
                        })
                    else:
                        print("SKIPPED")

                # --- FCN Balanced ---
                if "fcn_balanced" in args.methods:
                    print(f"      [FCN_Bal]    ", end="", flush=True)
                    t0 = time.time()
                    mean_f1, std_f1, folds = run_fcn(
                        X_a_f, X_b_f, y_f, n_classes, pca_dim, n_folds=k, balanced=True)
                    elapsed = time.time() - t0
                    if mean_f1 is not None:
                        folds_str = " ".join(f"{s:.4f}" for s in folds)
                        print(f"{mean_f1:.4f} ±{std_f1:.4f} ({elapsed:.0f}s) [{folds_str}]")
                        all_results.append({
                            "pair": pair_name, "method": "FCN_Balanced", "pca_dim": pca_dim,
                            "k": k, "threshold": threshold, "n_classes": n_classes,
                            "n_samples": len(y_f), "mean_f1": mean_f1,
                            "std_f1": std_f1, "fold_scores": folds,
                        })
                    else:
                        print("SKIPPED")

    total_elapsed = time.time() - total_start

    # =====================================================
    # FINAL RESULTS TABLE
    # =====================================================

    print(f"\n\n{'=' * 120}")
    print(f"  FINAL RESULTS — COMPREHENSIVE FUSION COMPARISON (total: {total_elapsed:.0f}s)")
    print(f"{'=' * 120}")
    print(f"{'Pair':<20} | {'Method':<13} | {'PCA':<4} | {'Thresh':<6} | {'Classes':<7} | "
          f"{'Macro-F1':<9} | {'Std':<6}")
    print(f"{'─' * 120}")

    for r in sorted(all_results, key=lambda x: (-x["threshold"], -x["mean_f1"])):
        print(
            f"{r['pair']:<20} | {r['method']:<13} | {r['pca_dim']:<4} | {r['threshold']:<6} | "
            f"{r['n_classes']:<7} | {r['mean_f1']:<9.4f} | {r['std_f1']:<6.4f}"
        )

    print(f"{'=' * 120}")

    # =====================================================
    # BEST PER PAIR (across all PCA dims and methods)
    # =====================================================

    print(f"\n  === BEST METHOD PER PAIR (across PCA dims & methods) ===")
    print(f"  {'─' * 90}")
    print(f"  {'Pair':<20} | {'Method':<13} | {'PCA':<4} | {'Macro-F1':<9} | {'Std':<6}")
    print(f"  {'─' * 90}")
    for pair_name in pairs_to_run:
        subset = [r for r in all_results if r["pair"] == pair_name]
        if subset:
            best = max(subset, key=lambda x: x["mean_f1"])
            print(f"  {pair_name:<20} | {best['method']:<13} | {best['pca_dim']:<4} | "
                  f"{best['mean_f1']:<9.4f} | {best['std_f1']:<6.4f}")

    # =====================================================
    # BEST PER PCA DIM
    # =====================================================

    print(f"\n  === BEST RESULT PER PCA DIM ===")
    print(f"  {'─' * 90}")
    for pca_dim in pca_dims:
        subset = [r for r in all_results if r["pca_dim"] == pca_dim]
        if subset:
            best = max(subset, key=lambda x: x["mean_f1"])
            print(f"  PCA={pca_dim:<3}: {best['pair']:<20} {best['method']:<13} → "
                  f"{best['mean_f1']:.4f} ±{best['std_f1']:.4f}")

    # =====================================================
    # TOP 10 OVERALL
    # =====================================================

    print(f"\n  === TOP 10 RESULTS OVERALL ===")
    print(f"  {'─' * 105}")
    print(f"  {'Rank':<4} | {'Pair':<20} | {'Method':<13} | {'PCA':<4} | {'Thresh':<6} | {'Classes':<7} | {'Macro-F1':<9} | {'Std':<6}")
    print(f"  {'─' * 105}")
    top10 = sorted(all_results, key=lambda x: -x["mean_f1"])[:10]
    for i, r in enumerate(top10, 1):
        print(f"  {i:<4} | {r['pair']:<20} | {r['method']:<13} | {r['pca_dim']:<4} | "
              f"{r['threshold']:<6} | {r['n_classes']:<7} | {r['mean_f1']:<9.4f} | {r['std_f1']:<6.4f}")

    # =====================================================
    # vs SINGLE-MODALITY BASELINES
    # =====================================================

    print(f"\n  === vs SINGLE-MODALITY BASELINES (SVM-RBF best) ===")
    print(f"  {'─' * 90}")
    baselines = {
        10: {"HuBERT": 0.6870, "DINOv2": 0.6810, "ConvNeXt": 0.6578, "Swin": 0.6470,
             "EfficientNet": 0.6463, "MobileNet": 0.6434, "VGG19": 0.6300, "ViT": 0.6155,
             "WavLM": 0.6286, "Wav2Vec2": 0.5727},
        8: {"HuBERT": 0.6831, "DINOv2": 0.6816, "ConvNeXt": 0.6512, "Swin": 0.6340,
            "EfficientNet": 0.6373, "MobileNet": 0.6373, "VGG19": 0.6145, "ViT": 0.6125,
            "WavLM": 0.6189, "Wav2Vec2": 0.5663},
        5: {"HuBERT": 0.5970, "DINOv2": 0.5845, "ConvNeXt": 0.5613, "Swin": 0.5547,
            "EfficientNet": 0.5553, "MobileNet": 0.5507, "VGG19": 0.5393, "ViT": 0.5314,
            "WavLM": 0.5399, "Wav2Vec2": 0.4948},
    }

    for threshold in thresholds:
        if threshold not in baselines:
            continue
        print(f"\n  Threshold >= {threshold}:")
        for pair_name in pairs_to_run:
            subset = [r for r in all_results if r["pair"] == pair_name and r["threshold"] == threshold]
            if not subset:
                continue
            best = max(subset, key=lambda x: x["mean_f1"])
            # Get best single-modality baseline for constituent models
            pair_models = pair_name.split("+")
            best_single = max(baselines[threshold].get(m, 0) for m in pair_models)
            diff = best["mean_f1"] - best_single
            arrow = "↑" if diff > 0 else "↓"
            print(f"    {pair_name:<20}: {best['mean_f1']:.4f} vs best_single={best_single:.4f} "
                  f"({arrow}{abs(diff):.4f}) [{best['method']}, PCA={best['pca_dim']}]")

    print(f"\n  Total time: {total_elapsed:.0f}s")
    print()


if __name__ == "__main__":
    main()
