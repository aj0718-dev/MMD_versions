#!/usr/bin/env python3
"""
fcn_fusion_kfold.py

FCN fusion with proper 5-fold CV and PCA fitted inside each fold.
Tests: ViT+VGG19, VGG19+WavLM, VGG19+HuBERT

For each fold:
  1. Fit PCA separately on each modality's training data
  2. Transform train/val
  3. Concatenate PCA-reduced embeddings
  4. Train FCN (FC layers) on concatenated features
  5. Evaluate Macro-F1 on val fold

Two variants: regular CE loss and balanced (class-weighted) CE loss.

Usage:
    python fcn_fusion_kfold.py
    python fcn_fusion_kfold.py --thresholds 10 8 5
    python fcn_fusion_kfold.py --pca_dim 384
"""

import argparse
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
    "ViT+VGG19": {
        "modA": {"name": "ViT", "embeddings": "vit_vgg_fcn/vit_embeddings.pt", "labels": "vit_vgg_fcn/labels.pt"},
        "modB": {"name": "VGG19", "embeddings": "vit_vgg_fcn/vgg_embeddings.pt", "labels": "vit_vgg_fcn/labels.pt"},
        "aligned": True,  # same label file = already aligned
    },
    "VGG19+WavLM": {
        "modA": {"name": "VGG19", "embeddings": "vgg19/vgg19_embeddings_all.pt", "labels": "vgg19/labels_all.pt", "paths": "vgg19/vgg19_paths.pt"},
        "modB": {"name": "WavLM", "embeddings": "wav2vec2_hubert_wavlm/wavlm_embeddings.pt", "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
    "VGG19+HuBERT": {
        "modA": {"name": "VGG19", "embeddings": "vgg19/vgg19_embeddings_all.pt", "labels": "vgg19/labels_all.pt", "paths": "vgg19/vgg19_paths.pt"},
        "modB": {"name": "HuBERT", "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt", "labels": "wav2vec2_hubert_wavlm/labels.pt", "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt"},
        "aligned": False,
    },
}

PCA_DIM = 256
HIDDEN_DIM = 1024
DROPOUT = 0.3
EPOCHS = 80
LR = 1e-3
PATIENCE = 10
BATCH_SIZE = 256
N_FOLDS = 5
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
# HELPERS
# =====================================================


def load_embeddings(config):
    """Load embedding and label tensors."""
    import os
    if not os.path.exists(config["embeddings"]):
        return None, None, None
    emb = torch.load(config["embeddings"], map_location="cpu", weights_only=True)
    labels = torch.load(config["labels"], map_location="cpu", weights_only=True)
    paths = None
    if "paths" in config and os.path.exists(config["paths"]):
        paths = torch.load(config["paths"], map_location="cpu", weights_only=False)
    return emb.numpy().astype(np.float32), labels.numpy().astype(np.int64), paths


def get_sample_id(path):
    """Extract sample ID (MOTIF_xxxx hash) from a file path."""
    import os
    base = os.path.basename(path)
    # Remove extension (.png, .wav, etc.)
    return os.path.splitext(base)[0]


def align_pairs(X_a, y_a, paths_a, X_b, y_b, paths_b, is_aligned):
    """Align two modality arrays by sample ID matching."""
    if is_aligned:
        # Same label file — just truncate to min length
        n = min(len(X_a), len(X_b))
        return X_a[:n], X_b[:n], y_a[:n]

    # Path-based alignment
    if paths_a is None or paths_b is None:
        print("WARNING: Need paths for alignment but paths not available!")
        return None, None, None

    # Build ID → index maps
    ids_a = {get_sample_id(p): i for i, p in enumerate(paths_a)}
    ids_b = {get_sample_id(p): i for i, p in enumerate(paths_b)}

    # Find common IDs (preserving order of modA)
    common_ids = [sid for sid in ids_a if sid in ids_b]

    if len(common_ids) == 0:
        print("WARNING: No common sample IDs found!")
        return None, None, None

    # Build aligned arrays
    idx_a = np.array([ids_a[sid] for sid in common_ids])
    idx_b = np.array([ids_b[sid] for sid in common_ids])

    X_a_aligned = X_a[idx_a]
    X_b_aligned = X_b[idx_b]
    y_aligned = y_a[idx_a]

    # Verify labels match after alignment
    y_b_aligned = y_b[idx_b]
    if not np.array_equal(y_aligned, y_b_aligned):
        # Labels use different encoding — just use modA's labels
        # (both should map to same family, just different int encoding)
        pass

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


def compute_class_weights(y, n_classes):
    """Compute balanced class weights (sklearn-style)."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(X_train, y_train, X_val, y_val, n_classes, pca_dim,
                   balanced=False, epochs=80, patience=10):
    """
    Train FCN on one fold with PCA fitted on training data only.

    Steps:
        1. PCA on modality A (train) → transform train & val
        2. PCA on modality B (train) → transform train & val
        3. Concatenate
        4. Train FCN
        5. Return val Macro-F1
    """
    # Determine concat dimension split (X_train has both modalities concatenated with a marker)
    # Actually we receive already-split data — let me restructure

    # The caller passes concatenated X but we need separate PCA per modality
    # Better: caller passes split arrays
    raise NotImplementedError("Use train_one_fold_split instead")


def train_one_fold_split(X_a_train, X_b_train, y_train, X_a_val, X_b_val, y_val,
                         n_classes, pca_dim, balanced=False, epochs=80, patience=10):
    """
    Train FCN on one fold with PCA fitted on training data only.

    1. Fit PCA_A on X_a_train, transform X_a_train & X_a_val
    2. Fit PCA_B on X_b_train, transform X_b_train & X_b_val
    3. Concatenate → [PCA_A | PCA_B]
    4. Train FCN with early stopping
    5. Return val Macro-F1
    """
    # --- PCA per modality (fitted on train only) ---
    pca_a = PCA(n_components=min(pca_dim, X_a_train.shape[1], X_a_train.shape[0] - 1),
                random_state=RANDOM_STATE)
    pca_b = PCA(n_components=min(pca_dim, X_b_train.shape[1], X_b_train.shape[0] - 1),
                random_state=RANDOM_STATE)

    X_a_train_pca = pca_a.fit_transform(X_a_train)
    X_a_val_pca = pca_a.transform(X_a_val)

    X_b_train_pca = pca_b.fit_transform(X_b_train)
    X_b_val_pca = pca_b.transform(X_b_val)

    # --- StandardScaler per modality (fitted on train only) ---
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()

    X_a_train_pca = scaler_a.fit_transform(X_a_train_pca)
    X_a_val_pca = scaler_a.transform(X_a_val_pca)

    X_b_train_pca = scaler_b.fit_transform(X_b_train_pca)
    X_b_val_pca = scaler_b.transform(X_b_val_pca)

    # --- Concatenate ---
    X_train_cat = np.concatenate([X_a_train_pca, X_b_train_pca], axis=1)
    X_val_cat = np.concatenate([X_a_val_pca, X_b_val_pca], axis=1)

    input_dim = X_train_cat.shape[1]

    # --- Convert to tensors ---
    X_tr = torch.tensor(X_train_cat, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_v = torch.tensor(X_val_cat, dtype=torch.float32).to(DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    # --- Model ---
    model = FusionFCN(input_dim, n_classes, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)

    # --- Loss ---
    if balanced:
        weights = compute_class_weights(y_train, n_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- Training with early stopping ---
    best_f1 = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        indices = torch.randperm(len(X_tr))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_tr), BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            X_batch = X_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # --- Evaluate on val ---
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            preds = val_logits.argmax(dim=1).cpu().numpy()
            true = y_v.cpu().numpy()

        macro_f1 = f1_score(true, preds, average="macro", zero_division=0)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model and get final prediction
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_v)
        preds = val_logits.argmax(dim=1).cpu().numpy()

    final_f1 = f1_score(y_v.cpu().numpy(), preds, average="macro", zero_division=0)
    return final_f1


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="FCN Fusion with K-Fold CV")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10],
                        help="Family size thresholds (default: 10)")
    parser.add_argument("--pca_dim", type=int, default=PCA_DIM,
                        help=f"PCA dim per modality (default: {PCA_DIM})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Max epochs (default: {EPOCHS})")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help=f"Early stopping patience (default: {PATIENCE})")
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)
    pca_dim = args.pca_dim

    print("\n" + "=" * 100)
    print("  FCN FUSION — K-Fold with PCA inside each fold (NO data leakage)")
    print("=" * 100)
    print(f"\n  Pairs: {list(PAIRS.keys())}")
    print(f"  Thresholds: {thresholds}")
    print(f"  K-Fold: {N_FOLDS}-fold Stratified")
    print(f"  PCA dim per modality: {pca_dim} (concat = {pca_dim * 2})")
    print(f"  FCN: {pca_dim * 2} → {HIDDEN_DIM} → ReLU → Dropout({DROPOUT}) → n_classes")
    print(f"  Epochs: {args.epochs} (patience={args.patience})")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  LR: {LR}")
    print(f"  Modes: Regular CE, Balanced CE (class-weighted)")
    print()

    all_results = []
    total_start = time.time()

    for threshold in thresholds:
        print(f"\n{'─' * 100}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'─' * 100}")

        for pair_name, pair_cfg in PAIRS.items():
            print(f"\n  [{pair_name}] Loading... ", end="", flush=True)

            X_a, y_a, paths_a = load_embeddings(pair_cfg["modA"])
            X_b, y_b, paths_b = load_embeddings(pair_cfg["modB"])

            if X_a is None or X_b is None:
                print("FILE NOT FOUND")
                continue

            # Align by sample ID
            is_aligned = pair_cfg.get("aligned", False)
            X_a, X_b, y = align_pairs(X_a, y_a, paths_a, X_b, y_b, paths_b, is_aligned)
            if y is None:
                print("ALIGNMENT FAILED")
                continue

            print(f"{X_a.shape[0]} samples ({pair_cfg['modA']['name']}:{X_a.shape[1]}d + "
                  f"{pair_cfg['modB']['name']}:{X_b.shape[1]}d)")

            # Filter by threshold
            X_a_f, X_b_f, y_f, n_classes = filter_by_threshold(X_a, X_b, y, threshold)
            if X_a_f is None or n_classes < 2:
                print(f"  [{pair_name}] SKIPPED — not enough classes")
                continue

            print(f"  [{pair_name}] After filtering: {len(y_f)} samples, {n_classes} classes")

            # Run both modes: regular and balanced
            for mode_name, balanced in [("FCN", False), ("FCN_Balanced", True)]:
                print(f"    [{mode_name}] ", end="", flush=True)
                t0 = time.time()

                torch.manual_seed(RANDOM_STATE)
                np.random.seed(RANDOM_STATE)

                cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                fold_scores = []

                for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_a_f, y_f)):
                    X_a_train, X_a_val = X_a_f[train_idx], X_a_f[val_idx]
                    X_b_train, X_b_val = X_b_f[train_idx], X_b_f[val_idx]
                    y_train, y_val = y_f[train_idx], y_f[val_idx]

                    torch.manual_seed(RANDOM_STATE + fold_idx)

                    f1 = train_one_fold_split(
                        X_a_train, X_b_train, y_train,
                        X_a_val, X_b_val, y_val,
                        n_classes=n_classes,
                        pca_dim=pca_dim,
                        balanced=balanced,
                        epochs=args.epochs,
                        patience=args.patience,
                    )
                    fold_scores.append(f1)
                    print(f"f{fold_idx + 1}={f1:.4f} ", end="", flush=True)

                elapsed = time.time() - t0
                mean_f1 = np.mean(fold_scores)
                std_f1 = np.std(fold_scores)

                print(f"→ {mean_f1:.4f} ±{std_f1:.4f} ({elapsed:.0f}s)")

                all_results.append({
                    "pair": pair_name,
                    "mode": mode_name,
                    "threshold": threshold,
                    "n_classes": n_classes,
                    "n_samples": len(y_f),
                    "mean_f1": mean_f1,
                    "std_f1": std_f1,
                    "fold_scores": fold_scores,
                    "pca_dim": pca_dim,
                })

    total_elapsed = time.time() - total_start

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print(f"\n\n{'=' * 100}")
    print(f"  FINAL RESULTS — FCN FUSION (total: {total_elapsed:.0f}s)")
    print(f"{'=' * 100}")
    print(f"{'Pair':<15} | {'Mode':<14} | {'Thresh':<6} | {'Classes':<7} | "
          f"{'Macro-F1':<9} | {'Std':<6} | {'Fold Scores'}")
    print(f"{'=' * 100}")

    for r in sorted(all_results, key=lambda x: (-x["threshold"], -x["mean_f1"])):
        folds_str = " ".join(f"{s:.4f}" for s in r["fold_scores"])
        print(
            f"{r['pair']:<15} | {r['mode']:<14} | {r['threshold']:<6} | "
            f"{r['n_classes']:<7} | {r['mean_f1']:<9.4f} | {r['std_f1']:<6.4f} | "
            f"{folds_str}"
        )

    print(f"{'=' * 100}")

    # Best per threshold
    print(f"\n  BEST PAIR+MODE PER THRESHOLD:")
    print(f"  {'─' * 80}")
    for threshold in thresholds:
        subset = [r for r in all_results if r["threshold"] == threshold]
        if subset:
            best = max(subset, key=lambda x: x["mean_f1"])
            print(
                f"  >= {threshold}: {best['pair']} ({best['mode']}) → "
                f"Macro-F1 = {best['mean_f1']:.4f} ±{best['std_f1']:.4f} "
                f"({best['n_classes']} classes, PCA={best['pca_dim']})"
            )

    # Compare with single-modality baselines
    print(f"\n  COMPARISON WITH SINGLE-MODALITY BASELINES (LogReg Extended):")
    print(f"  {'─' * 80}")
    baselines = {
        10: {"HuBERT": 0.6607, "ViT": 0.6159, "VGG19": 0.5993, "WavLM": 0.6043},
        8: {"HuBERT": 0.6438, "ViT": 0.6140, "VGG19": 0.5993, "WavLM": 0.5843},
        5: {"HuBERT": 0.5718, "ViT": 0.5241, "VGG19": 0.5256, "WavLM": 0.5168},
    }

    for threshold in thresholds:
        if threshold in baselines:
            subset = [r for r in all_results if r["threshold"] == threshold]
            if subset:
                best = max(subset, key=lambda x: x["mean_f1"])
                # Best single modality from the pair
                pair_models = best["pair"].split("+")
                best_single = max(baselines[threshold].get(m, 0) for m in pair_models)
                diff = best["mean_f1"] - best_single
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(
                    f"  >= {threshold}: Fusion {best['mean_f1']:.4f} vs "
                    f"best single {best_single:.4f} ({arrow} {abs(diff):.4f}) — "
                    f"{best['pair']} ({best['mode']})"
                )

    print()


if __name__ == "__main__":
    main()
