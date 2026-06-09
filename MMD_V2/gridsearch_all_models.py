#!/usr/bin/env python3
"""
gridsearch_all_models.py

Runs GridSearchCV with Pipeline(PCA → Normalize → LogisticRegression)
on all 6 pretrained model embeddings with K-Fold cross-validation.

PCA is fitted INSIDE each fold (no data leakage).

Configurable family threshold: only keeps families with >= N samples.

Usage:
    python gridsearch_all_models.py
    python gridsearch_all_models.py --thresholds 10 8 5
"""

import argparse
import time
import warnings
from collections import Counter

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

warnings.filterwarnings("ignore")


# =====================================================
# CONFIG
# =====================================================

MODELS = {
    "ViT": {
        "embeddings": "vit_vgg_fcn/vit_embeddings.pt",
        "labels": "vit_vgg_fcn/labels.pt",
    },
    "VGG19": {
        "embeddings": "vgg19/vgg19_embeddings_all.pt",
        "labels": "vgg19/labels_all.pt",
    },
    "RegNet": {
        "embeddings": "regnety_040/regnet_embeddings_all.pt",
        "labels": "regnety_040/labels_all.pt",
    },
    "Wav2Vec2": {
        "embeddings": "wav2vec2_hubert_wavlm/wav2vec2_embeddings.pt",
        "labels": "wav2vec2_hubert_wavlm/labels.pt",
    },
    "HuBERT": {
        "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
        "labels": "wav2vec2_hubert_wavlm/labels.pt",
    },
    "WavLM": {
        "embeddings": "wav2vec2_hubert_wavlm/wavlm_embeddings.pt",
        "labels": "wav2vec2_hubert_wavlm/labels.pt",
    },
}

# GridSearch parameter grid
PARAM_GRID = {
    "pca__n_components": [64, 128, 256],
    "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "clf__solver": ["lbfgs", "saga"],
    "clf__class_weight": [None, "balanced"],
}

N_FOLDS = 5
RANDOM_STATE = 42
SCORING = "f1_macro"


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def load_embeddings(model_config):
    """Load embedding and label tensors, convert to numpy."""
    import os
    if not os.path.exists(model_config["embeddings"]):
        return None, None
    if not os.path.exists(model_config["labels"]):
        return None, None

    emb = torch.load(model_config["embeddings"], map_location="cpu", weights_only=True)
    labels = torch.load(model_config["labels"], map_location="cpu", weights_only=True)

    X = emb.numpy().astype(np.float32)
    y = labels.numpy().astype(np.int64)

    return X, y


def filter_by_threshold(X, y, threshold):
    """
    Keep only samples whose class has >= threshold samples.
    Remap labels to contiguous integers.
    """
    counts = Counter(y)
    keep_classes = {cls for cls, cnt in counts.items() if cnt >= threshold}

    if not keep_classes:
        return None, None, 0

    mask = np.array([label in keep_classes for label in y])
    X_filtered = X[mask]
    y_filtered = y[mask]

    # Remap to contiguous labels (required for stratified k-fold)
    unique_labels = sorted(set(y_filtered))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[label] for label in y_filtered])

    return X_filtered, y_remapped, len(unique_labels)


def adjust_pca_components(param_grid, n_samples, n_features, n_folds=5):
    """
    Ensure PCA n_components doesn't exceed min(n_train_samples, n_features).
    n_train_samples = n_samples * (n_folds-1) / n_folds (smallest training fold).
    """
    # In k-fold, training set size is n_samples * (k-1)/k
    n_train = int(n_samples * (n_folds - 1) / n_folds)
    max_components = min(n_train, n_features) - 1  # -1 for safety

    valid_components = [c for c in param_grid["pca__n_components"] if c < max_components]

    if not valid_components:
        # Use something reasonable
        valid_components = [min(32, max_components - 1)]
        if valid_components[0] < 2:
            return None  # Too few samples

    adjusted = param_grid.copy()
    adjusted["pca__n_components"] = valid_components
    return adjusted


def run_gridsearch(X, y, param_grid, n_folds=5):
    """
    Run GridSearchCV with Pipeline(PCA → Normalize → LogisticRegression).
    PCA is fitted inside each fold = no data leakage.
    """
    # Check minimum samples per class for k-fold
    min_count = min(Counter(y).values())
    actual_folds = min(n_folds, min_count)

    if actual_folds < 2:
        return None

    # Adjust PCA components if needed
    param_grid = adjust_pca_components(param_grid, X.shape[0], X.shape[1], n_folds=actual_folds)

    if param_grid is None:
        return None

    # Build pipeline
    pipe = Pipeline([
        ("pca", PCA(random_state=RANDOM_STATE)),
        ("norm", Normalizer(norm="l2")),
        ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
    ])

    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_STATE)

    # GridSearchCV
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )

    grid.fit(X, y)

    return grid


def print_separator():
    print("=" * 90)


def print_results_table(results):
    """Print a formatted comparison table."""
    print_separator()
    print(f"{'Model':<10} | {'Threshold':<5} | {'Classes':<7} | {'Samples':<7} | "
          f"{'Best Macro-F1':<13} | {'Std':<6} | {'Best Params'}")
    print_separator()

    for r in results:
        params_str = (
            f"PCA={r['pca']}, C={r['C']}, "
            f"solver={r['solver']}, weight={r['class_weight']}"
        )
        print(
            f"{r['model']:<10} | {r['threshold']:<5} | {r['n_classes']:<7} | "
            f"{r['n_samples']:<7} | {r['best_score']:<13.4f} | "
            f"{r['std']:<6.4f} | {params_str}"
        )

    print_separator()


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="GridSearchCV on all 6 models")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[10, 8, 5],
        help="Family size thresholds to try (default: 10 8 5)",
    )
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)  # start with strictest

    print("\n" + "=" * 90)
    print("  MULTIMODAL MALWARE DETECTION — GridSearchCV with K-Fold + PCA (No Leakage)")
    print("=" * 90)
    print(f"\n  Models: {list(MODELS.keys())}")
    print(f"  Thresholds: {thresholds}")
    print(f"  K-Fold: {N_FOLDS}-fold Stratified")
    print(f"  Scoring: {SCORING}")
    print(f"  PCA components: {PARAM_GRID['pca__n_components']}")
    print(f"  C values: {PARAM_GRID['clf__C']}")
    print(f"  Solvers: {PARAM_GRID['clf__solver']}")
    print(f"  Class weights: {PARAM_GRID['clf__class_weight']}")
    total_combos = (
        len(PARAM_GRID["pca__n_components"])
        * len(PARAM_GRID["clf__C"])
        * len(PARAM_GRID["clf__solver"])
        * len(PARAM_GRID["clf__class_weight"])
    )
    print(f"  Total param combinations per model: {total_combos}")
    print(f"  Total fits per model: {total_combos} x {N_FOLDS} folds = {total_combos * N_FOLDS}")
    print()

    all_results = []

    for threshold in thresholds:
        print(f"\n{'─' * 90}")
        print(f"  THRESHOLD >= {threshold} samples per family")
        print(f"{'─' * 90}")

        for model_name, config in MODELS.items():
            print(f"\n  [{model_name}] Loading embeddings... ", end="", flush=True)

            X, y = load_embeddings(config)

            if X is None:
                print(f"FILE NOT FOUND — run extraction script first")
                print(f"    Missing: {config['embeddings']}")
                continue

            print(f"raw shape: {X.shape}, {len(np.unique(y))} classes")

            # Filter by threshold
            X_filt, y_filt, n_classes = filter_by_threshold(X, y, threshold)

            if X_filt is None or n_classes < 2:
                print(f"  [{model_name}] SKIPPED — not enough classes after filtering")
                continue

            print(f"  [{model_name}] After filtering: {X_filt.shape[0]} samples, {n_classes} classes")

            # Run GridSearchCV
            print(f"  [{model_name}] Running GridSearchCV... ", end="", flush=True)
            t0 = time.time()

            grid = run_gridsearch(X_filt, y_filt, PARAM_GRID, n_folds=N_FOLDS)

            elapsed = time.time() - t0

            if grid is None:
                print(f"SKIPPED (some classes have <2 samples)")
                continue

            best = grid.best_params_
            best_score = grid.best_score_
            best_std = grid.cv_results_["std_test_score"][grid.best_index_]

            print(f"done in {elapsed:.1f}s")
            print(f"  [{model_name}] Best Macro-F1: {best_score:.4f} (+/- {best_std:.4f})")
            print(f"  [{model_name}] Best params: {best}")

            all_results.append({
                "model": model_name,
                "threshold": threshold,
                "n_classes": n_classes,
                "n_samples": X_filt.shape[0],
                "best_score": best_score,
                "std": best_std,
                "pca": best["pca__n_components"],
                "C": best["clf__C"],
                "solver": best["clf__solver"],
                "class_weight": best["clf__class_weight"],
            })

    # =====================================================
    # FINAL COMPARISON TABLE
    # =====================================================

    print("\n\n")
    print("  FINAL COMPARISON TABLE")
    print()
    print_results_table(all_results)

    # =====================================================
    # BEST MODEL PER THRESHOLD
    # =====================================================

    print("\n  BEST MODEL PER THRESHOLD:")
    print("  " + "-" * 60)

    for threshold in thresholds:
        subset = [r for r in all_results if r["threshold"] == threshold]
        if subset:
            best = max(subset, key=lambda x: x["best_score"])
            print(
                f"  Threshold >= {threshold}: {best['model']} "
                f"(Macro-F1 = {best['best_score']:.4f}, "
                f"{best['n_classes']} classes, {best['n_samples']} samples)"
            )

    print()


if __name__ == "__main__":
    main()
