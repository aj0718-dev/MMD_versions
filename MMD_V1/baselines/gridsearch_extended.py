#!/usr/bin/env python3
"""
gridsearch_extended.py

Extended grid search for the top 4 models (HuBERT, ViT, VGG19, WavLM)
with wider hyperparameter ranges beyond the initial search boundaries.

Uses Pipeline(PCA → Normalize → LogisticRegression) with StratifiedKFold.
PCA is fitted INSIDE each fold — no data leakage.

Usage:
    python gridsearch_extended.py
    python gridsearch_extended.py --thresholds 10 8 5
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
# CONFIG — TOP 4 MODELS ONLY
# =====================================================

MODELS = {
    "HuBERT": {
        "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
        "labels": "wav2vec2_hubert_wavlm/labels.pt",
    },
    "ViT": {
        "embeddings": "vit_vgg_fcn/vit_embeddings.pt",
        "labels": "vit_vgg_fcn/labels.pt",
    },
    "VGG19": {
        "embeddings": "vit_vgg_fcn/vgg_embeddings.pt",
        "labels": "vit_vgg_fcn/labels.pt",
    },
    "WavLM": {
        "embeddings": "wav2vec2_hubert_wavlm/wavlm_embeddings.pt",
        "labels": "wav2vec2_hubert_wavlm/labels.pt",
    },
}

# EXTENDED parameter grid — goes beyond initial boundaries
PARAM_GRID = {
    "pca__n_components": [256, 384, 512],
    "clf__C": [100, 500, 1000, 5000],
    "clf__solver": ["lbfgs"],
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

    # Remap to contiguous labels
    unique_labels = sorted(set(y_filtered))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[label] for label in y_filtered])

    return X_filtered, y_remapped, len(unique_labels)


def adjust_pca_components(param_grid, n_samples, n_features, n_folds=5):
    """
    Ensure PCA n_components < min(n_train_samples, n_features).
    Train fold size = n_samples * (n_folds - 1) / n_folds.
    """
    n_train = int(n_samples * (n_folds - 1) / n_folds)
    max_components = min(n_train, n_features) - 1

    valid_components = [c for c in param_grid["pca__n_components"] if c < max_components]

    if not valid_components:
        valid_components = [min(128, max_components - 1)]
        if valid_components[0] < 2:
            return None

    adjusted = param_grid.copy()
    adjusted["pca__n_components"] = valid_components
    return adjusted


def run_gridsearch(X, y, param_grid, n_folds=5):
    """
    Run GridSearchCV with Pipeline(PCA → Normalize → LogisticRegression).

    KEY POINT: PCA is inside the Pipeline, so sklearn fits PCA only on
    the training fold and transforms the validation fold — NO DATA LEAKAGE.
    """
    # Determine actual folds based on minimum class count
    min_count = min(Counter(y).values())
    actual_folds = min(n_folds, min_count)

    if actual_folds < 2:
        return None

    # Adjust PCA components for train fold size
    param_grid = adjust_pca_components(
        param_grid, X.shape[0], X.shape[1], n_folds=actual_folds
    )

    if param_grid is None:
        return None

    # Pipeline: PCA (fit per fold) → L2 Normalize → LogisticRegression
    pipe = Pipeline([
        ("pca", PCA(random_state=RANDOM_STATE)),
        ("norm", Normalizer(norm="l2")),
        ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
    ])

    # Stratified K-Fold ensures each fold has proportional class distribution
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_STATE)

    # GridSearchCV tries every combination and reports best via cross-validation
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    grid.fit(X, y)

    return grid


def print_separator():
    print("=" * 100)


def print_results_table(results):
    """Print a formatted comparison table."""
    print_separator()
    print(
        f"{'Model':<8} | {'Thresh':<6} | {'Classes':<7} | {'Samples':<7} | "
        f"{'Macro-F1':<9} | {'Std':<6} | {'PCA':<4} | {'C':<7} | "
        f"{'Solver':<6} | {'Weight'}"
    )
    print_separator()

    for r in results:
        weight_str = str(r["class_weight"]) if r["class_weight"] else "None"
        print(
            f"{r['model']:<8} | {r['threshold']:<6} | {r['n_classes']:<7} | "
            f"{r['n_samples']:<7} | {r['best_score']:<9.4f} | {r['std']:<6.4f} | "
            f"{r['pca']:<4} | {r['C']:<7} | {r['solver']:<6} | {weight_str}"
        )

    print_separator()


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="Extended GridSearchCV — top 4 models")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[10, 8, 5],
        help="Family size thresholds (default: 10 8 5)",
    )
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)

    print("\n" + "=" * 100)
    print("  EXTENDED GRID SEARCH — HuBERT, ViT, VGG19, WavLM")
    print("  K-Fold with PCA inside Pipeline (NO data leakage)")
    print("=" * 100)
    print(f"\n  Thresholds: {thresholds}")
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
    print(f"\n  Param combinations: {total_combos}")
    print(f"  Fits per model: {total_combos} combos x {N_FOLDS} folds = {total_combos * N_FOLDS}")
    print(f"  Total fits: {total_combos * N_FOLDS} x {len(MODELS)} models x {len(thresholds)} thresholds = {total_combos * N_FOLDS * len(MODELS) * len(thresholds)}")
    print()

    all_results = []
    total_start = time.time()

    for threshold in thresholds:
        print(f"\n{'─' * 100}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'─' * 100}")

        for model_name, config in MODELS.items():
            print(f"\n  [{model_name}] Loading... ", end="", flush=True)

            X, y = load_embeddings(config)
            print(f"shape: {X.shape}", end="")

            # Filter by threshold
            X_filt, y_filt, n_classes = filter_by_threshold(X, y, threshold)

            if X_filt is None or n_classes < 2:
                print(f" → SKIPPED (not enough classes)")
                continue

            print(f" → filtered: {X_filt.shape[0]} samples, {n_classes} classes")

            # Run GridSearchCV
            print(f"  [{model_name}] GridSearchCV running... ", end="", flush=True)
            t0 = time.time()

            grid = run_gridsearch(X_filt, y_filt, PARAM_GRID, n_folds=N_FOLDS)

            elapsed = time.time() - t0

            if grid is None:
                print(f"SKIPPED")
                continue

            best = grid.best_params_
            best_score = grid.best_score_
            best_std = grid.cv_results_["std_test_score"][grid.best_index_]

            print(f"done ({elapsed:.0f}s)")
            print(f"  [{model_name}] ★ Macro-F1: {best_score:.4f} (±{best_std:.4f})")
            print(f"  [{model_name}]   Params: PCA={best['pca__n_components']}, "
                  f"C={best['clf__C']}, solver={best['clf__solver']}, "
                  f"weight={best['clf__class_weight']}")

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

    total_elapsed = time.time() - total_start

    # =====================================================
    # FINAL TABLE
    # =====================================================

    print(f"\n\n  Total time: {total_elapsed / 60:.1f} minutes\n")
    print("  FINAL RESULTS — EXTENDED GRID SEARCH")
    print()
    print_results_table(all_results)

    # Best per threshold
    print("\n  BEST MODEL PER THRESHOLD:")
    print("  " + "-" * 70)
    for threshold in thresholds:
        subset = [r for r in all_results if r["threshold"] == threshold]
        if subset:
            best = max(subset, key=lambda x: x["best_score"])
            print(
                f"  >= {threshold}: {best['model']} → Macro-F1 = {best['best_score']:.4f} "
                f"(PCA={best['pca']}, C={best['C']}, {best['n_classes']} classes)"
            )

    # Check if we're still at grid boundary
    print("\n  BOUNDARY CHECK (did we find true optimum?):")
    print("  " + "-" * 70)
    max_pca = max(PARAM_GRID["pca__n_components"])
    max_c = max(PARAM_GRID["clf__C"])

    for r in all_results:
        at_boundary = []
        if r["pca"] == max_pca:
            at_boundary.append(f"PCA={r['pca']} (max)")
        if r["C"] == max_c:
            at_boundary.append(f"C={r['C']} (max)")
        if at_boundary:
            print(f"  ⚠  {r['model']} (>={r['threshold']}): still at boundary → {', '.join(at_boundary)}")

    no_boundary = [r for r in all_results if r["pca"] != max_pca and r["C"] != max_c]
    if no_boundary:
        print(f"  ✓  {len(no_boundary)} model(s) found optimum within grid")

    print()


if __name__ == "__main__":
    main()
