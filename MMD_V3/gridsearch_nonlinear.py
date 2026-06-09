#!/usr/bin/env python3
"""
gridsearch_nonlinear.py

Tests non-linear classifiers and alternative pipelines to push baselines higher.
Includes: SVM (RBF), MLP, LogReg without PCA, different scalers.

Usage:
    python gridsearch_nonlinear.py --thresholds 10 8 5
    python gridsearch_nonlinear.py --thresholds 10 --models HuBERT ViT
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# =====================================================
# CONFIG
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
        "embeddings": "vgg19/vgg19_embeddings_all.pt",
        "labels": "vgg19/labels_all.pt",
    },
    "WavLM": {
        "embeddings": "wav2vec2_hubert_wavlm/wavlm_embeddings.pt",
        "labels": "wav2vec2_hubert_wavlm/labels.pt",
    },
}

# Multiple pipeline configurations to test
PIPELINES = {
    # 1. LogReg WITHOUT PCA — just scale and classify
    "LogReg_NoPCA": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=42)),
        ]),
        "params": {
            "clf__C": [100, 500, 1000, 5000],
            "clf__class_weight": [None, "balanced"],
        },
    },
    # 2. LogReg with PCA + StandardScaler (instead of L2 norm)
    "LogReg_PCA_StdScale": {
        "pipe": Pipeline([
            ("pca", PCA(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=42)),
        ]),
        "params": {
            "pca__n_components": [384, 512],
            "clf__C": [500, 1000, 5000],
            "clf__class_weight": [None, "balanced"],
        },
    },
    # 3. SVM with RBF kernel — non-linear
    "SVM_RBF": {
        "pipe": Pipeline([
            ("pca", PCA(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42)),
        ]),
        "params": {
            "pca__n_components": [128, 256, 384, 512],
            "clf__C": [1, 10, 50, 100, 500, 1000],
            "clf__gamma": ["scale", "auto"],
            "clf__class_weight": [None, "balanced"],
        },
    },
    # 4. MLP — small neural network
    "MLP": {
        "pipe": Pipeline([
            ("pca", PCA(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(max_iter=500, random_state=42, early_stopping=True)),
        ]),
        "params": {
            "pca__n_components": [384, 512],
            "clf__hidden_layer_sizes": [(256,), (512,), (256, 128)],
            "clf__alpha": [0.0001, 0.001, 0.01],
            "clf__learning_rate_init": [0.001, 0.0005],
        },
    },
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
    """Keep classes with >= threshold samples, remap to contiguous."""
    counts = Counter(y)
    keep_classes = {cls for cls, cnt in counts.items() if cnt >= threshold}

    if not keep_classes:
        return None, None, 0

    mask = np.array([label in keep_classes for label in y])
    X_filtered = X[mask]
    y_filtered = y[mask]

    unique_labels = sorted(set(y_filtered))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[label] for label in y_filtered])

    return X_filtered, y_remapped, len(unique_labels)


def adjust_pca_components(param_grid, n_samples, n_features, n_folds=5):
    """Ensure PCA n_components < min(n_train_samples, n_features)."""
    if "pca__n_components" not in param_grid:
        return param_grid

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


def run_gridsearch(X, y, pipe, param_grid, n_folds=5):
    """Run GridSearchCV with given pipeline and params."""
    min_count = min(Counter(y).values())
    actual_folds = min(n_folds, min_count)

    if actual_folds < 2:
        return None

    # Adjust PCA if present
    param_grid = adjust_pca_components(param_grid, X.shape[0], X.shape[1], n_folds=actual_folds)
    if param_grid is None:
        return None

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_STATE)

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


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="Non-linear GridSearchCV")
    parser.add_argument(
        "--thresholds", nargs="+", type=int, default=[10, 8, 5],
        help="Family size thresholds (default: 10 8 5)",
    )
    parser.add_argument(
        "--models", nargs="+", type=str, default=None,
        help="Models to test (default: all). E.g. --models HuBERT ViT",
    )
    parser.add_argument(
        "--pipelines", nargs="+", type=str, default=None,
        help="Pipelines to test (default: all). E.g. --pipelines SVM_RBF MLP",
    )
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)
    models_to_run = args.models if args.models else list(MODELS.keys())
    pipes_to_run = args.pipelines if args.pipelines else list(PIPELINES.keys())

    print("\n" + "=" * 100)
    print("  NON-LINEAR GRID SEARCH — Pushing Baselines Higher")
    print("  K-Fold with PCA inside Pipeline (NO data leakage)")
    print("=" * 100)
    print(f"\n  Models: {models_to_run}")
    print(f"  Pipelines: {pipes_to_run}")
    print(f"  Thresholds: {thresholds}")
    print(f"  K-Fold: {N_FOLDS}-fold Stratified")
    print(f"  Scoring: {SCORING}")

    for pipe_name in pipes_to_run:
        cfg = PIPELINES[pipe_name]
        n_combos = 1
        for vals in cfg["params"].values():
            n_combos *= len(vals)
        print(f"\n  {pipe_name}: {n_combos} combos × {N_FOLDS} folds = {n_combos * N_FOLDS} fits/model")

    print()

    all_results = []
    total_start = time.time()

    for threshold in thresholds:
        print(f"\n{'─' * 100}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'─' * 100}")

        for model_name in models_to_run:
            if model_name not in MODELS:
                print(f"\n  [{model_name}] NOT FOUND — skipping")
                continue

            config = MODELS[model_name]
            print(f"\n  [{model_name}] Loading... ", end="", flush=True)

            X, y = load_embeddings(config)
            if X is None:
                print("FILE NOT FOUND")
                continue

            X_filt, y_filt, n_classes = filter_by_threshold(X, y, threshold)
            if X_filt is None or n_classes < 2:
                print("SKIPPED (not enough classes)")
                continue

            print(f"{X_filt.shape[0]} samples, {n_classes} classes")

            for pipe_name in pipes_to_run:
                if pipe_name not in PIPELINES:
                    continue

                cfg = PIPELINES[pipe_name]
                print(f"    [{pipe_name}] running... ", end="", flush=True)
                t0 = time.time()

                # Clone pipeline to avoid state leakage between runs
                from sklearn.base import clone
                pipe_clone = clone(cfg["pipe"])

                grid = run_gridsearch(X_filt, y_filt, pipe_clone, cfg["params"].copy(), n_folds=N_FOLDS)
                elapsed = time.time() - t0

                if grid is None:
                    print("SKIPPED")
                    continue

                best_score = grid.best_score_
                best_std = grid.cv_results_["std_test_score"][grid.best_index_]
                best_params = grid.best_params_

                print(f"F1={best_score:.4f} ±{best_std:.4f} ({elapsed:.0f}s)")

                # Format params compactly
                params_short = {k.split("__")[-1]: v for k, v in best_params.items()}
                print(f"           params: {params_short}")

                all_results.append({
                    "model": model_name,
                    "pipeline": pipe_name,
                    "threshold": threshold,
                    "n_classes": n_classes,
                    "n_samples": X_filt.shape[0],
                    "best_score": best_score,
                    "std": best_std,
                    "params": best_params,
                })

    total_elapsed = time.time() - total_start

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print(f"\n\n{'=' * 100}")
    print(f"  FINAL RESULTS — NON-LINEAR GRID SEARCH (total: {total_elapsed:.0f}s)")
    print(f"{'=' * 100}")
    print(f"{'Model':<8} | {'Pipeline':<20} | {'Thresh':<6} | {'Classes':<7} | "
          f"{'Macro-F1':<9} | {'Std':<6} | {'Key Params'}")
    print(f"{'=' * 100}")

    for r in sorted(all_results, key=lambda x: (-x["threshold"], -x["best_score"])):
        params_short = ", ".join(
            f"{k.split('__')[-1]}={v}" for k, v in r["params"].items()
        )
        # Truncate params if too long
        if len(params_short) > 50:
            params_short = params_short[:47] + "..."
        print(
            f"{r['model']:<8} | {r['pipeline']:<20} | {r['threshold']:<6} | "
            f"{r['n_classes']:<7} | {r['best_score']:<9.4f} | {r['std']:<6.4f} | "
            f"{params_short}"
        )

    print(f"{'=' * 100}")

    # Best per threshold
    print(f"\n  BEST MODEL+PIPELINE PER THRESHOLD:")
    print(f"  {'─' * 80}")

    for threshold in thresholds:
        subset = [r for r in all_results if r["threshold"] == threshold]
        if subset:
            best = max(subset, key=lambda x: x["best_score"])
            print(
                f"  >= {threshold}: {best['model']} + {best['pipeline']} → "
                f"Macro-F1 = {best['best_score']:.4f} ({best['n_classes']} classes)"
            )

    # Compare with previous best (LogReg extended)
    print(f"\n  COMPARISON WITH PREVIOUS BEST (LogReg Extended):")
    print(f"  {'─' * 80}")
    prev_best = {10: 0.6607, 8: 0.6438, 5: 0.5718}

    for threshold in thresholds:
        subset = [r for r in all_results if r["threshold"] == threshold]
        if subset:
            best = max(subset, key=lambda x: x["best_score"])
            prev = prev_best.get(threshold, 0)
            diff = best["best_score"] - prev
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(
                f"  >= {threshold}: {best['best_score']:.4f} vs {prev:.4f} "
                f"({arrow} {abs(diff):.4f}) — {best['model']}+{best['pipeline']}"
            )

    print()


if __name__ == "__main__":
    main()
