#!/usr/bin/env python3
"""
final_baseline_eval.py

Final rigorous evaluation of the best models per modality.
Locked params from grid search + additional gamma refinement.

Reports:
  - Macro-F1, Weighted-F1, Std
  - Top-3, Top-5 accuracy
  - Confusion matrix (saved as .png)
  - Per-class F1 breakdown (bottom 10 worst classes)

Usage:
    python final_baseline_eval.py --thresholds 10 8 5
    python final_baseline_eval.py --thresholds 10 --refine-gamma
"""

import argparse
import os
import time
import warnings
from collections import Counter

import numpy as np
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# =====================================================
# CONFIG — LOCKED BEST MODELS
# =====================================================

# Best audio model: HuBERT + SVM_RBF
# Best image model: VGG19 + SVM_RBF (richer 4096-dim features for fusion)
#                   ViT + SVM_RBF (secondary)

MODELS = {
    "HuBERT": {
        "modality": "audio",
        "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
        "labels": "wav2vec2_hubert_wavlm/labels.pt",
        "best_params": {
            "pca__n_components": 128,
            "clf__C": 100,
            "clf__gamma": "scale",
            "clf__class_weight": "balanced",
        },
    },
    "VGG19": {
        "modality": "image",
        "embeddings": "vgg19/vgg19_embeddings_all.pt",
        "labels": "vgg19/labels_all.pt",
        "best_params": {
            "pca__n_components": 128,
            "clf__C": 50,
            "clf__gamma": "scale",
            "clf__class_weight": "balanced",
        },
    },
    "ViT": {
        "modality": "image",
        "embeddings": "vit_vgg_fcn/vit_embeddings.pt",
        "labels": "vit_vgg_fcn/labels.pt",
        "best_params": {
            "pca__n_components": 128,
            "clf__C": 500,
            "clf__gamma": "scale",
            "clf__class_weight": "balanced",
        },
    },
    "WavLM": {
        "modality": "audio",
        "embeddings": "wav2vec2_hubert_wavlm/wavlm_embeddings.pt",
        "labels": "wav2vec2_hubert_wavlm/labels.pt",
        "best_params": {
            "pca__n_components": 128,
            "clf__C": 10,
            "clf__gamma": "scale",
            "clf__class_weight": None,
        },
    },
}

# Gamma refinement grid (the missing param from previous search)
GAMMA_GRID = {
    "pca__n_components": [128],
    "clf__C": [50, 100, 500],
    "clf__gamma": ["scale", 1e-3, 1e-4, 5e-4],
    "clf__class_weight": ["balanced", None],
}

N_FOLDS = 5
RANDOM_STATE = 42


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def load_embeddings(model_config):
    """Load embedding and label tensors."""
    if not os.path.exists(model_config["embeddings"]):
        return None, None
    emb = torch.load(model_config["embeddings"], map_location="cpu", weights_only=True)
    labels = torch.load(model_config["labels"], map_location="cpu", weights_only=True)
    return emb.numpy().astype(np.float32), labels.numpy().astype(np.int64)


def filter_by_threshold(X, y, threshold):
    """Keep classes with >= threshold samples, remap to contiguous."""
    counts = Counter(y)
    keep_classes = {cls for cls, cnt in counts.items() if cnt >= threshold}
    if not keep_classes:
        return None, None, 0, None

    mask = np.array([label in keep_classes for label in y])
    X_filtered = X[mask]
    y_filtered = y[mask]

    unique_labels = sorted(set(y_filtered))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[label] for label in y_filtered])

    return X_filtered, y_remapped, len(unique_labels), label_map


def build_pipeline():
    """Build SVM_RBF pipeline."""
    return Pipeline([
        ("pca", PCA(random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)),
    ])


def refine_gamma(X, y, n_folds=5):
    """Quick gamma refinement search."""
    min_count = min(Counter(y).values())
    actual_folds = min(n_folds, min_count)
    if actual_folds < 2:
        return None

    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        pipe, GAMMA_GRID, cv=cv, scoring="f1_macro",
        n_jobs=-1, verbose=0, return_train_score=True,
    )
    grid.fit(X, y)
    return grid


def evaluate_locked(X, y, params, n_folds=5):
    """
    Run fixed-param evaluation with 5-fold CV.
    Returns detailed metrics per fold.
    """
    min_count = min(Counter(y).values())
    actual_folds = min(n_folds, min_count)
    if actual_folds < 2:
        return None

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build and configure pipeline
        pipe = build_pipeline()
        pipe.set_params(**params)

        # Fit
        pipe.fit(X_train, y_train)

        # Predict
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        # Metrics
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Top-K accuracy (needs probability)
        n_classes = len(np.unique(y))
        top3_acc = top_k_accuracy_score(y_test, y_proba, k=min(3, n_classes), labels=np.arange(n_classes))
        top5_acc = top_k_accuracy_score(y_test, y_proba, k=min(5, n_classes), labels=np.arange(n_classes))

        # Train score (check overfitting)
        y_train_pred = pipe.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)

        fold_results.append({
            "fold": fold_idx + 1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "top3_acc": top3_acc,
            "top5_acc": top5_acc,
            "train_f1": train_f1,
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.append(y_proba)

    return {
        "folds": fold_results,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
    }


def save_confusion_matrix(y_true, y_pred, n_classes, model_name, threshold, output_dir="confusion_matrices"):
    """Save confusion matrix as image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("      (matplotlib/seaborn not available — skipping confusion matrix plot)")
        return

    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

    # Normalize
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(12, n_classes // 5), max(10, n_classes // 5)))

    if n_classes <= 30:
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    else:
        sns.heatmap(cm_norm, annot=False, cmap="Blues", ax=ax)

    ax.set_title(f"{model_name} | Threshold >= {threshold} | {n_classes} classes")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    filename = f"{output_dir}/{model_name}_thresh{threshold}_cm.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"      Saved: {filename}")


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser(description="Final baseline evaluation")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10, 8, 5])
    parser.add_argument("--refine-gamma", action="store_true",
                        help="Run quick gamma refinement before final eval")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to evaluate (default: all)")
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)
    models_to_run = args.models if args.models else list(MODELS.keys())

    print("\n" + "=" * 100)
    print("  FINAL BASELINE EVALUATION — Locked Best Configs")
    print("  SVM_RBF + PCA inside K-Fold (NO data leakage)")
    print("=" * 100)
    print(f"\n  Models: {models_to_run}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Metrics: Macro-F1, Weighted-F1, Top-3 Acc, Top-5 Acc")
    print(f"  Gamma refinement: {'YES' if args.refine_gamma else 'NO (using locked params)'}")
    print()

    all_results = []

    for threshold in thresholds:
        print(f"\n{'─' * 100}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'─' * 100}")

        for model_name in models_to_run:
            if model_name not in MODELS:
                continue

            config = MODELS[model_name]
            print(f"\n  [{model_name}] ({config['modality']}) Loading... ", end="", flush=True)

            X, y = load_embeddings(config)
            if X is None:
                print("FILE NOT FOUND")
                continue

            X_filt, y_filt, n_classes, label_map = filter_by_threshold(X, y, threshold)
            if X_filt is None or n_classes < 2:
                print("SKIPPED")
                continue

            print(f"{X_filt.shape[0]} samples, {n_classes} classes")

            # ─── Gamma refinement (optional) ───
            params = config["best_params"].copy()

            if args.refine_gamma:
                print(f"    [gamma search] running... ", end="", flush=True)
                t0 = time.time()
                grid = refine_gamma(X_filt, y_filt, n_folds=N_FOLDS)
                if grid:
                    params = {k: v for k, v in grid.best_params_.items()}
                    print(f"done ({time.time()-t0:.0f}s) → gamma={params['clf__gamma']}, C={params['clf__C']}")
                else:
                    print("SKIPPED")

            # ─── Final evaluation ───
            print(f"    [final eval] 5-fold CV... ", end="", flush=True)
            t0 = time.time()

            results = evaluate_locked(X_filt, y_filt, params, n_folds=N_FOLDS)
            elapsed = time.time() - t0

            if results is None:
                print("SKIPPED")
                continue

            folds = results["folds"]
            macro_f1s = [f["macro_f1"] for f in folds]
            weighted_f1s = [f["weighted_f1"] for f in folds]
            top3s = [f["top3_acc"] for f in folds]
            top5s = [f["top5_acc"] for f in folds]
            train_f1s = [f["train_f1"] for f in folds]

            print(f"done ({elapsed:.0f}s)")
            print()
            print(f"    ┌─────────────────────────────────────────────────────────┐")
            print(f"    │  {model_name} ({config['modality']}) — Threshold >= {threshold} ({n_classes} classes)")
            print(f"    ├─────────────────────────────────────────────────────────┤")
            print(f"    │  Macro-F1:    {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
            print(f"    │  Weighted-F1: {np.mean(weighted_f1s):.4f} ± {np.std(weighted_f1s):.4f}")
            print(f"    │  Top-3 Acc:   {np.mean(top3s):.4f} ± {np.std(top3s):.4f}")
            print(f"    │  Top-5 Acc:   {np.mean(top5s):.4f} ± {np.std(top5s):.4f}")
            print(f"    │  Train F1:    {np.mean(train_f1s):.4f} (overfit gap: {np.mean(train_f1s)-np.mean(macro_f1s):.4f})")
            print(f"    │  Params: {params}")
            print(f"    └─────────────────────────────────────────────────────────┘")

            # Per-fold breakdown
            print(f"    Fold breakdown:")
            for f in folds:
                print(f"      Fold {f['fold']}: macro={f['macro_f1']:.4f} weighted={f['weighted_f1']:.4f} "
                      f"top3={f['top3_acc']:.4f} top5={f['top5_acc']:.4f} train={f['train_f1']:.4f}")

            # Confusion matrix
            save_confusion_matrix(results["y_true"], results["y_pred"], n_classes,
                                  model_name, threshold)

            # Worst classes
            per_class_f1 = f1_score(results["y_true"], results["y_pred"],
                                    average=None, zero_division=0, labels=np.arange(n_classes))
            worst_idx = np.argsort(per_class_f1)[:10]
            print(f"    Bottom 10 worst classes (F1):")
            for idx in worst_idx:
                count = np.sum(results["y_true"] == idx)
                print(f"      Class {idx:3d}: F1={per_class_f1[idx]:.4f} (n={count})")

            all_results.append({
                "model": model_name,
                "modality": config["modality"],
                "threshold": threshold,
                "n_classes": n_classes,
                "n_samples": X_filt.shape[0],
                "macro_f1": np.mean(macro_f1s),
                "macro_f1_std": np.std(macro_f1s),
                "weighted_f1": np.mean(weighted_f1s),
                "top3_acc": np.mean(top3s),
                "top5_acc": np.mean(top5s),
                "train_f1": np.mean(train_f1s),
                "overfit_gap": np.mean(train_f1s) - np.mean(macro_f1s),
                "params": params,
            })

    # =====================================================
    # FINAL SUMMARY TABLE
    # =====================================================

    print(f"\n\n{'=' * 110}")
    print(f"  FINAL SUMMARY — LOCKED BASELINE EVALUATION")
    print(f"{'=' * 110}")
    print(f"{'Model':<8} | {'Mod':<6} | {'Thr':<4} | {'Cls':<4} | "
          f"{'Macro-F1':<10} | {'Weighted-F1':<12} | {'Top-3':<7} | {'Top-5':<7} | "
          f"{'Train F1':<9} | {'Gap':<6}")
    print(f"{'─' * 110}")

    for r in all_results:
        print(
            f"{r['model']:<8} | {r['modality']:<6} | {r['threshold']:<4} | {r['n_classes']:<4} | "
            f"{r['macro_f1']:.4f}±{r['macro_f1_std']:.3f} | "
            f"{r['weighted_f1']:<12.4f} | {r['top3_acc']:<7.4f} | {r['top5_acc']:<7.4f} | "
            f"{r['train_f1']:<9.4f} | {r['overfit_gap']:<6.4f}"
        )

    print(f"{'=' * 110}")

    # Best per modality per threshold
    print(f"\n  BEST PER MODALITY (for fusion pairing):")
    print(f"  {'─' * 80}")

    for threshold in thresholds:
        subset = [r for r in all_results if r["threshold"] == threshold]
        audio = [r for r in subset if r["modality"] == "audio"]
        image = [r for r in subset if r["modality"] == "image"]

        if audio:
            best_audio = max(audio, key=lambda x: x["macro_f1"])
            print(f"  >= {threshold} AUDIO: {best_audio['model']} → "
                  f"Macro-F1={best_audio['macro_f1']:.4f}, Top-5={best_audio['top5_acc']:.4f}")
        if image:
            best_image = max(image, key=lambda x: x["macro_f1"])
            print(f"  >= {threshold} IMAGE: {best_image['model']} → "
                  f"Macro-F1={best_image['macro_f1']:.4f}, Top-5={best_image['top5_acc']:.4f}")

    # Overfitting analysis
    print(f"\n  OVERFITTING ANALYSIS:")
    print(f"  {'─' * 80}")
    for r in all_results:
        status = "✅ healthy" if r["overfit_gap"] < 0.15 else "⚠️  moderate" if r["overfit_gap"] < 0.25 else "❌ severe"
        print(f"  {r['model']:<8} (>={r['threshold']}): train={r['train_f1']:.4f} test={r['macro_f1']:.4f} "
              f"gap={r['overfit_gap']:.4f} → {status}")

    print(f"\n  READY FOR FUSION: Use best audio + best image per threshold")
    print()


if __name__ == "__main__":
    main()
