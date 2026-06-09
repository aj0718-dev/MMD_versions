#!/usr/bin/env python3
"""
Evaluate all image PTM embeddings with LogReg + SVM-RBF.
Runs 5-fold stratified CV with PCA fitted inside each fold.

Usage:
    python eval_image_ptms.py --threshold 10
    python eval_image_ptms.py --threshold 10 --models resnet50 convnext_base
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from collections import Counter

EMB_DIR = Path("image_embeddings")
K_FOLDS = 5
RANDOM_STATE = 42

# Classifiers to evaluate
CLASSIFIERS = {
    # LogReg: C=1000 and C=100, both balanced and None
    "LR_bal_C1k": lambda: LogisticRegression(
        C=1000, class_weight="balanced", solver="lbfgs",
        max_iter=5000, random_state=RANDOM_STATE
    ),
    "LR_C1k": lambda: LogisticRegression(
        C=1000, class_weight=None, solver="lbfgs",
        max_iter=5000, random_state=RANDOM_STATE
    ),
    "LR_bal_C100": lambda: LogisticRegression(
        C=100, class_weight="balanced", solver="lbfgs",
        max_iter=5000, random_state=RANDOM_STATE
    ),
    "LR_C100": lambda: LogisticRegression(
        C=100, class_weight=None, solver="lbfgs",
        max_iter=5000, random_state=RANDOM_STATE
    ),
    # SVM-RBF: balanced and None
    "SVM_bal": lambda: SVC(
        C=100, kernel="rbf", gamma="scale", class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "SVM": lambda: SVC(
        C=100, kernel="rbf", gamma="scale", class_weight=None,
        random_state=RANDOM_STATE
    ),
    # MLP (no class_weight in sklearn MLPClassifier)
    "MLP": lambda: MLPClassifier(
        hidden_layer_sizes=(512, 128), activation="relu",
        solver="adam", alpha=1e-4, batch_size=32,
        max_iter=200, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=10, random_state=RANDOM_STATE
    ),
}

# PCA dims to try per classifier
PCA_DIMS = {
    "LR_bal_C1k": [256, 384, 512],
    "LR_C1k": [256, 384, 512],
    "LR_bal_C100": [256, 384, 512],
    "LR_C100": [256, 384, 512],
    "SVM_bal": [128, 256],
    "SVM": [128, 256],
    "MLP": [256, 512],
}

# Models to evaluate (image + audio)
IMAGE_MODELS = [
    "resnet50", "efficientnet_b0", "convnext_base", "swin_base",
    "mobilenetv3_large", "dinov2", "beit", "deit_base",
    "vgg19", "vit",
]
AUDIO_MODELS = ["wav2vec2", "hubert", "wavlm"]
ALL_MODELS = IMAGE_MODELS + AUDIO_MODELS

AUDIO_EMB_DIR = Path("wav2vec2_hubert_wavlm")

# Special paths for legacy models (full 3094 samples)
SPECIAL_PATHS = {
    "vgg19": (Path("vgg19/vgg19_embeddings_all.pt"), Path("vgg19/labels_all.pt")),
    "vit": (Path("vit_vgg_fcn/vit_embeddings.pt"), Path("vit_vgg_fcn/labels.pt")),
}


def load_embeddings(model_name: str):
    """Load embeddings and labels for a model."""
    if model_name in SPECIAL_PATHS:
        emb_path, lab_path = SPECIAL_PATHS[model_name]
    elif model_name in AUDIO_MODELS:
        emb_path = AUDIO_EMB_DIR / f"{model_name}_embeddings.pt"
        lab_path = AUDIO_EMB_DIR / "labels.pt"
    else:
        emb_path = EMB_DIR / f"{model_name}_embeddings.pt"
        lab_path = EMB_DIR / f"{model_name}_labels.pt"

    if not emb_path.exists():
        return None, None

    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
    labels = torch.load(lab_path, map_location="cpu", weights_only=True)

    if not isinstance(embeddings, np.ndarray):
        embeddings = embeddings.numpy()
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy()

    return embeddings, labels


def filter_by_threshold(X, y, threshold):
    """Keep only classes with >= threshold samples."""
    counts = Counter(y)
    valid_classes = {c for c, n in counts.items() if n >= threshold}
    mask = np.array([yi in valid_classes for yi in y])

    X_filt = X[mask]
    y_filt = y[mask]

    # Remap labels to 0..n_classes-1
    unique = sorted(set(y_filt))
    label_map = {old: new for new, old in enumerate(unique)}
    y_filt = np.array([label_map[yi] for yi in y_filt])

    return X_filt, y_filt, len(unique)


def evaluate_model(model_name: str, threshold: int):
    """Run full evaluation for one model."""
    X, y = load_embeddings(model_name)
    if X is None:
        print(f"  [SKIP] {model_name} — embeddings not found at {EMB_DIR}")
        return None

    X, y, n_classes = filter_by_threshold(X, y, threshold)
    print(f"  [{model_name}] {X.shape[0]} samples, {n_classes} classes, {X.shape[1]}d")

    results = {}
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for clf_name, clf_fn in CLASSIFIERS.items():
        for pca_dim in PCA_DIMS[clf_name]:
            if pca_dim >= X.shape[1]:
                pca_dim = X.shape[1]  # Don't expand

            fold_scores = []
            t0 = time.time()

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # PCA fitted on train only
                pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)

                # Normalization: L2 for LogReg variants, StandardScaler for SVM/MLP
                if clf_name.startswith("LR"):
                    norm = Normalizer(norm="l2")
                    X_train_sc = norm.fit_transform(X_train_pca)
                    X_test_sc = norm.transform(X_test_pca)
                else:
                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train_pca)
                    X_test_sc = scaler.transform(X_test_pca)

                # Fit classifier
                clf = clf_fn()
                clf.fit(X_train_sc, y_train)
                y_pred = clf.predict(X_test_sc)

                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                fold_scores.append(f1)

            elapsed = time.time() - t0
            mean_f1 = np.mean(fold_scores)
            std_f1 = np.std(fold_scores)

            key = f"{clf_name}_pca{pca_dim}"
            results[key] = (mean_f1, std_f1, fold_scores, elapsed)
            print(f"    {key:25s} → {mean_f1:.4f} ±{std_f1:.4f} ({elapsed:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, nargs="+", default=[10])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    args = parser.parse_args()

    for threshold in args.threshold:
        print(f"\n{'='*70}")
        print(f"  IMAGE+AUDIO PTM EVALUATION — Threshold ≥{threshold}, {K_FOLDS}-Fold CV")
        print(f"  PCA fitted inside each fold (no data leakage)")
        print(f"{'='*70}\n")

        all_results = {}

        for model_name in args.models:
            results = evaluate_model(model_name, threshold)
            if results:
                all_results[model_name] = results

        # Final summary sorted by best F1
        print(f"\n{'='*70}")
        print(f"  SUMMARY — Best Config Per Model (Threshold ≥{threshold})")
        print(f"{'='*70}")
        print(f"  {'Model':<20s} | {'Best Config':<25s} | {'Macro-F1':<10s} | Std")
        print(f"  {'─'*75}")

        summary = []
        for model_name, results in all_results.items():
            best_key = max(results, key=lambda k: results[k][0])
            mean_f1, std_f1, _, _ = results[best_key]
            summary.append((model_name, best_key, mean_f1, std_f1))

        summary.sort(key=lambda x: x[2], reverse=True)
        for model_name, best_key, mean_f1, std_f1 in summary:
            print(f"  {model_name:<20s} | {best_key:<25s} | {mean_f1:.4f}     | {std_f1:.4f}")


if __name__ == "__main__":
    main()
