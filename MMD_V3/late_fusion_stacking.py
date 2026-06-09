"""
Phase 2: Late Fusion / Stacking
================================
Instead of concatenating raw embeddings, we train per-modality SVM classifiers
and fuse their out-of-fold probability vectors using various meta-learners.

Combinations tested:
- HuBERT + DINOv2
- HuBERT + DINOv2 + Swin
- HuBERT + DINOv2 + ConvNeXt
- HuBERT + DINOv2 + Swin + ConvNeXt

Meta-fusion methods:
1. Average probabilities
2. Weighted average (learned on val)
3. Stacked LogisticRegression
4. Stacked MLP
"""

import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# CONFIGURATION
# =====================================================

MODELS = {
    "HuBERT": {
        "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
        "labels": "wav2vec2_hubert_wavlm/labels.pt",
        "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt",
    },
    "DINOv2": {
        "embeddings": "image_embeddings/dinov2_embeddings.pt",
        "labels": "image_embeddings/dinov2_labels.pt",
        "paths": "image_embeddings/dinov2_paths.pt",
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

COMBOS = {
    "HuBERT+DINOv2": ["HuBERT", "DINOv2"],
    "HuBERT+DINOv2+Swin": ["HuBERT", "DINOv2", "Swin"],
    "HuBERT+DINOv2+ConvNeXt": ["HuBERT", "DINOv2", "ConvNeXt"],
    "HuBERT+DINOv2+Swin+ConvNeXt": ["HuBERT", "DINOv2", "Swin", "ConvNeXt"],
}

K_FOLDS = 5
RANDOM_STATE = 42
PCA_DIM = 128  # SVM works best at lower PCA for probability calibration


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def load_embeddings(config):
    emb = torch.load(config["embeddings"], map_location="cpu", weights_only=True)
    labels = torch.load(config["labels"], map_location="cpu", weights_only=True)
    paths = None
    if "paths" in config and os.path.exists(config["paths"]):
        paths = torch.load(config["paths"], map_location="cpu", weights_only=False)
    return emb.numpy().astype(np.float32), labels.numpy().astype(np.int64), paths


def get_sample_id(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def align_all_models(model_data):
    """Align all models by common sample IDs."""
    # Collect IDs per model
    id_maps = {}
    for name, (X, y, paths) in model_data.items():
        if paths is not None:
            id_maps[name] = {get_sample_id(p): i for i, p in enumerate(paths)}
        else:
            id_maps[name] = {str(i): i for i in range(len(X))}

    # Find common IDs across all models
    common_ids = set(list(id_maps.values())[0].keys())
    for ids in id_maps.values():
        common_ids &= set(ids.keys())
    common_ids = sorted(common_ids)

    # Build aligned arrays
    aligned = {}
    for name, (X, y, paths) in model_data.items():
        idx = np.array([id_maps[name][sid] for sid in common_ids])
        aligned[name] = X[idx]

    # Use labels from first model
    first_name = list(model_data.keys())[0]
    _, y_first, _ = model_data[first_name]
    first_idx = np.array([id_maps[first_name][sid] for sid in common_ids])
    labels = y_first[first_idx]

    # Sanity check: ensure all models agree on labels after alignment
    for name, (X, y, paths) in model_data.items():
        idx = np.array([id_maps[name][sid] for sid in common_ids])
        if not np.array_equal(y[idx], labels):
            print(f"WARNING: Label mismatch detected for {name}")

    return aligned, labels


def filter_by_threshold(labels, threshold):
    """Keep only classes with >= threshold samples."""
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= threshold]
    mask = np.isin(labels, valid_classes)
    return mask, valid_classes


# =====================================================
# OUT-OF-FOLD PROBABILITY GENERATION
# =====================================================

def generate_oof_probabilities(X_dict, y, n_classes, pca_dim=128):
    """
    For each model, train SVM with probability=True on each fold,
    produce out-of-fold probability vectors.
    Returns dict: model_name -> (n_samples, n_classes) OOF prob array
    """
    n_samples = len(y)
    oof_probs = {name: np.zeros((n_samples, n_classes), dtype=np.float32) for name in X_dict}

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y)):
        print(f"    Fold {fold_idx + 1}/{K_FOLDS}...", end=" ", flush=True)

        for name, X in X_dict.items():
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            # PCA
            pca = PCA(n_components=min(pca_dim, X_train.shape[1], X_train.shape[0] - 1),
                      random_state=RANDOM_STATE)
            X_train_pca = pca.fit_transform(X_train)
            X_val_pca = pca.transform(X_val)

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_pca)
            X_val_scaled = scaler.transform(X_val_pca)

            # SVM with probability
            svm = SVC(
                C=100, kernel="rbf", gamma="scale",
                class_weight="balanced", probability=True,
                random_state=RANDOM_STATE, cache_size=1000
            )
            svm.fit(X_train_scaled, y_train)

            # Get probability predictions for val
            probs = svm.predict_proba(X_val_scaled)  # (n_val, n_classes_seen)

            # Map to full class space (some classes may not appear in fold)
            full_probs = np.zeros((len(val_idx), n_classes), dtype=np.float32)
            for ci, cls_label in enumerate(svm.classes_):
                full_probs[:, cls_label] = probs[:, ci]

            oof_probs[name][val_idx] = full_probs

        print("done")

    return oof_probs


# =====================================================
# META-FUSION METHODS
# =====================================================

def fuse_average(oof_probs_list, y, combo_name):
    """Simple average of probability vectors."""
    avg_probs = np.mean(oof_probs_list, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return f1


def fuse_weighted_average(oof_probs_list, y, model_names):
    """
    Weight each model's probabilities by its individual OOF macro-F1.
    """
    # Compute individual F1s
    weights = []
    for probs in oof_probs_list:
        preds = np.argmax(probs, axis=1)
        f1 = f1_score(y, preds, average="macro", zero_division=0)
        weights.append(f1)
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Weighted average
    weighted_probs = np.zeros_like(oof_probs_list[0])
    for w, probs in zip(weights, oof_probs_list):
        weighted_probs += w * probs

    preds = np.argmax(weighted_probs, axis=1)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return f1, weights


def _train_base_svms(X_dict, y_train, n_classes, pca_dim):
    """Train base SVM classifiers and return fitted models + transforms."""
    models = {}
    for name, X in X_dict.items():
        pca = PCA(n_components=min(pca_dim, X.shape[1], X.shape[0] - 1),
                  random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pca)
        svm = SVC(
            C=100, kernel="rbf", gamma="scale",
            class_weight="balanced", probability=True,
            random_state=RANDOM_STATE, cache_size=1000
        )
        svm.fit(X_scaled, y_train)
        models[name] = {"pca": pca, "scaler": scaler, "svm": svm}
    return models


def _get_probs(models, X_dict, n_classes):
    """Get probability predictions from fitted base models."""
    all_probs = []
    for name in models:
        X = X_dict[name]
        m = models[name]
        X_pca = m["pca"].transform(X)
        X_scaled = m["scaler"].transform(X_pca)
        probs = m["svm"].predict_proba(X_scaled)
        full_probs = np.zeros((len(X), n_classes), dtype=np.float32)
        for ci, cls_label in enumerate(m["svm"].classes_):
            full_probs[:, cls_label] = probs[:, ci]
        all_probs.append(full_probs)
    return np.hstack(all_probs)


def _generate_inner_oof(X_dict_train, y_train, n_classes, pca_dim, n_inner_folds=4):
    """Generate inner OOF probability vectors for meta-learner training."""
    n_samples = len(y_train)
    inner_oof = np.zeros((n_samples, n_classes * len(X_dict_train)), dtype=np.float32)
    model_names = list(X_dict_train.keys())

    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=RANDOM_STATE + 10)
    for inner_train_idx, inner_val_idx in skf.split(np.zeros(n_samples), y_train):
        col_offset = 0
        for name in model_names:
            X = X_dict_train[name]
            X_tr, X_val = X[inner_train_idx], X[inner_val_idx]
            y_tr = y_train[inner_train_idx]

            pca = PCA(n_components=min(pca_dim, X_tr.shape[1], X_tr.shape[0] - 1),
                      random_state=RANDOM_STATE)
            X_tr_pca = pca.fit_transform(X_tr)
            X_val_pca = pca.transform(X_val)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_pca)
            X_val_s = scaler.transform(X_val_pca)

            svm = SVC(
                C=100, kernel="rbf", gamma="scale",
                class_weight="balanced", probability=True,
                random_state=RANDOM_STATE, cache_size=1000
            )
            svm.fit(X_tr_s, y_tr)
            probs = svm.predict_proba(X_val_s)

            for ci, cls_label in enumerate(svm.classes_):
                inner_oof[inner_val_idx, col_offset + cls_label] = probs[:, ci]

            col_offset += n_classes

    return inner_oof


def fuse_stacked_logreg(X_dict, y, n_classes, pca_dim, model_list):
    """
    Fully nested stacking with LogisticRegression meta-learner.
    Outer fold: train/test split
      Inner: generate OOF probs on train → fit meta-LR
      Test: train base SVMs on full outer-train → predict test → meta-LR predicts
    """
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_f1s = []

    X_sub = {name: X_dict[name] for name in model_list}

    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        y_train, y_test = y[train_idx], y[test_idx]
        X_train = {name: X[train_idx] for name, X in X_sub.items()}
        X_test = {name: X[test_idx] for name, X in X_sub.items()}

        # Inner OOF for meta-learner training
        inner_oof = _generate_inner_oof(X_train, y_train, n_classes, pca_dim)

        # Train meta-learner
        scaler = StandardScaler()
        inner_oof_s = scaler.fit_transform(inner_oof)
        lr = LogisticRegression(
            C=10, max_iter=2000, class_weight="balanced",
            solver="lbfgs", random_state=RANDOM_STATE
        )
        lr.fit(inner_oof_s, y_train)

        # Train base SVMs on full outer-train, predict test
        base_models = _train_base_svms(X_train, y_train, n_classes, pca_dim)
        test_probs = _get_probs(base_models, X_test, n_classes)
        test_probs_s = scaler.transform(test_probs)

        preds = lr.predict(test_probs_s)
        fold_f1s.append(f1_score(y_test, preds, average="macro", zero_division=0))

    return np.mean(fold_f1s), np.std(fold_f1s)


def fuse_stacked_mlp(X_dict, y, n_classes, pca_dim, model_list):
    """
    Fully nested stacking with MLP meta-learner.
    Same nested protocol as LogReg version.
    """
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_f1s = []

    X_sub = {name: X_dict[name] for name in model_list}

    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        y_train, y_test = y[train_idx], y[test_idx]
        X_train = {name: X[train_idx] for name, X in X_sub.items()}
        X_test = {name: X[test_idx] for name, X in X_sub.items()}

        # Inner OOF for meta-learner training
        inner_oof = _generate_inner_oof(X_train, y_train, n_classes, pca_dim)

        # Train meta-learner
        scaler = StandardScaler()
        inner_oof_s = scaler.fit_transform(inner_oof)
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
            batch_size=64,
        )
        mlp.fit(inner_oof_s, y_train)

        # Train base SVMs on full outer-train, predict test
        base_models = _train_base_svms(X_train, y_train, n_classes, pca_dim)
        test_probs = _get_probs(base_models, X_test, n_classes)
        test_probs_s = scaler.transform(test_probs)

        preds = mlp.predict(test_probs_s)
        fold_f1s.append(f1_score(y_test, preds, average="macro", zero_division=0))

    return np.mean(fold_f1s), np.std(fold_f1s)


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Late Fusion / Stacking (Phase 2)")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[10],
                        help="Family count thresholds (default: 10)")
    parser.add_argument("--pca", type=int, default=128,
                        help="PCA dim for base SVM classifiers (default: 128)")
    args = parser.parse_args()

    global PCA_DIM
    PCA_DIM = args.pca

    print("=" * 70)
    print("  PHASE 2: LATE FUSION / STACKING")
    print("=" * 70)
    print(f"  Models: {list(MODELS.keys())}")
    print(f"  Combos: {list(COMBOS.keys())}")
    print(f"  Base SVM PCA dim: {PCA_DIM}")
    print(f"  K-Folds: {K_FOLDS}")
    print(f"  Thresholds: {args.thresholds}")
    print("=" * 70)

    # Load all model data
    print("\n[1] Loading embeddings...")
    raw_data = {}
    for name, config in MODELS.items():
        X, y, paths = load_embeddings(config)
        print(f"    {name}: {X.shape}")
        raw_data[name] = (X, y, paths)

    # Align all models
    print("\n[2] Aligning samples across all models...")
    aligned_X, aligned_y = align_all_models(raw_data)
    print(f"    Aligned samples: {len(aligned_y)}")
    print(f"    Unique classes: {len(np.unique(aligned_y))}")

    all_results = []

    for threshold in args.thresholds:
        print(f"\n{'=' * 70}")
        print(f"  THRESHOLD ≥ {threshold}")
        print(f"{'=' * 70}")

        # Filter by threshold
        mask, valid_classes = filter_by_threshold(aligned_y, threshold)
        y_filtered = aligned_y[mask]
        X_filtered = {name: X[mask] for name, X in aligned_X.items()}

        # Re-encode labels to 0..n_classes-1
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_filtered)
        n_classes = len(le.classes_)
        n_samples = len(y_encoded)
        print(f"  Classes: {n_classes}, Samples: {n_samples}")

        # Generate OOF probabilities for all models
        print(f"\n  [3] Generating out-of-fold SVM probabilities (PCA={PCA_DIM})...")
        oof_probs = generate_oof_probabilities(X_filtered, y_encoded, n_classes, PCA_DIM)

        # Print individual model OOF F1
        print(f"\n  Individual model OOF Macro-F1:")
        for name, probs in oof_probs.items():
            preds = np.argmax(probs, axis=1)
            f1 = f1_score(y_encoded, preds, average="macro", zero_division=0)
            print(f"    {name}: {f1:.4f}")

        # Test each combination
        for combo_name, model_list in COMBOS.items():
            print(f"\n  {'─' * 60}")
            print(f"  Combo: {combo_name}")
            print(f"  {'─' * 60}")

            probs_list = [oof_probs[m] for m in model_list]

            # Method 1: Average
            f1_avg = fuse_average(probs_list, y_encoded, combo_name)
            print(f"    Average Prob:      {f1_avg:.4f}")
            all_results.append({
                "threshold": threshold, "n_classes": n_classes,
                "combo": combo_name, "method": "Avg_Prob",
                "mean_f1": f1_avg, "std_f1": 0.0
            })

            # Method 2: Weighted Average
            f1_wavg, weights = fuse_weighted_average(probs_list, y_encoded, model_list)
            w_str = ", ".join(f"{m}={w:.3f}" for m, w in zip(model_list, weights))
            print(f"    Weighted Avg:      {f1_wavg:.4f}  (weights: {w_str})")
            all_results.append({
                "threshold": threshold, "n_classes": n_classes,
                "combo": combo_name, "method": "Weighted_Avg",
                "mean_f1": f1_wavg, "std_f1": 0.0
            })

            # Method 3: Stacked LogReg (fully nested)
            print(f"    Stacked LogReg:    ", end="", flush=True)
            f1_lr, std_lr = fuse_stacked_logreg(X_filtered, y_encoded, n_classes, PCA_DIM, model_list)
            print(f"{f1_lr:.4f} ± {std_lr:.4f}")
            all_results.append({
                "threshold": threshold, "n_classes": n_classes,
                "combo": combo_name, "method": "Stack_LogReg",
                "mean_f1": f1_lr, "std_f1": std_lr
            })

            # Method 4: Stacked MLP (fully nested)
            print(f"    Stacked MLP:       ", end="", flush=True)
            f1_mlp, std_mlp = fuse_stacked_mlp(X_filtered, y_encoded, n_classes, PCA_DIM, model_list)
            print(f"{f1_mlp:.4f} ± {std_mlp:.4f}")
            all_results.append({
                "threshold": threshold, "n_classes": n_classes,
                "combo": combo_name, "method": "Stack_MLP",
                "mean_f1": f1_mlp, "std_f1": std_mlp
            })

    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    print(f"\n\n{'=' * 70}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'─' * 95}")
    print(f"  {'Rank':<4} | {'Combo':<30} | {'Method':<13} | {'Thresh':<6} | {'Classes':<7} | {'Macro-F1':<9} | {'Std':<6}")
    print(f"  {'─' * 95}")

    sorted_results = sorted(all_results, key=lambda x: -x["mean_f1"])
    for i, r in enumerate(sorted_results[:20], 1):
        print(f"  {i:<4} | {r['combo']:<30} | {r['method']:<13} | {r['threshold']:<6} | "
              f"{r['n_classes']:<7} | {r['mean_f1']:<9.4f} | {r['std_f1']:<6.4f}")

    # Compare with baseline
    print(f"\n  {'─' * 50}")
    print(f"  COMPARISON WITH EARLY FUSION BASELINE:")
    print(f"  DINOv2+HuBERT FCN_Balanced PCA=256 = 0.7223")
    print(f"  {'─' * 50}")
    best = sorted_results[0]
    delta = best["mean_f1"] - 0.7223
    sign = "+" if delta >= 0 else ""
    print(f"  Best late fusion: {best['combo']} {best['method']} = {best['mean_f1']:.4f} ({sign}{delta:.4f})")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
