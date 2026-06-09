#!/usr/bin/env python3
"""
FOCA-style CNN classifier on frozen PTM embeddings.
Architecture: Conv1D(64,k=3) → MaxPool → Conv1D(128,k=3) → MaxPool → FC(128) → Output
Training: Adam, lr=1e-5, batch=32, max 50 epochs, early stopping, dropout.
Evaluation: 5-Fold Stratified CV, Macro-F1.

Usage:
    python eval_foca_cnn.py --threshold 10
    python eval_foca_cnn.py --threshold 10 8 5
    python eval_foca_cnn.py --threshold 10 --models resnet50 hubert
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── Config ───────────────────────────────────────────────────────────────────
K_FOLDS = 5
RANDOM_STATE = 42
BATCH_SIZE = 32
LR = 1e-3
MAX_EPOCHS = 50
PATIENCE = 7  # early stopping patience
DROPOUT = 0.3
VAL_SPLIT = 0.15  # fraction of train fold for validation (early stopping)
PCA_DIM = 256  # PCA dimensionality reduction inside each fold

IMAGE_EMB_DIR = Path("image_embeddings")
AUDIO_EMB_DIR = Path("wav2vec2_hubert_wavlm")

# All available models
IMAGE_MODELS = [
    "resnet50", "efficientnet_b0", "convnext_base", "swin_base",
    "mobilenetv3_large", "dinov2", "beit", "deit_base",
    "vgg19", "vit",
]
AUDIO_MODELS = ["wav2vec2", "hubert", "wavlm"]
ALL_MODELS = IMAGE_MODELS + AUDIO_MODELS

# Special paths for legacy models (full 3094 samples)
SPECIAL_PATHS = {
    "vgg19": (Path("vgg19/vgg19_embeddings_all.pt"), Path("vgg19/labels_all.pt")),
    "vit": (Path("vit_vgg_fcn/vit_embeddings.pt"), Path("vit_vgg_fcn/labels.pt")),
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─── FOCA CNN Architecture ────────────────────────────────────────────────────
class FOCA_CNN(nn.Module):
    """
    FOCA-style CNN block on top of frozen PTM embeddings.
    Input: (batch, embed_dim) → reshape to (batch, 1, embed_dim)
    Conv1D(1→64, k=3) → ReLU → MaxPool(2)
    Conv1D(64→128, k=3) → ReLU → MaxPool(2)
    Flatten → Dropout → FC(128) → ReLU → Dropout → FC(n_classes)
    """

    def __init__(self, embed_dim: int, n_classes: int, dropout: float = DROPOUT):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Compute flattened size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, embed_dim)
            out = self.conv_block(dummy)
            flat_dim = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (batch, embed_dim) → (batch, 1, embed_dim)
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_embeddings(model_name: str):
    """Load embeddings and labels for a model."""
    if model_name in SPECIAL_PATHS:
        emb_path, lab_path = SPECIAL_PATHS[model_name]
    elif model_name in AUDIO_MODELS:
        emb_path = AUDIO_EMB_DIR / f"{model_name}_embeddings.pt"
        lab_path = AUDIO_EMB_DIR / "labels.pt"
    else:
        emb_path = IMAGE_EMB_DIR / f"{model_name}_embeddings.pt"
        lab_path = IMAGE_EMB_DIR / f"{model_name}_labels.pt"

    if not emb_path.exists():
        return None, None

    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
    labels = torch.load(lab_path, map_location="cpu", weights_only=True)

    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    return embeddings.float(), labels.long()


def filter_by_threshold(X, y, threshold):
    """Keep only classes with >= threshold samples, remap labels."""
    y_np = y.numpy()
    counts = Counter(y_np)
    valid_classes = {c for c, n in counts.items() if n >= threshold}
    mask = torch.tensor([yi.item() in valid_classes for yi in y])

    X_filt = X[mask]
    y_filt = y[mask]

    # Remap labels to 0..n_classes-1
    unique = sorted(set(y_filt.numpy()))
    label_map = {int(old): new for new, old in enumerate(unique)}
    y_filt = torch.tensor([label_map[int(yi)] for yi in y_filt], dtype=torch.long)

    return X_filt, y_filt, len(unique)


# ─── Training ─────────────────────────────────────────────────────────────────
def compute_class_weights(y_train):
    """Compute inverse-frequency class weights for balanced loss."""
    counts = Counter(y_train.numpy())
    n_samples = len(y_train)
    n_classes = len(counts)
    weights = torch.zeros(n_classes)
    for cls, cnt in counts.items():
        weights[cls] = n_samples / (n_classes * cnt)
    return weights


def train_one_fold(X_train, y_train, X_test, y_test, embed_dim, n_classes):
    """Train FOCA CNN on one fold with early stopping, return test F1."""

    # ─── PCA + StandardScaler fitted on training fold only ─────────────────
    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()

    # PCA (reduce dimensionality, remove noise)
    pca_dim = min(PCA_DIM, X_train_np.shape[1], X_train_np.shape[0] - 1)
    pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
    X_train_np = pca.fit_transform(X_train_np)
    X_test_np = pca.transform(X_test_np)

    # StandardScaler (zero mean, unit variance)
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np = scaler.transform(X_test_np)

    # Convert back to tensors with new dim
    X_train = torch.from_numpy(X_train_np).float()
    X_test = torch.from_numpy(X_test_np).float()
    actual_dim = pca_dim  # CNN input dim is now PCA_DIM

    # Split training into train_inner + val for early stopping
    n_val = max(1, int(len(X_train) * VAL_SPLIT))
    indices = torch.randperm(len(X_train), generator=torch.Generator().manual_seed(RANDOM_STATE))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    # Class weights for balanced loss
    class_weights = compute_class_weights(y_tr).to(DEVICE)

    # Model
    model = FOCA_CNN(actual_dim, n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        # Shuffle training data
        perm = torch.randperm(len(X_tr))
        X_tr_shuf = X_tr[perm]
        y_tr_shuf = y_tr[perm]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_tr_shuf), BATCH_SIZE):
            batch_x = X_tr_shuf[i:i + BATCH_SIZE].to(DEVICE)
            batch_y = y_tr_shuf[i:i + BATCH_SIZE].to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(DEVICE))
            val_loss = criterion(val_logits, y_val.to(DEVICE)).item()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Load best model and evaluate on test fold
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(DEVICE))
        y_pred = test_logits.argmax(dim=1).cpu().numpy()

    f1 = f1_score(y_test.numpy(), y_pred, average="macro", zero_division=0)
    return f1, epoch + 1  # return epochs trained


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model_name: str, threshold: int):
    """Run 5-fold FOCA CNN evaluation for one model at one threshold."""
    X, y = load_embeddings(model_name)
    if X is None:
        print(f"  [SKIP] {model_name} — embeddings not found")
        return None

    X, y, n_classes = filter_by_threshold(X, y, threshold)
    embed_dim = X.shape[1]
    print(f"  [{model_name}] {X.shape[0]} samples, {n_classes} classes, {embed_dim}d")

    if n_classes < 2:
        print(f"  [SKIP] {model_name} — fewer than 2 classes after filtering")
        return None

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    fold_epochs = []
    t0 = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X.numpy(), y.numpy())):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        f1, epochs = train_one_fold(X_train, y_train, X_test, y_test, embed_dim, n_classes)
        fold_scores.append(f1)
        fold_epochs.append(epochs)

    elapsed = time.time() - t0
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    avg_epochs = np.mean(fold_epochs)

    print(f"    FOCA_CNN → {mean_f1:.4f} ±{std_f1:.4f} "
          f"(avg {avg_epochs:.0f} epochs, {elapsed:.1f}s)")

    return {
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "fold_scores": fold_scores,
        "avg_epochs": avg_epochs,
        "elapsed": elapsed,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FOCA CNN evaluation on PTM embeddings")
    parser.add_argument("--threshold", type=int, nargs="+", default=[10],
                        help="Family threshold(s), e.g. --threshold 10 8 5")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="Models to evaluate")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print(f"\n{'='*70}")
    print(f"  FOCA CNN EVALUATION — {K_FOLDS}-Fold Stratified CV")
    print(f"  Architecture: Conv1D(64,k=3)→MaxPool→Conv1D(128,k=3)→MaxPool→FC(128)")
    print(f"  Preprocessing: PCA({PCA_DIM}) + StandardScaler inside each fold")
    print(f"  Training: Adam lr={LR}, batch={BATCH_SIZE}, max {MAX_EPOCHS} epochs")
    print(f"  Early stopping: patience={PATIENCE}, val_split={VAL_SPLIT}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}\n")

    all_results = {}

    for threshold in args.threshold:
        print(f"\n{'─'*70}")
        print(f"  Threshold ≥{threshold}")
        print(f"{'─'*70}")

        results_at_threshold = {}
        for model_name in args.models:
            result = evaluate_model(model_name, threshold)
            if result:
                results_at_threshold[model_name] = result

        all_results[threshold] = results_at_threshold

    # ─── Final Summary ────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  FINAL SUMMARY — FOCA CNN Macro-F1")
    print(f"{'='*70}")

    # Print table header
    thresholds = args.threshold
    header = f"  {'Model':<20s}"
    for t in thresholds:
        header += f" | {'≥' + str(t):^12s}"
    print(header)
    print(f"  {'─' * (22 + 15 * len(thresholds))}")

    # Collect all models that have at least one result
    all_model_names = set()
    for t in thresholds:
        all_model_names.update(all_results[t].keys())

    # Sort by best F1 at first threshold
    def sort_key(m):
        if m in all_results[thresholds[0]]:
            return all_results[thresholds[0]][m]["mean_f1"]
        return 0.0

    for model_name in sorted(all_model_names, key=sort_key, reverse=True):
        row = f"  {model_name:<20s}"
        for t in thresholds:
            if model_name in all_results[t]:
                r = all_results[t][model_name]
                row += f" | {r['mean_f1']:.4f}±{r['std_f1']:.4f}"
            else:
                row += f" | {'—':^12s}"
        print(row)

    print()


if __name__ == "__main__":
    main()
