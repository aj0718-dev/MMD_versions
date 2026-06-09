#!/usr/bin/env python3
"""
cross_attention_hyperbolic_svm.py

Cross-Attention Fusion in Hyperbolic Space + SVM-RBF Classifier.
Uses all 4 modalities: HuBERT + WavLM + Wav2Vec2 (audio) + VGG19 (image).

Architecture:
  1. PCA per modality (fit on train) → no data leakage
  2. Cross-attention fuses audio (3 tokens) with image (1 token)
  3. Fused embedding projected to hyperbolic space (Poincaré ball)
  4. Log-map back to tangent space → SVM-RBF classifier
  
The key insight: neural cross-attention learns the REPRESENTATION,
but SVM-RBF (proven superior for this data size) does CLASSIFICATION.

Also includes a simple concat+SVM-RBF baseline for comparison.

Usage:
    python cross_attention_hyperbolic_svm.py --thresholds 10 8 5
"""

import argparse
import os
import time
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")  # SVM is CPU-only anyway
EPS = 1e-8


# =====================================================
# HYPERBOLIC OPERATIONS
# =====================================================


def exp_map(x):
    """Exponential map: Euclidean → Poincaré ball."""
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    return torch.tanh(norm) * x / norm


def log_map(x):
    """Logarithmic map: Poincaré ball → Euclidean."""
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    norm = torch.clamp(norm, max=1 - 1e-5)
    return torch.atanh(norm) * x / norm


# =====================================================
# CROSS-ATTENTION FEATURE EXTRACTOR
# =====================================================


class CrossAttentionEncoder(nn.Module):
    """
    Lightweight cross-attention to fuse 3 audio tokens with 1 image token.
    Output is a fixed-dim fused embedding (NOT a classifier).
    Trained with a proxy classification loss, then used as feature extractor.
    """

    def __init__(self, input_dim, proj_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj_dim = proj_dim

        # Shared projection (all modalities already PCA'd to same dim)
        self.audio_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # Modality tokens for the 3 audio models
        self.audio_tokens = nn.Parameter(torch.randn(3, proj_dim) * 0.02)

        # Single cross-attention layer (keep it simple!)
        self.cross_attn = nn.MultiheadAttention(
            proj_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(proj_dim)
        self.ff = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(proj_dim)

    def forward(self, audio_embs, image_emb):
        """
        audio_embs: (B, 3, input_dim)
        image_emb: (B, input_dim)
        Returns: fused embedding (B, proj_dim) in hyperbolic tangent space
        """
        B = audio_embs.shape[0]

        # Project
        a = self.audio_proj(audio_embs) + self.audio_tokens.unsqueeze(0)  # (B, 3, proj_dim)
        i = self.image_proj(image_emb).unsqueeze(1)  # (B, 1, proj_dim)

        # Concatenate all tokens: [audio1, audio2, audio3, image]
        all_tokens = torch.cat([a, i], dim=1)  # (B, 4, proj_dim)

        # Self-attention over all modality tokens
        attn_out, _ = self.cross_attn(all_tokens, all_tokens, all_tokens)
        all_tokens = self.norm(all_tokens + attn_out)
        all_tokens = self.norm_ff(all_tokens + self.ff(all_tokens))

        # Mean pool all tokens → fused embedding
        fused = all_tokens.mean(dim=1)  # (B, proj_dim)

        # Hyperbolic round-trip: exp_map → log_map (learn in curved space)
        h = exp_map(fused)
        e = log_map(h)

        return e


class CrossAttentionWithHead(nn.Module):
    """Encoder + classification head for training."""

    def __init__(self, input_dim, proj_dim=128, num_heads=4, dropout=0.1, num_classes=89):
        super().__init__()
        self.encoder = CrossAttentionEncoder(input_dim, proj_dim, num_heads, dropout)
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, audio_embs, image_emb):
        e = self.encoder(audio_embs, image_emb)
        return self.head(e)

    def extract(self, audio_embs, image_emb):
        """Extract fused embedding without classification."""
        return self.encoder(audio_embs, image_emb)


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def load_and_align_embeddings():
    """Load all 4 modalities and align by file hash."""
    hub_emb = torch.load("wav2vec2_hubert_wavlm/hubert_embeddings.pt", map_location="cpu", weights_only=True)
    wavlm_emb = torch.load("wav2vec2_hubert_wavlm/wavlm_embeddings.pt", map_location="cpu", weights_only=True)
    wav2vec_emb = torch.load("wav2vec2_hubert_wavlm/wav2vec2_embeddings.pt", map_location="cpu", weights_only=True)
    vgg_emb = torch.load("vgg19/vgg19_embeddings_all.pt", map_location="cpu", weights_only=True)

    audio_paths = torch.load("wav2vec2_hubert_wavlm/wavlm_paths.pt", weights_only=False)
    img_paths = torch.load("vgg19/vgg19_paths.pt", weights_only=False)

    audio_labels = torch.load("wav2vec2_hubert_wavlm/labels.pt", map_location="cpu", weights_only=True)
    img_labels = torch.load("vgg19/labels_all.pt", map_location="cpu", weights_only=True)

    def get_hash(p):
        return os.path.basename(str(p)).replace(".wav", "").replace(".png", "")

    audio_dict = {get_hash(p): i for i, p in enumerate(audio_paths)}
    img_dict = {get_hash(p): i for i, p in enumerate(img_paths)}

    common_hashes = sorted(set(audio_dict.keys()) & set(img_dict.keys()))
    audio_indices = [audio_dict[h] for h in common_hashes]
    img_indices = [img_dict[h] for h in common_hashes]

    X_hub = hub_emb[audio_indices].numpy().astype(np.float32)
    X_wavlm = wavlm_emb[audio_indices].numpy().astype(np.float32)
    X_wav2vec = wav2vec_emb[audio_indices].numpy().astype(np.float32)
    X_image = vgg_emb[img_indices].numpy().astype(np.float32)
    y = audio_labels[audio_indices].numpy().astype(np.int64)

    y_img = img_labels[img_indices].numpy().astype(np.int64)
    assert np.all(y == y_img), "Label mismatch!"

    return X_hub, X_wavlm, X_wav2vec, X_image, y


def filter_by_threshold(X_hub, X_wavlm, X_wav2vec, X_image, y, threshold):
    """Keep classes with >= threshold samples, remap labels."""
    counts = Counter(y)
    keep = {c for c, n in counts.items() if n >= threshold}
    if not keep:
        return None, None, None, None, None, 0

    mask = np.array([l in keep for l in y])
    X_h, X_w, X_v, X_i, y_f = X_hub[mask], X_wavlm[mask], X_wav2vec[mask], X_image[mask], y[mask]

    unique = sorted(set(y_f))
    remap = {old: new for new, old in enumerate(unique)}
    y_r = np.array([remap[l] for l in y_f])
    return X_h, X_w, X_v, X_i, y_r, len(unique)


def train_encoder_and_extract(audio_train, image_train, y_train,
                              audio_test, image_test,
                              pca_dim, proj_dim, n_classes, epochs=30, lr=1e-3):
    """
    Train the cross-attention encoder on training data,
    then extract fused embeddings for both train and test.
    """
    model = CrossAttentionWithHead(
        input_dim=pca_dim, proj_dim=proj_dim,
        num_heads=4, dropout=0.1, num_classes=n_classes
    )

    # Class weights
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes
    class_weights = torch.tensor(weights, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    audio_t = torch.tensor(audio_train, dtype=torch.float32)
    image_t = torch.tensor(image_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    n = len(y_train)
    batch_size = 64

    # Quick training (just enough to learn good alignment, not overfit)
    model.train()
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            logits = model(audio_t[idx], image_t[idx])
            loss = criterion(logits, y_t[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        audio_test_t = torch.tensor(audio_test, dtype=torch.float32)
        image_test_t = torch.tensor(image_test, dtype=torch.float32)

        train_emb = model.extract(audio_t, image_t).numpy()
        test_emb = model.extract(audio_test_t, image_test_t).numpy()

    return train_emb, test_emb


def run_fold(X_h_tr, X_w_tr, X_v_tr, X_i_tr, y_tr,
             X_h_te, X_w_te, X_v_te, X_i_te, y_te,
             n_classes, pca_dim=128, proj_dim=128,
             encoder_epochs=30, svm_C=100, svm_gamma="scale"):
    """
    Full pipeline for one fold:
    1. PCA per modality (fit on train)
    2. Train cross-attention encoder → extract fused embeddings
    3. SVM-RBF on fused embeddings
    """
    # ─── PCA per modality ───
    pca_h = PCA(n_components=pca_dim, random_state=42)
    pca_w = PCA(n_components=pca_dim, random_state=42)
    pca_v = PCA(n_components=pca_dim, random_state=42)
    pca_i = PCA(n_components=pca_dim, random_state=42)

    Xh_tr = pca_h.fit_transform(X_h_tr)
    Xh_te = pca_h.transform(X_h_te)
    Xw_tr = pca_w.fit_transform(X_w_tr)
    Xw_te = pca_w.transform(X_w_te)
    Xv_tr = pca_v.fit_transform(X_v_tr)
    Xv_te = pca_v.transform(X_v_te)
    Xi_tr = pca_i.fit_transform(X_i_tr)
    Xi_te = pca_i.transform(X_i_te)

    # ─── Normalize ───
    sc_h, sc_w, sc_v, sc_i = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
    Xh_tr = sc_h.fit_transform(Xh_tr)
    Xh_te = sc_h.transform(Xh_te)
    Xw_tr = sc_w.fit_transform(Xw_tr)
    Xw_te = sc_w.transform(Xw_te)
    Xv_tr = sc_v.fit_transform(Xv_tr)
    Xv_te = sc_v.transform(Xv_te)
    Xi_tr = sc_i.fit_transform(Xi_tr)
    Xi_te = sc_i.transform(Xi_te)

    # ─── METHOD 1: Simple concat + SVM-RBF (baseline ceiling) ───
    concat_tr = np.concatenate([Xh_tr, Xw_tr, Xv_tr, Xi_tr], axis=1)
    concat_te = np.concatenate([Xh_te, Xw_te, Xv_te, Xi_te], axis=1)

    svm_concat = SVC(C=svm_C, kernel="rbf", gamma=svm_gamma,
                     class_weight="balanced", random_state=42)
    svm_concat.fit(concat_tr, y_tr)
    concat_preds = svm_concat.predict(concat_te)
    concat_f1 = f1_score(y_te, concat_preds, average="macro", zero_division=0)
    concat_wf1 = f1_score(y_te, concat_preds, average="weighted", zero_division=0)
    concat_train_f1 = f1_score(y_tr, svm_concat.predict(concat_tr), average="macro", zero_division=0)

    # ─── METHOD 2: Cross-Attention Hyperbolic + SVM-RBF ───
    # Stack audio as (N, 3, pca_dim)
    audio_tr = np.stack([Xh_tr, Xw_tr, Xv_tr], axis=1)
    audio_te = np.stack([Xh_te, Xw_te, Xv_te], axis=1)

    # Train encoder and extract fused embeddings
    fused_tr, fused_te = train_encoder_and_extract(
        audio_tr, Xi_tr, y_tr, audio_te, Xi_te,
        pca_dim=pca_dim, proj_dim=proj_dim,
        n_classes=n_classes, epochs=encoder_epochs, lr=1e-3,
    )

    # Normalize fused embeddings
    sc_fused = StandardScaler()
    fused_tr = sc_fused.fit_transform(fused_tr)
    fused_te = sc_fused.transform(fused_te)

    # SVM-RBF on fused embeddings
    svm_fused = SVC(C=svm_C, kernel="rbf", gamma=svm_gamma,
                    class_weight="balanced", random_state=42)
    svm_fused.fit(fused_tr, y_tr)
    fused_preds = svm_fused.predict(fused_te)
    fused_f1 = f1_score(y_te, fused_preds, average="macro", zero_division=0)
    fused_wf1 = f1_score(y_te, fused_preds, average="weighted", zero_division=0)
    fused_train_f1 = f1_score(y_tr, svm_fused.predict(fused_tr), average="macro", zero_division=0)

    # ─── METHOD 3: Concat + Hyperbolic projection + SVM-RBF ───
    # Project concat to hyperbolic space (static, no learning)
    concat_tr_t = torch.tensor(concat_tr, dtype=torch.float32)
    concat_te_t = torch.tensor(concat_te, dtype=torch.float32)
    hyp_tr = log_map(exp_map(concat_tr_t)).numpy()
    hyp_te = log_map(exp_map(concat_te_t)).numpy()

    sc_hyp = StandardScaler()
    hyp_tr = sc_hyp.fit_transform(hyp_tr)
    hyp_te = sc_hyp.transform(hyp_te)

    svm_hyp = SVC(C=svm_C, kernel="rbf", gamma=svm_gamma,
                  class_weight="balanced", random_state=42)
    svm_hyp.fit(hyp_tr, y_tr)
    hyp_preds = svm_hyp.predict(hyp_te)
    hyp_f1 = f1_score(y_te, hyp_preds, average="macro", zero_division=0)
    hyp_wf1 = f1_score(y_te, hyp_preds, average="weighted", zero_division=0)

    return {
        "concat_f1": concat_f1, "concat_wf1": concat_wf1, "concat_train_f1": concat_train_f1,
        "fused_f1": fused_f1, "fused_wf1": fused_wf1, "fused_train_f1": fused_train_f1,
        "hyp_concat_f1": hyp_f1, "hyp_concat_wf1": hyp_wf1,
    }


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10, 8, 5])
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=30)
    parser.add_argument("--svm-C", type=float, default=100)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)

    print("\n" + "=" * 100)
    print("  CROSS-ATTENTION HYPERBOLIC FUSION + SVM-RBF")
    print("  HuBERT + WavLM + Wav2Vec2 + VGG19 | K-Fold + PCA per fold (NO leakage)")
    print("=" * 100)
    print(f"\n  Config:")
    print(f"    Thresholds: {thresholds}")
    print(f"    PCA dim: {args.pca_dim} | Proj dim: {args.proj_dim}")
    print(f"    Encoder epochs: {args.encoder_epochs}")
    print(f"    SVM: C={args.svm_C}, gamma={args.svm_gamma}, balanced")
    print(f"    K-Fold: {args.n_folds}-fold Stratified")
    print()

    # Load
    print("  Loading embeddings... ", end="", flush=True)
    X_hub, X_wavlm, X_wav2vec, X_image, y = load_and_align_embeddings()
    print(f"done ({X_hub.shape[0]} samples)")
    print(f"    Audio: 3×{X_hub.shape[1]}d | Image: {X_image.shape[1]}d")
    print()

    for threshold in thresholds:
        print(f"\n{'─' * 100}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'─' * 100}")

        X_h, X_w, X_v, X_i, y_f, nc = filter_by_threshold(
            X_hub, X_wavlm, X_wav2vec, X_image, y, threshold
        )
        if X_h is None:
            print("  SKIPPED")
            continue

        print(f"  Samples: {X_h.shape[0]}, Classes: {nc}")

        min_count = min(Counter(y_f).values())
        n_folds = min(args.n_folds, min_count)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        all_concat, all_fused, all_hyp = [], [], []
        all_concat_w, all_fused_w = [], []
        all_concat_train, all_fused_train = [], []

        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_h, y_f)):
            t0 = time.time()

            result = run_fold(
                X_h[tr_idx], X_w[tr_idx], X_v[tr_idx], X_i[tr_idx], y_f[tr_idx],
                X_h[te_idx], X_w[te_idx], X_v[te_idx], X_i[te_idx], y_f[te_idx],
                n_classes=nc, pca_dim=args.pca_dim, proj_dim=args.proj_dim,
                encoder_epochs=args.encoder_epochs, svm_C=args.svm_C,
                svm_gamma=args.svm_gamma,
            )

            elapsed = time.time() - t0
            all_concat.append(result["concat_f1"])
            all_fused.append(result["fused_f1"])
            all_hyp.append(result["hyp_concat_f1"])
            all_concat_w.append(result["concat_wf1"])
            all_fused_w.append(result["fused_wf1"])
            all_concat_train.append(result["concat_train_f1"])
            all_fused_train.append(result["fused_train_f1"])

            print(f"    Fold {fold_idx+1}: Concat={result['concat_f1']:.4f}  "
                  f"CrossAttn+Hyp={result['fused_f1']:.4f}  "
                  f"HypConcat={result['hyp_concat_f1']:.4f}  ({elapsed:.0f}s)")

        print()
        print(f"  ┌─────────────────────────────────────────────────────────────────────┐")
        print(f"  │  Threshold >= {threshold} ({nc} classes, {X_h.shape[0]} samples)")
        print(f"  ├─────────────────────────────────────────────────────────────────────┤")
        print(f"  │  [1] Concat+SVM:            {np.mean(all_concat):.4f} ± {np.std(all_concat):.4f}  "
              f"(train: {np.mean(all_concat_train):.4f})")
        print(f"  │  [2] CrossAttn+Hyp+SVM:     {np.mean(all_fused):.4f} ± {np.std(all_fused):.4f}  "
              f"(train: {np.mean(all_fused_train):.4f})")
        print(f"  │  [3] HypConcat+SVM:         {np.mean(all_hyp):.4f} ± {np.std(all_hyp):.4f}")
        print(f"  ├─────────────────────────────────────────────────────────────────────┤")
        baselines = {10: 0.6870, 8: 0.6831, 5: 0.5992}
        best_single = baselines[threshold]
        print(f"  │  Best single-modality:      HuBERT SVM-RBF = {best_single:.4f}")
        best_fusion = max(np.mean(all_concat), np.mean(all_fused), np.mean(all_hyp))
        gain = best_fusion - best_single
        print(f"  │  Best fusion gain:          {'↑' if gain > 0 else '↓'} {abs(gain):.4f}")
        gap = 0.80 - best_fusion
        print(f"  │  Gap to 0.80 target:        {gap:.4f}")
        print(f"  └─────────────────────────────────────────────────────────────────────┘")

    print(f"\n{'=' * 100}\n")


if __name__ == "__main__":
    main()
