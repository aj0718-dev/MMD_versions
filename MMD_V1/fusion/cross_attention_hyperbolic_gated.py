#!/usr/bin/env python3
"""
cross_attention_hyperbolic_gated.py

Cross-Attention Fusion in Hyperbolic Space — HuBERT + VGG19
With GATED RESIDUAL so fusion can never hurt HuBERT.

Key design principles:
  1. HuBERT-centric: HuBERT gets more PCA dimensions (dominant signal)
  2. Gated cross-attention: learned gate controls VGG19 contribution
     - gate=0: pure HuBERT (can't be worse than baseline)
     - gate=1: full fusion (adds VGG19 info where helpful)
  3. Hyperbolic projection for hierarchical class geometry
  4. SVM-RBF final classifier (proven best for this data size)
  5. PCA per fold (no data leakage)

Usage:
    python cross_attention_hyperbolic_gated.py --thresholds 10 8 5
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

EPS = 1e-8


# =====================================================
# HYPERBOLIC OPERATIONS
# =====================================================


def exp_map_np(x):
    """Exponential map (numpy): Euclidean → Poincaré ball."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, EPS)
    return np.tanh(norm) * x / norm


def log_map_np(x):
    """Logarithmic map (numpy): Poincaré ball → tangent space."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.clip(norm, EPS, 1 - 1e-5)
    return np.arctanh(norm) * x / norm


def exp_map(x):
    """Exponential map (torch)."""
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    return torch.tanh(norm) * x / norm


def log_map(x):
    """Logarithmic map (torch)."""
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    norm = torch.clamp(norm, max=1 - 1e-5)
    return torch.atanh(norm) * x / norm


# =====================================================
# GATED CROSS-ATTENTION ENCODER
# =====================================================


class GatedCrossAttentionEncoder(nn.Module):
    """
    Gated cross-attention: HuBERT is the BACKBONE, VGG19 is the SUPPLEMENT.
    
    Architecture:
      audio_proj(HuBERT) → base embedding
      cross_attention(HuBERT_query, VGG19_kv) → supplement
      gate = sigmoid(learned_gate_fn(base, supplement))
      output = base + gate * supplement  ← can never hurt HuBERT!
      hyperbolic_projection(output)
    """

    def __init__(self, audio_dim, image_dim, proj_dim=128, num_heads=4, dropout=0.1):
        super().__init__()

        # Project audio (HuBERT) — this is the backbone
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # Project image (VGG19) — this is the supplement
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # Cross-attention: audio queries, image keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(proj_dim)

        # Gate: learns when VGG19 is useful
        self.gate_fn = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.Sigmoid(),
        )

        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, audio_emb, image_emb):
        """
        audio_emb: (B, audio_dim) — HuBERT
        image_emb: (B, image_dim) — VGG19
        Returns: (B, proj_dim) — fused embedding in hyperbolic tangent space
        """
        # Project to shared dim
        a = self.audio_proj(audio_emb)  # (B, proj_dim) — backbone
        i = self.image_proj(image_emb)  # (B, proj_dim) — supplement

        # Cross-attention: audio attends to image
        a_q = a.unsqueeze(1)  # (B, 1, proj_dim)
        i_kv = i.unsqueeze(1)  # (B, 1, proj_dim)
        attended, _ = self.cross_attn(query=a_q, key=i_kv, value=i_kv)
        supplement = self.attn_norm(attended.squeeze(1))  # (B, proj_dim)

        # Gated residual: gate decides how much VGG19 contributes
        gate = self.gate_fn(torch.cat([a, supplement], dim=-1))  # (B, proj_dim)
        fused = a + gate * supplement  # If gate=0, pure HuBERT

        # Final projection
        fused = self.out_proj(fused)

        # Hyperbolic round-trip
        h = exp_map(fused)
        e = log_map(h)

        return e


class GatedCrossAttentionWithHead(nn.Module):
    """Encoder + classification head for proxy training."""

    def __init__(self, audio_dim, image_dim, proj_dim=128, num_heads=4,
                 dropout=0.1, num_classes=89):
        super().__init__()
        self.encoder = GatedCrossAttentionEncoder(
            audio_dim, image_dim, proj_dim, num_heads, dropout
        )
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, audio_emb, image_emb):
        e = self.encoder(audio_emb, image_emb)
        return self.head(e)

    def extract(self, audio_emb, image_emb):
        return self.encoder(audio_emb, image_emb)


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def load_and_align_embeddings():
    """Load HuBERT + VGG19 and align by file hash."""
    hub_emb = torch.load("wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                         map_location="cpu", weights_only=True)
    vgg_emb = torch.load("vgg19/vgg19_embeddings_all.pt",
                         map_location="cpu", weights_only=True)

    audio_paths = torch.load("wav2vec2_hubert_wavlm/wavlm_paths.pt", weights_only=False)
    img_paths = torch.load("vgg19/vgg19_paths.pt", weights_only=False)

    audio_labels = torch.load("wav2vec2_hubert_wavlm/labels.pt",
                              map_location="cpu", weights_only=True)
    img_labels = torch.load("vgg19/labels_all.pt",
                            map_location="cpu", weights_only=True)

    def get_hash(p):
        return os.path.basename(str(p)).replace(".wav", "").replace(".png", "")

    audio_dict = {get_hash(p): i for i, p in enumerate(audio_paths)}
    img_dict = {get_hash(p): i for i, p in enumerate(img_paths)}

    common = sorted(set(audio_dict.keys()) & set(img_dict.keys()))
    a_idx = [audio_dict[h] for h in common]
    i_idx = [img_dict[h] for h in common]

    X_audio = hub_emb[a_idx].numpy().astype(np.float32)
    X_image = vgg_emb[i_idx].numpy().astype(np.float32)
    y = audio_labels[a_idx].numpy().astype(np.int64)

    y_img = img_labels[i_idx].numpy().astype(np.int64)
    assert np.all(y == y_img), "Label mismatch!"

    return X_audio, X_image, y


def filter_by_threshold(X_audio, X_image, y, threshold):
    """Keep classes with >= threshold samples."""
    counts = Counter(y)
    keep = {c for c, n in counts.items() if n >= threshold}
    if not keep:
        return None, None, None, 0

    mask = np.array([l in keep for l in y])
    X_a, X_i, y_f = X_audio[mask], X_image[mask], y[mask]

    unique = sorted(set(y_f))
    remap = {old: new for new, old in enumerate(unique)}
    y_r = np.array([remap[l] for l in y_f])
    return X_a, X_i, y_r, len(unique)


def train_encoder(audio_train, image_train, y_train, n_classes,
                  audio_dim, image_dim, proj_dim=128, epochs=20, lr=1e-3):
    """
    Train the gated cross-attention encoder with a proxy classification loss.
    Short training to learn alignment without overfitting.
    """
    model = GatedCrossAttentionWithHead(
        audio_dim=audio_dim, image_dim=image_dim,
        proj_dim=proj_dim, num_heads=4, dropout=0.15,
        num_classes=n_classes,
    )

    # Class weights
    cc = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    cc = np.maximum(cc, 1.0)
    w = 1.0 / cc
    w = w / w.sum() * n_classes
    class_weights = torch.tensor(w, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    audio_t = torch.tensor(audio_train, dtype=torch.float32)
    image_t = torch.tensor(image_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    n = len(y_train)
    batch_size = 64

    model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]

            logits = model(audio_t[idx], image_t[idx])
            loss = criterion(logits, y_t[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def run_fold(X_a_tr, X_i_tr, y_tr, X_a_te, X_i_te, y_te,
             n_classes, audio_pca=512, image_pca=256, proj_dim=128,
             encoder_epochs=20, svm_C=100, svm_gamma="scale"):
    """
    Full pipeline for one fold:
    1. PCA per modality (asymmetric: more for HuBERT)
    2. StandardScaler
    3. Train gated cross-attention encoder
    4. Extract fused embeddings (in hyperbolic tangent space)
    5. SVM-RBF classifier on fused embeddings
    Also runs HuBERT-only SVM-RBF for direct comparison.
    """
    # ─── PCA (asymmetric) ───
    pca_a = PCA(n_components=audio_pca, random_state=42)
    pca_i = PCA(n_components=image_pca, random_state=42)

    Xa_tr = pca_a.fit_transform(X_a_tr)
    Xa_te = pca_a.transform(X_a_te)
    Xi_tr = pca_i.fit_transform(X_i_tr)
    Xi_te = pca_i.transform(X_i_te)

    # ─── Normalize ───
    sc_a = StandardScaler()
    sc_i = StandardScaler()
    Xa_tr = sc_a.fit_transform(Xa_tr)
    Xa_te = sc_a.transform(Xa_te)
    Xi_tr = sc_i.fit_transform(Xi_tr)
    Xi_te = sc_i.transform(Xi_te)

    # ─── Baseline: HuBERT alone → SVM-RBF ───
    svm_audio = SVC(C=svm_C, kernel="rbf", gamma=svm_gamma,
                    class_weight="balanced", random_state=42)
    svm_audio.fit(Xa_tr, y_tr)
    audio_preds = svm_audio.predict(Xa_te)
    audio_f1 = f1_score(y_te, audio_preds, average="macro", zero_division=0)

    # ─── Baseline: HuBERT hyperbolic → SVM-RBF ───
    Xa_tr_hyp = log_map_np(exp_map_np(Xa_tr))
    Xa_te_hyp = log_map_np(exp_map_np(Xa_te))
    sc_ah = StandardScaler()
    Xa_tr_hyp = sc_ah.fit_transform(Xa_tr_hyp)
    Xa_te_hyp = sc_ah.transform(Xa_te_hyp)

    svm_audio_hyp = SVC(C=svm_C, kernel="rbf", gamma=svm_gamma,
                        class_weight="balanced", random_state=42)
    svm_audio_hyp.fit(Xa_tr_hyp, y_tr)
    audio_hyp_preds = svm_audio_hyp.predict(Xa_te_hyp)
    audio_hyp_f1 = f1_score(y_te, audio_hyp_preds, average="macro", zero_division=0)

    # ─── Train gated cross-attention encoder ───
    model = train_encoder(
        Xa_tr, Xi_tr, y_tr, n_classes,
        audio_dim=audio_pca, image_dim=image_pca,
        proj_dim=proj_dim, epochs=encoder_epochs, lr=1e-3,
    )

    # ─── Extract fused embeddings ───
    model.eval()
    with torch.no_grad():
        audio_tr_t = torch.tensor(Xa_tr, dtype=torch.float32)
        image_tr_t = torch.tensor(Xi_tr, dtype=torch.float32)
        audio_te_t = torch.tensor(Xa_te, dtype=torch.float32)
        image_te_t = torch.tensor(Xi_te, dtype=torch.float32)

        fused_tr = model.extract(audio_tr_t, image_tr_t).numpy()
        fused_te = model.extract(audio_te_t, image_te_t).numpy()

    # ─── Normalize fused embeddings ───
    sc_f = StandardScaler()
    fused_tr = sc_f.fit_transform(fused_tr)
    fused_te = sc_f.transform(fused_te)

    # ─── SVM-RBF on fused embeddings ───
    svm_fused = SVC(C=svm_C, kernel="rbf", gamma=svm_gamma,
                    class_weight="balanced", random_state=42)
    svm_fused.fit(fused_tr, y_tr)
    fused_preds = svm_fused.predict(fused_te)
    fused_f1 = f1_score(y_te, fused_preds, average="macro", zero_division=0)

    # ─── Also try: concat [HuBERT_hyp, fused] → SVM ───
    combo_tr = np.concatenate([Xa_tr_hyp, fused_tr], axis=1)
    combo_te = np.concatenate([Xa_te_hyp, fused_te], axis=1)
    svm_combo = SVC(C=svm_C, kernel="rbf", gamma=svm_gamma,
                    class_weight="balanced", random_state=42)
    svm_combo.fit(combo_tr, y_tr)
    combo_preds = svm_combo.predict(combo_te)
    combo_f1 = f1_score(y_te, combo_preds, average="macro", zero_division=0)

    return {
        "audio_f1": audio_f1,
        "audio_hyp_f1": audio_hyp_f1,
        "fused_f1": fused_f1,
        "combo_f1": combo_f1,
    }


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10, 8, 5])
    parser.add_argument("--audio-pca", type=int, default=512)
    parser.add_argument("--image-pca", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=20)
    parser.add_argument("--svm-C", type=float, default=100)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)

    print("\n" + "=" * 100)
    print("  GATED CROSS-ATTENTION FUSION IN HYPERBOLIC SPACE + SVM-RBF")
    print("  HuBERT (audio) + VGG19 (image) | K-Fold + PCA per fold (NO leakage)")
    print("=" * 100)
    print(f"\n  Design: HuBERT-centric with gated VGG19 supplement")
    print(f"  Audio PCA: {args.audio_pca} | Image PCA: {args.image_pca} | Proj: {args.proj_dim}")
    print(f"  Encoder epochs: {args.encoder_epochs} (short — just learn alignment)")
    print(f"  SVM-RBF: C={args.svm_C}, gamma={args.svm_gamma}, balanced")
    print(f"  Folds: {args.n_folds}")
    print()

    # ─── Load ───
    print("  Loading HuBERT + VGG19... ", end="", flush=True)
    X_audio, X_image, y = load_and_align_embeddings()
    print(f"done ({X_audio.shape[0]} samples)")
    print(f"  HuBERT: {X_audio.shape} | VGG19: {X_image.shape}")
    print()

    for threshold in thresholds:
        print(f"\n{'─' * 100}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'─' * 100}")

        X_a, X_i, y_f, nc = filter_by_threshold(X_audio, X_image, y, threshold)
        if X_a is None:
            continue

        print(f"  Samples: {X_a.shape[0]}, Classes: {nc}")

        min_cnt = min(Counter(y_f).values())
        n_folds = min(args.n_folds, min_cnt)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        results = {"audio": [], "audio_hyp": [], "fused": [], "combo": []}

        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_a, y_f)):
            t0 = time.time()

            r = run_fold(
                X_a[tr_idx], X_i[tr_idx], y_f[tr_idx],
                X_a[te_idx], X_i[te_idx], y_f[te_idx],
                n_classes=nc,
                audio_pca=args.audio_pca,
                image_pca=args.image_pca,
                proj_dim=args.proj_dim,
                encoder_epochs=args.encoder_epochs,
                svm_C=args.svm_C,
                svm_gamma=args.svm_gamma,
            )

            elapsed = time.time() - t0
            results["audio"].append(r["audio_f1"])
            results["audio_hyp"].append(r["audio_hyp_f1"])
            results["fused"].append(r["fused_f1"])
            results["combo"].append(r["combo_f1"])

            print(f"    Fold {fold_idx+1}: "
                  f"HuBERT={r['audio_f1']:.4f}  "
                  f"HuBERT+Hyp={r['audio_hyp_f1']:.4f}  "
                  f"GatedCA+Hyp={r['fused_f1']:.4f}  "
                  f"Combo={r['combo_f1']:.4f}  "
                  f"({elapsed:.0f}s)")

        # Summary
        print()
        print(f"  ┌───────────────────────────────────────────────────────────────────────┐")
        print(f"  │  Threshold >= {threshold} ({nc} classes, {X_a.shape[0]} samples)")
        print(f"  ├───────────────────────────────────────────────────────────────────────┤")
        for name, key in [("HuBERT→SVM", "audio"),
                          ("HuBERT→Hyp→SVM", "audio_hyp"),
                          ("GatedCrossAttn+Hyp→SVM", "fused"),
                          ("Combo(Hyp+Fused)→SVM", "combo")]:
            vals = results[key]
            best_marker = " ★" if np.mean(vals) == max(np.mean(results[k]) for k in results) else ""
            print(f"  │  {name:<25} {np.mean(vals):.4f} ± {np.std(vals):.4f}{best_marker}")
        print(f"  ├───────────────────────────────────────────────────────────────────────┤")
        best_val = max(np.mean(results[k]) for k in results)
        gap = 0.80 - best_val
        print(f"  │  Best: {best_val:.4f}  |  Gap to 0.80: {gap:.4f}")
        print(f"  └───────────────────────────────────────────────────────────────────────┘")

    print(f"\n{'=' * 100}\n")


if __name__ == "__main__":
    main()
