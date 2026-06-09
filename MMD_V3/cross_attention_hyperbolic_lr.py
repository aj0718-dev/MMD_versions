#!/usr/bin/env python3
"""
cross_attention_hyperbolic_lr.py

Cross-Attention Fusion in Hyperbolic Space + LogReg/SVM classifier.
Uses all 4 modalities: HuBERT + WavLM + Wav2Vec2 + VGG19.

KEY FIX: Global PCA on raw concatenated features (6400d → N components)
instead of per-modality PCA which dilutes strong modality signal.

Methods tested:
  [1] Global concat → PCA → LogReg (best LR config)
  [2] Global concat → PCA → SVM-RBF  
  [3] Cross-Attention on per-modality PCA → Hyperbolic → LogReg
  [4] Cross-Attention on per-modality PCA → Hyperbolic → SVM-RBF

Usage:
    python cross_attention_hyperbolic_lr.py --thresholds 10 8 5
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

EPS = 1e-8


# =====================================================
# HYPERBOLIC OPS
# =====================================================

def exp_map(x):
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    return torch.tanh(norm) * x / norm

def log_map(x):
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    norm = torch.clamp(norm, max=1 - 1e-5)
    return torch.atanh(norm) * x / norm


# =====================================================
# CROSS-ATTENTION ENCODER (lightweight)
# =====================================================

class CrossAttentionEncoder(nn.Module):
    def __init__(self, input_dim, proj_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU()
        )
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU()
        )
        self.audio_tokens = nn.Parameter(torch.randn(3, proj_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.ff = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.norm_ff = nn.LayerNorm(proj_dim)

    def forward(self, audio_embs, image_emb):
        a = self.audio_proj(audio_embs) + self.audio_tokens.unsqueeze(0)
        i = self.image_proj(image_emb).unsqueeze(1)
        tokens = torch.cat([a, i], dim=1)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        tokens = self.norm_ff(tokens + self.ff(tokens))
        fused = tokens.mean(dim=1)
        h = exp_map(fused)
        return log_map(h)


class CrossAttentionWithHead(nn.Module):
    def __init__(self, input_dim, proj_dim=128, num_heads=4, dropout=0.1, num_classes=89):
        super().__init__()
        self.encoder = CrossAttentionEncoder(input_dim, proj_dim, num_heads, dropout)
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, audio_embs, image_emb):
        return self.head(self.encoder(audio_embs, image_emb))

    def extract(self, audio_embs, image_emb):
        return self.encoder(audio_embs, image_emb)


# =====================================================
# LOAD DATA
# =====================================================

def load_and_align():
    hub = torch.load("wav2vec2_hubert_wavlm/hubert_embeddings.pt", map_location="cpu", weights_only=True)
    wavlm = torch.load("wav2vec2_hubert_wavlm/wavlm_embeddings.pt", map_location="cpu", weights_only=True)
    wav2vec = torch.load("wav2vec2_hubert_wavlm/wav2vec2_embeddings.pt", map_location="cpu", weights_only=True)
    vgg = torch.load("vgg19/vgg19_embeddings_all.pt", map_location="cpu", weights_only=True)

    audio_paths = torch.load("wav2vec2_hubert_wavlm/wavlm_paths.pt", weights_only=False)
    img_paths = torch.load("vgg19/vgg19_paths.pt", weights_only=False)
    audio_labels = torch.load("wav2vec2_hubert_wavlm/labels.pt", map_location="cpu", weights_only=True)
    img_labels = torch.load("vgg19/labels_all.pt", map_location="cpu", weights_only=True)

    def get_hash(p):
        return os.path.basename(str(p)).replace(".wav", "").replace(".png", "")

    audio_dict = {get_hash(p): i for i, p in enumerate(audio_paths)}
    img_dict = {get_hash(p): i for i, p in enumerate(img_paths)}
    common = sorted(set(audio_dict) & set(img_dict))
    ai = [audio_dict[h] for h in common]
    ii = [img_dict[h] for h in common]

    X_hub = hub[ai].numpy().astype(np.float32)
    X_wavlm = wavlm[ai].numpy().astype(np.float32)
    X_wav2vec = wav2vec[ai].numpy().astype(np.float32)
    X_img = vgg[ii].numpy().astype(np.float32)
    y = audio_labels[ai].numpy().astype(np.int64)
    assert np.all(y == img_labels[ii].numpy()), "Label mismatch!"

    return X_hub, X_wavlm, X_wav2vec, X_img, y


def filter_threshold(X_h, X_w, X_v, X_i, y, threshold):
    counts = Counter(y)
    keep = {c for c, n in counts.items() if n >= threshold}
    if not keep:
        return [None]*5 + [0]
    mask = np.array([l in keep for l in y])
    yf = y[mask]
    unique = sorted(set(yf))
    remap = {old: new for new, old in enumerate(unique)}
    yr = np.array([remap[l] for l in yf])
    return X_h[mask], X_w[mask], X_v[mask], X_i[mask], yr, len(unique)


# =====================================================
# FOLD RUNNER
# =====================================================

def run_fold(X_h_tr, X_w_tr, X_v_tr, X_i_tr, y_tr,
             X_h_te, X_w_te, X_v_te, X_i_te, y_te,
             n_classes, global_pca_dim=256, modal_pca_dim=128,
             proj_dim=128, encoder_epochs=25):
    """Run all methods for one fold."""

    results = {}

    # ═══ GLOBAL CONCAT approach ═══
    # Concat raw → single PCA (captures best components across all modalities)
    raw_tr = np.concatenate([X_h_tr, X_w_tr, X_v_tr, X_i_tr], axis=1)  # 6400d
    raw_te = np.concatenate([X_h_te, X_w_te, X_v_te, X_i_te], axis=1)

    pca_global = PCA(n_components=global_pca_dim, random_state=42)
    G_tr = pca_global.fit_transform(raw_tr)
    G_te = pca_global.transform(raw_te)

    sc_g = StandardScaler()
    G_tr = sc_g.fit_transform(G_tr)
    G_te = sc_g.transform(G_te)

    # [1] Global PCA → LogReg
    lr = LogisticRegression(C=1000, solver="lbfgs", max_iter=5000,
                            class_weight="balanced", random_state=42)
    lr.fit(G_tr, y_tr)
    results["global_lr"] = f1_score(y_te, lr.predict(G_te), average="macro", zero_division=0)
    results["global_lr_train"] = f1_score(y_tr, lr.predict(G_tr), average="macro", zero_division=0)

    # [2] Global PCA → SVM-RBF
    svm = SVC(C=100, kernel="rbf", gamma="scale", class_weight="balanced", random_state=42)
    svm.fit(G_tr, y_tr)
    results["global_svm"] = f1_score(y_te, svm.predict(G_te), average="macro", zero_division=0)
    results["global_svm_train"] = f1_score(y_tr, svm.predict(G_tr), average="macro", zero_division=0)

    # [3] Global PCA → Hyperbolic → SVM-RBF
    G_tr_t = torch.tensor(G_tr, dtype=torch.float32)
    G_te_t = torch.tensor(G_te, dtype=torch.float32)
    H_tr = log_map(exp_map(G_tr_t)).numpy()
    H_te = log_map(exp_map(G_te_t)).numpy()
    sc_h = StandardScaler()
    H_tr = sc_h.fit_transform(H_tr)
    H_te = sc_h.transform(H_te)

    svm_h = SVC(C=100, kernel="rbf", gamma="scale", class_weight="balanced", random_state=42)
    svm_h.fit(H_tr, y_tr)
    results["global_hyp_svm"] = f1_score(y_te, svm_h.predict(H_te), average="macro", zero_division=0)

    # [4] Global PCA → Hyperbolic → LogReg
    lr_h = LogisticRegression(C=1000, solver="lbfgs", max_iter=5000,
                              class_weight="balanced", random_state=42)
    lr_h.fit(H_tr, y_tr)
    results["global_hyp_lr"] = f1_score(y_te, lr_h.predict(H_te), average="macro", zero_division=0)

    # ═══ CROSS-ATTENTION approach ═══
    # Per-modality PCA → cross-attention encoder → extract → classify
    pca_h = PCA(n_components=modal_pca_dim, random_state=42)
    pca_w = PCA(n_components=modal_pca_dim, random_state=42)
    pca_v = PCA(n_components=modal_pca_dim, random_state=42)
    pca_i = PCA(n_components=modal_pca_dim, random_state=42)

    Mh_tr = pca_h.fit_transform(X_h_tr); Mh_te = pca_h.transform(X_h_te)
    Mw_tr = pca_w.fit_transform(X_w_tr); Mw_te = pca_w.transform(X_w_te)
    Mv_tr = pca_v.fit_transform(X_v_tr); Mv_te = pca_v.transform(X_v_te)
    Mi_tr = pca_i.fit_transform(X_i_tr); Mi_te = pca_i.transform(X_i_te)

    # Normalize
    for arr_pair in [(Mh_tr, Mh_te), (Mw_tr, Mw_te), (Mv_tr, Mv_te), (Mi_tr, Mi_te)]:
        sc = StandardScaler()
        arr_pair_list = list(arr_pair)
        # Need to reassign
    sc1 = StandardScaler(); Mh_tr = sc1.fit_transform(Mh_tr); Mh_te = sc1.transform(Mh_te)
    sc2 = StandardScaler(); Mw_tr = sc2.fit_transform(Mw_tr); Mw_te = sc2.transform(Mw_te)
    sc3 = StandardScaler(); Mv_tr = sc3.fit_transform(Mv_tr); Mv_te = sc3.transform(Mv_te)
    sc4 = StandardScaler(); Mi_tr = sc4.fit_transform(Mi_tr); Mi_te = sc4.transform(Mi_te)

    # Stack audio
    audio_tr = np.stack([Mh_tr, Mw_tr, Mv_tr], axis=1)  # (N, 3, modal_pca_dim)
    audio_te = np.stack([Mh_te, Mw_te, Mv_te], axis=1)

    # Train encoder
    model = CrossAttentionWithHead(modal_pca_dim, proj_dim, num_heads=4, dropout=0.1, num_classes=n_classes)
    class_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    w = 1.0 / class_counts; w = w / w.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    audio_t = torch.tensor(audio_tr, dtype=torch.float32)
    image_t = torch.tensor(Mi_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.long)

    model.train()
    for epoch in range(encoder_epochs):
        idx = np.random.permutation(len(y_tr))
        for s in range(0, len(y_tr), 64):
            e = min(s + 64, len(y_tr))
            b = idx[s:e]
            logits = model(audio_t[b], image_t[b])
            loss = criterion(logits, y_t[b])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Extract fused embeddings
    model.eval()
    with torch.no_grad():
        audio_te_t = torch.tensor(audio_te, dtype=torch.float32)
        image_te_t = torch.tensor(Mi_te, dtype=torch.float32)
        F_tr = model.extract(audio_t, image_t).numpy()
        F_te = model.extract(audio_te_t, image_te_t).numpy()

    sc_f = StandardScaler()
    F_tr = sc_f.fit_transform(F_tr)
    F_te = sc_f.transform(F_te)

    # [5] CrossAttn+Hyp → LogReg
    lr_ca = LogisticRegression(C=1000, solver="lbfgs", max_iter=5000,
                               class_weight="balanced", random_state=42)
    lr_ca.fit(F_tr, y_tr)
    results["ca_hyp_lr"] = f1_score(y_te, lr_ca.predict(F_te), average="macro", zero_division=0)
    results["ca_hyp_lr_train"] = f1_score(y_tr, lr_ca.predict(F_tr), average="macro", zero_division=0)

    # [6] CrossAttn+Hyp → SVM-RBF
    svm_ca = SVC(C=100, kernel="rbf", gamma="scale", class_weight="balanced", random_state=42)
    svm_ca.fit(F_tr, y_tr)
    results["ca_hyp_svm"] = f1_score(y_te, svm_ca.predict(F_te), average="macro", zero_division=0)
    results["ca_hyp_svm_train"] = f1_score(y_tr, svm_ca.predict(F_tr), average="macro", zero_division=0)

    return results


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10, 8, 5])
    parser.add_argument("--global-pca", type=int, default=256)
    parser.add_argument("--modal-pca", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=25)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)

    print("\n" + "=" * 100)
    print("  CROSS-ATTENTION HYPERBOLIC FUSION — ALL CLASSIFIERS COMPARISON")
    print("  HuBERT + WavLM + Wav2Vec2 + VGG19 | K-Fold + PCA per fold")
    print("=" * 100)
    print(f"  Global PCA: {args.global_pca} | Per-modal PCA: {args.modal_pca} | Proj: {args.proj_dim}")
    print(f"  Encoder epochs: {args.encoder_epochs} | Folds: {args.n_folds}")
    print()

    print("  Loading... ", end="", flush=True)
    X_h, X_w, X_v, X_i, y = load_and_align()
    print(f"done ({X_h.shape[0]} samples, 6400d total)")
    print()

    baselines = {10: 0.6870, 8: 0.6831, 5: 0.5992}
    methods = ["global_lr", "global_svm", "global_hyp_svm", "global_hyp_lr", "ca_hyp_lr", "ca_hyp_svm"]
    labels = {
        "global_lr": "GlobalPCA→LR",
        "global_svm": "GlobalPCA→SVM",
        "global_hyp_svm": "GlobalPCA→Hyp→SVM",
        "global_hyp_lr": "GlobalPCA→Hyp→LR",
        "ca_hyp_lr": "CrossAttn+Hyp→LR",
        "ca_hyp_svm": "CrossAttn+Hyp→SVM",
    }

    for threshold in thresholds:
        print(f"{'─' * 100}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'─' * 100}")

        Xh, Xw, Xv, Xi, yf, nc = filter_threshold(X_h, X_w, X_v, X_i, y, threshold)
        if Xh is None:
            continue

        print(f"  Samples: {Xh.shape[0]}, Classes: {nc}")

        min_count = min(Counter(yf).values())
        nf = min(args.n_folds, min_count)
        cv = StratifiedKFold(n_splits=nf, shuffle=True, random_state=42)

        all_results = {m: [] for m in methods}

        for fi, (tr, te) in enumerate(cv.split(Xh, yf)):
            t0 = time.time()
            r = run_fold(
                Xh[tr], Xw[tr], Xv[tr], Xi[tr], yf[tr],
                Xh[te], Xw[te], Xv[te], Xi[te], yf[te],
                nc, args.global_pca, args.modal_pca, args.proj_dim, args.encoder_epochs
            )
            elapsed = time.time() - t0

            for m in methods:
                all_results[m].append(r[m])

            print(f"    Fold {fi+1}: GlobLR={r['global_lr']:.4f} GlobSVM={r['global_svm']:.4f} "
                  f"GlobHypSVM={r['global_hyp_svm']:.4f} CA+LR={r['ca_hyp_lr']:.4f} "
                  f"CA+SVM={r['ca_hyp_svm']:.4f} ({elapsed:.0f}s)")

        print()
        print(f"  ┌─────────────────────────────────────────────────────────────────────────┐")
        print(f"  │  Threshold >= {threshold} ({nc} classes)")
        print(f"  ├─────────────────────────────────────────────────────────────────────────┤")

        best_method = None
        best_score = 0
        for m in methods:
            scores = all_results[m]
            mean = np.mean(scores)
            std = np.std(scores)
            marker = ""
            if mean > best_score:
                best_score = mean
                best_method = m
            print(f"  │  {labels[m]:<22s} {mean:.4f} ± {std:.4f}")

        bl = baselines[threshold]
        gain = best_score - bl
        print(f"  ├─────────────────────────────────────────────────────────────────────────┤")
        print(f"  │  HuBERT alone (baseline):  {bl:.4f}")
        print(f"  │  Best fusion ({labels[best_method]}): {best_score:.4f}  "
              f"({'↑' if gain > 0 else '↓'} {abs(gain):.4f})")
        print(f"  │  Gap to 0.80:              {0.80 - best_score:.4f}")
        print(f"  └─────────────────────────────────────────────────────────────────────────┘")
        print()

    print("=" * 100)


if __name__ == "__main__":
    main()
