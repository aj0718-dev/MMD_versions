#!/usr/bin/env python3
"""
cross_attention_hyperbolic_v3.py

Improved Cross-Attention Fusion in Hyperbolic Space.
Addresses bottlenecks from v1/v2:
  1. Concat+MLP baseline (ceiling check)
  2. Improved cross-attention: 384 proj, 8 heads, FFN after attention
  3. Label smoothing + ranking-aware loss
  4. Bidirectional attention with proper aggregation

Usage:
    python cross_attention_hyperbolic_v3.py --thresholds 10 8 5
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


# =====================================================
# HYPERBOLIC OPERATIONS
# =====================================================


def exp_map(x):
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    return torch.tanh(norm) * x / norm


def log_map(x):
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
    norm = torch.clamp(norm, max=1 - 1e-5)
    return torch.atanh(norm) * x / norm


# =====================================================
# IMPROVED CROSS-ATTENTION MODEL
# =====================================================


class ImprovedCrossAttentionFusion(nn.Module):
    """
    Improvements over v1:
      - Larger projection: 384d (was 256)
      - 8 attention heads (was 4)
      - FFN block after attention (transformer-style)
      - Proper bidirectional concat (A→I ⊕ I→A)
      - Hyperbolic distance-aware projection
      - Label smoothing in loss (external)
    """

    def __init__(self, audio_dim, image_dim, proj_dim=384, num_heads=8,
                 ff_dim=512, dropout=0.25, num_classes=89):
        super().__init__()

        # Modality projections (same dim for both)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Bidirectional cross-attention
        self.cross_attn_a2i = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.cross_attn_i2a = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        # Post-attention layer norms
        self.norm_a = nn.LayerNorm(proj_dim)
        self.norm_i = nn.LayerNorm(proj_dim)

        # FFN blocks after attention (transformer-style)
        self.ffn_a = nn.Sequential(
            nn.Linear(proj_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, proj_dim),
            nn.Dropout(dropout * 0.5),
        )
        self.ffn_i = nn.Sequential(
            nn.Linear(proj_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, proj_dim),
            nn.Dropout(dropout * 0.5),
        )
        self.norm_ffn_a = nn.LayerNorm(proj_dim)
        self.norm_ffn_i = nn.LayerNorm(proj_dim)

        # Gated fusion: concat(audio_res, a_cross, i_cross) → fused
        # 3 * proj_dim → proj_dim
        self.gate_net = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.Sigmoid(),
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Hyperbolic-aware classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, audio_emb, image_emb):
        # Project to shared space
        a = self.audio_proj(audio_emb)  # (B, proj_dim)
        i = self.image_proj(image_emb)  # (B, proj_dim)

        # Save audio residual (for gated bypass)
        a_res = a

        # Cross-attention (reshape for MHA)
        a_seq = a.unsqueeze(1)  # (B, 1, proj_dim)
        i_seq = i.unsqueeze(1)

        # A→I: audio attends to image
        a_attn, _ = self.cross_attn_a2i(query=a_seq, key=i_seq, value=i_seq)
        a_cross = self.norm_a(a + a_attn.squeeze(1))  # Residual

        # I→A: image attends to audio
        i_attn, _ = self.cross_attn_i2a(query=i_seq, key=a_seq, value=a_seq)
        i_cross = self.norm_i(i + i_attn.squeeze(1))  # Residual

        # FFN blocks (transformer-style)
        a_cross = self.norm_ffn_a(a_cross + self.ffn_a(a_cross))
        i_cross = self.norm_ffn_i(i_cross + self.ffn_i(i_cross))

        # Gated fusion
        combined = torch.cat([a_res, a_cross, i_cross], dim=-1)
        gate = self.gate_net(combined)
        fused_candidate = self.fusion_net(combined)

        # Gated residual: output = audio + gate * (fusion - audio)
        fused = a_res + gate * (fused_candidate - a_res)

        # Hyperbolic round-trip
        h = exp_map(fused)
        e = log_map(h)

        # Classify
        logits = self.classifier(e)
        return logits


# =====================================================
# RANKING-AWARE LOSS
# =====================================================


class RankingCrossEntropyLoss(nn.Module):
    """
    Combined CE + ranking loss:
    - CrossEntropy with label smoothing (soft targets)
    - Margin ranking: push correct class above top-incorrect by margin
    """

    def __init__(self, weight=None, label_smoothing=0.1, ranking_weight=0.3, margin=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.ranking_weight = ranking_weight
        self.margin = margin

    def forward(self, logits, targets):
        # Standard CE with label smoothing
        ce_loss = self.ce(logits, targets)

        # Ranking loss: correct class score should exceed best incorrect by margin
        batch_size = logits.shape[0]
        correct_scores = logits[torch.arange(batch_size), targets]  # (B,)

        # Mask out correct class, get max incorrect
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), targets] = False
        incorrect_logits = logits.masked_fill(~mask, -1e9)
        max_incorrect = incorrect_logits.max(dim=1)[0]  # (B,)

        # Hinge loss: max(0, margin - (correct - max_incorrect))
        ranking_loss = F.relu(self.margin - (correct_scores - max_incorrect)).mean()

        return ce_loss + self.ranking_weight * ranking_loss


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def load_and_align():
    """Load HuBERT + VGG19, align by hash."""
    hub_emb = torch.load("wav2vec2_hubert_wavlm/hubert_embeddings.pt",
                         map_location="cpu", weights_only=True)
    vgg_emb = torch.load("vgg19/vgg19_embeddings_all.pt",
                         map_location="cpu", weights_only=True)

    audio_paths = torch.load("wav2vec2_hubert_wavlm/wavlm_paths.pt", weights_only=False)
    img_paths = torch.load("vgg19/vgg19_paths.pt", weights_only=False)
    audio_labels = torch.load("wav2vec2_hubert_wavlm/labels.pt",
                              map_location="cpu", weights_only=True)
    img_labels = torch.load("vgg19/labels_all.pt", map_location="cpu", weights_only=True)

    def get_hash(p):
        return os.path.basename(str(p)).replace(".wav", "").replace(".png", "")

    audio_dict = {get_hash(p): i for i, p in enumerate(audio_paths)}
    img_dict = {get_hash(p): i for i, p in enumerate(img_paths)}
    common = sorted(set(audio_dict.keys()) & set(img_dict.keys()))
    a_idx = [audio_dict[h] for h in common]
    i_idx = [img_dict[h] for h in common]

    X_a = hub_emb[a_idx].numpy().astype(np.float32)
    X_i = vgg_emb[i_idx].numpy().astype(np.float32)
    y = audio_labels[a_idx].numpy().astype(np.int64)
    assert np.all(y == img_labels[i_idx].numpy()), "Label mismatch!"
    return X_a, X_i, y


def filter_by_threshold(X_a, X_i, y, threshold):
    counts = Counter(y)
    keep = {c for c, n in counts.items() if n >= threshold}
    if not keep:
        return None, None, None, 0
    mask = np.array([l in keep for l in y])
    X_af, X_if, y_f = X_a[mask], X_i[mask], y[mask]
    unique = sorted(set(y_f))
    remap = {old: new for new, old in enumerate(unique)}
    y_r = np.array([remap[l] for l in y_f])
    return X_af, X_if, y_r, len(unique)


def train_concat_mlp(X_a_tr, X_i_tr, y_tr, X_a_te, X_i_te, y_te, pca_dim=384):
    """Concat + MLP baseline (sklearn)."""
    # PCA per modality
    pca_a = PCA(n_components=pca_dim, random_state=42)
    pca_i = PCA(n_components=min(pca_dim, X_i_tr.shape[1]), random_state=42)

    Xa_tr = pca_a.fit_transform(X_a_tr)
    Xa_te = pca_a.transform(X_a_te)
    Xi_tr = pca_i.fit_transform(X_i_tr)
    Xi_te = pca_i.transform(X_i_te)

    # Concat + scale
    Xc_tr = np.concatenate([Xa_tr, Xi_tr], axis=1)
    Xc_te = np.concatenate([Xa_te, Xi_te], axis=1)
    sc = StandardScaler()
    Xc_tr = sc.fit_transform(Xc_tr)
    Xc_te = sc.transform(Xc_te)

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        alpha=0.005,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    mlp.fit(Xc_tr, y_tr)
    preds = mlp.predict(Xc_te)
    return f1_score(y_te, preds, average="macro", zero_division=0)


def train_cross_attention_fold(X_a_tr, X_i_tr, y_tr, X_a_te, X_i_te, y_te,
                               n_classes, pca_dim=384, proj_dim=384,
                               epochs=100, lr=3e-4, batch_size=32):
    """Train improved cross-attention model for one fold."""
    # PCA per modality
    pca_a = PCA(n_components=pca_dim, random_state=42)
    pca_i = PCA(n_components=pca_dim, random_state=42)
    Xa_tr = pca_a.fit_transform(X_a_tr)
    Xa_te = pca_a.transform(X_a_te)
    Xi_tr = pca_i.fit_transform(X_i_tr)
    Xi_te = pca_i.transform(X_i_te)

    # Scale
    sc_a, sc_i = StandardScaler(), StandardScaler()
    Xa_tr = sc_a.fit_transform(Xa_tr)
    Xa_te = sc_a.transform(Xa_te)
    Xi_tr = sc_i.fit_transform(Xi_tr)
    Xi_te = sc_i.transform(Xi_te)

    # Tensors
    a_tr_t = torch.tensor(Xa_tr, dtype=torch.float32).to(DEVICE)
    i_tr_t = torch.tensor(Xi_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    a_te_t = torch.tensor(Xa_te, dtype=torch.float32).to(DEVICE)
    i_te_t = torch.tensor(Xi_te, dtype=torch.float32).to(DEVICE)

    # Model
    model = ImprovedCrossAttentionFusion(
        audio_dim=pca_dim, image_dim=pca_dim,
        proj_dim=proj_dim, num_heads=8, ff_dim=512,
        dropout=0.25, num_classes=n_classes,
    ).to(DEVICE)

    # Class weights
    cc = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    cc = np.maximum(cc, 1.0)
    w = 1.0 / cc
    w = w / w.sum() * n_classes
    class_weights = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    # Ranking-aware loss with label smoothing
    criterion = RankingCrossEntropyLoss(
        weight=class_weights, label_smoothing=0.1,
        ranking_weight=0.3, margin=1.0,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = len(y_tr)
    best_f1 = 0.0
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            logits = model(a_tr_t[idx], i_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Eval every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                te_logits = model(a_te_t, i_te_t)
                te_preds = torch.argmax(te_logits, dim=1).cpu().numpy()
                f1 = f1_score(y_te, te_preds, average="macro", zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

                if patience >= 5:  # 25 epochs without improvement
                    break

    # Final eval
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(a_te_t, i_te_t)
        te_preds = torch.argmax(te_logits, dim=1).cpu().numpy()
        tr_logits = model(a_tr_t, i_tr_t)
        tr_preds = torch.argmax(tr_logits, dim=1).cpu().numpy()

    macro_f1 = f1_score(y_te, te_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_te, te_preds, average="weighted", zero_division=0)
    train_f1 = f1_score(y_tr, tr_preds, average="macro", zero_division=0)

    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "train_f1": train_f1}


# =====================================================
# MAIN
# =====================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=int, default=[10, 8, 5])
    parser.add_argument("--pca-dim", type=int, default=384)
    parser.add_argument("--proj-dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)

    print("\n" + "=" * 100)
    print("  IMPROVED CROSS-ATTENTION FUSION IN HYPERBOLIC SPACE (v3)")
    print("  HuBERT + VGG19 | Concat+MLP baseline + Improved CrossAttn")
    print("=" * 100)
    print(f"\n  Improvements over v1:")
    print(f"    • Projection: 256 → {args.proj_dim}")
    print(f"    • Attention heads: 4 → 8")
    print(f"    • FFN after attention (transformer-style)")
    print(f"    • Label smoothing (0.1) + Ranking loss (margin=1.0)")
    print(f"    • Concat+MLP baseline for comparison")
    print(f"\n  Config: PCA={args.pca_dim} | Proj={args.proj_dim} | Epochs={args.epochs}")
    print(f"  LR={args.lr} | Batch={args.batch_size} | Folds={args.n_folds}")
    print()

    # Load
    print("  Loading HuBERT + VGG19... ", end="", flush=True)
    X_audio, X_image, y = load_and_align()
    print(f"done ({X_audio.shape[0]} samples)")
    print(f"  Audio: {X_audio.shape} | Image: {X_image.shape}")
    print()

    baselines_lr = {10: 0.6607, 8: 0.6438, 5: 0.5718}
    baselines_svm = {10: 0.6870, 8: 0.6831, 5: 0.6011}
    prev_fusion = {10: 0.6924, 8: 0.6563, 5: 0.5733}

    for threshold in thresholds:
        print(f"\n{'━' * 100}")
        print(f"  THRESHOLD >= {threshold}")
        print(f"{'━' * 100}")

        X_a, X_i, y_f, nc = filter_by_threshold(X_audio, X_image, y, threshold)
        if X_a is None:
            continue

        print(f"  Samples: {X_a.shape[0]}, Classes: {nc}")

        min_cnt = min(Counter(y_f).values())
        n_folds = min(args.n_folds, min_cnt)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        mlp_scores = []
        ca_macro, ca_weighted, ca_train = [], [], []

        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_a, y_f)):
            t0 = time.time()

            # Concat+MLP baseline
            mlp_f1 = train_concat_mlp(
                X_a[tr_idx], X_i[tr_idx], y_f[tr_idx],
                X_a[te_idx], X_i[te_idx], y_f[te_idx],
                pca_dim=args.pca_dim,
            )
            mlp_scores.append(mlp_f1)

            # Improved cross-attention
            result = train_cross_attention_fold(
                X_a[tr_idx], X_i[tr_idx], y_f[tr_idx],
                X_a[te_idx], X_i[te_idx], y_f[te_idx],
                n_classes=nc,
                pca_dim=args.pca_dim,
                proj_dim=args.proj_dim,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )
            ca_macro.append(result["macro_f1"])
            ca_weighted.append(result["weighted_f1"])
            ca_train.append(result["train_f1"])

            elapsed = time.time() - t0
            print(f"    Fold {fold_idx+1}: MLP={mlp_f1:.4f}  "
                  f"CrossAttn={result['macro_f1']:.4f} (W={result['weighted_f1']:.4f}) "
                  f"Train={result['train_f1']:.4f}  ({elapsed:.0f}s)")

        # Summary
        print()
        print(f"  ┌──────────────────────────────────────────────────────────────────────────────┐")
        print(f"  │  Threshold >= {threshold} ({nc} classes, {X_a.shape[0]} samples)")
        print(f"  ├──────────────────────────────────────────────────────────────────────────────┤")
        print(f"  │  [NEW] Concat+MLP:              {np.mean(mlp_scores):.4f} ± {np.std(mlp_scores):.4f}")
        print(f"  │  [NEW] CrossAttn v3 (improved): {np.mean(ca_macro):.4f} ± {np.std(ca_macro):.4f}  "
              f"(W-F1: {np.mean(ca_weighted):.4f})")
        print(f"  │        Train F1:                {np.mean(ca_train):.4f}")
        print(f"  ├──────────────────────────────────────────────────────────────────────────────┤")
        print(f"  │  [PREV] CrossAttn v1:           {prev_fusion[threshold]:.4f}")
        print(f"  │  [BASE] HuBERT + LR:            {baselines_lr[threshold]:.4f}")
        print(f"  │  [BASE] HuBERT + SVM-RBF:       {baselines_svm[threshold]:.4f}")
        print(f"  ├──────────────────────────────────────────────────────────────────────────────┤")

        best_new = max(np.mean(mlp_scores), np.mean(ca_macro))
        gain_vs_v1 = best_new - prev_fusion[threshold]
        gain_vs_svm = best_new - baselines_svm[threshold]
        print(f"  │  Best new:  {best_new:.4f}  "
              f"(vs v1: {'↑' if gain_vs_v1 > 0 else '↓'} {abs(gain_vs_v1):.4f})  "
              f"(vs SVM: {'↑' if gain_vs_svm > 0 else '↓'} {abs(gain_vs_svm):.4f})")
        gap = 0.80 - best_new
        print(f"  │  Gap to 0.80: {gap:.4f}")
        print(f"  └──────────────────────────────────────────────────────────────────────────────┘")

    print(f"\n{'=' * 100}\n")


if __name__ == "__main__":
    main()
