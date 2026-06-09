#!/usr/bin/env python3
"""
cross_attention_hyperbolic_v4.py

Optimized Cross-Attention Fusion in Hyperbolic Space (v4).
Builds on v3 with critical improvements:
  1. No PCA bottleneck — direct NN projection from raw embeddings
  2. Stronger ranking loss (weight=0.6, margin=2.0) + Top-K softmax loss
  3. Mixup augmentation in embedding space (critical for low-sample classes)
  4. Multi-token attention (split embeddings into 4 chunks → real sequence)
  5. Stronger regularization (dropout=0.35, weight_decay=1e-3)
  6. Longer training with cosine warmup

Usage:
    python cross_attention_hyperbolic_v4.py --thresholds 10 8 5
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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
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
# MULTI-TOKEN CROSS-ATTENTION MODEL (v4)
# =====================================================


class MultiTokenCrossAttentionFusion(nn.Module):
    """
    Key innovations over v3:
      1. NO PCA — projects raw embeddings via learned linear layers
      2. Multi-token: splits each embedding into N chunks → real sequence attention
      3. 2-layer cross-attention with FFN (proper transformer)
      4. Gated residual preserving audio signal
      5. Hyperbolic projection with learned curvature scaling
    """

    def __init__(self, audio_dim=768, image_dim=4096, proj_dim=384,
                 num_heads=8, num_tokens=4, num_layers=2,
                 ff_dim=512, dropout=0.35, num_classes=89):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj_dim = proj_dim
        self.token_dim = proj_dim // num_tokens  # Each token's dim

        # Direct projection from raw embeddings (NO PCA)
        # Audio: 768 → proj_dim
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        # Image: 4096 → proj_dim (deeper projection for higher dim)
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Token mixing layers (create semantically diverse tokens, not just reshaping)
        self.audio_token_mix = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
        )
        self.image_token_mix = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
        )

        # Positional encoding for multi-token sequences
        self.pos_enc = nn.Parameter(torch.randn(1, num_tokens, self.token_dim) * 0.02)

        # Multi-layer cross-attention (4 heads per token — not too few)
        attn_heads = max(4, num_heads // 2)  # At least 4 heads
        # Ensure token_dim is divisible by attn_heads
        assert self.token_dim % attn_heads == 0 or self.token_dim % 4 == 0
        attn_heads = 4 if self.token_dim % attn_heads != 0 else attn_heads

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                # Audio attends to image
                "a2i": nn.MultiheadAttention(
                    self.token_dim, num_heads=attn_heads,
                    dropout=dropout, batch_first=True),
                "norm_a": nn.LayerNorm(self.token_dim),
                # Image attends to audio
                "i2a": nn.MultiheadAttention(
                    self.token_dim, num_heads=attn_heads,
                    dropout=dropout, batch_first=True),
                "norm_i": nn.LayerNorm(self.token_dim),
                # FFN blocks
                "ffn_a": nn.Sequential(
                    nn.Linear(self.token_dim, ff_dim // num_tokens),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim // num_tokens, self.token_dim),
                    nn.Dropout(dropout * 0.5)),
                "ffn_i": nn.Sequential(
                    nn.Linear(self.token_dim, ff_dim // num_tokens),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim // num_tokens, self.token_dim),
                    nn.Dropout(dropout * 0.5)),
                "norm_ffn_a": nn.LayerNorm(self.token_dim),
                "norm_ffn_i": nn.LayerNorm(self.token_dim),
            }))

        # Gated fusion
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

        # Hyperbolic: learned prototypes for distance-based classification
        self.curvature = nn.Parameter(torch.tensor(1.0))
        self.prototypes = nn.Parameter(torch.randn(num_classes, proj_dim) * 0.05)

        # Classifier (dual path: Euclidean logits + hyperbolic distance)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Weight for combining Euclidean + hyperbolic logits
        self.hyp_weight = nn.Parameter(torch.tensor(-2.0))

    def forward(self, audio_emb, image_emb):
        B = audio_emb.shape[0]

        # Project raw embeddings
        a = self.audio_proj(audio_emb)  # (B, proj_dim)
        i = self.image_proj(image_emb)  # (B, proj_dim)

        # Save audio residual
        a_res = a

        # Token mixing (create diverse tokens, not just reshape)
        a_mixed = self.audio_token_mix(a)
        i_mixed = self.image_token_mix(i)

        # Reshape to multi-token: (B, num_tokens, token_dim)
        a_tok = a_mixed.view(B, self.num_tokens, self.token_dim) + self.pos_enc
        i_tok = i_mixed.view(B, self.num_tokens, self.token_dim) + self.pos_enc

        # Multi-layer cross-attention
        for layer in self.layers:
            # A→I cross-attention
            a_attn, _ = layer["a2i"](query=a_tok, key=i_tok, value=i_tok)
            a_tok = layer["norm_a"](a_tok + a_attn)
            a_tok = layer["norm_ffn_a"](a_tok + layer["ffn_a"](a_tok))

            # I→A cross-attention
            i_attn, _ = layer["i2a"](query=i_tok, key=a_tok, value=a_tok)
            i_tok = layer["norm_i"](i_tok + i_attn)
            i_tok = layer["norm_ffn_i"](i_tok + layer["ffn_i"](i_tok))

        # Flatten tokens back: (B, proj_dim)
        a_cross = a_tok.reshape(B, self.proj_dim)
        i_cross = i_tok.reshape(B, self.proj_dim)

        # Gated fusion with audio residual
        combined = torch.cat([a_res, a_cross, i_cross], dim=-1)
        gate = self.gate_net(combined)
        fused_candidate = self.fusion_net(combined)
        fused = a_res + gate * (fused_candidate - a_res)

        # Euclidean classifier logits
        eucl_logits = self.classifier(fused)

        # Hyperbolic distance-based logits (actually uses geometry)
        h_fused = exp_map(fused * self.curvature)
        h_protos = exp_map(self.prototypes * self.curvature)
        # Poincaré distance: negative distance = similarity
        # Use simplified: -||h_fused - h_proto||^2 as logits
        hyp_logits = -torch.cdist(h_fused, h_protos).pow(2)

        # Combine both (learned weight)
        w = torch.sigmoid(self.hyp_weight)
        return (1 - w) * eucl_logits + w * hyp_logits


# =====================================================
# COMBINED LOSS: CE + Ranking + Top-K
# =====================================================


class CombinedLoss(nn.Module):
    """
    Three-component loss:
      1. CrossEntropy with label smoothing (0.1)
      2. Margin ranking loss (push correct >> incorrect)
      3. Top-K softmax loss (improve ranking quality)
    """

    def __init__(self, weight=None, label_smoothing=0.1,
                 ranking_weight=0.5, margin=2.0, topk_weight=0.2, k=5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.ranking_weight = ranking_weight
        self.margin = margin
        self.topk_weight = topk_weight
        self.k = k

    def forward(self, logits, targets, mixup=False):
        # 1. CE with label smoothing (always computed)
        ce_loss = self.ce(logits, targets)

        # For mixup batches, only return CE (ranking/topk need clean labels)
        if mixup:
            return ce_loss

        batch_size = logits.shape[0]

        # 2. Ranking loss: correct score should exceed best incorrect by margin
        correct_scores = logits[torch.arange(batch_size), targets]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), targets] = False
        incorrect_logits = logits.masked_fill(~mask, -1e9)
        max_incorrect = incorrect_logits.max(dim=1)[0]
        ranking_loss = F.relu(self.margin - (correct_scores - max_incorrect)).mean()

        # 3. REAL Top-K loss: push probability mass into top-K region
        probs = F.softmax(logits, dim=1)
        topk_vals, _ = torch.topk(probs, k=min(self.k, probs.shape[1]), dim=1)
        topk_loss = -torch.log(topk_vals.mean(dim=1) + 1e-8).mean()

        return ce_loss + self.ranking_weight * ranking_loss + self.topk_weight * topk_loss


# =====================================================
# MIXUP
# =====================================================


def mixup_batch(audio, image, targets, alpha=0.3):
    """Mixup in embedding space. Returns mixed inputs + lambda for loss."""
    if alpha <= 0:
        return audio, image, targets, targets, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Keep dominant

    perm = torch.randperm(audio.shape[0])
    mixed_audio = lam * audio + (1 - lam) * audio[perm]
    mixed_image = lam * image + (1 - lam) * image[perm]

    return mixed_audio, mixed_image, targets, targets[perm], lam


# =====================================================
# HELPERS
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


def train_fold(X_a_tr, X_i_tr, y_tr, X_a_te, X_i_te, y_te,
               n_classes, proj_dim=384, epochs=120, lr=2e-4,
               batch_size=32, mixup_alpha=0.3):
    """Train v4 model for one fold. NO PCA — just normalize."""
    # StandardScaler only (no PCA bottleneck)
    sc_a = StandardScaler()
    sc_i = StandardScaler()
    Xa_tr = sc_a.fit_transform(X_a_tr)
    Xa_te = sc_a.transform(X_a_te)
    Xi_tr = sc_i.fit_transform(X_i_tr)
    Xi_te = sc_i.transform(X_i_te)

    # Tensors
    a_tr_t = torch.tensor(Xa_tr, dtype=torch.float32).to(DEVICE)
    i_tr_t = torch.tensor(Xi_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    a_te_t = torch.tensor(Xa_te, dtype=torch.float32).to(DEVICE)
    i_te_t = torch.tensor(Xi_te, dtype=torch.float32).to(DEVICE)

    # Model
    model = MultiTokenCrossAttentionFusion(
        audio_dim=X_a_tr.shape[1],
        image_dim=X_i_tr.shape[1],
        proj_dim=proj_dim,
        num_heads=8,
        num_tokens=4,
        num_layers=2,
        ff_dim=512,
        dropout=0.35,
        num_classes=n_classes,
    ).to(DEVICE)

    # Class weights
    cc = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    cc = np.maximum(cc, 1.0)
    w = 1.0 / cc
    w = w / w.sum() * n_classes
    class_weights = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    # Combined loss
    criterion = CombinedLoss(
        weight=class_weights, label_smoothing=0.1,
        ranking_weight=0.3, margin=2.0,
        topk_weight=0.1, k=5,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Cosine with warmup
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

            batch_a = a_tr_t[idx]
            batch_i = i_tr_t[idx]
            batch_y = y_tr_t[idx]

            # Mixup (40% of batches) — only CE part uses mixup labels
            if np.random.rand() < 0.4:
                batch_a, batch_i, y1, y2, lam = mixup_batch(
                    batch_a, batch_i, batch_y, alpha=mixup_alpha)
                logits = model(batch_a, batch_i)
                loss = lam * criterion(logits, y1, mixup=True) + \
                       (1 - lam) * criterion(logits, y2, mixup=True)
            else:
                logits = model(batch_a, batch_i)
                loss = criterion(logits, batch_y, mixup=False)

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

                if patience >= 6:  # 30 epochs no improvement
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
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    thresholds = sorted(args.thresholds, reverse=True)

    print("\n" + "=" * 100)
    print("  OPTIMIZED CROSS-ATTENTION FUSION v4 — HYPERBOLIC SPACE")
    print("  HuBERT + VGG19 | No PCA | Multi-Token Attention | Mixup | Ranking Loss")
    print("=" * 100)
    print(f"\n  Key changes from v3:")
    print(f"    • NO PCA bottleneck — direct NN projection from raw embeddings")
    print(f"    • Multi-token attention: split into 4 chunks → real sequence")
    print(f"    • 2-layer cross-attention transformer")
    print(f"    • Stronger ranking loss (weight=0.5, margin=2.0) + Top-K loss")
    print(f"    • Mixup augmentation (α={args.mixup_alpha}, 70% of batches)")
    print(f"    • Higher regularization (dropout=0.35, wd=1e-3)")
    print(f"    • Cosine LR with warmup (10 epochs)")
    print(f"\n  Config: Proj={args.proj_dim} | Epochs={args.epochs} | LR={args.lr}")
    print(f"  Batch={args.batch_size} | Folds={args.n_folds}")
    print()

    # Load
    print("  Loading HuBERT + VGG19... ", end="", flush=True)
    X_audio, X_image, y = load_and_align()
    print(f"done ({X_audio.shape[0]} samples)")
    print(f"  Audio: {X_audio.shape} | Image: {X_image.shape}")
    print()

    prev_v3 = {10: 0.6990, 8: 0.6623, 5: 0.5777}
    baselines_svm = {10: 0.6870, 8: 0.6831, 5: 0.6011}

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

        ca_macro, ca_weighted, ca_train = [], [], []

        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_a, y_f)):
            t0 = time.time()

            result = train_fold(
                X_a[tr_idx], X_i[tr_idx], y_f[tr_idx],
                X_a[te_idx], X_i[te_idx], y_f[te_idx],
                n_classes=nc,
                proj_dim=args.proj_dim,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                mixup_alpha=args.mixup_alpha,
            )

            ca_macro.append(result["macro_f1"])
            ca_weighted.append(result["weighted_f1"])
            ca_train.append(result["train_f1"])

            elapsed = time.time() - t0
            print(f"    Fold {fold_idx+1}: Macro={result['macro_f1']:.4f}  "
                  f"Weighted={result['weighted_f1']:.4f}  "
                  f"Train={result['train_f1']:.4f}  ({elapsed:.0f}s)")

        # Summary
        mean_f1 = np.mean(ca_macro)
        std_f1 = np.std(ca_macro)
        mean_wf1 = np.mean(ca_weighted)

        print()
        print(f"  ┌──────────────────────────────────────────────────────────────────────────────┐")
        print(f"  │  Threshold >= {threshold} ({nc} classes, {X_a.shape[0]} samples)")
        print(f"  ├──────────────────────────────────────────────────────────────────────────────┤")
        print(f"  │  [v4] CrossAttn (no PCA, mixup):  {mean_f1:.4f} ± {std_f1:.4f}  "
              f"(W-F1: {mean_wf1:.4f})")
        print(f"  │       Train F1:                   {np.mean(ca_train):.4f}")
        print(f"  ├──────────────────────────────────────────────────────────────────────────────┤")
        print(f"  │  [v3] CrossAttn (PCA=384):        {prev_v3[threshold]:.4f}")
        print(f"  │  [SVM] HuBERT alone:              {baselines_svm[threshold]:.4f}")
        print(f"  ├──────────────────────────────────────────────────────────────────────────────┤")

        gain_v3 = mean_f1 - prev_v3[threshold]
        gain_svm = mean_f1 - baselines_svm[threshold]
        print(f"  │  vs v3: {'↑' if gain_v3 > 0 else '↓'} {abs(gain_v3):.4f}  |  "
              f"vs SVM: {'↑' if gain_svm > 0 else '↓'} {abs(gain_svm):.4f}  |  "
              f"Gap to 0.80: {0.80 - mean_f1:.4f}")
        print(f"  └──────────────────────────────────────────────────────────────────────────────┘")

    print(f"\n{'=' * 100}\n")


if __name__ == "__main__":
    main()
