"""
Phase 3: Advanced Loss Functions for DINOv2+HuBERT Fusion
==========================================================
Test improved training objectives on the best fusion pair (DINOv2+HuBERT).

Methods:
1. CE_Balanced (baseline, reproduces 0.7223)
2. CE_Balanced + SupCon (supervised contrastive)
3. ArcFace head (angular margin classifier)
4. CE_Balanced + Prototype loss (class centroid regularization)

All use: PCA=256, 5-fold stratified CV, same alignment as fusion_all_pairs.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# CONFIGURATION
# =====================================================

DEVICE = "cpu"
RANDOM_STATE = 42
K_FOLDS = 5
PCA_DIM = 256
HIDDEN_DIM = 1024
EMBED_DIM = 256  # embedding dimension for contrastive/arcface
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 256
LR = 1e-3

# DINOv2 + HuBERT paths
DINO_CONFIG = {
    "embeddings": "image_embeddings/dinov2_embeddings.pt",
    "labels": "image_embeddings/dinov2_labels.pt",
    "paths": "image_embeddings/dinov2_paths.pt",
}
HUBERT_CONFIG = {
    "embeddings": "wav2vec2_hubert_wavlm/hubert_embeddings.pt",
    "labels": "wav2vec2_hubert_wavlm/labels.pt",
    "paths": "wav2vec2_hubert_wavlm/wavlm_paths.pt",
}

# SupCon hyperparameters to sweep
SUPCON_LAMBDAS = [0.05, 0.1, 0.2]
SUPCON_TEMPS = [0.07, 0.1]

# ArcFace hyperparameters
ARCFACE_MARGINS = [0.3, 0.5]
ARCFACE_SCALES = [30.0, 64.0]


# =====================================================
# DATA LOADING
# =====================================================

def load_embeddings(config):
    emb = torch.load(config["embeddings"], map_location="cpu", weights_only=True)
    labels = torch.load(config["labels"], map_location="cpu", weights_only=True)
    paths = None
    if "paths" in config and os.path.exists(config["paths"]):
        paths = torch.load(config["paths"], map_location="cpu", weights_only=False)
    return emb.numpy().astype(np.float32), labels.numpy().astype(np.int64), paths


def get_sample_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def align_pair(X_a, y_a, paths_a, X_b, y_b, paths_b):
    ids_a = {get_sample_id(p): i for i, p in enumerate(paths_a)}
    ids_b = {get_sample_id(p): i for i, p in enumerate(paths_b)}
    common_ids = sorted(set(ids_a.keys()) & set(ids_b.keys()))
    idx_a = np.array([ids_a[sid] for sid in common_ids])
    idx_b = np.array([ids_b[sid] for sid in common_ids])
    return X_a[idx_a], X_b[idx_b], y_a[idx_a]


def filter_by_threshold(labels, threshold):
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= threshold]
    mask = np.isin(labels, valid_classes)
    return mask


# =====================================================
# MODEL ARCHITECTURES
# =====================================================

class OriginalFusionFCN(nn.Module):
    """Original FCN_Balanced architecture (matches fusion_all_pairs.py exactly).
    input → Linear(1024) → ReLU → Dropout(0.3) → Linear(n_classes)
    """
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits, None


class FusionFCN3Layer(nn.Module):
    """3-layer FCN_Balanced (deeper capacity ablation).
    input → Linear(1024) → ReLU → Dropout → Linear(512) → ReLU → Dropout → Linear(n_classes)
    """
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits, None


class FusionEncoder(nn.Module):
    """Shared encoder: concat PCA features -> hidden -> embedding (bottleneck)."""
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class CEClassifier(nn.Module):
    """Bottleneck encoder + linear classifier (for SupCon/Proto that need embeddings)."""
    def __init__(self, input_dim, hidden_dim, embed_dim, n_classes):
        super().__init__()
        self.encoder = FusionEncoder(input_dim, hidden_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits, emb


class ArcFaceHead(nn.Module):
    """ArcFace angular margin classification head."""
    def __init__(self, embed_dim, n_classes, margin=0.5, scale=64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = np.cos(margin)
        self.sin_m = np.sin(margin)
        # Threshold to avoid numerical issues
        self.th = np.cos(np.pi - margin)
        self.mm = np.sin(np.pi - margin) * margin

    def forward(self, embeddings, labels=None):
        # Normalize
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb_norm, w_norm)

        if labels is None:
            return cosine * self.scale

        # Add angular margin to target class
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


class ArcFaceModel(nn.Module):
    """Encoder + ArcFace head."""
    def __init__(self, input_dim, hidden_dim, embed_dim, n_classes, margin=0.5, scale=64.0):
        super().__init__()
        self.encoder = FusionEncoder(input_dim, hidden_dim, embed_dim)
        self.arcface = ArcFaceHead(embed_dim, n_classes, margin, scale)

    def forward(self, x, labels=None):
        emb = self.encoder(x)
        logits = self.arcface(emb, labels)
        return logits, emb


# =====================================================
# LOSS FUNCTIONS
# =====================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        # Similarity matrix
        sim = torch.mm(features, features.t()) / self.temperature

        # Mask: same class = 1, different = 0, diagonal = 0
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()
        mask.fill_diagonal_(0)

        # Mask to exclude self-similarity (non-inplace for autograd safety)
        diag_mask = 1.0 - torch.eye(batch_size, device=features.device)

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        # Compute log-sum-exp over all non-self entries
        exp_sim = torch.exp(sim) * diag_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs
        pos_count = mask.sum(dim=1)
        # Avoid division by zero for classes with single sample in batch
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        mean_log_prob = (mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
        loss = -mean_log_prob[valid].mean()
        return loss


class PrototypeLoss(nn.Module):
    """Push embeddings toward their class prototype (centroid)."""
    def __init__(self):
        super().__init__()

    def forward(self, features, labels, n_classes):
        features = F.normalize(features, p=2, dim=1)
        # Compute class prototypes
        prototypes = torch.zeros(n_classes, features.size(1), device=features.device)
        counts = torch.zeros(n_classes, device=features.device)
        for i in range(len(labels)):
            prototypes[labels[i]] += features[i]
            counts[labels[i]] += 1
        counts = counts.clamp(min=1)
        prototypes = prototypes / counts.unsqueeze(1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        # Distance to own prototype (want to minimize)
        target_protos = prototypes[labels]
        # Cosine distance
        cos_sim = (features * target_protos).sum(dim=1)
        loss = (1 - cos_sim).mean()
        return loss


# =====================================================
# TRAINING FUNCTIONS
# =====================================================

def train_original_fcn(X_train, y_train, X_val, y_val, n_classes, input_dim):
    """Train original FCN_Balanced (same arch as fusion_all_pairs.py = 0.7223)."""
    model = OriginalFusionFCN(input_dim, HIDDEN_DIM, n_classes)

    class_weights = compute_class_weights(y_train, n_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    best_f1, best_state, patience_ctr = 0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        f1 = evaluate(model, X_val, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return best_f1


def train_fcn_3layer(X_train, y_train, X_val, y_val, n_classes, input_dim):
    """Train 3-layer FCN_Balanced (deeper capacity ablation)."""
    model = FusionFCN3Layer(input_dim, HIDDEN_DIM, n_classes)

    class_weights = compute_class_weights(y_train, n_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    best_f1, best_state, patience_ctr = 0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        f1 = evaluate(model, X_val, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return best_f1


def train_ce_balanced(X_train, y_train, X_val, y_val, n_classes, input_dim):
    """Train bottleneck CE_Balanced (encoder→256d→classifier)."""
    model = CEClassifier(input_dim, HIDDEN_DIM, EMBED_DIM, n_classes)

    class_weights = compute_class_weights(y_train, n_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    best_f1, best_state, patience_ctr = 0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        f1 = evaluate(model, X_val, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return best_f1


def train_ce_supcon(X_train, y_train, X_val, y_val, n_classes, input_dim,
                    lam=0.1, temperature=0.07):
    """Train CE_Balanced + λ * SupCon loss."""
    model = CEClassifier(input_dim, HIDDEN_DIM, EMBED_DIM, n_classes)

    class_weights = compute_class_weights(y_train, n_classes)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    supcon_loss_fn = SupConLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    best_f1, best_state, patience_ctr = 0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            logits, emb = model(xb)
            loss_ce = ce_loss_fn(logits, yb)
            loss_sc = supcon_loss_fn(emb, yb)
            loss = loss_ce + lam * loss_sc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        f1 = evaluate(model, X_val, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return best_f1


def train_arcface(X_train, y_train, X_val, y_val, n_classes, input_dim,
                  margin=0.5, scale=64.0):
    """Train with ArcFace angular margin head."""
    model = ArcFaceModel(input_dim, HIDDEN_DIM, EMBED_DIM, n_classes, margin, scale)

    class_weights = compute_class_weights(y_train, n_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    best_f1, best_state, patience_ctr = 0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            logits, _ = model(xb, labels=yb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # At eval, no margin
        f1 = evaluate_arcface(model, X_val, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return best_f1


def train_ce_prototype(X_train, y_train, X_val, y_val, n_classes, input_dim,
                       lam=0.1):
    """Train CE_Balanced + λ * Prototype loss."""
    model = CEClassifier(input_dim, HIDDEN_DIM, EMBED_DIM, n_classes)

    class_weights = compute_class_weights(y_train, n_classes)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    proto_loss_fn = PrototypeLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    best_f1, best_state, patience_ctr = 0, None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            logits, emb = model(xb)
            loss_ce = ce_loss_fn(logits, yb)
            loss_proto = proto_loss_fn(emb, yb, n_classes)
            loss = loss_ce + lam * loss_proto
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        f1 = evaluate(model, X_val, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return best_f1


# =====================================================
# HELPERS
# =====================================================

def compute_class_weights(y_train, n_classes):
    classes = np.arange(n_classes)
    present = np.unique(y_train)
    weights = np.ones(n_classes, dtype=np.float32)
    if len(present) > 1:
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=present, y=y_train)
        for i, c in enumerate(present):
            weights[c] = cw[i]
    return torch.FloatTensor(weights)


def make_loader(X, y, batch_size, shuffle=False):
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_val)
        logits, _ = model(X_t)
        preds = logits.argmax(dim=1).numpy()
    return f1_score(y_val, preds, average="macro", zero_division=0)


def evaluate_arcface(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_val)
        logits, _ = model(X_t, labels=None)
        preds = logits.argmax(dim=1).numpy()
    return f1_score(y_val, preds, average="macro", zero_division=0)


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: SupCon/ArcFace/Prototype")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--pca", type=int, default=256)
    args = parser.parse_args()

    global PCA_DIM
    PCA_DIM = args.pca

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("=" * 70)
    print("  PHASE 3: ADVANCED LOSS FUNCTIONS (DINOv2 + HuBERT)")
    print("=" * 70)
    print(f"  PCA: {PCA_DIM}, Threshold: ≥{args.threshold}")
    print(f"  Hidden: {HIDDEN_DIM}, Embed: {EMBED_DIM}")
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}, LR: {LR}")
    print("=" * 70)

    # Load and align
    print("\n[1] Loading DINOv2 + HuBERT...")
    X_dino, y_dino, paths_dino = load_embeddings(DINO_CONFIG)
    X_hub, y_hub, paths_hub = load_embeddings(HUBERT_CONFIG)
    X_dino, X_hub, labels = align_pair(X_dino, y_dino, paths_dino, X_hub, y_hub, paths_hub)
    print(f"    Aligned: {len(labels)} samples")

    # Filter threshold
    mask = filter_by_threshold(labels, args.threshold)
    X_dino, X_hub, labels = X_dino[mask], X_hub[mask], labels[mask]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    print(f"    After threshold ≥{args.threshold}: {len(y)} samples, {n_classes} classes")

    # CV setup
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_results = []

    # Define experiments
    experiments = []
    # Baselines / architecture ablations
    experiments.append(("FCN_Balanced (original)", {}))  # exact reproduction of 0.7223 arch
    experiments.append(("FCN_Balanced (3-layer)", {}))    # deeper capacity ablation
    experiments.append(("CE_Balanced (bottleneck)", {}))  # bottleneck for SupCon/ArcFace/Proto
    # SupCon
    for lam in SUPCON_LAMBDAS:
        for temp in SUPCON_TEMPS:
            experiments.append((f"CE+SupCon(λ={lam},τ={temp})", {"lam": lam, "temp": temp}))
    # ArcFace
    for margin in ARCFACE_MARGINS:
        for scale in ARCFACE_SCALES:
            experiments.append((f"ArcFace(m={margin},s={scale})", {"margin": margin, "scale": scale}))
    # Prototype
    for lam in [0.05, 0.1, 0.2]:
        experiments.append((f"CE+Proto(λ={lam})", {"lam": lam}))

    print(f"\n[2] Running {len(experiments)} experiments × {K_FOLDS} folds...")
    print(f"    Total training runs: {len(experiments) * K_FOLDS}")

    for exp_name, exp_params in experiments:
        print(f"\n  {'─' * 60}")
        print(f"  {exp_name}")
        print(f"  {'─' * 60}", flush=True)

        fold_f1s = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            # Split
            X_dino_tr, X_dino_val = X_dino[train_idx], X_dino[val_idx]
            X_hub_tr, X_hub_val = X_hub[train_idx], X_hub[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # PCA per modality (fit on train only)
            pca_d = PCA(n_components=min(PCA_DIM, X_dino_tr.shape[1], X_dino_tr.shape[0] - 1),
                        random_state=RANDOM_STATE)
            pca_h = PCA(n_components=min(PCA_DIM, X_hub_tr.shape[1], X_hub_tr.shape[0] - 1),
                        random_state=RANDOM_STATE)

            Xd_tr = pca_d.fit_transform(X_dino_tr)
            Xd_val = pca_d.transform(X_dino_val)
            Xh_tr = pca_h.fit_transform(X_hub_tr)
            Xh_val = pca_h.transform(X_hub_val)

            # Scale
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(np.hstack([Xd_tr, Xh_tr]))
            X_val = scaler.transform(np.hstack([Xd_val, Xh_val]))
            input_dim = X_tr.shape[1]

            # Train
            torch.manual_seed(RANDOM_STATE + fold_idx)

            if exp_name == "FCN_Balanced (original)":
                f1 = train_original_fcn(X_tr, y_train, X_val, y_val, n_classes, input_dim)
            elif exp_name == "FCN_Balanced (3-layer)":
                f1 = train_fcn_3layer(X_tr, y_train, X_val, y_val, n_classes, input_dim)
            elif exp_name == "CE_Balanced (bottleneck)":
                f1 = train_ce_balanced(X_tr, y_train, X_val, y_val, n_classes, input_dim)
            elif exp_name.startswith("CE+SupCon"):
                f1 = train_ce_supcon(X_tr, y_train, X_val, y_val, n_classes, input_dim,
                                     lam=exp_params["lam"], temperature=exp_params["temp"])
            elif exp_name.startswith("ArcFace"):
                f1 = train_arcface(X_tr, y_train, X_val, y_val, n_classes, input_dim,
                                   margin=exp_params["margin"], scale=exp_params["scale"])
            elif exp_name.startswith("CE+Proto"):
                f1 = train_ce_prototype(X_tr, y_train, X_val, y_val, n_classes, input_dim,
                                        lam=exp_params["lam"])

            fold_f1s.append(f1)
            print(f"    Fold {fold_idx+1}: {f1:.4f}", flush=True)

        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)
        print(f"    → Mean: {mean_f1:.4f} ± {std_f1:.4f}")

        all_results.append({
            "name": exp_name,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "fold_f1s": fold_f1s,
        })

    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    print(f"\n\n{'=' * 70}")
    print(f"  FINAL RESULTS (Threshold ≥{args.threshold}, {n_classes} classes)")
    print(f"{'=' * 70}")
    print(f"  {'─' * 65}")
    print(f"  {'Rank':<4} | {'Method':<30} | {'Macro-F1':<9} | {'Std':<6}")
    print(f"  {'─' * 65}")

    sorted_results = sorted(all_results, key=lambda x: -x["mean_f1"])
    for i, r in enumerate(sorted_results, 1):
        marker = " ★" if r["mean_f1"] > 0.7223 else ""
        print(f"  {i:<4} | {r['name']:<30} | {r['mean_f1']:<9.4f} | {r['std_f1']:<6.4f}{marker}")

    # Show internal baseline comparison
    orig_baseline = next((r for r in all_results if r["name"] == "FCN_Balanced (original)"), None)
    print(f"\n  {'─' * 55}")
    print(f"  REFERENCE: DINOv2+HuBERT FCN_Balanced (fusion_all_pairs) = 0.7223")
    if orig_baseline:
        print(f"  INTERNAL:  FCN_Balanced (original, this script)        = {orig_baseline['mean_f1']:.4f}")
    best = sorted_results[0]
    delta_ref = best["mean_f1"] - 0.7223
    sign_ref = "+" if delta_ref >= 0 else ""
    print(f"  BEST:      {best['name']:<35} = {best['mean_f1']:.4f} ({sign_ref}{delta_ref:.4f} vs 0.7223)")
    if orig_baseline and best["name"] != "FCN_Balanced (original)":
        delta_int = best["mean_f1"] - orig_baseline["mean_f1"]
        sign_int = "+" if delta_int >= 0 else ""
        print(f"           {'':<35}          ({sign_int}{delta_int:.4f} vs internal baseline)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
