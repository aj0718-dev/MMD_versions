#!/usr/bin/env python3
"""
Extract embeddings from 10 image PTMs (frozen backbone).
All weights loaded from local cache (~/.cache/torch/hub/checkpoints/).

Usage:
    python extract_image_embeddings.py --model resnet50
    python extract_image_embeddings.py --model all
    python extract_image_embeddings.py --model resnet50 dinov2 clip

Models (10 total):
  torchvision: resnet50, densenet121, efficientnet_b0, convnext_base,
               swin_base, mobilenetv3_large
  special:     dinov2, clip, beit, deit_base
"""

import os
import argparse
import ssl
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time

# Fix SSL cert issues with corporate proxy (set SSL_NO_VERIFY=1 if needed)
if os.environ.get("SSL_NO_VERIFY"):
    ssl._create_default_https_context = ssl._create_unverified_context

# ================= CONFIG =================
DATA_DIR = "family_rgb_images"
OUT_DIR = Path("image_embeddings")
BATCH_SIZE = 32
CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# ==========================================

# Model registry: name -> embedding_dim
MODEL_REGISTRY = {
    "resnet50":          2048,
    "densenet121":       1024,
    "efficientnet_b0":   1280,
    "convnext_base":     1024,
    "swin_base":         1024,
    "mobilenetv3_large": 960,
    "dinov2":            768,
    "clip":              512,
    "beit":              768,
    "deit_base":         768,
}

ALL_MODELS = list(MODEL_REGISTRY.keys())


# ================= MODEL BUILDERS =================

def build_resnet50():
    m = models.resnet50(weights=None)
    m.load_state_dict(torch.load(CACHE_DIR / "resnet50-11ad3fa6.pth", map_location="cpu", weights_only=True))
    m.fc = nn.Identity()
    return m

def build_densenet121():
    m = models.densenet121(weights=None)
    m.load_state_dict(torch.load(CACHE_DIR / "densenet121-a639ec97.pth", map_location="cpu", weights_only=True))
    m.classifier = nn.Identity()
    return m

def build_efficientnet_b0():
    m = models.efficientnet_b0(weights=None)
    m.load_state_dict(torch.load(CACHE_DIR / "efficientnet_b0_rwightman-7f5810bc.pth", map_location="cpu", weights_only=True))
    m.classifier = nn.Identity()
    return m

def build_convnext_base():
    m = models.convnext_base(weights=None)
    m.load_state_dict(torch.load(CACHE_DIR / "convnext_base-6075fbad.pth", map_location="cpu", weights_only=True))
    m.classifier[2] = nn.Identity()  # Remove final Linear, keep LayerNorm+Flatten
    return m

def build_swin_base():
    m = models.swin_b(weights=None)
    m.load_state_dict(torch.load(CACHE_DIR / "swin_b-68c6b09e.pth", map_location="cpu", weights_only=True))
    m.head = nn.Identity()
    return m

def build_mobilenetv3_large():
    m = models.mobilenet_v3_large(weights=None)
    m.load_state_dict(torch.load(CACHE_DIR / "mobilenet_v3_large-5c1a4163.pth", map_location="cpu", weights_only=True))
    m.classifier = nn.Identity()
    return m

def build_dinov2():
    """DINOv2 ViT-B/14 — self-supervised vision transformer."""
    import timm
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    state = torch.load(CACHE_DIR / "dinov2_vitb14_pretrain.pth", map_location="cpu", weights_only=False)
    # DINOv2 official weights may not match timm exactly — try direct load
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing:
        print(f"    [DINOv2] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return m

def build_clip():
    """CLIP ViT-B/32 — OpenAI contrastive vision encoder."""
    import timm
    m = timm.create_model("vit_base_patch32_clip_224.openai", pretrained=False, num_classes=0)
    state = torch.load(CACHE_DIR / "ViT-B-32.pt", map_location="cpu", weights_only=False)
    # CLIP checkpoint contains full model — extract visual encoder
    if "visual" in dir(state) or "state_dict" in state:
        state = state.get("state_dict", state)
    # Try direct load first
    try:
        missing, unexpected = m.load_state_dict(state, strict=False)
        if len(missing) > len(state) * 0.5:
            raise RuntimeError("Too many missing keys")
    except Exception:
        # Extract visual keys with prefix removal
        visual_state = {}
        for k, v in state.items():
            if k.startswith("visual."):
                visual_state[k.replace("visual.", "")] = v
        if visual_state:
            missing, unexpected = m.load_state_dict(visual_state, strict=False)
    if missing:
        print(f"    [CLIP] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return m

def build_beit():
    """BEiT-Base — Microsoft masked image modeling."""
    import timm
    m = timm.create_model("beit_base_patch16_224", pretrained=False, num_classes=0)
    state = torch.load(CACHE_DIR / "beit_base_patch16_224_pt22k_ft22kto1k.pth", map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing:
        print(f"    [BEiT] Missing: {len(missing)} (head removed)")
    return m

def build_deit_base():
    """DeiT-Base — Facebook data-efficient image transformer."""
    import timm
    m = timm.create_model("deit_base_patch16_224", pretrained=False, num_classes=0)
    state = torch.load(CACHE_DIR / "deit_base_patch16_224-b5f2ef4d.pth", map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing:
        print(f"    [DeiT] Missing: {len(missing)} (head removed)")
    return m


MODEL_BUILDERS = {
    "resnet50": build_resnet50,
    "densenet121": build_densenet121,
    "efficientnet_b0": build_efficientnet_b0,
    "convnext_base": build_convnext_base,
    "swin_base": build_swin_base,
    "mobilenetv3_large": build_mobilenetv3_large,
    "dinov2": build_dinov2,
    "clip": build_clip,
    "beit": build_beit,
    "deit_base": build_deit_base,
}


# ================= EXTRACTION =================

def get_transform(model_name: str):
    """Get appropriate transform. DINOv2 uses 518x518, others 224x224."""
    if model_name == "dinov2":
        size = 518
    else:
        size = 224
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def extract_one_model(model_name: str, data_dir: str, batch_size: int):
    """Extract embeddings for a single model."""
    if model_name not in MODEL_REGISTRY:
        print(f"ERROR: Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
        return None

    expected_dim = MODEL_REGISTRY[model_name]
    print(f"\n{'='*70}")
    print(f"  Extracting: {model_name}")
    print(f"  Expected dim: {expected_dim}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}\n")

    # Load dataset
    dataset = datasets.ImageFolder(data_dir, transform=get_transform(model_name))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"  Dataset: {len(dataset)} samples, {len(dataset.classes)} classes")

    # Build model from local weights
    print(f"  Loading weights from cache...")
    model = MODEL_BUILDERS[model_name]()
    model = model.to(DEVICE)
    model.eval()

    # Verify output dim
    with torch.no_grad():
        img_size = 518 if model_name == "dinov2" else 224
        dummy = torch.randn(1, 3, img_size, img_size).to(DEVICE)
        dummy_out = model(dummy)
        actual_dim = dummy_out.shape[-1] if dummy_out.dim() == 2 else dummy_out.shape[1]
        print(f"  Actual embedding dim: {actual_dim}")

    # Extract
    all_embeddings = []
    all_labels = []
    all_paths = []

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            embeddings = model(images)

            # Ensure 2D output (batch, dim)
            if embeddings.dim() > 2:
                embeddings = embeddings.mean(dim=list(range(2, embeddings.dim())))

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

            # Get file paths for this batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + images.shape[0]
            batch_paths = [dataset.samples[i][0] for i in range(start_idx, end_idx)]
            all_paths.extend(batch_paths)

            if (batch_idx + 1) % 20 == 0:
                elapsed_so_far = time.time() - t0
                print(f"    Batch {batch_idx+1}/{len(loader)} ({elapsed_so_far:.1f}s)")

    elapsed = time.time() - t0

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Embeddings shape: {all_embeddings.shape}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = OUT_DIR / f"{model_name}_embeddings.pt"
    lab_path = OUT_DIR / f"{model_name}_labels.pt"
    path_path = OUT_DIR / f"{model_name}_paths.pt"

    torch.save(all_embeddings, emb_path)
    torch.save(all_labels, lab_path)
    torch.save(all_paths, path_path)

    print(f"  Saved: {emb_path} ({all_embeddings.shape[0]}x{all_embeddings.shape[1]})")
    return all_embeddings.shape


def main():
    parser = argparse.ArgumentParser(description="Extract image PTM embeddings")
    parser.add_argument("--model", nargs="+", default=["all"],
                        help="Model name(s) or 'all'")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if "all" in args.model:
        models_to_run = ALL_MODELS
    else:
        models_to_run = args.model

    print(f"Models to extract: {models_to_run}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Device: {DEVICE}")

    results = {}
    for model_name in models_to_run:
        try:
            shape = extract_one_model(model_name, args.data_dir, args.batch_size)
            if shape:
                results[model_name] = shape
        except Exception as e:
            print(f"\n  ERROR: {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for name, shape in results.items():
        print(f"  {name:20s} -> {shape[0]} samples x {shape[1]}d")
    failed = [m for m in models_to_run if m not in results]
    if failed:
        print(f"\n  FAILED: {failed}")
    print(f"\n  All saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
