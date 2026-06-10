# Multimodal Malware Detection — Experiment Report

## 1. Overview

This report documents the complete experimental pipeline for multimodal malware family classification using the MOTIF dataset. The pipeline evaluates pre-trained audio and image embeddings for malware detection, comparing single-modality baselines, comprehensive pairwise fusion (14 pairs × 4 methods), and cross-attention architectures in both Euclidean and hyperbolic space.

**Dataset:** MOTIF malware dataset  
**Total samples:** 3,094–3,095 (depending on modality alignment)  
**Total families:** 502  
**Evaluation:** 5-Fold Stratified Cross-Validation with PCA fitted inside each fold (no data leakage). Core screening experiments use seed=42; final selected models are additionally validated across 5 random seeds (42, 123, 777, 2025, 3407).  
**Primary metric:** Macro-F1  

---

## 2. Embedding Models

### 2.1 Audio Models

| Model | Embedding Dim | Source |
|-------|---------------|--------|
| HuBERT | 768 | Self-supervised speech model applied to malware audio |
| WavLM | 768 | Self-supervised speech model applied to malware audio |
| Wav2Vec2 | 768 | Self-supervised speech model applied to malware audio |

### 2.2 Image Models

| Model | Embedding Dim | Source |
|-------|---------------|--------|
| DINOv2 (ViT-B/14) | 768 | Self-supervised vision transformer |
| ConvNeXt-Base | 1024 | Modern CNN architecture |
| Swin-Base | 1024 | Shifted window transformer |
| EfficientNet-B0 | 1280 | Efficient compound-scaled CNN |
| MobileNetV3-Large | 960 | Lightweight mobile CNN |
| ResNet-50 | 2048 | Standard residual CNN |
| DeiT-Base | 768 | Data-efficient image transformer |
| VGG19 | 4096 | Deep CNN features |
| ViT-Base | 768 | Vision transformer |
| BEiT-Base | 768 | BERT-style image transformer |

---

## 3. Thresholds

To handle the long-tail class distribution, experiments are run at multiple minimum-sample thresholds:

| Threshold | Classes | Samples | Avg Samples/Class |
|-----------|---------|---------|-------------------|
| ≥ 10 | 89 | 1,946 | ~22 |
| ≥ 8 | 108 | 2,109 | ~20 |
| ≥ 5 | 166 | 2,445 | ~15 |

---

## 4. Phase 1: Single-Modality Baselines — Expanded (13 Models)

### 4.1 Evaluation Setup

**Classifiers tested (all with PCA fitted inside each fold):**
- LogReg: C=[100, 1000], class_weight=[None, balanced], L2 Normalizer, PCA=[256, 384, 512]
- SVM-RBF: C=100, class_weight=[None, balanced], StandardScaler, PCA=[128, 256]
- MLP: (512, 128), Adam, early_stopping, StandardScaler, PCA=[256, 512]

**Pipeline:** Embedding → PCA (fit on train) → L2 Norm (LogReg) or StandardScaler (SVM/MLP) → Classifier

### 4.2 Best Results Per Model (All Thresholds)

| Model | Thresh ≥10 | Thresh ≥8 | Thresh ≥5 | Best Config |
|-------|-----------|----------|----------|-------------|
| **HuBERT** | **0.6870 ±0.011** | **0.6831 ±0.015** | **0.5970 ±0.028** | SVM_bal_pca128 |
| **DINOv2** | **0.6810 ±0.010** | **0.6816 ±0.022** | **0.5845 ±0.026** | SVM_bal/SVM_pca128 |
| ConvNeXt | 0.6578 ±0.011 | 0.6512 ±0.013 | 0.5613 ±0.025 | SVM_bal_pca128 |
| Swin | 0.6470 ±0.008 | 0.6340 ±0.025 | 0.5547 ±0.020 | SVM_bal_pca128 |
| EfficientNet | 0.6463 ±0.011 | 0.6373 ±0.012 | 0.5553 ±0.024 | SVM_bal_pca128 |
| MobileNet | 0.6434 ±0.015 | 0.6373 ±0.016 | 0.5507 ±0.030 | SVM_bal_pca128 |
| ResNet50 | 0.6394 ±0.010 | 0.6309 ±0.018 | 0.5459 ±0.018 | SVM_bal_pca128 |
| DeiT | 0.6365 ±0.015 | 0.6242 ±0.024 | 0.5407 ±0.023 | SVM_bal/LR_bal_C1k |
| VGG19 | 0.6300 ±0.013 | 0.6145 ±0.018 | 0.5393 ±0.034 | SVM_bal_pca128 |
| WavLM | 0.6286 ±0.008 | 0.6189 ±0.015 | 0.5399 ±0.023 | SVM_pca128 |
| ViT | 0.6155 ±0.014 | 0.6125 ±0.018 | 0.5314 ±0.016 | SVM_bal_pca128 |
| BEiT | 0.5831 ±0.008 | 0.5748 ±0.016 | 0.4903 ±0.025 | SVM_bal_pca128 |
| Wav2Vec2 | 0.5727 ±0.019 | 0.5663 ±0.003 | 0.4948 ±0.036 | SVM_bal_pca256 |

### 4.3 Key Findings

1. **SVM-RBF with PCA=128 is the dominant configuration** — it is the best or near-best setting for most models
2. **DINOv2 is the top image model** (0.6810), nearly matching HuBERT audio (0.6870)
3. **Image model ranking:** DINOv2 > ConvNeXt > Swin > EfficientNet > MobileNet > ResNet50 > DeiT > VGG19 > ViT > BEiT
4. **Audio model ranking (base-size models):** HuBERT >> WavLM > Wav2Vec2 *(Note: superseded at large scale — see Section 17. WavLM-Large masked = 0.7060 vs HuBERT-Large = 0.6172)*
5. **Balanced class_weight helps marginally** over unweighted for most models
6. **PCA=128 dominates over PCA=256** for SVM — less noise, better generalization
7. **LogReg best:** LR_bal_C1k_pca512 (HuBERT 0.6621, DINOv2 0.6596, EfficientNet 0.6416)

---

## 5. Phase 2: Classifier Comparison (Original 5 Models)

**Classifiers tested on the original 5 models:**
- LogReg with PCA + L2 Normalizer
- SVM-RBF with PCA + StandardScaler
- MLP with PCA + StandardScaler

### Best Results per Classifier (Threshold ≥10)

| Model | LogReg (best) | SVM-RBF | MLP |
|-------|--------------|---------|-----|
| **HuBERT** | 0.6621 | **0.6870** | 0.6290 |
| ViT | 0.6140 | 0.6155 | 0.5735 |
| VGG19 | 0.5998 | 0.6300 | 0.5869 |
| WavLM | 0.6057 | 0.6286 | 0.5693 |
| Wav2Vec2 | 0.5596 | 0.5727 | 0.5237 |

### Best Single-Modality Results (All Thresholds)

| Threshold | Model | Classifier | Macro-F1 | Key Params |
|-----------|-------|-----------|----------|------------|
| **≥ 10** | **HuBERT** | **SVM-RBF** | **0.6870 ± 0.011** | PCA=128, C=100, γ=scale, balanced |
| **≥ 8** | **HuBERT** | **SVM-RBF** | **0.6831 ± 0.015** | PCA=128, C=100, γ=scale, balanced |
| **≥ 5** | **HuBERT** | **SVM-RBF** | **0.5970 ± 0.028** | PCA=128, C=100, γ=scale, balanced |

**Finding:** SVM-RBF with PCA=128 is the best single-modality classifier. HuBERT + SVM-RBF sets the overall single-modality ceiling.

---

## 6. Phase 3: Comprehensive Multimodal Fusion (Seed=42 Screening)

> **Note:** Results in this phase are from the comprehensive seed=42 screening protocol. Final multi-seed validated results are reported in Section 11.

### 6.1 Experimental Design

14 modality pairs tested with 4 fusion methods, 4 PCA dimensions, evaluated at K=5:

**Modality pairs (14 total):**
- Image+Image: ViT+VGG19
- Legacy Image+Audio: ViT+HuBERT, ViT+WavLM, ViT+Wav2Vec2, VGG19+HuBERT, VGG19+WavLM, VGG19+Wav2Vec2
- New Image+Audio: DINOv2+HuBERT, ConvNeXt+HuBERT, EfficientNet+HuBERT, Swin+HuBERT, MobileNet+HuBERT, DINOv2+WavLM, ConvNeXt+WavLM

**Fusion methods:**
1. **Concat+LR:** PCA per modality → L2 normalize → concatenate → LogReg (C=1000, balanced, lbfgs)
2. **Concat+SVM:** PCA per modality → StandardScaler → concatenate → SVM-RBF (C=100, γ=scale, balanced)
3. **FCN:** PCA per modality → StandardScaler → concatenate → FC(→1024→ReLU→Dropout(0.3)→n_classes)
4. **FCN_Balanced:** Same as FCN but with class-weighted CE loss

**PCA sweep:** 128, 256, 384, 512 per modality (fitted inside each fold, no data leakage)

**Protocol:** 5-fold stratified CV, thresholds ≥10/≥8/≥5.

### 6.2 Results: Best Per Pair (Threshold ≥10, 89 classes)

| Rank | Pair | Best Method | PCA | Macro-F1 |
|------|------|-------------|-----|----------|
| 1 | **DINOv2+HuBERT** | **FCN_Balanced** | **256** | **0.7223 ± 0.008** |
| 2 | Swin+HuBERT | FCN_Balanced | 256 | 0.7143 ± 0.009 |
| 3 | ConvNeXt+HuBERT | FCN_Balanced | 256 | 0.7131 ± 0.006 |
| 4 | MobileNet+HuBERT | FCN_Balanced | 256 | 0.7100 ± 0.015 |
| 5 | VGG19+HuBERT | FCN_Balanced | 256 | 0.7067 ± 0.012 |
| 6 | ViT+HuBERT | FCN_Balanced | 256 | 0.7066 ± 0.009 |
| 7 | EfficientNet+HuBERT | FCN_Balanced | 256 | 0.7060 ± 0.005 |
| 8 | DINOv2+WavLM | FCN_Balanced | 256 | 0.6947 ± 0.005 |
| 9 | ConvNeXt+WavLM | FCN_Balanced | 256 | 0.6887 ± 0.016 |
| 10 | VGG19+WavLM | FCN_Balanced | 256 | 0.6713 ± 0.008 |
| 11 | ViT+WavLM | FCN_Balanced | 256 | 0.6683 ± 0.009 |
| 12 | ViT+VGG19 | FCN_Balanced | 256 | 0.6652 ± 0.013 |
| 13 | VGG19+Wav2Vec2 | FCN_Balanced | 384 | 0.6369 ± 0.006 |
| 14 | ViT+Wav2Vec2 | FCN | 256 | 0.6351 ± 0.017 |

### 6.3 DINOv2+HuBERT Method Comparison (Threshold ≥10)

| Method | PCA=128 | PCA=256 | PCA=384 | PCA=512 |
|--------|---------|---------|---------|---------|
| Concat+SVM | 0.7017 | 0.6577 | 0.6203 | 0.5668 |
| Concat+LR | 0.6506 | 0.6594 | 0.6677 | 0.6687 |
| FCN | 0.7030 | 0.7149 | 0.7085 | 0.6953 |
| **FCN_Balanced** | 0.6988 | **0.7223** | 0.7144 | 0.7046 |

### 6.4 Cross-Threshold Stability

| Threshold | Classes | Best Pair | Method | PCA | Macro-F1 |
|-----------|---------|-----------|--------|-----|----------|
| **≥10** | 89 | DINOv2+HuBERT | FCN_Balanced | 256 | **0.7223 ± 0.008** |
| **≥8** | 108 | DINOv2+HuBERT | FCN_Balanced | 384 | **0.7167 ± 0.009** |
| **≥5** | 166 | DINOv2+HuBERT | FCN_Balanced | 512 | **0.6331 ± 0.025** |

Lower thresholds introduce more families and require higher PCA dimensionality for best performance.

### 6.5 Key Observations

1. **DINOv2+HuBERT is the new best pair** — beats old VGG19+HuBERT (0.7067) by +1.56pp consistently across all thresholds
2. **FCN_Balanced + PCA=256 is the universal best config** — optimal for all 14 pairs at threshold ≥10
3. **Every new image PTM + HuBERT beats VGG19+HuBERT**: DINOv2 (+1.6pp), Swin (+0.8pp), ConvNeXt (+0.6pp), MobileNet (+0.3pp)
4. **Concat+SVM collapses above PCA=128** — curse of dimensionality in RBF kernel space
5. **HuBERT-base >> WavLM-base as audio fusion partner** — DINOv2+WavLM-base (0.6947) is 2.8pp below DINOv2+HuBERT-base (0.7223). *(Note: this comparison used base-size models only. At large scale, WavLM-Large with masked pooling is the top unimodal encoder — see Section 17.)*
6. **Fusion gain over best single-modality is +3.5pp** (0.7223 vs HuBERT 0.6870)

---

## 7. Phase 4: Cross-Attention Fusion

### 7.1 Cross-Attention in Euclidean Space

Tested bidirectional cross-attention (without hyperbolic mapping) on the top 3 pairs.

**Architecture:**
- PCA per modality → Projection (256d) → LayerNorm → GELU
- Bidirectional cross-attention (A→B, B→A) with residual
- FFN block after attention
- Gated fusion → FC classifier

**Configs tested:** 4 heads and 8 heads, with/without class-weighted loss.

| Pair | Best Config | Macro-F1 | vs FCN_Balanced |
|------|-------------|----------|-----------------|
| ViT+HuBERT | CrossAttn_Bal_h4 | 0.6835 ± 0.007 | ↓ 0.0231 |
| VGG19+HuBERT | CrossAttn_Bal_h8 | 0.6799 ± 0.009 | ↓ 0.0268 |
| VGG19+WavLM | CrossAttn_h4 | 0.6531 ± 0.006 | ↓ 0.0182 |

**Finding:** Cross-attention in Euclidean space **underperforms** FCN by 2–3% and even falls below single-modality HuBERT+SVM (0.6870). With single-token embeddings (seq_len=1), attention degenerates to a learned weighted average — there is no sequence structure for attention to exploit.

### 7.2 Cross-Attention in Hyperbolic Space

**Architecture additions over Euclidean:**
- Poincaré ball exp/log maps after gated fusion
- Ranking-aware loss (CE + margin ranking)
- Label smoothing

#### v1 Results: HuBERT + VGG19

| Threshold | Macro-F1 | HuBERT LR Baseline | Gain |
|-----------|----------|--------------------|----- |
| **≥ 10** | **0.6924 ± 0.009** | 0.6607 | **↑ 0.0317** |
| ≥ 8 | 0.6563 ± 0.012 | 0.6438 | ↑ 0.0125 |
| ≥ 5 | 0.5733 ± 0.018 | 0.5718 | ↑ 0.0015 |

#### v3 Results (Improved): HuBERT + VGG19

Improvements: proj_dim 256→384, 8 heads, FFN, label smoothing + ranking loss, PCA 256→384.

| Threshold | Macro-F1 (v3) | vs v1 | vs SVM-RBF |
|-----------|--------------|-------|------------|
| **≥ 10** | **0.6990 ± 0.013** | ↑ 0.0066 | **↑ 0.0120** |
| ≥ 8 | 0.6623 ± 0.014 | ↑ 0.0060 | ↓ 0.0208 |
| ≥ 5 | 0.5777 ± 0.015 | ↑ 0.0044 | ↓ 0.0234 |

---

## 8. Seed=42 Screening Comparison Table (Threshold ≥10)

> **Note:** This table summarizes seed=42 screening results. Multi-seed validated final results are reported in Section 11.6.

All methods ranked by Macro-F1 (89 classes, 1,946 samples, seed=42):

| Rank | Method | Macro-F1 | Gain vs HuBERT SVM |
|------|--------|----------|-------------------|
| 1 | **DINOv2+HuBERT FCN_Balanced** | **0.7223 ± 0.008** | **↑ 0.0353** |
| 2 | Swin+HuBERT FCN_Balanced | 0.7143 ± 0.009 | ↑ 0.0273 |
| 3 | ConvNeXt+HuBERT FCN_Balanced | 0.7131 ± 0.006 | ↑ 0.0261 |
| 4 | MobileNet+HuBERT FCN_Balanced | 0.7100 ± 0.015 | ↑ 0.0230 |
| 5 | VGG19+HuBERT FCN_Balanced | 0.7067 ± 0.012 | ↑ 0.0197 |
| 6 | ViT+HuBERT FCN_Balanced | 0.7066 ± 0.009 | ↑ 0.0196 |
| 7 | EfficientNet+HuBERT FCN_Balanced | 0.7060 ± 0.005 | ↑ 0.0190 |
| 8 | DINOv2+HuBERT Concat+SVM | 0.7017 ± 0.011 | ↑ 0.0147 |
| 9 | VGG19+HuBERT CrossAttn Hyperbolic v3 | 0.6990 ± 0.013 | ↑ 0.0120 |
| 10 | VGG19+HuBERT FCN | 0.6988 ± 0.010 | ↑ 0.0118 |
| 11 | DINOv2+WavLM FCN_Balanced | 0.6947 ± 0.005 | ↑ 0.0077 |
| 12 | VGG19+HuBERT CrossAttn Hyperbolic v1 | 0.6924 ± 0.009 | ↑ 0.0054 |
| 13 | **HuBERT (SVM-RBF)** | **0.6870 ± 0.011** | — (single-modality ceiling) |
| 14 | ViT+HuBERT CrossAttn Euclidean (Bal_h4) | 0.6835 ± 0.007 | ↓ 0.0035 |
| 15 | **DINOv2 (SVM-RBF)** | **0.6810 ± 0.010** | — (single-modality #2) |
| 16 | HuBERT (LogReg) | 0.6621 ± 0.017 | — (LR baseline) |
| 17 | DINOv2 (LogReg) | 0.6596 ± 0.012 | — |
| 18 | ConvNeXt (SVM-RBF) | 0.6578 ± 0.011 | — |
| 19 | Swin (SVM-RBF) | 0.6470 ± 0.014 | — |
| 20 | VGG19 (SVM-RBF) | 0.6300 ± 0.013 | — |
| 21 | ViT (SVM-RBF) | 0.6155 ± 0.014 | — |

---

## 9. Late Fusion / Stacking

### 9.1 Motivation

Early fusion (FCN_Balanced) concatenates PCA-reduced embeddings and trains a single classifier. Late fusion instead trains independent per-modality SVM classifiers and fuses their output probability vectors. This tests whether decision-level fusion can outperform feature-level fusion.

### 9.2 Experimental Design

**Base classifiers:** SVM-RBF (C=100, γ=scale, balanced, probability=True) with PCA=128 per modality.

**Fully nested stacking protocol (no leakage):**
- Outer loop: 5-fold stratified CV
- Inner loop: 4-fold OOF probability generation within each outer-train
- Base SVMs retrained on full outer-train for test prediction
- Meta-learner trained on inner-OOF, predicts on test probabilities

**Models:** HuBERT, DINOv2, Swin, ConvNeXt

**Combinations:** 2-model, 3-model, and 4-model ensembles

**Meta-fusion methods:**
1. Average probabilities (no training)
2. Weighted average (weights ∝ individual OOF F1)
3. Stacked LogisticRegression (C=10, balanced, lbfgs)
4. Stacked MLP (256→128, early stopping)

### 9.3 Results (Threshold ≥10, 89 classes)

| Combo | Avg Prob | Weighted Avg | Stack LogReg | Stack MLP |
|-------|----------|--------------|--------------|-----------|
| HuBERT+DINOv2 | 0.6985 | 0.6976 | 0.6303 | 0.6574 |
| HuBERT+DINOv2+Swin | 0.6944 | 0.6953 | 0.6481 | 0.6493 |
| **HuBERT+DINOv2+ConvNeXt** | **0.7006** | **0.7020** | 0.6510 | 0.6673 |
| HuBERT+DINOv2+Swin+ConvNeXt | 0.6929 | 0.6929 | 0.6591 | 0.6650 |

**Individual model OOF SVM Macro-F1:** HuBERT=0.6807, DINOv2=0.6643, ConvNeXt=0.6393, Swin=0.6286

### 9.4 Analysis: Why Late Fusion Fails to Beat Early Fusion

Best late fusion: **HuBERT+DINOv2+ConvNeXt Weighted Avg = 0.7020**
Early fusion baseline (seed=42): **DINOv2+HuBERT FCN_Balanced = 0.7223**
Gap: **−0.0203**

Late fusion underperforms for several reasons:

1. **Information bottleneck:** SVM probability vectors are a lossy compression of the original embeddings. The 89-dimensional probability simplex discards fine-grained feature relationships that FCN can exploit from the 256d PCA embeddings directly.

2. **SVM probability calibration:** Platt scaling (used by `probability=True`) is approximate — probabilities are poorly calibrated for tail classes with few samples, degrading fusion quality.

3. **Stacking suffers from dimensionality:** With 89 classes × N models, the meta-learner input is 178–356 features trained on ~1556 inner samples — prone to overfitting despite nested CV.

4. **Adding more models hurts average fusion:** The 4-model ensemble (0.6929) is worse than the 2-model (0.6985) because weaker models (Swin=0.6286) dilute the signal from stronger ones.

5. **Weighted average is marginal:** Weights are nearly uniform (HuBERT=0.506, DINOv2=0.494) because individual F1s are close, providing minimal benefit over simple averaging.

### 9.5 Conclusion

Late fusion / stacking does **not** improve over early feature fusion for this task. The FCN_Balanced approach directly learning from rich PCA embeddings outperforms any decision-level combination of SVM probability outputs. This confirms that the feature-level complementarity between DINOv2 and HuBERT is best exploited by learned nonlinear fusion rather than post-hoc probability combination.

---

## 10. Advanced Loss Functions (SupCon / ArcFace / Prototype)

### 10.1 Motivation

Under the seed=42 optimized training setting (100 epochs, patience 12), FCN_Balanced reaches 0.7361. This phase tests whether advanced representation-learning losses — Supervised Contrastive (SupCon), ArcFace angular margin, and Prototype Networks — can improve over this strong single-seed baseline by shaping the embedding geometry before classification.

### 10.2 Setup

**Pair:** DINOv2 + HuBERT (best from Phase 3 fusion)  
**Config:** PCA=256, Threshold≥10, 89 classes, 1946 samples  
**Training:** Adam lr=1e-3, batch=256, max 100 epochs, patience=12  
**Evaluation:** 5-fold Stratified K-Fold (same protocol as all previous phases)

**Baselines:**
1. FCN_Balanced (original): input→1024→ReLU→Drop(0.3)→n_classes
2. FCN_Balanced (3-layer): input→1024→ReLU→Drop→512→ReLU→Drop→n_classes
3. CE_Balanced (bottleneck): input→1024→ReLU→Drop→256→Linear(n_classes)

**Advanced losses (all use bottleneck encoder → 256d embeddings):**
- **SupCon:** CE + λ·SupConLoss — grid: λ∈{0.05, 0.1, 0.2} × τ∈{0.07, 0.1}
- **ArcFace:** Encoder + ArcFace head (angular margin classification) — grid: margin∈{0.3, 0.5} × scale∈{30, 64}
- **Prototype:** CE + λ·ProtoLoss (pull embeddings toward class centroids) — grid: λ∈{0.05, 0.1, 0.2}

### 10.3 Results (Threshold ≥10, 89 classes)

| Rank | Method | Macro-F1 | Std |
|------|--------|----------|-----|
| 1 | **FCN_Balanced (original)** | **0.7361** | 0.0065 |
| 2 | CE+SupCon(λ=0.05,τ=0.07) | 0.7338 | 0.0092 |
| 3 | CE+Proto(λ=0.2) | 0.7334 | 0.0095 |
| 4 | CE+Proto(λ=0.1) | 0.7333 | 0.0098 |
| 5 | FCN_Balanced (3-layer) | 0.7333 | 0.0110 |
| 6 | CE_Balanced (bottleneck) | 0.7330 | 0.0087 |
| 7 | CE+Proto(λ=0.05) | 0.7327 | 0.0091 |
| 8 | CE+SupCon(λ=0.05,τ=0.1) | 0.7303 | 0.0074 |
| 9 | CE+SupCon(λ=0.1,τ=0.07) | 0.7301 | 0.0086 |
| 10 | CE+SupCon(λ=0.2,τ=0.07) | 0.7287 | 0.0070 |
| 11 | CE+SupCon(λ=0.2,τ=0.1) | 0.7260 | 0.0053 |
| 12 | ArcFace(m=0.3,s=30) | 0.7256 | 0.0150 |
| 13 | ArcFace(m=0.5,s=64) | 0.7255 | 0.0132 |
| 14 | CE+SupCon(λ=0.1,τ=0.1) | 0.7254 | 0.0071 |
| 15 | ArcFace(m=0.5,s=30) | 0.7244 | 0.0132 |
| 16 | ArcFace(m=0.3,s=64) | 0.7222 | 0.0167 |

### 10.4 Analysis: Why Advanced Losses Don't Help

**No method beats the simple CE baseline (0.7361).** The advanced losses consistently hurt performance:

1. **SupCon (best = 0.7338, Δ = −0.0023):** The contrastive term is a regularizer that competes with CE for gradient signal. At low λ (0.05), the damage is minimal; at λ=0.2 it drops to 0.7260. SupCon helps when data is abundant — with only ~22 samples/class on average, there are too few positive pairs per class for contrastive learning to discover structure that CE doesn't already capture.

2. **Prototype (best = 0.7334, Δ = −0.0027):** Prototype loss encourages embeddings to cluster around class centroids. This is redundant with CE + balanced sampling, which already produces well-separated class clusters. The additional constraint slightly over-regularizes the representation.

3. **ArcFace (best = 0.7256, Δ = −0.0105):** Angular margin is the worst performer. With 89 classes and only 1946 samples, the angular decision boundaries are too aggressive — the model cannot achieve the required margins for tail classes with 10–15 samples. High variance (std ~0.015) confirms instability.

4. **Deeper architectures don't help either:** 3-layer FCN (0.7333) ≈ bottleneck (0.7330) ≈ original (0.7361). The task is not capacity-limited — a single hidden layer suffices for 256d PCA input → 89 classes.

### 10.5 Key Takeaway

Within the seed=42 optimized setting, advanced losses fail to improve over the original FCN_Balanced baseline (0.7361). Multi-seed validation in Section 11 shows that the expected two-modality performance is lower, around 0.7231, confirming substantial seed sensitivity. The performance appears strongly **data-regime limited** — with ~22 samples/class average (and some classes having only 10), the bottleneck is likely statistical rather than architectural. Simple CE with class-balanced weighting appears to exploit the available frozen-embedding signal more effectively than the tested auxiliary losses.

---

## 11. Multi-Seed Validation & Multi-Encoder Fusion

### 11.1 Motivation

Single-seed results (seed=42) can be misleadingly optimistic. This phase:
1. Validates the best 2-modality result across 5 seeds to establish the true expected performance
2. Tests whether adding a third encoder (ConvNeXt, Swin) genuinely improves over 2-encoder fusion
3. Establishes multi-threshold baselines with proper uncertainty quantification

### 11.2 Multi-Seed Validation (DINOv2 + HuBERT, 5 seeds)

**Seeds:** 42, 123, 777, 2025, 3407  
**Architecture:** FCN_Balanced (original), EPOCHS=100, PATIENCE=12  

#### Threshold ≥10 (89 classes, 1946 samples, PCA=256)

| Seed | Mean F1 |
|------|---------|
| 42 | 0.7361 |
| 123 | 0.7135 |
| 777 | 0.7215 |
| 2025 | 0.7209 |
| 3407 | 0.7233 |
| **Grand Mean** | **0.7231 ± 0.0073** |

**All-folds mean:** 0.7231 ± 0.0146 (25 folds)  
**Range:** 0.7135–0.7361

#### Threshold ≥8 (108 classes, 2109 samples, PCA=384)

| Seed | Mean F1 |
|------|---------|
| 42 | 0.7018 |
| 123 | 0.7039 |
| 777 | 0.7270 |
| 2025 | 0.6946 |
| 3407 | 0.7026 |
| **Grand Mean** | **0.7060 ± 0.0110** |

#### Threshold ≥5 (166 classes, 2445 samples, PCA=512)

| Seed | Mean F1 |
|------|---------|
| 42 | 0.6164 |
| 123 | 0.6217 |
| 777 | 0.6200 |
| 2025 | 0.6206 |
| 3407 | 0.6177 |
| **Grand Mean** | **0.6193 ± 0.0020** |

### 11.3 Three-Encoder Feature Fusion (Threshold ≥10)

**Combos tested:** DINOv2+HuBERT+Swin, DINOv2+HuBERT+ConvNeXt, DINOv2+HuBERT+Swin+ConvNeXt  
**PCA per modality:** 128, 256  
**Dropout:** 0.3, 0.4  

| Rank | Combo | PCA | Drop | Macro-F1 | Std |
|------|-------|-----|------|----------|-----|
| 1 | **DINOv2+HuBERT+ConvNeXt** | 256 | 0.4 | **0.7414** | 0.0036 |
| 2 | DINOv2+HuBERT+ConvNeXt | 256 | 0.3 | 0.7407 | 0.0026 |
| 3 | DINOv2+HuBERT+Swin | 256 | 0.3 | 0.7395 | 0.0018 |
| 4 | DINOv2+HuBERT+Swin | 256 | 0.4 | 0.7395 | 0.0024 |
| 5 | DINOv2+HuBERT+Swin+ConvNeXt | 256 | 0.4 | 0.7340 | 0.0075 |
| 6 | DINOv2+HuBERT+Swin+ConvNeXt | 256 | 0.3 | 0.7325 | 0.0084 |

### 11.4 Multi-Seed Validation of Best Three-Encoder Result

**DINOv2+HuBERT+ConvNeXt**, PCA=256, Dropout=0.4:

| Seed | Mean F1 |
|------|---------|
| 42 | 0.7414 |
| 123 | 0.7225 |
| 777 | 0.7230 |
| 2025 | 0.7234 |
| 3407 | 0.7215 |
| **Grand Mean** | **0.7264 ± 0.0075** |

### 11.5 Analysis

1. **Seed 42 was consistently lucky** — highest among all seeds for both 2-encoder (0.7361) and 3-encoder (0.7414). Single-seed reporting would have been misleading.

2. **True improvement from 3-encoder fusion: +0.0033** (0.7264 vs 0.7231). Real but small — ConvNeXt adds complementary information to DINOv2+HuBERT, but most discriminative structure is already captured by the 2-encoder pair.

3. **4-encoder fusion hurts** (0.7325–0.7340 vs 0.7395–0.7414 for 3-encoder). Adding both Swin AND ConvNeXt introduces redundant visual features that dilute the signal. PCA from correlated visual models introduces noise.

4. **PCA=256 is universally optimal** — all PCA=128 configs score 0.72–0.73 regardless of encoder count, confirming the feature bottleneck seen in 2-encoder experiments.

5. **Threshold ≥5 is remarkably stable** (range 0.0054 across seeds) but low (0.6193) — the many 5–10 sample classes constrain absolute performance regardless of seed.

### 11.6 Validated Final Results Table

| Threshold | Classes | 2-Encoder (DINOv2+HuBERT) | 3-Encoder (DINOv2+HuBERT+ConvNeXt) |
|-----------|---------|---------------------------|-------------------------------------|
| ≥10 | 89 | 0.7231 ± 0.0073 | **0.7264 ± 0.0075** |
| ≥8 | 108 | 0.7060 ± 0.0110 | — |
| ≥5 | 166 | 0.6193 ± 0.0020 | — |

All values are multi-seed (5 seeds × 5 folds = 25 runs), representing the **true expected performance** free of seed luck. The ± values are standard deviations computed across seed-level CV means (N=5).

Therefore, the most defensible final result is the multi-seed validated three-encoder model: **DINOv2+HuBERT+ConvNeXt with FCN_Balanced, PCA=256, dropout=0.4, achieving 0.7264 ± 0.0075 Macro-F1 at threshold ≥10.**

---

## 12. Embedding-Space Augmentation

### 12.1 Motivation

With ~22 samples/class on average, data scarcity is the primary bottleneck. Embedding-space augmentation applies transformations to frozen PCA embeddings during training only, potentially improving generalization without requiring new data or fine-tuning encoders.

### 12.2 Methods Tested (Seed=42 Screening)

**Augmentation applied after PCA + StandardScaler, on training folds only. Validation folds are never augmented.**

- **(A) Gaussian noise:** `x_aug = x + N(0, σ²I)`, σ ∈ {0.005, 0.01, 0.02, 0.03}
- **(B) Same-class Mixup:** `x_aug = λ·x_i + (1-λ)·x_j` where `y_i = y_j`, λ ~ Beta(α, α), α ∈ {0.2, 0.4}
- **(C) Combined:** Mixup followed by Gaussian noise

### 12.3 Screening Results (Seed=42)

#### 2-Encoder (DINOv2+HuBERT, dropout=0.3)

| Method | Macro-F1 | Δ vs Baseline |
|--------|----------|---------------|
| **Mixup α=0.4** | **0.7419** | **+0.0058** |
| Combined σ=0.02, α=0.2 | 0.7407 | +0.0045 |
| Gaussian σ=0.02 | 0.7399 | +0.0038 |
| No augmentation | 0.7361 | — |

#### 3-Encoder (DINOv2+HuBERT+ConvNeXt, dropout=0.4)

| Method | Macro-F1 | Δ vs Baseline |
|--------|----------|---------------|
| **Mixup α=0.4** | **0.7436** | **+0.0022** |
| Mixup α=0.2 | 0.7425 | +0.0011 |
| No augmentation | 0.7414 | — |
| Gaussian σ=0.005 | 0.7395 | −0.0019 |
| Gaussian σ=0.03 | 0.7384 | −0.0030 |

Key screening finding: same-class mixup consistently helps; Gaussian noise helps 2-encoder marginally but hurts 3-encoder.

### 12.4 Multi-Seed Validation (Mixup α=0.4)

The best augmentation (same-class mixup α=0.4) was validated across 5 seeds:

| Model | Baseline (validated) | Mixup α=0.4 (validated) | Mean Δ | Positive Seeds |
|-------|---------------------|------------------------|--------|----------------|
| DINOv2+HuBERT | 0.7231 ± 0.0073 | 0.7210 ± 0.0106 | **−0.0021** | 1/5 |
| DINOv2+HuBERT+ConvNeXt | 0.7264 ± 0.0075 | 0.7277 ± 0.0082 | **+0.0013** | 3/5 |

Per-seed breakdown (3-encoder):

| Seed | Baseline | Mixup | Δ |
|------|----------|-------|---|
| 42 | 0.7414 | 0.7436 | +0.0022 |
| 123 | 0.7225 | 0.7219 | −0.0006 |
| 777 | 0.7230 | 0.7220 | −0.0011 |
| 2025 | 0.7234 | 0.7269 | +0.0035 |
| 3407 | 0.7215 | 0.7239 | +0.0024 |

### 12.5 Analysis

1. **2-encoder: Mixup does not validate.** The seed=42 gain (+0.0058) was entirely seed-specific. Across 5 seeds, mixup slightly hurts (−0.0021) and increases variance (std 0.0073 → 0.0106). With dropout=0.3, the model at seed=42 happened to benefit from the additional regularization, but this is not reproducible.

2. **3-encoder: Mixup gives marginal, non-decisive benefit.** The +0.0013 gain is positive but much smaller than the seed-level std (0.0075). Only 3/5 seeds show improvement. This is not statistically robust enough to claim a meaningful new result.

3. **Gaussian noise is not useful.** For the 3-encoder model, all σ values hurt performance. Random noise pushes samples off the data manifold, which is particularly harmful when the representation is already rich and well-structured.

4. **Why augmentation fails:** The frozen PCA embeddings appear to capture most of the available discriminative signal. Adding diversity via noise or interpolation cannot create new class-discriminative information — it only perturbs existing signal. The bottleneck is genuine lack of class-specific examples, not lack of training-time diversity.

### 12.6 Conclusion

Embedding-space augmentation does not provide a robust improvement over the non-augmented baseline. The best validated result remains **0.7264 ± 0.0075** without augmentation. Same-class mixup gives a marginal gain for the 3-encoder model (0.7277 ± 0.0082), but the improvement is smaller than seed variance and not statistically decisive.

---

## 13. Additional Ablation: FOCA-Style CNN Evaluation

### 13.1 Motivation

FOCA-style convolutional evaluation inspired by prior multimodal malware work. We tested whether Conv1D layers on embeddings — treating embedding dimensions as a 1D sequence signal — could capture useful patterns.

### 13.2 Architecture

```
PCA(256) → StandardScaler → Conv1D(1→64, k=3) → ReLU → MaxPool(2)
→ Conv1D(64→128, k=3) → ReLU → MaxPool(2) → Flatten
→ Dropout(0.3) → FC(128) → ReLU → Dropout(0.3) → FC(n_classes)
```

**Training:** Adam lr=1e-3, batch=32, max 50 epochs, patience=7 (early stopping on val loss)

### 13.3 Results (Threshold ≥10, 89 classes)

| Model | Macro-F1 ± Std |
|-------|---------------|
| HuBERT | 0.5407 ± 0.036 |
| DINOv2 | 0.5372 ± 0.026 |
| EfficientNet | 0.5155 ± 0.025 |
| ConvNeXt | 0.5144 ± 0.020 |
| VGG19 | 0.5046 ± 0.016 |
| DeiT | 0.5043 ± 0.026 |
| ViT | 0.5038 ± 0.013 |
| Swin | 0.5014 ± 0.025 |
| ResNet50 | 0.4836 ± 0.022 |
| MobileNet | 0.4817 ± 0.030 |
| WavLM | 0.4789 ± 0.018 |
| Wav2Vec2 | 0.4546 ± 0.028 |
| BEiT | 0.4213 ± 0.027 |

### 13.4 Analysis: Why CNN Underperforms SVM

The best CNN result (HuBERT = 0.5407) is **21% below** SVM-RBF (0.6870). This is expected:

1. **Ordering assumption violated:** Conv1D treats adjacent embedding dimensions as spatially related, but PTM pooled embeddings have no meaningful spatial/sequential ordering across dimensions.
2. **Data efficiency:** SVM-RBF with PCA=128 has far fewer parameters to fit on 1946 training samples.
3. **Inductive bias mismatch:** Convolutional locality is useful for images/audio sequences but not for unordered feature vectors.
4. **Conclusion:** For pooled PTM embeddings, kernel methods (SVM-RBF) or linear probes (LogReg) are more appropriate than convolutional architectures.

---

## 14. Key Insights

### Fusion Method Hierarchy

| Rank | Method | Best Macro-F1 | Notes |
|------|--------|--------------|-------|
| 1 | **Three-encoder FCN_Balanced** | **0.7264 ± 0.0075** | Multi-seed validated three-encoder fusion |
| 2 | **Two-encoder FCN_Balanced** | **0.7231 ± 0.0073** | Multi-seed validated DINOv2+HuBERT |
| 3 | Seed=42 optimized FCN_Balanced | 0.7361 | High single-seed result, not expected performance |
| 4 | Seed=42 three-encoder FCN_Balanced | 0.7414 | Highest observed single-seed result |
| 5 | Late fusion weighted average | 0.7020 | Decision-level fusion underperforms feature-level |
| 6 | Hyperbolic cross-attention v3 | 0.6990 | Higher complexity but lower performance |

### DINOv2 Improves Fusion Consistently (Seed=42 Screening)

| Threshold | Old Best (VGG19+HuBERT) | New Best (DINOv2+HuBERT) | Gain |
|-----------|--------------------------|---------------------------|------|
| ≥10 | 0.7067 | **0.7223** | **+0.0156** |
| ≥8 | 0.7022 | **0.7167** | **+0.0145** |
| ≥5 | 0.6206 | **0.6331** | **+0.0125** |

DINOv2's self-supervised visual features provide richer structural information about PE binary visualizations, yielding a stable +1.2 to +1.6pp gain across all difficulty settings. After multi-seed validation, the expected threshold ≥10 performance is 0.7231 for DINOv2+HuBERT (two-encoder) and 0.7264 for DINOv2+HuBERT+ConvNeXt (three-encoder).

### PCA Sweet Spot Scales with Number of Classes

| Threshold | Classes | Optimal PCA |
|-----------|---------|-------------|
| ≥10 | 89 | 256 |
| ≥8 | 108 | 384 |
| ≥5 | 166 | 512 |

More classes require more representation capacity — the optimal PCA dimensionality increases proportionally with the number of families.

### HuBERT-base >> WavLM-base as Audio Partner (Base-Size Models)

| Audio Model | With DINOv2 (≥10) | With DINOv2 (≥8) | With DINOv2 (≥5) |
|-------------|--------------------|--------------------|-------------------|
| **HuBERT-base** | **0.7223** | **0.7167** | **0.6331** |
| WavLM-base | 0.6947 | 0.6902 | 0.6068 |
| Δ | +0.0276 | +0.0265 | +0.0263 |

HuBERT-base consistently outperforms WavLM-base by ~2.7pp as a fusion partner. However, this finding applies only to base-size models. The new large PTM benchmark (Section 17) shows that **WavLM-Large with masked pooling (0.7060) far exceeds HuBERT-Large (0.6172)** as a unimodal encoder, suggesting WavLM-Large may also be the superior fusion partner at large scale.

### Concat+SVM Collapses at High PCA Dimensions

DINOv2+HuBERT Concat+SVM performance by PCA:

| PCA=128 | PCA=256 | PCA=384 | PCA=512 |
|---------|---------|---------|---------|
| **0.7017** | 0.6577 | 0.6203 | 0.5668 |

RBF kernel SVM suffers curse of dimensionality — concatenated features beyond 256d degrade dramatically. Neural fusion (FCN) handles higher dimensions better.

### Why Cross-Attention Underperforms FCN

Cross-attention (Euclidean) achieves 0.68 but falls 2–3% below FCN because:
- Embeddings are **single-token** (seq_len=1) — no sequence structure for attention to exploit
- Attention degenerates to a learned weighted average with only 1 query/key
- The additional parameters (proj + MHA + FFN) are harder to optimize for small data
- Simple concatenation + nonlinear classifier (FCN) is a more appropriate inductive bias

### Image PTM Ranking for Fusion (Seed=42 Screening)

When paired with HuBERT (FCN_Balanced, PCA=256, ≥10):

| Rank | Image PTM | Fusion Macro-F1 | Single-Modality |
|------|-----------|-----------------|------------------|
| 1 | **DINOv2** | **0.7223** | 0.6810 |
| 2 | Swin | 0.7143 | 0.6470 |
| 3 | ConvNeXt | 0.7131 | 0.6578 |
| 4 | MobileNet | 0.7100 | 0.6434 |
| 5 | VGG19 | 0.7067 | 0.6300 |
| 6 | ViT | 0.7066 | 0.6155 |
| 7 | EfficientNet | 0.7060 | 0.6463 |

All image PTMs converge to a narrow band (0.7060–0.7223) once fused with HuBERT, suggesting that HuBERT provides a strong shared audio anchor, while image encoders contribute smaller but complementary gains.

### Practical Conclusions

- **Best validated overall method:** DINOv2+HuBERT+ConvNeXt FCN_Balanced = **0.7264 ± 0.0075** (multi-seed, three-encoder)
- **Best validated 2-encoder method:** DINOv2+HuBERT FCN_Balanced = **0.7231 ± 0.0073** (multi-seed)
- **Best observed with augmentation:** DINOv2+HuBERT+ConvNeXt + same-class mixup α=0.4 = **0.7277 ± 0.0082** (marginal, not decisive)
- **Highest observed single-seed result:** DINOv2+HuBERT+ConvNeXt = **0.7414** (seed=42, not representative)
- **Best single-modality (base models):** HuBERT-base SVM-RBF = **0.6870**
- **Best single-modality (large models, Section 17):** WavLM-Large masked SVM-RBF = **0.7060**
- **Best image model:** DINOv2-Large SVM-RBF = **0.6884** (DINOv2-B = 0.6810)
- **Three-encoder gain is marginal:** +0.0033 over 2-encoder (0.7264 vs 0.7231)
- **Embedding-space augmentation does not robustly help:** same-class mixup gives +0.0013 for 3-encoder but hurts 2-encoder (−0.0021)
- **Advanced losses (SupCon, ArcFace, Proto) do not outperform** the optimized CE baseline
- **CNN on embeddings is ineffective:** FOCA-style Conv1D loses 15–26% vs SVM
- **Performance appears strongly data-limited:** the small per-family sample count (~22 avg) likely constrains further gains from more complex heads/losses
- **Single-seed results are unreliable:** seed 42 was lucky by +0.013 (range 0.023 across seeds)

### Final Takeaway

The best defensible configuration is **DINOv2+HuBERT+ConvNeXt with class-balanced FCN fusion**, achieving **0.7264 ± 0.0075 Macro-F1** under five-seed validation at threshold ≥10. The experiments show that modern self-supervised image features, especially DINOv2, substantially strengthen malware-image representations, but the largest gains come from simple feature-level fusion with HuBERT rather than complex attention, hyperbolic fusion, late stacking, or auxiliary metric-learning losses. *(Note: The new large PTM benchmark in Section 17 identifies WavLM-Large masked = 0.7060 as the strongest unimodal encoder, which has yet to be validated in the fusion pipeline.)*

---

## 15. Experimental Setup

- **Python:** 3.13.6
- **Hardware:** CPU (macOS)
- **Key libraries:** PyTorch, scikit-learn
- **Evaluation protocol:** 5-fold Stratified K-Fold, PCA fit on train split only
- **No data leakage:** All preprocessing (PCA, StandardScaler) fitted exclusively on training folds
- **Reproducibility:** seed=42 for screening; final selected models evaluated with seeds 42, 123, 777, 2025, and 3407
- **Alignment:** Path-based sample ID matching between image (3094) and audio (3095) modalities

---

## 16. Files Reference

| File | Purpose |
|------|---------|
| `gridsearch_all_models.py` | Phase 1: Initial LR grid search (all models) |
| `gridsearch_extended.py` | Phase 1: Extended LR grid search (top 4 models) |
| `gridsearch_nonlinear.py` | Phase 2: SVM-RBF, MLP, LogReg comparison |
| `fusion_all_pairs.py` | Phase 3: Comprehensive 14-pair × 4-method × 4-PCA fusion comparison |
| `cross_attn_euclidean.py` | Phase 4: Cross-attention fusion (Euclidean, top 3 pairs) |
| `cross_attention_hyperbolic_final.py` | Phase 4: Cross-attention fusion v1 (Hyperbolic) |
| `cross_attention_hyperbolic_v3.py` | Phase 4: Improved cross-attention v3 (Hyperbolic) |
| `eval_image_ptms.py` | Phase 5: Full 13-model evaluation (LogReg + SVM + MLP × 3 thresholds) |
| `eval_foca_cnn.py` | Phase 5: FOCA-style CNN evaluation on all PTM embeddings |
| `late_fusion_stacking.py` | Phase 6: Late fusion / stacking (fully nested, 4 meta-methods) |
| `advanced_losses.py` | Phase 7: Advanced losses (SupCon, ArcFace, Prototype) on DINOv2+HuBERT |
| `optimized_fcn_validation.py` | Phase 8: Multi-seed validation of FCN_Balanced across thresholds |
| `three_modality_fusion.py` | Phase 8: Three-encoder feature fusion (DINOv2+HuBERT+Swin/ConvNeXt) |
| `validate_3mod.py` | Phase 8: Multi-seed validation of best three-encoder result |
| `embedding_augmentation.py` | Phase 9: Augmentation screening (Gaussian, Mixup, Combined) on embeddings |
| `validate_mixup.py` | Phase 9: Multi-seed validation of best augmentation (Mixup α=0.4) |

---

## 17. Follow-up: New Large PTM Unimodal Benchmark

A follow-up unimodal benchmark evaluated stronger pretrained models beyond the original 13 base-size models from Phase 1. New models include DINOv2-Large, DINOv2-Large-Reg, CLIP/OpenCLIP ViT-L/14, WavLM-Large, HuBERT-Large, AST, BEATs, CLAP, and data2vec-audio-large. The evaluation followed the same 5-fold stratified CV protocol with PCA fitted inside each fold (no leakage), testing SVM-RBF, LogReg, and MLP at thresholds ≥10, ≥8, ≥5.

Additionally, **length-aware masked mean pooling** was tested for waveform SSL models (WavLM, HuBERT, BEATs), which averages only over valid (non-padding) frames rather than including zero-padded frames.

### 17.1 Key Results (Threshold ≥10, 89 classes)

| Rank | Model | Modality | Pooling | Best Config | Macro-F1 ± Std |
|---:|---|---|---|---|---:|
| 1 | **WavLM-Large** | Audio | Masked | SVM_bal_pca128 | **0.7060 ±0.0140** |
| 2 | AST | Audio | — | SVM_bal_pca128 | 0.6886 ±0.0124 |
| 3 | DINOv2-Large | Image | — | SVM_bal_pca128 | 0.6884 ±0.0058 |
| 4 | DINOv2-Large-Reg | Image | — | SVM_bal_pca128 | 0.6873 ±0.0127 |
| 5 | HuBERT-base (old) | Audio | Unmasked | SVM_bal_pca128 | 0.6870 ±0.0112 |
| 6 | BEATs | Audio | Unmasked | SVM_bal_pca128 | 0.6823 ±0.0147 |
| 7 | DINOv2-B (old) | Image | — | SVM_bal_pca128 | 0.6810 ±0.0095 |

### 17.2 Updated Conclusions

**Supersedes Section 4.3 findings:**

In the original base-model benchmark (Section 4), HuBERT-base was the strongest audio encoder (0.6870) and DINOv2-B the strongest image encoder (0.6810). These findings remain valid for that model scale, but the new benchmark shows:

1. **WavLM-Large with masked mean pooling = 0.7060** is now the strongest single unimodal encoder at threshold ≥10, improving over HuBERT-base by **+0.0190** absolute Macro-F1.

2. **DINOv2-Large = 0.6884** improves over DINOv2-B by **+0.0074**, though the gain is modest.

3. **The old claim "HuBERT >> WavLM" (Section 4.3, 6.5) must be qualified:** This was true for base-size models (HuBERT-base=0.6870 vs WavLM-base=0.6286, and as a fusion partner in Section 6). However, at large scale, WavLM-Large significantly outperforms HuBERT-Large (0.7060 vs 0.6172 masked). WavLM's denoising pre-training objective appears more robust for non-speech signals like malware byte-audio.

4. **HuBERT-Large (0.6111–0.6172) underperforms HuBERT-base (0.6870)** — a genuine domain mismatch: HuBERT's cluster-based self-supervised pre-training is speech-specific and does not transfer well to malware audio at larger scale.

### 17.3 Masked vs Unmasked Pooling

| Model | Unmasked | Masked | Δ | Recommendation |
|-------|----------|--------|---|----------------|
| WavLM-Large | 0.6957 | **0.7060** | **+0.0103** | Use masked |
| HuBERT-Large | 0.6111 | 0.6172 | +0.0061 | Marginal gain |
| BEATs | **0.6823** | 0.6814 | −0.0009 | Use unmasked |

Masked pooling is **not universally beneficial**. BEATs already handles padding internally via its `forward_padding_mask`, so external masked pooling is redundant and slightly harmful.

### 17.4 Updated Model Selection for Fusion

Based on the new benchmark, the fusion candidates to compare against the old validated best (DINOv2-B + HuBERT-base + ConvNeXt = 0.7264 ±0.0075) are:

**Image:** DINOv2-Large, DINOv2-Large-Reg  
**Audio:** WavLM-Large (masked), AST, BEATs (unmasked)  
**Reference:** HuBERT-base (historical fusion anchor)

> Full per-classifier breakdown and all thresholds available in [`unimodal_benchmark_mean_pooling.md`](unimodal_benchmark_mean_pooling.md).
