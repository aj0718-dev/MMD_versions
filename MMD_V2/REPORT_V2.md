# Multimodal Malware Detection — Experiment Report

## 1. Overview

This report documents the complete experimental pipeline for multimodal malware family classification using the MOTIF dataset. The pipeline evaluates pre-trained audio and image embeddings for malware detection, comparing single-modality baselines, comprehensive pairwise fusion (7 pairs × 4 methods), and cross-attention architectures in both Euclidean and hyperbolic space.

**Dataset:** MOTIF malware dataset  
**Total samples:** 3,094–3,095 (depending on modality alignment)  
**Total families:** 502  
**Evaluation:** 5-Fold Stratified Cross-Validation with PCA fitted inside each fold (no data leakage)  
**Primary metric:** Macro-F1  

---

## 2. Embedding Models

| Model | Modality | Embedding Dim | Source Representation |
|-------|----------|---------------|----------------------|
| HuBERT | Audio | 768 | Self-supervised speech model applied to malware audio |
| WavLM | Audio | 768 | Self-supervised speech model applied to malware audio |
| Wav2Vec2 | Audio | 768 | Self-supervised speech model applied to malware audio |
| VGG19 | Image | 4096 | CNN features from malware visualizations |
| ViT | Image | 768 | Vision Transformer features from malware visualizations |

---

## 3. Thresholds

To handle the long-tail class distribution, experiments are run at multiple minimum-sample thresholds:

| Threshold | Classes | Samples | Avg Samples/Class |
|-----------|---------|---------|-------------------|
| ≥ 10 | 89 | 1,946 | ~22 |
| ≥ 8 | 108 | 2,109 | ~20 |
| ≥ 5 | 166 | 2,445 | ~15 |

---

## 4. Phase 1: Single-Modality Baselines (Logistic Regression)

### 4.1 Initial Grid Search

**Classifier:** LogisticRegression  
**Grid:**
- PCA components: [64, 128, 256]
- C: [0.01, 0.1, 1.0, 10.0, 100.0]
- Solver: [lbfgs, saga]
- Class weight: [None, balanced]

| Model | Thresh ≥10 | Thresh ≥8 | Thresh ≥5 |
|-------|-----------|----------|----------|
| **HuBERT** | **0.6376** | **0.6198** | **0.5533** |
| ViT | 0.5937 | 0.5872 | 0.4993 |
| VGG19 | 0.5863 | 0.5896 | 0.5110 |
| WavLM | 0.5937 | 0.5728 | 0.5086 |
| Wav2Vec2 | 0.5370 | 0.5288 | 0.4643 |

**Finding:** All models hit grid boundary at PCA=256, C=100.

### 4.2 Extended Grid Search (Top 4 Models)

**Grid:**
- PCA components: [256, 384, 512]
- C: [100, 500, 1000, 5000]
- Solver: [lbfgs]
- Class weight: [None, balanced]

| Model | Thresh ≥10 | Thresh ≥8 | Thresh ≥5 | Best PCA | Best C |
|-------|-----------|----------|----------|----------|--------|
| **HuBERT** | **0.6607 ± 0.019** | **0.6438 ± 0.010** | **0.5718 ± 0.029** | 512 | 1000 |
| ViT | 0.6159 ± 0.008 | 0.6140 ± 0.017 | 0.5241 ± 0.021 | 512 | 1000 |
| VGG19 | 0.5993 ± 0.009 | 0.5993 ± 0.010 | 0.5256 ± 0.027 | 384 | 1000 |
| WavLM | 0.6043 ± 0.012 | 0.5843 ± 0.010 | 0.5168 ± 0.017 | 384 | 1000 |

**Finding:** HuBERT dominates all thresholds. Best single-modality LR: **HuBERT = 0.6607** (thresh ≥10).

---

## 5. Phase 2: Non-Linear Classifiers (Single-Modality)

**Classifiers tested:**
- LogReg without PCA (just StandardScaler)
- LogReg with PCA + StandardScaler
- SVM-RBF (PCA + StandardScaler + RBF kerne)
- MLP (PCA + StandardScaler + neural network)

### Best Results per Classifier (Threshold ≥10)

| Model | LogReg (best) | SVM-RBF | MLP | 
|-------|--------------|---------|-----|
| **HuBERT** | 0.6607 | **0.6870** | 0.6330 |
| ViT | 0.6159 | 0.6160 | 0.5538 |
| VGG19 | 0.5993 | 0.6301 | 0.5804 |
| WavLM | 0.6043 | 0.6295 | 0.5799 |

### Best Single-Modality Results (All Thresholds)

| Threshold | Model | Classifier | Macro-F1 | Key Params |
|-----------|-------|-----------|----------|------------|
| **≥ 10** | **HuBERT** | **SVM-RBF** | **0.6870 ± 0.011** | PCA=128, C=100, γ=scale, balanced |
| **≥ 8** | **HuBERT** | **SVM-RBF** | **0.6831 ± 0.015** | PCA=128, C=100, γ=scale, balanced |
| **≥ 5** | **HuBERT** | **SVM-RBF** | **0.6011 ± 0.029** | PCA=128, C=1000, γ=scale, balanced |

**Finding:** SVM-RBF with PCA=128 is the best single-modality classifier. HuBERT + SVM-RBF sets the overall single-modality ceiling.

---

## 6. Phase 3: Comprehensive Multimodal Fusion

### 6.1 Experimental Design

All 7 modality pairs tested with 4 fusion methods, evaluated at K=5 and K=10:

**Modality pairs:**
- Image+Image: ViT+VGG19
- Image+Audio: ViT+HuBERT, ViT+WavLM, ViT+Wav2Vec2, VGG19+HuBERT, VGG19+WavLM, VGG19+Wav2Vec2

**Fusion methods:**
1. **Concat+LR:** PCA per modality → L2 normalize → concatenate → LogReg (C=1000, balanced, lbfgs)
2. **Concat+SVM:** PCA per modality → L2 normalize → concatenate → SVM-RBF (C=100, γ=scale, balanced)
3. **FCN:** PCA per modality → StandardScaler → concatenate → FC(512→1024→ReLU→Dropout→n_classes)
4. **FCN_Balanced:** Same as FCN but with class-weighted CE loss

**Protocol:** PCA fitted inside each fold (no data leakage), 5-fold and 10-fold stratified CV.

### 6.2 Results: All Pairs × All Methods (K=5, Threshold ≥10)

| Pair | Concat+LR | Concat+SVM | FCN | FCN_Balanced |
|------|-----------|-----------|-----|-------------|
| **VGG19+HuBERT** | 0.6535 | 0.6826 | 0.6988 | **0.7067** |
| **ViT+HuBERT** | 0.6633 | 0.6899 | 0.6932 | **0.7066** |
| VGG19+WavLM | 0.6330 | 0.6596 | 0.6667 | 0.6713 |
| ViT+WavLM | 0.6305 | 0.6669 | 0.6658 | 0.6683 |
| ViT+VGG19 | 0.6177 | 0.6614 | 0.6598 | 0.6652 |
| VGG19+Wav2Vec2 | 0.6070 | 0.6344 | 0.6311 | 0.6354 |
| ViT+Wav2Vec2 | 0.6260 | 0.6231 | 0.6351 | 0.6330 |

### 6.3 Results: K=10 Validation (Top Pairs)

| Pair | Method | K=5 | K=10 | Δ |
|------|--------|-----|------|---|
| VGG19+HuBERT | FCN_Balanced | 0.7067 ± 0.012 | 0.6997 ± 0.037 | −0.007 |
| ViT+HuBERT | FCN_Balanced | 0.7066 ± 0.009 | 0.6945 ± 0.037 | −0.012 |
| VGG19+HuBERT | FCN | 0.6988 ± 0.010 | 0.6957 ± 0.041 | −0.003 |

K=10 yields marginally lower mean (less training data per fold) with higher variance (smaller val sets), but confirms K=5 results.

### 6.4 Key Findings from Comprehensive Comparison

1. **FCN_Balanced is the best fusion method** — class-weighted loss consistently adds +0.5–1%
2. **HuBERT is the critical ingredient** — every top pair includes HuBERT
3. **Both VGG19 and ViT pair well with HuBERT** — tied at 0.7067 vs 0.7066
4. **Every fusion pair improves over single-modality LR baselines** (↑2–7%)
5. **Method ranking is stable:** FCN_Balanced > FCN > Concat+SVM > Concat+LR

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

## 8. Final Comparison Table (Threshold ≥10)

All methods ranked by Macro-F1 (89 classes, 1,946 samples):

| Rank | Method | Macro-F1 | Gain vs HuBERT LR |
|------|--------|----------|-------------------|
| 1 | **VGG19+HuBERT FCN_Balanced** | **0.7067 ± 0.012** | **↑ 0.0460** |
| 2 | ViT+HuBERT FCN_Balanced | 0.7066 ± 0.009 | ↑ 0.0459 |
| 3 | VGG19+HuBERT CrossAttn Hyperbolic v3 | 0.6990 ± 0.013 | ↑ 0.0383 |
| 4 | VGG19+HuBERT FCN | 0.6988 ± 0.010 | ↑ 0.0381 |
| 5 | ViT+HuBERT FCN | 0.6932 ± 0.007 | ↑ 0.0325 |
| 6 | VGG19+HuBERT CrossAttn Hyperbolic v1 | 0.6924 ± 0.009 | ↑ 0.0317 |
| 7 | ViT+HuBERT Concat+SVM | 0.6899 ± 0.006 | ↑ 0.0292 |
| 8 | **HuBERT (single-modality SVM-RBF)** | **0.6870 ± 0.011** | — |
| 9 | ViT+HuBERT CrossAttn Euclidean (Bal_h4) | 0.6835 ± 0.007 | ↑ 0.0228 |
| 10 | VGG19+HuBERT Concat+SVM | 0.6826 ± 0.010 | ↑ 0.0219 |
| 11 | VGG19+HuBERT CrossAttn Euclidean (Bal_h8) | 0.6799 ± 0.009 | ↑ 0.0192 |
| 12 | VGG19+WavLM FCN_Balanced | 0.6713 ± 0.008 | ↑ 0.0106 |
| 13 | HuBERT (single-modality LogReg) | 0.6607 ± 0.019 | — |
| 14 | ViT+VGG19 FCN_Balanced | 0.6652 ± 0.013 | ↑ 0.0045 |
| 15 | VGG19 (single-modality SVM-RBF) | 0.6301 ± 0.013 | — |
| 16 | ViT (single-modality SVM-RBF) | 0.6160 ± 0.015 | — |

---

## 9. Key Insights

### Fusion Method Hierarchy

| Rank | Method | Avg Macro-F1 (top pair) | Notes |
|------|--------|------------------------|-------|
| 1 | **FCN_Balanced** | 0.7067 | Class-weighted loss consistently helps |
| 2 | FCN | 0.6988 | Strong even without class weights |
| 3 | Concat+SVM | 0.6826 | No neural training required |
| 4 | CrossAttn_Balanced (Euclidean) | 0.6835 | Marginal over simple FCN for +HuBERT |
| 5 | CrossAttn_Hyperbolic v3 | 0.6990 | Matches FCN but higher complexity |
| 6 | Concat+LR | 0.6535 | Simplest, still improves over single-modality |

### Why Cross-Attention Underperforms FCN

Cross-attention (Euclidean) achieves 0.68 but falls 2–3% below FCN because:
- Embeddings are **single-token** (seq_len=1) — no sequence structure for attention to exploit
- Attention degenerates to a learned weighted average with only 1 query/key
- The additional parameters (proj + MHA + FFN) are harder to optimize for small data
- Simple concatenation + nonlinear classifier (FCN) is a more appropriate inductive bias

### Modality Pair Rankings

1. **{VGG19, ViT} + HuBERT** — top tier (0.70+)
2. **{VGG19, ViT} + WavLM** — mid tier (0.66–0.67)
3. **ViT + VGG19** — similar to WavLM pairs (0.66)
4. **{VGG19, ViT} + Wav2Vec2** — lowest (0.63–0.64)

### VGG19 vs ViT as Image Modality

Both yield nearly identical fusion performance with HuBERT (0.7067 vs 0.7066). The VGG19 4096d features and ViT 768d features provide equivalent complementary information when fused with audio.

### Practical Implications

- **Best overall method:** VGG19+HuBERT FCN_Balanced = **0.7067** (+4.6% over LR, +2.0% over SVM-RBF)
- **Simplest competitive method:** ViT+HuBERT Concat+SVM = **0.6899** (no training required beyond SVM)
- Cross-attention adds model complexity without F1 gain on this dataset
- Hyperbolic geometry provides marginal benefit over Euclidean cross-attention but does not outperform FCN

---

## 10. Experimental Setup

- **Python:** 3.13.6
- **Hardware:** CPU (macOS)
- **Key libraries:** PyTorch, scikit-learn
- **Evaluation protocol:** 5-fold Stratified K-Fold, PCA fit on train split only
- **No data leakage:** All preprocessing (PCA, StandardScaler) fitted exclusively on training folds
- **Reproducibility:** random_state=42 for all sklearn operations
- **Alignment:** Path-based sample ID matching between image (3094) and audio (3095) modalities

---

## 11. Files Reference

| File | Purpose |
|------|---------|
| `gridsearch_all_models.py` | Phase 1: Initial LR grid search (all models) |
| `gridsearch_extended.py` | Phase 1: Extended LR grid search (top 4 models) |
| `gridsearch_nonlinear.py` | Phase 2: SVM-RBF, MLP, LogReg comparison |
| `fusion_all_pairs.py` | Phase 3: Comprehensive 7-pair × 4-method fusion comparison |
| `cross_attn_euclidean.py` | Phase 4: Cross-attention fusion (Euclidean, top 3 pairs) |
| `cross_attention_hyperbolic_final.py` | Phase 4: Cross-attention fusion v1 (Hyperbolic) |
| `cross_attention_hyperbolic_v3.py` | Phase 4: Improved cross-attention v3 (Hyperbolic) |
