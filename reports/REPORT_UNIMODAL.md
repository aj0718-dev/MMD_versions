# REPORT_UNIMODAL — Unimodal Benchmarking Summary

---

## 1. Executive Summary

Four benchmark stages were conducted on the MOTIF Windows PE malware dataset to establish unimodal classification performance using frozen pretrained model (PTM) embeddings:

1. **Stage 1 — Direct/global pooled PTM embeddings** from raw audio/image samples, evaluated with sklearn classifiers (PCA → SVM/LR/MLP).
2. **Stage 2 — Saved sequence embeddings pooled with multiple strategies**, evaluated with sklearn classifiers (PCA → SVM/LR/MLP).
3. **Stage 3 — Saved sequence embeddings pooled and evaluated with PyTorch PooledMLP** neural heads.
4. **Stage 4 — Sequence-level CNN/attention heads** operating directly on full `[N, T, D]` sequence embeddings (no pre-pooling).

**Key results at threshold ≥10 (1,947 samples, 89 classes):**

| Metric | Method | Model / Pooling | Macro-F1 | Validation Status |
|--------|--------|-----------------|----------|------------------|
| Best classical unimodal | SVM_bal + PCA=128 | WavLM-Large / masked_mean_max | **0.7444 ±0.0151** | Seed=42 screening |
| Best validated unimodal | SVM_bal + PCA=128 | WavLM-Large / masked_mean_max | **0.7417 ±0.0051** | 5-seed validated (Section 19.6 of REPORT.md) |
| Best differentiable neural branch | PooledMLP | WavLM-Large / masked_max | **0.7207 ±0.0225** | Seed=42 screening |
| Best sequence-level head | CNN_GlobalMax | WavLM-Large / raw sequence | 0.6019 ±0.0235 | Seed=42 screening |

Note: A later 12-config PooledMLP architecture ablation reached 0.7222 on WavLM-Large masked_max, but the +0.0015 improvement over the main-grid 0.7207 result is negligible relative to fold variance; therefore 0.7207 is retained as the primary main-benchmark neural reference.

- SVM remains the strongest overall unimodal classifier.
- The best neural branch (PooledMLP) is within 0.021 of the SVM reference for the same model/pooling pair, making it a viable differentiable branch for future fusion. However, the PooledMLP std (0.0225) is large enough that this gap is not statistically negligible — it indicates PooledMLP is a practical fusion-compatible alternative, not a statistical equivalent to SVM.
- Sequence-level heads (Stage 4) dramatically underperform pooled methods (mean gap −0.14 vs SVM). Within the tested frozen-feature setup, sequence heads are not competitive.
- PooledFOCA_CNN_Adaptive (Conv1D over pooled embedding dimensions) was smoke-tested and showed severe underperformance (best ~0.32). It is not carried forward.
- Fusion experiments are not part of this report; only proposed next steps are listed.

---

## 2. Benchmark Naming and Scope

### 2.1 unimodal_benchmark_mean_pooling.md

**Meaning**: Frozen PTM → direct/global mean or pooled embedding from raw audio/image samples → PCA → LR/SVM/MLP.

This is the broadest benchmark (28 models across image and audio), but uses simple pooling only (CLS token, global average pooling, or frame-level mean). It does not explore masked or concatenated pooling from saved sequence tensors.

### 2.2 sequence_unimodal_benchmark_pooling.md

**Meaning**: Saved `[N, T, D]` sequence embeddings → multiple pooling variants (masked_mean, masked_max, masked_mean_max, mean, max, mean_max) → PCA → sklearn LR/SVM/MLP.

This benchmark demonstrates that **pooling strategy significantly affects classification quality**, and that masked_mean_max (concatenating masked-mean and masked-max into `[N, 2D]`) is the optimal classical pooling for audio models.

### 2.3 sequence_unimodal_benchmark_pooled_neural_heads.md

**Meaning**: Saved `[N, T, D]` sequence embeddings → selected best pooling variants → train-fold StandardScaler → PyTorch PooledMLP neural heads (no PCA).

This benchmark identifies the best differentiable unimodal branch for future trainable fusion, since SVM is not directly differentiable.

### 2.4 sequence_unimodal_benchmark_sequence_heads.md

**Meaning**: Saved `[N, T, D]` sequence embeddings → trainable CNN/attention heads operating directly on full sequences (no pre-pooling, no PCA, no external scaling).

This benchmark tests whether learned temporal/spatial aggregation can outperform simple pooling + SVM. Result: it cannot.

**This naming hierarchy prevents confusion between:**
- Direct pooled embeddings (Stage 1)
- Pooled-from-sequence classical evaluation (Stage 2)
- Pooled-from-sequence neural-head evaluation (Stage 3)
- Sequence-level neural head evaluation (Stage 4)

---

## 3. Dataset and Evaluation Protocol

### Dataset

- **Source**: MOTIF Windows PE malware dataset
- **Total samples**: 3,095 (audio) / 3,094 (image)
- **Total families**: 502
- **Representation**: RGB malware byte-images and 10-second audio waveforms (16 kHz)

### Thresholds

| Threshold | Samples | Classes | Used in |
|-----------|---------|---------|---------|
| ≥10 | 1,947 | 89 | Stages 1, 2, 3, 4 |
| ≥8 | 2,110 | 108 | Stages 1, 2 |
| ≥5 | 2,446 | 166 | Stages 1, 2 |

The neural-head benchmarks (Stages 3–4) report threshold ≥10 only.

### Shared Protocol

- **Metric**: Macro-F1
- **CV**: 5-fold StratifiedKFold, seed=42
- **PCA/Scalers**: Fitted inside each fold only (no leakage)
- All stages maintain strict train/test separation within each fold

---

## 4. Stage 1 — Direct/Global Pooled PTM Embeddings

**Source**: `unimodal_benchmark_mean_pooling.md`

### Method

- Frozen pretrained backbones produce global pooled embeddings from raw audio/image samples:
  - CLS token for ViT/DINOv2/AST
  - Mean pool over frames for HuBERT/WavLM/BEATs
  - Global average pooling for CNNs
  - Projected features for CLIP/OpenCLIP/CLAP
- Classifiers: SVM (PCA 128/256), LR (PCA 512), MLP (PCA 256)
- SVM_bal_pca128 dominated across almost all models

### Top Models at Threshold ≥10

| Rank | Model | Modality | Dim | Pooling | Best Config | Macro-F1 |
|------|-------|----------|-----|---------|-------------|----------|
| 1 | **WavLM-Large** | Audio | 1024 | Masked | SVM_bal_pca128 | **0.7060 ±0.0140** |
| 2 | AST | Audio | 768 | — | SVM_bal_pca128 | 0.6886 ±0.0124 |
| 3 | DINOv2-Large | Image | 1024 | — | SVM_bal_pca128 | 0.6884 ±0.0058 |
| 4 | DINOv2-Large-Reg | Image | 1024 | — | SVM_bal_pca128 | 0.6873 ±0.0127 |
| 5 | HuBERT-base | Audio | 768 | Unmasked | SVM_bal_pca128 | 0.6870 ±0.0112 |
| 6 | BEATs | Audio | 768 | Unmasked | SVM_bal_pca128 | 0.6823 ±0.0147 |
| 7 | ConvNeXt-B | Image | 1024 | — | SVM_bal_pca128 | 0.6578 ±0.0106 |

### Observations

- This stage established WavLM-Large as the strongest audio encoder and DINOv2-Large as the strongest image encoder.
- However, it used only direct/simple pooling and did not explore masked_max or masked_mean_max from saved sequence tensors.
- Stage 2 supersedes this for the top unimodal result, though Stage 1 remains a valid historical baseline and covers a broader set of 28 models.

---

## 5. Stage 2 — Pooled from Saved Sequence Embeddings with sklearn Classifiers

**Source**: `sequence_unimodal_benchmark_pooling.md`

### Method

- Saved `[N, T, D]` sequence embeddings were pooled into fixed-length vectors using multiple strategies.
- Audio pooling variants: masked_mean, masked_max, masked_mean_max, unmasked_mean
- Image pooling variants: mean, max, mean_max
- Classifiers: LR (C=100/1000, bal/unbal), SVM (bal/unbal), sklearn MLP
- PCA dims: LR [128, 256, 384, 512], SVM [128, 256], MLP [128, 256, 512]
- Total configs per threshold: ~575

### Models & Sequence Shapes

| Model | Modality | Shape | Dim | Pooling Variants |
|-------|----------|-------|-----|-----------------|
| WavLM-Large | Audio | [3095, 1499, 1024] | 1024 → 2048 | masked_mean, masked_max, masked_mean_max, unmasked_mean |
| HuBERT-base | Audio | [3095, 1499, 768] | 768 | masked_mean, masked_max, masked_mean_max, unmasked_mean |
| BEATs | Audio | [3095, 1496, 768] | 768 → 1536 | masked_mean, masked_max, masked_mean_max, unmasked_mean |
| AST | Audio | [3095, 1212, 768] | 768 → 1536 | masked_mean, masked_max, masked_mean_max, unmasked_mean |
| DINOv2-Large | Image | [3094, 1369, 1024] | 1024 → 2048 | mean, max, mean_max |
| DINOv2-Large-Reg | Image | [3094, 1369, 1024] | 1024 → 2048 | mean, max, mean_max |
| ConvNeXt-B | Image | [3094, 49, 1024] | 1024 → 2048 | mean, max, mean_max |

### Best per Model at Threshold ≥10

| Model | Pooling | Classifier | PCA | Macro-F1 | vs Stage 1 |
|-------|---------|-----------|-----|----------|-------------|
| **WavLM-Large** | masked_mean_max | SVM_bal | 128 | **0.7444** | +0.0384 |
| AST | masked_mean | SVM_bal | 128 | 0.7006 | +0.0120 |
| BEATs | masked_mean_max | SVM_bal | 128 | 0.6993 | +0.0170 |
| HuBERT-base | masked_mean | SVM_bal | 128 | 0.6916 | +0.0046 |
| DINOv2-Large | mean | SVM_bal | 128 | 0.6817 | −0.0067 |
| DINOv2-Large-Reg | mean_max | SVM_bal | 128 | 0.6806 | −0.0067 |
| ConvNeXt-B | mean | SVM_bal | 128 | 0.6730 | +0.0152 |

### WavLM-Large Breakdown by Pooling (t≥10)

| Pooling | Best Classifier | PCA | Macro-F1 |
|---------|----------------|-----|----------|
| masked_mean_max | SVM_bal | 128 | **0.7444** |
| masked_max | SVM_bal | 128 | 0.7413 |
| masked_mean | SVM_bal | 128 | 0.7260 |
| unmasked_mean | SVM_bal | 128 | 0.7060 |

### Cross-Threshold Summary

| Model | t≥10 | t≥8 | t≥5 |
|-------|------|------|------|
| WavLM-Large | **0.7444** | **0.7224** | **0.6571** |
| AST | 0.7006 | 0.6886 | 0.6023 |
| BEATs | 0.6993 | 0.6969 | 0.6139 |
| HuBERT-base | 0.6916 | 0.6751 | 0.6176 |
| DINOv2-Large | 0.6817 | 0.6735 | 0.5923 |
| DINOv2-Large-Reg | 0.6806 | 0.6727 | 0.5869 |
| ConvNeXt-B | 0.6730 | 0.6531 | 0.5623 |

### Key Observations

1. **WavLM-Large dominates** across all thresholds, holding all top-10 positions.
2. **masked_mean_max is the best pooling strategy** — concatenating masked-mean and masked-max consistently beats either alone (+0.0184 over masked_mean at t≥10).
3. **Masking matters** — for WavLM-Large: masked_mean (0.7260) vs unmasked_mean (0.7060) = +0.0200 gap. Padding tokens degrade representations.
4. **SVM_bal at PCA=128 is the best classifier** overall (wins 5/7 models at t≥10).
5. **Audio models benefit more from pooling exploration** than image models — WavLM gains +0.0384, while DINOv2 models show slight regressions (−0.0067) since their Stage 1 baselines already used near-optimal pooling.
6. **masked_max is extremely close to masked_mean_max** for WavLM at t≥10: 0.7413 vs 0.7444 (Δ=0.0031).

---

## 6. Stage 3 — Pooled Neural Heads on Saved Sequence Embeddings

**Source**: `sequence_unimodal_benchmark_pooled_neural_heads.md`

### Method

- Trainable PyTorch neural heads evaluated on pooled-from-sequence features.
- Each model/pooling pair uses the best pooling strategy from Stage 2.
- Head: PooledMLP (single architecture, grid-searched hyperparameters).
- Threshold ≥10 only.
- Total configs: 512 (8 model/pooling × 2 optimizers × 4 LR × 2 dropout × 4 WD).
- Runtime: 7,180 s (2.0 h) on Apple MPS.

### Architecture

```
PooledMLP:
  Input [B, D]
  → LayerNorm(D)
  → Linear(D → 512)
  → ReLU
  → Dropout(p)
  → Linear(512 → 128)
  → ReLU
  → Dropout(p)
  → Linear(128 → C)
  → raw logits
```

### Training Setup

- Loss: `CrossEntropyLoss` with inverse-frequency class weights (computed from training split only)
- batch_size: 32
- max_epochs: 50
- Patience: 7 (early stopping on validation macro-F1)
- Validation: 15% StratifiedShuffleSplit inside each train fold
- External scaling: StandardScaler fitted on training split only (no leakage)
- CV: 5-fold StratifiedKFold (seed=42)

### Best per Model/Pooling (t≥10)

| Model | Pooling | Macro-F1 | Std | Ref SVM | Δ vs SVM |
|-------|---------|----------|------|---------|----------|
| **WavLM-Large** | masked_max | **0.7207** | 0.0225 | 0.7413 | −0.0206 |
| WavLM-Large | masked_mean_max | 0.6994 | 0.0149 | 0.7444 | −0.0450 |
| BEATs | masked_mean_max | 0.6593 | 0.0198 | 0.6993 | −0.0399 |
| AST | masked_mean | 0.6458 | 0.0166 | 0.7006 | −0.0548 |
| DINOv2-Large-Reg | mean_max | 0.6268 | 0.0179 | 0.6806 | −0.0538 |
| HuBERT-base | masked_mean | 0.5957 | 0.0132 | 0.6916 | −0.0959 |
| DINOv2-Large | mean | 0.5954 | 0.0146 | 0.6817 | −0.0863 |
| ConvNeXt-B | mean | 0.5826 | 0.0208 | 0.6730 | −0.0904 |

### Unified Comparison: Neural Head vs sklearn Classifiers

| Model | Pooling | Neural Head | SVM | LR | sklearn MLP | NH vs SVM | NH vs MLP |
|-------|---------|-------------|------|------|------|-----------|-----------|
| **WavLM-Large** | masked_max | **0.7207** | 0.7413 | 0.7372 | 0.7143 | −0.0206 | **+0.0064** |
| WavLM-Large | masked_mean_max | 0.6994 | 0.7444 | 0.7313 | 0.7069 | −0.0450 | −0.0075 |
| BEATs | masked_mean_max | 0.6593 | 0.6993 | 0.6950 | 0.6609 | −0.0399 | −0.0015 |
| AST | masked_mean | 0.6458 | 0.7006 | 0.6889 | 0.6518 | −0.0548 | −0.0060 |
| DINOv2-Large-Reg | mean_max | 0.6268 | 0.6806 | 0.6629 | 0.6245 | −0.0538 | **+0.0023** |
| HuBERT-base | masked_mean | 0.5957 | 0.6916 | 0.6716 | 0.6443 | −0.0959 | −0.0486 |
| DINOv2-Large | mean | 0.5954 | 0.6817 | 0.6501 | 0.6427 | −0.0863 | −0.0472 |
| ConvNeXt-B | mean | 0.5826 | 0.6730 | 0.6339 | 0.6066 | −0.0904 | −0.0240 |

### Key Observations

1. **PooledMLP does not beat SVM** for any model/pooling pair (0/512 configs surpass their SVM reference).
2. **Best neural result is WavLM-Large masked_max** (0.7207), not masked_mean_max. The neural ranking differs from the SVM ranking: SVM favors masked_mean_max (2048-dim); PooledMLP favors masked_max (1024-dim). *Interpretation*: the lower-dimensional masked_max representation may be easier for a shallow 2-layer MLP to classify than the concatenated 2048-dim vector.
3. **Best PooledMLP is within 0.021 of its SVM reference** (19/512 configs are within 0.03), making it a viable differentiable branch.
4. **PooledMLP beats sklearn MLP for 2/8 pairs only**: WavLM-Large/masked_max (+0.0064) and DINOv2-Large-Reg/mean_max (+0.0023). For all other pairs, even sklearn MLP is superior.

> **Comparability note:** sklearn MLP uses PCA-reduced inputs (128–512 dims); PooledMLP operates on full-dimensional StandardScaler-normalized vectors (1024–2048 dims). This is a practical comparison ("best classical vs best neural"), not a pure architecture-vs-architecture comparison controlling for input dimensionality.

5. **Optimal hyperparameters**: adamw, lr=3e-4, dropout=0.3, wd=0.

### PooledMLP Architecture Ablation

A 12-config follow-up ablation tested three architecture variants (`current_relu_512_128`, `gelu_512_128`, `gelu_1024_256`) on the best audio neural branch (WavLM-Large / masked_max) and best image neural branch (DINOv2-Large-Reg / mean_max), with fixed optimal HPs (lr=3e-4, dropout=0.3, wd=0.0, standard scaling). Script: `eval_pooled_mlp_arch_ablation.py`.

**Results**:
- **WavLM-Large masked_max**: improved only marginally from 0.7207 to 0.7222 (gelu_512_128 + adam, Δ=+0.0015), negligible relative to fold variance (std=0.0200).
- **DINOv2-Large-Reg mean_max**: improved from 0.6268 to 0.6395 (gelu_1024_256, Δ=+0.0127), a meaningful gain driven by increased hidden width (1024→256).
- No architecture beat SVM for either pair.

**Decision**: Use gelu_1024_256 as the image fusion projection branch. The audio branch does not benefit materially from architecture changes. Further unimodal architecture search is not warranted.

*Full tables in `sequence_unimodal_benchmark_pooled_neural_heads.md` → "Architecture Ablation: PooledMLP Variants".*

---

## 7. CNN / FOCA-Style Pooled Branch Diagnostic

**Status**: Diagnostic smoke test only. Not a full benchmark.

PooledFOCA_CNN_Adaptive reshapes the pooled vector `[B, D]` into `[B, 1, D]` and applies Conv1D layers over the embedding dimension. This was smoke-tested on WavLM-Large pooled features:

| Model | Pooling | Head | Input Norm | Best Macro-F1 | SVM Ref | Δ |
|-------|---------|------|-----------|---------------|---------|------|
| WavLM-Large | masked_mean_max | PooledFOCA_CNN_Adaptive | standard | 0.3236 ±0.0203 | 0.7444 | −0.4208 |
| WavLM-Large | masked_max | PooledFOCA_CNN_Adaptive | standard | 0.3225 ±0.0434 | 0.7413 | −0.4188 |
| WavLM-Large | masked_max | PooledFOCA_CNN_Adaptive | layernorm | 0.1614 ±0.0856 | 0.7413 | −0.5799 |

*Source: `results/archive_pooled_neural_heads_cnn_failed_20260530_042224/`*

**Conclusion**: CNN over pooled embedding dimensions was not competitive in this setup. The pooled vector does not have spatial/temporal structure that Conv1D can exploit. This approach is not carried forward.

This is **not** the same as sequence-level CNN/attention over `[B, T, D]`, which is evaluated in Stage 4 below.

---

## 8. Stage 4 — Sequence-Level Neural Heads on Raw `[N, T, D]` Embeddings

**Source**: `sequence_unimodal_benchmark_sequence_heads.md`

### Method

- Trainable CNN and attention heads operate **directly** on full `[B, T, D]` sequence tensors — no pre-pooling, no PCA, no external scaling.
- 7 head architectures: CNN_GlobalMax, CNN_AvgMax, CNN_Dilated, CNN_FOCA_Flatten, CNN_AttnPool, AttentionPool, MeanMaxMLP.
- 7 models (same as Stages 2–3): WavLM-Large, AST, BEATs, HuBERT-base, DINOv2-Large, DINOv2-Large-Reg, ConvNeXt-B.
- Total configs: 294 (7 models × 7 heads × 3 LR × 2 dropout).
- Runtime: 88,657 s (24.6 h) on Apple MPS.

### Training Setup

- Loss: `CrossEntropyLoss` with inverse-frequency class weights (computed from training split only)
- Optimizer: AdamW (weight_decay=0.01, default — not grid-searched)
- LR: [0.001, 0.0003, 0.0001]
- Dropout: [0.3, 0.5]
- batch_size: 16, max_epochs: 50, patience: 7
- CV: 5-fold StratifiedKFold (seed=42)

### Best per Model

> **Note:** "Ref SVM" values are the Stage 2 best SVM results for the same model/pooling pair (seed=42), used as the comparison target. These are the same values reported in the Stage 2 tables above, not values printed by eval_sequence_heads.py itself.

| Model | Best Head | LR | Drop | Macro-F1 | Std | Ref SVM | Δ vs SVM |
|-------|-----------|-------|------|----------|------|---------|----------|
| **WavLM-Large** | CNN_GlobalMax | 3e-4 | 0.3 | **0.6019** | 0.0235 | 0.7444 | −0.1425 |
| BEATs | MeanMaxMLP | 3e-4 | 0.3 | 0.5783 | 0.0099 | 0.6993 | −0.1210 |
| AST | AttentionPool | 3e-4 | 0.3 | 0.5751 | 0.0040 | 0.7006 | −0.1255 |
| HuBERT-base | CNN_GlobalMax | 3e-4 | 0.3 | 0.5694 | 0.0165 | 0.6916 | −0.1222 |
| DINOv2-Large | CNN_GlobalMax | 3e-4 | 0.3 | 0.5225 | 0.0250 | 0.6817 | −0.1592 |
| DINOv2-Large-Reg | CNN_GlobalMax | 3e-4 | 0.3 | 0.5220 | 0.0122 | 0.6806 | −0.1586 |
| ConvNeXt-B | MeanMaxMLP | 3e-4 | 0.3 | 0.4990 | 0.0164 | 0.6730 | −0.1740 |

Mean gap vs SVM: **−0.1433** across all 7 models.

### Unified Comparison: Sequence Heads vs Pooled Methods

| Model | Seq Head (best) | PooledMLP | SVM | SeqHead vs SVM | SeqHead vs PooledMLP |
|-------|-----------------|-----------|------|----------------|---------------------|
| **WavLM-Large** | 0.6019 | 0.7207 | 0.7444 | −0.1425 | −0.1188 |
| BEATs | 0.5783 | 0.6593 | 0.6993 | −0.1210 | −0.0810 |
| AST | 0.5751 | 0.6458 | 0.7006 | −0.1255 | −0.0707 |
| HuBERT-base | 0.5694 | 0.5957 | 0.6916 | −0.1222 | −0.0263 |
| DINOv2-Large | 0.5225 | 0.5954 | 0.6817 | −0.1592 | −0.0729 |
| DINOv2-Large-Reg | 0.5220 | 0.6268 | 0.6806 | −0.1586 | −0.1048 |
| ConvNeXt-B | 0.4990 | 0.5826 | 0.6730 | −0.1740 | −0.0836 |

### Training Stability

- 23/294 configs (7.8%) failed catastrophically (F1 < 0.10)
- Most failures: DINOv2-Large (8), BEATs (8), AST (4)
- Most failure-prone head: MeanMaxMLP (9 catastrophic configs)
- WavLM-Large and HuBERT-base had **zero** catastrophic failures

### Key Observations

1. **Sequence heads dramatically underperform pooled methods.** Mean gap to SVM is −0.14; gap to PooledMLP is −0.03 to −0.12.
2. **CNN_GlobalMax is the best head** — 2-layer Conv1d (64→128, k=3) + global max pool wins for 4/7 models.
3. **lr=3e-4, dropout=0.3 are universally optimal** (present in all 7 models' best configs).
4. **Within the tested design family, this closes the frozen sequence-head direction.** No architecture/HP combination approaches pooled baselines. Further gains likely require backbone fine-tuning or substantially different head design.

---

## 9. Cross-Stage Comparison

| Stage | Input Representation | Best Method | Best Model / Pooling | Best Macro-F1 | Main Takeaway |
|-------|---------------------|-------------|---------------------|---------------|---------------|
| 1 | Direct/global pooled PTM embeddings | SVM_bal + PCA=128 | WavLM-Large / masked_mean | 0.7060 | Broad PTM survey; established WavLM as best |
| 2 | Pooled-from-sequence (classical) | SVM_bal + PCA=128 | WavLM-Large / masked_mean_max | **0.7444** | Pooling exploration unlocks +0.0384 gain |
| 3 | Pooled-from-sequence (neural) | PooledMLP | WavLM-Large / masked_max | 0.7207 | Best differentiable branch; within 0.021 of SVM |
| 4 | Raw sequence (neural) | CNN_GlobalMax | WavLM-Large / raw [1499, 1024] | 0.6019 | Tested sequence heads not competitive; −0.14 vs SVM |

- **Stage 2** provides the strongest classical unimodal benchmark (0.7444).
- **Stage 3** provides the strongest differentiable neural branch (0.7207).
- **Stage 4** shows that frozen sequence-level heads are not competitive within the tested architectures (−0.14 gap to SVM).
- **Stage 1** remains a useful historical baseline covering 28 models, but is no longer the top unimodal result.

---

## 10. Final Unimodal Conclusions

1. **Best seed=42 classical unimodal screening result**: WavLM-Large masked_mean_max + SVM_bal_pca128 = **0.7444 ±0.0151** at t≥10. The 5-seed validated equivalent is **0.7417 ±0.0051** (reported in REPORT.md Section 19.6).

2. **Best main-grid differentiable neural branch**: WavLM-Large masked_max + PooledMLP = **0.7207 ±0.0225** at t≥10. A small architecture ablation reached 0.7222 ±0.0200 with gelu_512_128, but the +0.0015 gain is negligible relative to fold variance.

3. **Best image classical candidates**: DINOv2-Large mean (0.6817) and DINOv2-Large-Reg mean_max (0.6806), both with SVM_bal_pca128.

4. **Best image neural branch**: DINOv2-Large-Reg mean_max + PooledMLP (gelu_1024_256) = **0.6395 ±0.0136** (architecture ablation improved from 0.6268).

5. **SVM remains stronger than neural heads** across all model/pooling pairs. The gap ranges from −0.021 (WavLM/masked_max) to −0.096 (HuBERT/masked_mean).

6. **Neural heads are still valuable for future trainable fusion** because SVM is not directly differentiable. The PooledMLP branch can be end-to-end fine-tuned in a multimodal fusion network.

7. **Sequence-level heads (Stage 4) are not viable in the tested frozen-feature setup.** Operating CNN/attention heads directly on frozen `[N, T, D]` sequences yields mean F1 gap of −0.14 vs SVM across all 7 models (294 configs total). The best result (WavLM CNN_GlobalMax = 0.6019) even trails PooledMLP by −0.12. Further sequence-head work is not justified unless the design changes substantially (e.g., fine-tuned backbone or pre-trained temporal encoder).

8. **Pooled CNN (PooledFOCA_CNN_Adaptive) should not be used** as a main pooled-vector fusion branch. It scored ~0.32 on the same data where PooledMLP scores 0.72.

9. **Within the tested frozen-feature family, the practical ceiling is approximately 0.74** (WavLM SVM). No pooling strategy, classifier, neural head architecture, or sequence-level model tested can break through this ceiling without unfreezing backbone parameters. Note: this does not preclude that novel architectures or much larger datasets could exceed this ceiling with frozen features.

---

## 11. Recommended Next Steps

The following are **proposed** experiments. None have been completed as part of this unimodal report. (Some have since been tested in REPORT.md Section 19.)

### Fusion Candidates (Audio + Image)

| Priority | Audio Branch | Image Branch | Status |
|----------|-------------|--------------|--------|
| 1 | WavLM-Large / masked_max (1024-dim) | DINOv2-Large-Reg / mean_max (2048-dim) | Tested — see REPORT.md §19 |
| 2 | WavLM-Large / masked_max (1024-dim) | DINOv2-Large / mean (1024-dim) | Tested — see REPORT.md §19 |
| 3 | WavLM-Large / masked_max (1024-dim) | ConvNeXt-B / mean (1024-dim) | Tested — see REPORT.md §19 |
| Ablation | WavLM-Large / masked_mean_max (2048-dim) | DINOv2-Large-Reg / mean_max (2048-dim) | Tested — see REPORT.md §19 |

### Fusion Methods (Proposed Order)

1. **Concat FCN_Balanced** — concatenate pooled features, train balanced MLP
2. **Gated fusion** — learned gates on each modality branch
3. **Euclidean cross-attention** over MLP-projected pseudo-tokens
4. **Hyperbolic cross-attention** — only after concat/gated baselines establish a reference

> **Update (REPORT.md §19):** Fusion candidates 1–4 and concat SVM/FCN methods were tested. Multi-seed validation showed fusion does not reliably improve over audio-only WavLM-Large. See REPORT.md Section 19 for full results.

The **highest-priority next step** is fine-tuning WavLM-Large top transformer layers. Secondary priorities:
- Do not run more sequence-head or pooled-CNN grids unless design changes substantially
- If fusion is revisited, use PooledMLP/linear projection branches (not pooled Conv1D)
- Perform per-class complementarity analysis before further fusion attempts

---

## 12. Source Files

### Primary Benchmark Reports

- `Unimodal Benchmarking/unimodal_benchmark_mean_pooling.md`
- `Unimodal Benchmarking/sequence_unimodal_benchmark_pooling.md`
- `Unimodal Benchmarking/sequence_unimodal_benchmark_pooled_neural_heads.md`
- `Unimodal Benchmarking/sequence_unimodal_benchmark_sequence_heads.md`

### Evaluation Scripts

- `eval_new_ptms.py` — Stage 1 direct/global pooled evaluation
- `eval_pooled_from_sequence_embeddings.py` — Stage 2 pooled-from-sequence classical
- `eval_pooled_neural_heads.py` — Stage 3 pooled-from-sequence neural heads
- `eval_pooled_mlp_arch_ablation.py` — PooledMLP architecture ablation (Stage 3 follow-up)
- `eval_sequence_heads.py` — Stage 4 sequence-level CNN/attention heads

### Result Artifacts

- `results/pooled_neural_heads_t10.csv`
- `results/pooled_neural_heads_t10.txt`
- `results/pooled_mlp_arch_ablation_t10.csv`
- `results/pooled_mlp_arch_ablation_t10.txt`
- `results/sequence_unimodal_heads.csv`
- `results/pooled_from_seq_emb_t10_checkpoint.json`
- `results/new_ptm_pooled_benchmark_t10_t8_t5.txt`
- `results/archive_pooled_neural_heads_cnn_failed_20260530_042224/` (CNN diagnostic archive)
