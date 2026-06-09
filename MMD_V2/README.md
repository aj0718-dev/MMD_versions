# Multimodal Malware Detection

Multimodal malware family classification using RGB image representations, audio representations, and interpretable static PE features on the MOTIF dataset.

This project explores how different modalities capture complementary structural, sequential, and semantic characteristics of malware binaries.

## Overview

This repository contains experiments for:
- Malware → RGB image conversion
- Malware → audio waveform conversion
- Image-based malware classification
- Audio-based malware classification

The work is based on the MOTIF malware dataset.

## Environment Setup

### Create Environment
```
conda create -n motif python=3.10
conda activate motif
```

### Install Dependencies
```
pip install torch torchvision torchaudio transformers timm scikit-learn matplotlib seaborn soundfile pefile pillow numpy pandas
```

## Dataset

### MOTIF dataset

Dataset used:
- MOTIF (Malware Open-source Threat Intelligence Family)

Characteristics:
- 3,095 disarmed malware samples
- 502 malware families (downloaders, ransomware, banking, backdoor, etc.)
- PE executables
- Family-level labels with _ground truth confidence_
- Highly imbalanced distribution
- Family labels were obtained by surveying thousands of open-source threat reports published by 14 major cybersecurity organizations between **Jan. 1st, 2016** and **Jan. 1st, 2021**.

The dataset should be organized family-wise.

Example structure:
```
family_samples/
├── family_1/
│   ├── sample_1
│   ├── sample_2
│   └── ...
├── family_2/
└── ...
```

Dataset Link: https://github.com/boozallen/MOTIF 

## RGB Image Representation Pipeline
### Purpose

Convert malware PE binaries into structured RGB images where:
- Red channel → PE headers
- Green channel → section data
- Blue channel → PE data directories/import/export/resource regions

This preserves structural information from binaries.

### Script
``` convert_motif_to_rgb.py ```

### Conversion Details
- Image size: 256×256
- Fixed byte length: 256×256=65536
- PE-aware channel mapping using ``` pefile ```
- Fallback grayscale conversion if PE parsing fails
  
### Running RGB Conversion

Update paths inside:
```
IN_DIR
OUT_DIR
```
Then run:
```
python convert_motif_to_rgb.py
```
Output structure:
```
family_rgb_images/
├── family_1/
│   ├── sample.png
│   └── ...
```

## Audio Representation Pipeline
### Purpose
Convert malware binaries into raw waveform audio representations. Each byte is interpreted as an 8-bit unsigned PCM audio. This preserves sequential byte relationships.

### Script
``` convert_to_audio.py ```

### Conversion Details
- Mono audio
- 8-bit PCM
- Sample rate: 8000 Hz
- Reversible mapping from bytes → waveform
- Running Audio Conversion

### Running Audio Conversion
Update:
```
INPUT_ROOT
OUTPUT_ROOT
```
Then:
```
python convert_to_audio.py
```
Output structure:
```
family_audio/
├── family_1/
│   ├── sample.wav
│   └── ...
```

## Image Classification (VGG19)
### Script
``` vgg19/linear_probe_vgg19.py ```

### Uses:
- Pretrained VGG19 backbone from ``` timm ```
- Frozen feature extractor
- Linear classifier trained on top

Feature dimension: 4096-d embeddings

### Training Details

Default config inside script:
```
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
NUM_CLASSES = 502
```
Train/test split:
- Random 80/20 split
- No stratification due to dataset imbalance

Evaluation:
- Macro-F1
- Weighted-F1
  
### Run VGG19 Baseline

Update:
```
DATA_DIR = "/path/to/family_rgb_images"
```
Then:
```
cd vgg19
python linear_probe_vgg19.py
```
Outputs:
- predictions
- embeddings
- labels
- paths

Saved files:
```
vgg19_embeddings_all.pt
labels_all.pt
vgg19_paths.pt
vgg_preds.pt
```

## Audio Classification (Wav2Vec2, HuBERT, WavLM)

Pretrained speech encoders are used as:
- frozen feature extractors
- linear probe backbones
  
### Audio Embedding Extraction
#### Script
``` wav2vec2_hubert_wavlm/extract_embeddings.py ```

Supported encoders: Wav2Vec2, HuBERT, WavLM

(Current script defaults to WavLM.)

Embedding extraction:
- Mean pooling over hidden states
- Output dimension: 768
  
#### Run Embedding Extraction
Update:
```
DATA_DIR
```
Then:
```
cd wav2vec2_hubert_wavlm
python extract_embeddings.py
```
Outputs:
```
wavlm_embeddings.pt
labels.pt
wavlm_paths.pt
```

### Audio Linear Probe Training
#### Script
``` train_classifier.py ```

This:
- freezes pretrained encoder
- trains only linear classifier layer

Supported models:
```
MODEL_NAME = "wav2vec2"
MODEL_NAME = "hubert"
MODEL_NAME = "wavlm"
```

#### Run Audio Baseline
```
cd wav2vec2_hubert_wavlm
python train_classifier.py --config configs/wavlm.yaml
```
Outputs:
- embeddings
- confusion matrices
- predictions

Metrics:
- Macro-F1
- Weighted-F1
  
### Logistic Regression Probe
#### Script
``` lin_probe.py ```

Uses:
- extracted embeddings
- sklearn Logistic Regression baseline

Also generates top-k malware family confusion matrices

## Experimental Results

| Model | Macro-F1 |
| :--- | :--- |
| **Image Models** | |
| ViT | 0.2852 |
| VGG19 | 0.3431 |
| RegNetY-040 | 0.2705 |
| **Audio Models** | |
| Wav2Vec2 | 0.2052 |
| HuBERT | 0.2302 |
| WavLM | 0.2845 |
| **Fusion Models** | |
| ViT + VGG (FCN) | 0.0116 |
| ViT + VGG (FCN, balanced) | 0.0946 |
| ViT + VGG (Cross-Attn) | 0.2661 |
| VGG19 + WavLM | 0.1592 |

Key observation:
- Image models outperform audio models consistently.
- VGG19 and WavLM perform best among image and speech encoders, respectively.
- Fusion models remain challenging.

## Fusion Experiments

Additional experiments explored:
- ViT + VGG concatenation
- Cross-attention fusion
- Image + Audio fusion
- Hyperbolic RGB fusion
- Static + Image fusion

Cross-attention provides the best fusion performance among tested approaches.

## Important Notes
### Dataset Challenges:
- Severe class imbalance
- Some malware families contain very few samples
- No stratified splitting used

### CUDA / Memory
Large models may require:
- lower batch size
- mixed precision
- reduced audio length

### PE Parsing Failures
Some binaries fail during PE parsing or RGB conversion. Current implementation skips problematic samples.

The current scripts use hardcoded parameters. Config files are provided primarily for reproducibility and experiment tracking. Scripts may require manual updates to load configs programmatically.

## Recommended Execution Order
For reproducing baselines:

1. Convert binaries to RGB
``` python convert_motif_to_rgb.py ```
2. Convert binaries to audio
``` python convert_to_audio.py ```
3. Train VGG19 baseline
``` python vgg19/linear_probe_vgg19.py ```
4. Extract audio embeddings
``` python wav2vec2_hubert_wavlm/extract_embeddings.py ```
5. Train audio classifier
``` python wav2vec2_hubert_wavlm/train_classifier.py ```
6. Run logistic regression probe
``` python wav2vec2_hubert_wavlm/lin_probe.py ```

## Ongoing Work & Future Directions
Static feature modeling

Static + image fusion

Hyperbolic multimodal fusion

Better cross-modal alignment

Advanced fusion architectures
