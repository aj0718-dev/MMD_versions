import torch
import timm
from transformers import AutoModel

print("========== DOWNLOADING AUDIO MODELS (HuggingFace) ==========")
audio_models = [
    "facebook/hubert-base-ls960", 
    "microsoft/wavlm-base", 
    "facebook/wav2vec2-base"
]

for model_id in audio_models:
    print(f"\n-> Downloading {model_id}...")
    model = AutoModel.from_pretrained(model_id)

print("\n========== DOWNLOADING IMAGE MODELS (TIMM) ==========")
image_models = [
    "vit_base_patch16_224",
    "vgg19",
    "regnety_040"
]

for model_name in image_models:
    print(f"\n-> Downloading {model_name}...")
    model = timm.create_model(model_name, pretrained=True)

print("\nAll required PTMs have been successfully downloaded and cached locally!")
