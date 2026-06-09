import torch

# Load one of your audio embeddings and one of your image embeddings
audio_emb = torch.load("path_to_your_audio_embedding.pt")
image_emb = torch.load("path_to_your_image_embedding.pt")

print(f"Audio shape: {audio_emb.shape}")
print(f"Image shape: {image_emb.shape}")