import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# ==============================
# CONFIG
# ==============================

DATA_DIR = "/home/aakanksha/MOTIF/family_rgb_images"
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# TRANSFORMS
# ==============================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Dataset size:", len(dataset))

# ==============================
# LOAD MODELS
# ==============================

vit = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=0
).to(DEVICE)

vgg = timm.create_model(
    "vgg19",
    pretrained=True,
    num_classes=0
).to(DEVICE)

vit.eval()
vgg.eval()

# ==============================
# EXTRACT EMBEDDINGS
# ==============================

vit_embeddings = []
vgg_embeddings = []
labels = []

with torch.no_grad():

    for images, y in loader:

        images = images.to(DEVICE)

        vit_embed = vit(images)
        vgg_embed = vgg(images)

        vit_embeddings.append(vit_embed.cpu())
        vgg_embeddings.append(vgg_embed.cpu())
        labels.append(y)

vit_embeddings = torch.cat(vit_embeddings)
vgg_embeddings = torch.cat(vgg_embeddings)
labels = torch.cat(labels)

print("ViT:", vit_embeddings.shape)
print("VGG:", vgg_embeddings.shape)
print("Labels:", labels.shape)

# ==============================
# SAVE
# ==============================

torch.save(vit_embeddings,"vit_embeddings.pt")
torch.save(vgg_embeddings,"vgg_embeddings.pt")
torch.save(labels,"labels.pt")

print("Saved embeddings for full dataset")
