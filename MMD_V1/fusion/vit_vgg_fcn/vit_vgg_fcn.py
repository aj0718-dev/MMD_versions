import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import numpy as np

# ==============================
# CONFIG
# ==============================

NUM_CLASSES = 502
EPOCHS = 25
LR = 1e-3
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)

# ==============================
# LOAD EMBEDDINGS
# ==============================

vit_embeddings = torch.load("/home/aakanksha/MOTIF/vit_vgg_fcn/vit_embeddings.pt")
vgg_embeddings = torch.load("/home/aakanksha/MOTIF/vit_vgg_fcn/vgg_embeddings.pt")
labels = torch.load("/home/aakanksha/MOTIF/vit_vgg_fcn/labels.pt")

print("ViT embeddings:", vit_embeddings.shape)
print("VGG embeddings:", vgg_embeddings.shape)
print(vit_embeddings.shape[0] == vgg_embeddings.shape[0] == labels.shape[0])
print((vit_embeddings[:5] == vit_embeddings[:5]).all())  # just to ensure consistency

# ==============================
# NORMALIZE EMBEDDINGS
# ==============================

vit_embeddings = F.normalize(vit_embeddings, dim=1)
vgg_embeddings = F.normalize(vgg_embeddings, dim=1)
from sklearn.decomposition import PCA

# convert to numpy first
vit_np = vit_embeddings.numpy()
vgg_np = vgg_embeddings.numpy()

DIM = 256

vit_pca = PCA(n_components=DIM).fit_transform(vit_np)
vgg_pca = PCA(n_components=DIM).fit_transform(vgg_np)

# normalize again
vit_pca /= np.linalg.norm(vit_pca, axis=1, keepdims=True) + 1e-8
vgg_pca /= np.linalg.norm(vgg_pca, axis=1, keepdims=True) + 1e-8

X = np.concatenate([vit_pca, vgg_pca], axis=1)  # 512 dim

# ==============================
# CONCATENATE
# ==============================

#fused_embeddings = torch.cat([vit_embeddings, vgg_embeddings], dim=1)
#fused_embeddings = torch.cat([vit_pca, vgg_pca], dim=1)
fused_embeddings = torch.tensor(X)
print("Fusion embedding shape:", fused_embeddings.shape)
torch.save(fused_embeddings, "vit_vgg_fusion_embeddings_balanced.pt")

# ==============================
# TRAIN / TEST SPLIT
# ==============================

#X = fused_embeddings.numpy()
y = labels.numpy()

train_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.2,
    random_state=SEED
)

X_train = torch.tensor(X[train_idx]).float().to(DEVICE)
X_test  = torch.tensor(X[test_idx]).float().to(DEVICE)

y_train = torch.tensor(y[train_idx]).long().to(DEVICE)
y_test  = torch.tensor(y[test_idx]).long().to(DEVICE)

# ==============================
# MODEL
# ==============================

class FusionClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim,1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024,num_classes)

        )

    def forward(self,x):
        return self.net(x)


model = FusionClassifier(
    input_dim=fused_embeddings.shape[1],
    num_classes=NUM_CLASSES
).to(DEVICE)

# ==============================
# TRAIN SETUP
# ==============================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

# ==============================
# TRAIN LOOP
# ==============================

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    logits = model(X_train)

    loss = criterion(logits,y_train)

    loss.backward()
    optimizer.step()

    # -------------------------
    # evaluation
    # -------------------------

    model.eval()

    with torch.no_grad():

        test_logits = model(X_test)

        preds = torch.argmax(test_logits,dim=1).cpu().numpy()
        true  = y_test.cpu().numpy()

    macro_f1 = f1_score(true,preds,average="macro",zero_division=0)
    weighted_f1 = f1_score(true,preds,average="weighted",zero_division=0)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("-"*40)

# ==============================
# SAVE RESULTS
# ==============================

torch.save(torch.tensor(preds),"fusion_preds_balanced.pt")

print("Fusion training complete")
