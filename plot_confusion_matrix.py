import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
from torchvision import datasets

DATA_DIR = "/home/aakanksha/MOTIF/family_rgb_images"

# ===============================
# LOAD LABELS + PREDICTIONS
# ===============================

# change filenames depending on model
labels = torch.load("/home/aakanksha/MOTIF/vit/labels.pt").numpy()
preds_vit = torch.load("/home/aakanksha/MOTIF/vit/vit_preds.pt").numpy()
preds_regnet = torch.load("/home/aakanksha/MOTIF/regnety_040/regnet_preds.pt").numpy()
preds_vgg = torch.load("/home/aakanksha/MOTIF/vgg19/vgg_preds.pt").numpy()

# ===============================
# FUNCTION TO COMPUTE MATRIX
# ===============================

'''def compute_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm'''
def compute_cm(y_true, y_pred, num_classes):
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm

NUM_CLASSES = 502

cm_vit = compute_cm(labels, preds_vit, NUM_CLASSES)
cm_regnet = compute_cm(labels, preds_regnet, NUM_CLASSES)
cm_vgg = compute_cm(labels, preds_vgg, NUM_CLASSES)

'''cm_vit = compute_cm(labels, preds_vit)
cm_regnet = compute_cm(labels, preds_regnet)
cm_vgg = compute_cm(labels, preds_vgg)'''

print("Matrix shape:", cm_vit.shape)   # should be (502,502)

dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes

# ===============================
# OPTIONAL: SHOW ONLY TOP-K CLASSES
# ===============================

import os

os.makedirs("confusion_matrices", exist_ok=True)

def plot_topk_cm(cm, labels, k=30, title="Confusion Matrix", filename="cm.png"):

    counts = Counter(labels)

    top_classes = [c for c,_ in counts.most_common(k)]
    top_classes = np.array(top_classes)
    labels_top = [class_names[c] for c in top_classes]
    cm_small = cm[np.ix_(top_classes, top_classes)]
    plt.figure(figsize=(10,8))

    sns.heatmap(cm_small,
            cmap="Blues",
            xticklabels=labels_top,
            yticklabels=labels_top)

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()

    plt.savefig(f"confusion_matrices/{filename}", dpi=300)
    plt.close()

plot_topk_cm(cm_vit, labels, k=30,
             title="ViT Confusion Matrix",
             filename="vit_cm.png")

plot_topk_cm(cm_regnet, labels, k=30,
             title="RegNet Confusion Matrix",
             filename="regnet_cm.png")

plot_topk_cm(cm_vgg, labels, k=30,
             title="VGG19 Confusion Matrix",
             filename="vgg_cm.png")

'''def plot_topk_cm(cm, labels, k=30, title="Confusion Matrix"):

    counts = Counter(labels)
    top_classes = [c for c,_ in counts.most_common(k)]
    top_classes = np.array(top_classes)

    cm_small = cm[np.ix_(top_classes, top_classes)]
    #cm_small = cm[top_classes][:, top_classes]

    plt.figure(figsize=(10,8))
    sns.heatmap(cm_small, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# ===============================
# PLOTS
# ===============================

plot_topk_cm(cm_vit, labels, k=30, title="ViT Confusion Matrix (Top 30)")
plot_topk_cm(cm_regnet, labels, k=30, title="RegNet Confusion Matrix (Top 30)")
plot_topk_cm(cm_vgg, labels, k=30, title="VGG19 Confusion Matrix (Top 30)")'''
