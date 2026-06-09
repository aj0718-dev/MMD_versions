import torch
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")   # important for SSH
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from collections import Counter
from torchvision import datasets

# ===============================
# CONFIG
# ===============================

DATA_DIR = "/home/aakanksha/MOTIF/family_rgb_images"
NUM_CLASSES = 502

# ===============================
# LOAD DATA
# ===============================

#labels = torch.load("labels.pt").numpy()
#preds  = torch.load("fusion_preds.pt").numpy()

labels_all = torch.load("labels.pt").numpy()
preds = torch.load("fusion_preds_balanced.pt").numpy()

# recreate same split
from sklearn.model_selection import train_test_split
import numpy as np

SEED = 42

_, test_idx = train_test_split(
    np.arange(len(labels_all)),
    test_size=0.2,
    random_state=SEED
)

labels = labels_all[test_idx]

print("Labels used:", labels.shape)
print("Preds:", preds.shape)

# ===============================
# CONFUSION MATRIX
# ===============================

cm = confusion_matrix(
    labels,
    preds,
    labels=list(range(NUM_CLASSES))
)

print("Confusion matrix shape:", cm.shape)

# ===============================
# CLASS NAME MAPPING
# ===============================

dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes

# ===============================
# TOP-30 CLASSES
# ===============================

counts = Counter(labels)
top_classes = [c for c,_ in counts.most_common(30)]

labels_top = [class_names[c] for c in top_classes]

cm_small = cm[np.ix_(top_classes, top_classes)]

# ===============================
# PLOT
# ===============================

plt.figure(figsize=(10,8))

sns.heatmap(
    cm_small,
    cmap="Blues",
    xticklabels=labels_top,
    yticklabels=labels_top
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Fusion Model Confusion Matrix (Top 30 Families)")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig("fusion_confusion_matrix_balanced.png", dpi=300)

print("Saved: fusion_confusion_matrix.png")

