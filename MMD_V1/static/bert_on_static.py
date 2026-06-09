#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import numpy as np
import random

# ================= CONFIG =================
MODEL_NAME = "bert-base-uncased"   # replace with MalBERT if available
BATCH_SIZE = 8
EPOCHS = 15
LR = 2e-5
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORPUS_PATH = "motif_bert_data/corpus.txt"
LABELS_PATH = "motif_bert_data/labels.txt"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================= DATASET =================

class MalwareDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ================= MODEL =================

class MalwareBERT(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        return logits, cls

# ================= LOAD DATA =================

#with open(CORPUS_PATH) as f:
#    texts = [line.strip() for line in f]

#with open(LABELS_PATH) as f:
#    labels = [line.strip() for line in f]

texts = []
labels = []

with open(CORPUS_PATH) as f_text, open(LABELS_PATH) as f_label:
    for t, l in zip(f_text, f_label):
        t = t.strip()
        l = l.strip()

        if t and l:   # skip empty lines
            texts.append(t)
            labels.append(l)

print(f"Loaded {len(texts)} aligned samples")

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

# -------- RANDOM 80/20 SPLIT --------
indices = list(range(len(texts)))
random.shuffle(indices)

split = int(0.8 * len(indices))
train_idx = indices[:split]
test_idx = indices[split:]

X_train = [texts[i] for i in train_idx]
y_train = [labels_encoded[i] for i in train_idx]

X_test = [texts[i] for i in test_idx]
y_test = [labels_encoded[i] for i in test_idx]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ================= TOKENIZER =================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = MalwareDataset(X_train, y_train, tokenizer)
test_ds = MalwareDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ================= INIT =================

model = MalwareBERT(MODEL_NAME, num_classes).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ================= TRAIN =================

def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # IMPORTANT: evaluate only on labels present in test
    unique_labels = np.unique(all_labels)
    return f1_score(all_labels, all_preds, average="macro", labels=unique_labels)

for epoch in range(EPOCHS):
    model.train()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    test_f1 = evaluate(test_loader)
    print(f"Epoch {epoch+1} | Test Macro F1: {test_f1:.4f}")

# ================= FINAL EVAL =================

final_f1 = evaluate(test_loader)
print(f"\nFinal Test Macro F1: {final_f1:.4f}")

# ================= EMBEDDINGS =================

model.eval()
all_embeddings = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        _, emb = model(input_ids, attention_mask)
        all_embeddings.append(emb.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0)
torch.save(all_embeddings, "motif_bert_data/test_embeddings.pt")

print("Saved test embeddings.")
