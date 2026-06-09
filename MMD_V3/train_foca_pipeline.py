import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from models_foca import MultimodalFOCA

# ==========================================
# 1. DATASETS & FILTERING
# ==========================================
def get_filtered_files(data_dir, threshold, extension):
    """Filters classes having >= threshold samples."""
    classes = sorted(os.listdir(data_dir))
    samples, labels = [], []
    class_to_idx = {}
    current_label = 0
    
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir): continue
        
        # Get matching files
        files = [f for f in os.listdir(cls_dir) if f.casefold().endswith(extension)]
        
        valid_files = []
        if extension in [".png", ".jpg", ".jpeg"]:
            for f in files:
                full_path = os.path.join(cls_dir, f)
                try:
                    with Image.open(full_path) as img:
                        # Force loading the image data to catch deeper corruption
                        img.convert('RGB').load()
                    valid_files.append(f)
                except Exception:
                    pass # silently drop bad images
        else:
            valid_files = files
            
        if len(valid_files) >= threshold:
            class_to_idx[cls_name] = current_label
            for f in valid_files:
                samples.append(os.path.join(cls_dir, f))
                labels.append(current_label)
            current_label += 1
            
    print(f"Dataset filter (>= {threshold}): Found {len(samples)} samples across {current_label} classes.")
    return samples, np.array(labels), class_to_idx

def stratified_split(samples, labels):
    """Produces 70/15/15 splits, enforcing classes with few samples into train."""
    unique_classes, counts = np.unique(labels, return_counts=True)
    rare_classes = unique_classes[counts < 3] # If < 3 samples, we can't reliably split into 3 sets safely by default
    
    X_train, y_train = [], []
    X_split, y_split = [], []
    
    # Pre-allocate rare classes entirely to training to preserve them
    for s, l in zip(samples, labels):
        if l in rare_classes:
            X_train.append(s)
            y_train.append(l)
        else:
            X_split.append(s)
            y_split.append(l)
            
    X_split = np.array(X_split)
    y_split = np.array(y_split)
    
    # 1. Split to (Train 70%, Rest 30%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    for t_idx, rest_idx in sss1.split(X_split, y_split):
        X_train.extend(X_split[t_idx])
        y_train.extend(y_split[t_idx])
        X_rest = X_split[rest_idx]
        y_rest = y_split[rest_idx]
        
    # 2. Split Rest to (Val 50%, Test 50%) -> effectively 15%/15% of total
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for v_idx, ts_idx in sss2.split(X_rest, y_rest):
        X_val, y_val = X_rest[v_idx], y_rest[v_idx]
        X_test, y_test = X_rest[ts_idx], y_rest[ts_idx]
        
    return X_train, X_val, X_test, y_train, y_val, y_test

class MalwareImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color='black') # Fallback if corruption missed
            
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class MalwareAudioDataset(Dataset):
    def __init__(self, file_paths, labels, max_length=160000): # 10 secs at 16000Hz
        self.file_paths = file_paths
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        try:
            waveform, sr = torchaudio.load(audio_path)
            # Reformat to match specifications
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                
            waveform = waveform.squeeze(0) # -> (length)
            
            # Truncate
            if waveform.shape[0] > self.max_length:
                waveform = waveform[:self.max_length]
            else:
                # Zero-pad
                padding = self.max_length - waveform.shape[0]
                waveform = F.pad(waveform, (0, padding))
        except Exception as e:
            # Fallback for completely busted files
            waveform = torch.zeros(self.max_length)
            
        return waveform, self.labels[idx]


# ==========================================
# 2. MAIN SCRIPT
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data")
    parser.add_argument("--modality", type=str, choices=["audio", "image"], required=True)
    parser.add_argument("--model", type=str, required=True, help="vit | vgg19 | regnety | wav2vec2 | hubert | wavlm")
    parser.add_argument("--threshold", type=int, default=10, help="Min samples per class")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="results/")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu") # CPU only
    print(f"Device enforced: {device}")
    
    # 1. Dataset setup
    ext = ".wav" if args.modality == "audio" else (".png", ".jpg", ".jpeg")
    samples, labels, class_to_idx = get_filtered_files(args.data_dir, args.threshold, ext)
    num_classes = len(class_to_idx)
    if num_classes == 0:
        print("No classes found above threshold.")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(samples, labels)
    print(f"Splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    if args.modality == "image":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_ds = MalwareImageDataset(X_train, y_train, transform)
        val_ds   = MalwareImageDataset(X_val, y_val, transform)
        test_ds  = MalwareImageDataset(X_test, y_test, transform)
    else:
        train_ds = MalwareAudioDataset(X_train, y_train)
        val_ds   = MalwareAudioDataset(X_val, y_val)
        test_ds  = MalwareAudioDataset(X_test, y_test)
        
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Model Initialization
    model = MultimodalFOCA(modality=args.modality, model_name=args.model.lower(), num_classes=num_classes)
    model.to(device)
    
    # 3. Shape Verification Block
    print("Verifying model output shape...")
    if args.modality == 'image':
        dummy = torch.randn(2, 3, 224, 224)
    else:
        dummy = torch.randn(2, 160000)
        
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, num_classes), f"Shape mismatch: {out.shape}"
    print(f"Shape OK: {out.shape}")
    
    # 4. Compute balanced class weights
    classes_present = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes_present, y=y_train)
    # Ensure all classes have a weight even if 0 samples ended up in train (failsafe)
    weight_tensor = torch.ones(num_classes, dtype=torch.float32)
    for cls_idx, w in zip(classes_present, weights):
        weight_tensor[cls_idx] = w
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    # Increased learning rate from 1e-5 to 1e-3 to give the random FOCA head enough 
    # momentum to actually converge on the frozen feature mappings
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3)
    
    # 5. Training Loop
    best_val_f1 = -1
    best_loss = float('inf')
    patience_counter = 0
    patience = 7
    save_path = os.path.join(args.save_dir, f"{args.model}_best.pt")
    
    print("\nStarting Phase 1 Training (Frozen Backbone)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval() # Enforcement
        train_loss = 0
        
        for batch_i, (X_b, y_b) in enumerate(train_loader):
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                val_loss += criterion(logits, y_b).item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(y_b.numpy())
                
        val_loss /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")
        
        # Save best model logic based on F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            
        # Early Stopping check using loss
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at Epoch {epoch} due to Val Loss plateau.")
                break
                
    # 6. Evaluation Script
    print(f"\nLoading best checkpoint from {save_path} for Final Test...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    test_preds, test_labels = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            logits = model(X_b)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.numpy())
            test_labels.extend(y_b.numpy())
            
    print("\n" + "="*50)
    print(f"FINAL TEST REPORT: {args.model.upper()}")
    print("="*50)
    print(classification_report(test_labels, test_preds, zero_division=0))
    mac_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    wei_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    print(f"Macro-F1: {mac_f1:.4f}")
    print(f"Weighted-F1: {wei_f1:.4f}")
    
    # Save results specifically format
    res_file = os.path.join(args.save_dir, f"results_{args.model}.csv")
    with open(res_file, 'w') as f:
        f.write(f"Model,Macro-F1,Weighted-F1\n")
        f.write(f"{args.model},{mac_f1},{wei_f1}\n")
    print(f"Results saved to {res_file}")

if __name__ == "__main__":
    main()