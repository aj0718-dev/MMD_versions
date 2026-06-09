import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import Wav2Vec2Model, HubertModel, WavLMModel

class FOCAHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.adaptive_pool(x)   # (batch, 128, 16) — fixed regardless of input length
        x = x.flatten(1)            # (batch, 2048)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))     # (batch, 128)
        return self.fc2(x)          # (batch, num_classes) — raw logits, no softmax


class MultimodalFOCA(nn.Module):
    def __init__(self, modality, model_name, num_classes):
        super().__init__()
        self.modality = modality
        self.model_name = model_name
        
        # 1. Load the specific backbone and assign in_channels
        if modality == 'image':
            if model_name == 'vit':
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
                in_channels = 768
            elif model_name == 'vgg19':
                self.backbone = timm.create_model('vgg19.tv_in1k', pretrained=True, num_classes=0, global_pool='')
                in_channels = 512
            elif model_name == 'regnety':
                self.backbone = timm.create_model('regnety_040', pretrained=True, num_classes=0, global_pool='')
                # We will verify exact channels for RegNetY dynamically below
            else:
                raise ValueError(f"Unknown image model: {model_name}")
                
        elif modality == 'audio':
            if model_name == 'wav2vec2':
                self.backbone = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
            elif model_name == 'hubert':
                self.backbone = HubertModel.from_pretrained('facebook/hubert-base-ls960')
            elif model_name == 'wavlm':
                self.backbone = WavLMModel.from_pretrained('microsoft/wavlm-base')
            else:
                raise ValueError(f"Unknown audio model: {model_name}")
            in_channels = 768
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # 2. Freeze all backbone weights explicitly
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # 3. Dynamic RegNetY channel check
        if modality == 'image' and model_name == 'regnety':
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                out = self.backbone.forward_features(dummy)
                in_channels = out.shape[1] # e.g. 440
                
        # 4. Attach FOCA Head
        self.head = FOCAHead(in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        # Force backbone to eval() and no_grad to save memory and skip BN updates
        self.backbone.eval()
        with torch.no_grad():
            if self.modality == 'image':
                feat = self.backbone.forward_features(x)
                
                # Handling Timms outputs differences
                if feat.dim() == 4:
                    # CNNs: (batch, channels, h, w) -> (batch, channels, h*w)
                    # VGG19 (batch, 512, 7, 7) -> (batch, 512, 49)
                    feat = feat.flatten(2)
                elif feat.dim() == 3:
                    # ViTs: (batch, seq_len, channels) -> (batch, channels, seq_len)
                    # ViT (batch, 197, 768) -> (batch, 768, 197)
                    feat = feat.transpose(1, 2)
            else:
                # Audio: (batch, seq_len, channels) -> (batch, channels, seq_len)
                feat = self.backbone(x).last_hidden_state
                feat = feat.transpose(1, 2)
                
        return self.head(feat)
