import torch
import torch.nn as nn
import torchvision.models as models

class VisionTransformerTumorClassifier(nn.Module):
    """
    Vision Transformer (ViT-B/16) for Cranial Tumor Classification.
    
    Transforms the MRI scan into 196 non-overlapping patches (16x16 px)
    processed through 12 Multi-Head Self-Attention (MHSA) Transformer blocks.
    
    Architecture:
    - Patch Resolution: 16x16
    - Sequence Length: 196 tokens + 1 [CLS] token = 197 tokens
    - Hidden Dimension: 768
    - Attention Heads: 12
    - Transformer Layers: 12
    - Input Size: (3, 224, 224)
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super(VisionTransformerTumorClassifier, self).__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)
        
        # Replace 1000-class ImageNet head with Clinical Classification Head
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize to 224x224 if necessary
        if x.shape[-2:] != (224, 224):
            x = nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        return self.backbone(x)


class DenseNetTumorClassifier(nn.Module):
    """
    DenseNet-121 Architecture for Cranial Tumor Classification.
    
    Employs dense iterative feature concatenation across 4 dense blocks:
    Every layer receives the direct concatenated feature outputs of all preceding layers,
    maximizing gradient flow and feature reuse for subtle tumor margins.
    
    Architecture:
    - Dense Blocks: 4 (6, 12, 24, 16 layers each)
    - Growth Rate: k = 32
    - Transition Layers: 3 (with 1x1 conv + 2x2 avg pooling)
    - Final Feature Dimension: 1024
    - Input Size: (3, 240, 240) or (3, 224, 224)
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super(DenseNetTumorClassifier, self).__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features  # 1024
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
