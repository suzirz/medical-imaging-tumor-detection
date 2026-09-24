import torch
import torch.nn as nn
import torchvision.models as models

class AdvancedTumorClassifier(nn.Module):
    """
    Production-grade Transfer Learning Intracranial Tumor Classifier.
    Backbone: Pretrained EfficientNet-B4 with regularized clinical classification head.
    
    Classes (4-Class):
        0: Glioma
        1: Meningioma
        2: Normal Tissue (notumor)
        3: Pituitary Adenoma
    """
    def __init__(self, num_classes=4, pretrained=False):
        super(AdvancedTumorClassifier, self).__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        return self.backbone.features(x)
