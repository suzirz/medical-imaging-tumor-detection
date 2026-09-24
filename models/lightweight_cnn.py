import torch
import torch.nn as nn

class LightweightTumorCNN(nn.Module):
    """
    Lightweight Convolutional Neural Network (CNN) for fast brain tumor detection:
    - Input: (3, 240, 240)
    - ZeroPadding2d(2, 2)
    - Conv2d(3 -> 32, kernel_size=7, stride=1) + BatchNorm2d + ReLU
    - MaxPool2d(4, 4)
    - MaxPool2d(4, 4)
    - Flatten (14 * 14 * 32 = 6,272 features)
    - Dense (Linear 6272 -> 1) with Sigmoid for binary tumor detection (Normal vs Tumor),
      or (6272 -> 4) for multi-class classification.
    """
    def __init__(self, num_classes: int = 2):
        super(LightweightTumorCNN, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Zero Padding (2, 2)
            nn.ZeroPad2d(2),
            # Conv Layer (32 filters, 7x7, stride 1)
            nn.Conv2d(3, 32, kernel_size=7, stride=1),
            # Batch Normalization
            nn.BatchNorm2d(32),
            # ReLU Activation
            nn.ReLU(inplace=True),
            # First Max Pooling (f=4, s=4) -> 240 -> 60
            nn.MaxPool2d(kernel_size=4, stride=4),
            # Second Max Pooling (f=4, s=4) -> 60 -> 15
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        # Flatten & Dense Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 1 if num_classes == 2 else num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        out = self.classifier(x)
        return out
