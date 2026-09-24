import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainTumorCustomCNN(nn.Module):
    """
    Custom Convolutional Neural Network (CNN) architecture built from scratch.
    Designed for fast CPU/laptop training and demonstration without external weights.
    Input shape: (batch_size, 4, 380, 380) or (batch_size, 3, 240, 240)
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 4):
        super(BrainTumorCustomCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling & Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(0.3)

        # Fully Connected (Dense) Layers
        self.fc1 = nn.Linear(256 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Layer 2: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Layer 3: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Layer 4: Conv -> BN -> ReLU -> AdaptivePool
        x = self.adaptive_pool(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        out = self.fc2(x)
        return out
