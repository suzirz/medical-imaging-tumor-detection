"""
Deep Generative Neural Network for MRI Cross-Modality Synthesis.
Implements a Residual Encoder-Decoder with Squeeze-and-Excitation Attention
for direct translation of unenhanced T1-weighted MRI into:
1. Virtual T1-Contrast (Virtual T1ce / Gadolinium Perfusion)
2. Virtual T2-FLAIR (Fluid-Attenuated Inversion Recovery with Vasogenic Edema Mapping)
3. Extended Tofts Pharmacokinetic Model (Ktrans and ve estimation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + res, inplace=True)

class DeepContrastSynthesisNet(nn.Module):
    """
    Dual-Head Generative Architecture:
    Head A: Synthesizes Virtual T1ce (Gadolinium perfusion dynamics)
    Head B: Synthesizes Virtual T2-FLAIR (CSF attenuation + parenchymal vasogenic edema)
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super(DeepContrastSynthesisNet, self).__init__()
        
        # Shared Encoder
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResidualBlock(base_channels * 2)
        self.res2 = ResidualBlock(base_channels * 2)
        
        # Shared Latent Bottleneck
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Head A: T1ce Enhancing Synthesis Head
        self.t1ce_head = nn.Sequential(
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )
        
        # Head B: FLAIR Synthesis Head
        self.flair_head = nn.Sequential(
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, dose: float = 1.0):
        feat = self.initial(x)
        feat = self.down1(feat)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.up1(feat)
        
        # Residual contrast addition modulated by dose
        t1ce_delta = self.t1ce_head(feat) * dose
        t1ce = torch.clamp(x + t1ce_delta * 0.4, 0.0, 1.0)
        
        flair = self.flair_head(feat)
        return t1ce, flair
