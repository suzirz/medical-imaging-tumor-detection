import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any

class DoubleConv(nn.Module):
    """
    [Conv2d -> BatchNorm2d -> ReLU] x 2
    Preserves spatial dimensions via padding=1.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate (AG) from Oktay et al., 2018 (Attention U-Net).
    
    Dynamically re-weights skip-connection spatial features (x) using gating signals (g)
    from deeper decoder layers. Suppresses non-target background tissue (calvarium, normal
    parenchyma) and amplifies salient neoplastic boundaries before feature concatenation.
    
    Formula:
    alpha = Sigmoid( psi( ReLU( W_g(g) + W_x(x) ) ) )
    output = x * alpha
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # In case spatial dimensions differ by 1 px due to odd resizing
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha, alpha


class AttentionUNet(nn.Module):
    """
    Attention U-Net for Cranial Lesion Semantic Segmentation.
    
    Architecture:
    - 4 Encoder stages with 2x2 Max Pooling (64 -> 128 -> 256 -> 512 -> 1024)
    - 4 Decoder stages with 2x2 Transposed Convolutions
    - 4 Attention Gates filtering skip connections at resolutions:
      (H/8, W/8), (H/4, W/4), (H/2, W/2), (H, W)
    - 1x1 Convolution Head outputting pixel-level tumor logits
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder with Attention Gates
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder 1
        d4 = self.up1(x5)
        x4_gated, a4 = self.ag1(g=d4, x=x4)
        d4 = torch.cat([x4_gated, d4], dim=1)
        d4 = self.conv1(d4)
        
        # Decoder 2
        d3 = self.up2(d4)
        x3_gated, a3 = self.ag2(g=d3, x=x3)
        d3 = torch.cat([x3_gated, d3], dim=1)
        d3 = self.conv2(d3)
        
        # Decoder 3
        d2 = self.up3(d3)
        x2_gated, a2 = self.ag3(g=d2, x=x2)
        d2 = torch.cat([x2_gated, d2], dim=1)
        d2 = self.conv3(d2)
        
        # Decoder 4
        d1 = self.up4(d2)
        x1_gated, a1 = self.ag4(g=d1, x=x1)
        d1 = torch.cat([x1_gated, d1], dim=1)
        d1 = self.conv4(d1)
        
        logits = self.outc(d1)
        
        if return_attention:
            return logits, [a1, a2, a3, a4]
        return logits


# ================= SEGMENTATION LOSSES & EVALUATION METRICS =================

def compute_dice_coefficient(pred_mask: torch.Tensor, target_mask: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Computes Dice Similarity Coefficient (DSC):
    DSC = (2 * |P ∩ G|) / (|P| + |G| + eps)
    """
    p = pred_mask.view(-1).float()
    g = target_mask.view(-1).float()
    intersection = (p * g).sum()
    dice = (2.0 * intersection + epsilon) / (p.sum() + g.sum() + epsilon)
    return float(dice.item())


def compute_iou_score(pred_mask: torch.Tensor, target_mask: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Computes Intersection over Union (IoU / Jaccard Index):
    IoU = |P ∩ G| / (|P ∪ G| + eps)
    """
    p = pred_mask.view(-1).float()
    g = target_mask.view(-1).float()
    intersection = (p * g).sum()
    union = p.sum() + g.sum() - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return float(iou.item())


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Soft Dice Loss for Lesion Segmentation.
    Balances pixel-level classification and region-level overlap.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, epsilon: float = 1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)
        
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        target_flat = target.view(-1)
        
        intersection = (probs_flat * target_flat).sum()
        dice = (2.0 * intersection + self.epsilon) / (probs_flat.sum() + target_flat.sum() + self.epsilon)
        dice_loss = 1.0 - dice
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
