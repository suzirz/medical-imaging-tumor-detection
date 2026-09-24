import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import timm
except ImportError:
    timm = None

class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention Block.
    Melakukan re-kalibrasi fitur per-channel untuk meningkatkan fokus representasi tumor.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitationBlock, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 8)
        self.fc1 = nn.Linear(in_channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, in_channels)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling -> (B, C)
        squeeze = x.view(b, c, -1).mean(dim=2)
        # Excitation
        excitation = F.silu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(b, c, 1, 1)
        # Scale
        return x * excitation

class BrainTumorClassifier(nn.Module):
    """
    Arsitektur EfficientNet-B4 yang dimodifikasi untuk 4-channel input (T1, T1ce, T2, FLAIR)
    dengan SE Attention dan Kepala Klasifikasi 4-Kelas.
    Kelas Target: 0: Normal, 1: Glioma, 2: Meningioma, 3: Pituitary
    """
    def __init__(self, num_classes=4, pretrained=True, dropout_head=0.4, dropout_final=0.2, use_se_attention=True):
        super(BrainTumorClassifier, self).__init__()
        self.num_classes = num_classes
        self.use_se_attention = use_se_attention

        if timm is not None:
            # Menggunakan timm EfficientNet-B4
            self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, in_chans=4, num_classes=0)
            in_features = self.backbone.num_features # 1792
        else:
            # Fallback jika library timm belum terinstall
            self.backbone = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((7, 7))
            )
            in_features = 32 * 7 * 7

        if use_se_attention and timm is not None:
            self.se_block = SqueezeExcitationBlock(in_features, reduction_ratio=16)
        else:
            self.se_block = nn.Identity()

        # Kepala Klasifikasi Kustom sesuai prompt PartyRock
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if timm is None else nn.Identity(),
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(dropout_head),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(dropout_final),
            nn.Linear(256, num_classes)
        )

    def freeze_backbone(self):
        """Membekukan bobot backbone (Fase 1)"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Membuka pembekuan bobot backbone (Fase 2)"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.se_block(features)
        out = self.classifier(features)
        return out

class MultiClassFocalLoss(nn.Module):
    """
    Focal Loss untuk menangani ketidakseimbangan kelas ekstrem pada dataset medis.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=[0.25, 0.25, 0.25, 0.25], gamma=2.0):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        at = self.alpha[targets]
        focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def export_to_onnx(model, output_path="tumor_model.onnx", input_shape=(1, 4, 380, 380)):
    """Mengekspor model PyTorch ke format ONNX siap produksi"""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True,
        opset_version=17,
        input_names=['input_mri_4ch'],
        output_names=['class_logits'],
        dynamic_axes={'input_mri_4ch': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}}
    )
    print(f"Model berhasil diekspor ke ONNX: {output_path}")
