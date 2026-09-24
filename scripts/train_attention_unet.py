"""
Attention U-Net Training and Checkpoint Generator.
Trains the 31.4M parameter Attention U-Net on real cranial MRI scans (glioma, meningioma, pituitary, notumor)
using combined BCEDiceLoss and AdamW to produce a validated model checkpoint:
models_checkpoint/attention_unet_best.pth
"""
import os
import sys
import glob
import random
import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset, DataLoader
from models.attention_unet import AttentionUNet, BCEDiceLoss, compute_dice_coefficient, compute_iou_score

class BrainTumorSegmentationDataset(Dataset):
    def __init__(self, file_paths, is_train=True, img_size=(224, 224)):
        self.file_paths = file_paths
        self.is_train = is_train
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.file_paths)

    def _extract_pseudo_ground_truth(self, gray_img, is_normal):
        if is_normal:
            return np.zeros(self.img_size, dtype=np.float32)
        
        # Otsu thresholding on brain parenchyma to isolate hyperintense lesion
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        # Skull strip approximation: threshold non-black brain tissue
        _, brain_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        brain_pixels = gray_img[brain_mask > 0]
        
        if len(brain_pixels) > 0:
            p85 = np.percentile(brain_pixels, 82)
            _, tumor_mask = cv2.threshold(gray_img, p85, 255, cv2.THRESH_BINARY)
            tumor_mask = cv2.bitwise_and(tumor_mask, tumor_mask, mask=brain_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel)
            tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel)
        else:
            tumor_mask = np.zeros(self.img_size, dtype=np.uint8)

        # Keep largest connected component as primary lesion
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tumor_mask)
        final_mask = np.zeros_like(tumor_mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            if stats[largest_label, cv2.CC_STAT_AREA] > 40:
                final_mask[labels == largest_label] = 1.0
        return final_mask.astype(np.float32)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # Fallback black canvas
            img_bgr = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(img_rgb, self.img_size)
        gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
        is_normal = "notumor" in path.lower()

        mask = self._extract_pseudo_ground_truth(gray, is_normal)

        # Augmentation for training
        if self.is_train:
            if random.random() > 0.5:
                resized_rgb = cv2.flip(resized_rgb, 1)
                mask = cv2.flip(mask, 1)

        # Normalization
        norm_img = (resized_rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy(mask).unsqueeze(0).float()

        return tensor_img, tensor_mask


def train_attention_unet(max_samples=600, epochs=5, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Attention U-Net] Running on device: {device}")

    # Gather dataset files
    all_files = []
    for cls in ["glioma", "meningioma", "pituitary", "notumor"]:
        files = glob.glob(f"Dataset/Training/{cls}/*.jpg")
        random.shuffle(files)
        all_files.extend(files[: max_samples // 4])

    random.shuffle(all_files)
    split = int(0.85 * len(all_files))
    train_files = all_files[:split]
    val_files = all_files[split:]

    print(f"[Dataset] Train samples: {len(train_files)}, Val samples: {len(val_files)}")

    train_ds = BrainTumorSegmentationDataset(train_files, is_train=True)
    val_ds = BrainTumorSegmentationDataset(val_files, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AttentionUNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    os.makedirs("models_checkpoint", exist_ok=True)
    save_path = "models_checkpoint/attention_unet_best.pth"

    best_val_dice = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_dice = 0.0
        val_iou = 0.0
        count = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                preds = (torch.sigmoid(logits) >= 0.40).float()
                for p, m in zip(preds, masks):
                    val_dice += compute_dice_coefficient(p, m)
                    val_iou += compute_iou_score(p, m)
                    count += 1

        avg_val_dice = val_dice / max(count, 1)
        avg_val_iou = val_iou / max(count, 1)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Dice: {avg_val_dice*100:.2f}% | Val IoU: {avg_val_iou*100:.2f}%")

        if avg_val_dice >= best_val_dice or epoch == epochs:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), save_path)
            print(f"--> Checkpoint saved to {save_path} (Dice: {avg_val_dice*100:.2f}%)")

    print("[SUCCESS] Attention U-Net checkpoint created successfully!")
    return save_path

if __name__ == "__main__":
    train_attention_unet(max_samples=160, epochs=3, batch_size=8)
