import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from models.lightweight_cnn import LightweightTumorCNN
from models.custom_nn import BrainTumorCustomCNN
from preprocessing.contour_cropper import crop_brain_contour

class LocalImageDataset(Dataset):
    """
    Loads brain MRI images from standard folder structures:
    - Binary: data/yes/ (tumor) and data/no/ (normal)
    - 4-Class: data/normal/, data/glioma/, data/meningioma/, data/pituitary/
    If folders do not exist or are empty, generates realistic synthetic scans for immediate testing.
    """
    def __init__(self, root_dir: str = "data", num_classes: int = 2, image_size: tuple = (240, 240)):
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.image_size = image_size
        self.samples = []
        self._scan_directory()

    def _scan_directory(self):
        if not os.path.exists(self.root_dir):
            return

        valid_exts = (".png", ".jpg", ".jpeg")
        if self.num_classes == 2:
            class_map = {"no": 0, "normal": 0, "yes": 1, "tumor": 1}
        else:
            class_map = {"normal": 0, "glioma": 1, "meningioma": 2, "pituitary": 3}

        for folder_name, class_id in class_map.items():
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.lower().endswith(valid_exts):
                        self.samples.append((os.path.join(folder_path, fname), class_id))

    def __len__(self):
        return len(self.samples) if len(self.samples) > 0 else 64  # fallback synthetic count

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            # Synthetic MRI scan generator
            img = np.zeros((*self.image_size, 3), dtype=np.uint8)
            cv2_center = (self.image_size[0] // 2, self.image_size[1] // 2)
            cv2_radius = self.image_size[0] // 3
            # Simple brain circle
            y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
            dist = np.sqrt((x - cv2_center[0])**2 + (y - cv2_center[1])**2)
            img[dist < cv2_radius] = 160
            label = idx % self.num_classes
            if label > 0:
                # Add synthetic tumor spot
                img[dist < (cv2_radius // 3)] = 240
            tensor_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return tensor_img, label

        img_path, label = self.samples[idx]
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)
        cropped = crop_brain_contour(img_np)
        cropped_pil = Image.fromarray(cropped).resize(self.image_size)
        tensor_img = transforms.ToTensor()(cropped_pil)
        return tensor_img, label

def train(architecture="lightweight", epochs=15, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {architecture.upper()} | Epochs: {epochs} | Batch Size: {batch_size}")

    if architecture == "lightweight":
        dataset = LocalImageDataset(root_dir="data", num_classes=2, image_size=(240, 240))
        model = LightweightTumorCNN(num_classes=2).to(device)
        criterion = nn.BCEWithLogitsLoss()
    else:
        dataset = LocalImageDataset(root_dir="data", num_classes=4, image_size=(240, 240))
        # CustomCNN 3-channel input adaptation
        model = BrainTumorCustomCNN(in_channels=3, num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss()

    # 80/20 train/validation split
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("models_checkpoint", exist_ok=True)
    best_val_acc = 0.0
    checkpoint_name = f"models_checkpoint/{architecture}_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            out = model(x)
            if architecture == "lightweight":
                target = y.float().unsqueeze(1).to(device)
                loss = criterion(out, target)
                preds = (torch.sigmoid(out) >= 0.5).long().squeeze(1)
            else:
                y = y.to(device)
                loss = criterion(out, y)
                _, preds = torch.max(out, 1)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (preds.cpu() == y.cpu()).sum().item()
            train_total += x.size(0)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                if architecture == "lightweight":
                    preds = (torch.sigmoid(out) >= 0.5).long().squeeze(1)
                else:
                    _, preds = torch.max(out, 1)
                val_correct += (preds.cpu() == y.cpu()).sum().item()
                val_total += x.size(0)

        train_acc = (train_correct / max(train_total, 1)) * 100
        val_acc = (val_correct / max(val_total, 1)) * 100
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_name)

    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {checkpoint_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["lightweight", "custom"], default="lightweight")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(architecture=args.arch, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
