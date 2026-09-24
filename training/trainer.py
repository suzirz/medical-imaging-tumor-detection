import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from dataclasses import dataclass
import logging
from tqdm import tqdm

from models.efficientnet_tumor_classifier import BrainTumorClassifier, MultiClassFocalLoss

@dataclass
class TrainingConfig:
    num_classes: int = 4
    batch_size: int = 16
    phase1_epochs: int = 20
    phase1_lr: float = 1e-3
    phase2_epochs: int = 80
    phase2_lr: float = 1e-5
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    early_stopping_patience: int = 15
    checkpoint_dir: str = "models_checkpoint"

class TwoPhaseTrainer:
    """
    Orkestrator pelatihan dua fase:
    Fase 1: Bekukan backbone, latih kepala klasifikasi (20 epoch, lr=1e-3, ReduceLROnPlateau).
    Fase 2: Buka semua layer, fine-tune keseluruhan (80 epoch, lr=1e-5, Cosine Annealing).
    """
    def __init__(self, model: BrainTumorClassifier, train_loader, val_loader, config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = MultiClassFocalLoss(gamma=config.focal_gamma)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def run_epoch(self, dataloader, optimizer=None, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.set_grad_enabled(is_train):
            for batch in tqdm(dataloader, desc="Training" if is_train else "Validating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                if is_train and optimizer:
                    optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                if is_train and optimizer:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()

                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = (correct / max(total, 1)) * 100
        return epoch_loss, epoch_acc

    def fit(self):
        print("=== Memulai Fase 1: Pelatihan Kepala Klasifikasi (Backbone Dibekukan) ===")
        self.model.freeze_backbone()
        optimizer_phase1 = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                 lr=self.config.phase1_lr, weight_decay=self.config.weight_decay)
        scheduler_phase1 = ReduceLROnPlateau(optimizer_phase1, mode='max', factor=0.5, patience=5)

        best_val_acc = 0.0

        for epoch in range(1, self.config.phase1_epochs + 1):
            train_loss, train_acc = self.run_epoch(self.train_loader, optimizer_phase1, is_train=True)
            val_loss, val_acc = self.run_epoch(self.val_loader, is_train=False)
            scheduler_phase1.step(val_acc)
            print(f"[Fase 1] Epoch {epoch}/{self.config.phase1_epochs} - Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, "best_phase1_head.pt"))

        print("\n=== Memulai Fase 2: Fine-Tuning Keseluruhan (Backbone Dibuka) ===")
        self.model.unfreeze_backbone()
        optimizer_phase2 = AdamW(self.model.parameters(), lr=self.config.phase2_lr, weight_decay=self.config.weight_decay)
        scheduler_phase2 = CosineAnnealingLR(optimizer_phase2, T_max=self.config.phase2_epochs, eta_min=1e-7)

        for epoch in range(1, self.config.phase2_epochs + 1):
            train_loss, train_acc = self.run_epoch(self.train_loader, optimizer_phase2, is_train=True)
            val_loss, val_acc = self.run_epoch(self.val_loader, is_train=False)
            scheduler_phase2.step()
            print(f"[Fase 2] Epoch {epoch}/{self.config.phase2_epochs} - Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(self.config.checkpoint_dir, "best_tumor_model.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"--> Bobot model terbaik disimpan ke: {best_path}")

        print(f"Pelatihan Selesai! Akurasi Validasi Terbaik: {best_val_acc:.2f}%")

if __name__ == "__main__":
    from preprocessing.brats_preprocessor import BraTS2023Dataset
    from torch.utils.data import DataLoader

    # Simulasi mandiri
    dummy_ds = BraTS2023Dataset(root_dir="./data/BraTS2023")
    loader = DataLoader(dummy_ds, batch_size=4, shuffle=True)
    model = BrainTumorClassifier(num_classes=4, pretrained=False)
    cfg = TrainingConfig(phase1_epochs=2, phase2_epochs=2)
    trainer = TwoPhaseTrainer(model, loader, loader, cfg)
    trainer.fit()
