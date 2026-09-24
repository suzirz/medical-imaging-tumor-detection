import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models import SimpleMedicalCNN

class SyntheticMedicalDataset(Dataset):
    """
    Demo dataset generator for synthetic medical scans (useful for quick testing without downloading 10GB datasets).
    """
    def __init__(self, num_samples=100, image_size=(224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random mock medical scan (noisy grayscale-ish tensor)
        img = torch.randn(3, *self.image_size) * 0.2 + 0.5
        label = 1 if idx % 2 == 0 else 0  # Alternating: 1 = Tumor, 0 = Normal
        return img, label

def train_pipeline(epochs=3, batch_size=8, lr=0.001):
    print("=== Memulai Pelatihan Pipeline Model Deteksi Medis ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device yang digunakan: {device}")

    # Dataset & DataLoader
    dataset = SyntheticMedicalDataset(num_samples=40)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Inisialisasi Model
    model = SimpleMedicalCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} | Akurasi: {epoch_acc:.2f}%")

    os.makedirs("models_checkpoint", exist_ok=True)
    checkpoint_path = os.path.join("models_checkpoint", "tumor_detector.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model berhasil disimpan ke: {checkpoint_path}")

if __name__ == "__main__":
    train_pipeline()
