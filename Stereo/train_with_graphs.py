
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

# ========== DATASET ==========
class FullImageDistanceDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.data = df[df['image'].apply(lambda x: os.path.isfile(os.path.join(image_dir, x)))].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        _, height, width = image.shape
        x = row['x'] / width
        y = row['y'] / height
        w = row['w'] / width
        h = row['h'] / height
        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)
        distance = torch.tensor(float(row['distance'])/100.0, dtype=torch.float32)
        return image, bbox, distance

# ========== MODEL ==========
class DistanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.fc = nn.Identity()
        self.bbox_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.final_fc = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, bbox):
        image_feat = self.cnn(image)
        bbox_feat = self.bbox_fc(bbox)
        combined = torch.cat((image_feat, bbox_feat), dim=1)
        return self.final_fc(combined).squeeze(-1)

# ========== TRENING ==========
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uporabljam napravo: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = FullImageDistanceDataset('podatki_aug.csv', 'slike_aug', transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True,)

    model = DistanceModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    losses = []
    maes = []

    for epoch in range(100):
        model.train()
        total_loss = 0
        mae_total = 0
        count = 0

        for imgs, bboxes, distances in dataloader:
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            distances = distances.to(device)

            preds = model(imgs, bboxes)
            loss = criterion(preds, distances)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mae_total += torch.abs(preds - distances).sum().item()
            count += distances.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_mae = mae_total / count
        losses.append(avg_loss)
        maes.append(avg_mae)
        print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | MAE: {avg_mae:.2f} m")

    torch.save(model.state_dict(), "model_distance2.pth")
    print("✅ Model shranjen v 'model_distance.pth'")

    # Graf: Loss
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Učenje: Izguba (Loss)")
    plt.grid()
    plt.savefig("loss_plot.png")

    # Graf: MAE
    plt.figure()
    plt.plot(range(1, len(maes) + 1), maes, label="MAE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (m)")
    plt.title("Učenje: Povprečna napaka (MAE)")
    plt.grid()
    plt.savefig("mae_plot.png")
    print("📈 Grafi shranjeni kot 'loss_plot.png' in 'mae_plot.png'")

if __name__ == "__main__":
    train()
