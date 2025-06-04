import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import sys

# ===== MODEL DEFINITION =====
class DistanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # Odstranimo zadnji sloj

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
        return self.final_fc(combined).squeeze()


# ===== INFERENCA =====
def predict(image_path, x, y, w, h):
    device = torch.device("cpu")
    print(f"Uporabljam napravo: {device}")

    # Naloži model
    model = DistanceModel().to(device)
    model.load_state_dict(torch.load("model_distance.pth", map_location=device))
    model.eval()

    # Naloži in obdelaj sliko
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Normalizacija bounding boxa
    width, height = image.size
    x_norm = x / width
    y_norm = y / height
    w_norm = w / width
    h_norm = h / height
    bbox_tensor = torch.tensor([[x_norm, y_norm, w_norm, h_norm]], dtype=torch.float32).to(device)

    # Napovej
    with torch.no_grad():
        prediction = model(img_tensor, bbox_tensor)

    print(f"📏 Ocenjena razdalja: {prediction.item():.2f} metra")

# ===== ZAGON =====
if __name__ == "__main__":
    # Primer: python test_model.py slike/L1.png 100 50 200 300
    if len(sys.argv) != 6:
        print("Uporaba: python test_model.py <pot_do_slike> x y w h")
        sys.exit(1)

    image_path = sys.argv[1]
    x, y, w, h = map(int, sys.argv[2:])
    predict(image_path, x, y, w, h)
