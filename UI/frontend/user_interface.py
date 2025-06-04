import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import base64
import threading
import paho.mqtt.client as mqtt
from ultralytics import YOLO

import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn

# === Naloži YOLO model ===
model = YOLO("best.pt")

# === Razredi ===
class_names = ['oseba_zelo_blizu', 'oseba_blizu', 'oseba_dalec',
               'vozilo_zelo_blizu', 'vozilo_blizu', 'vozilo_dalec',
               'ostalo_zelo_blizu', 'ostalo_blizu', 'ostalo_dalec']

# === Določanje stopnje nevarnosti ===
def get_alpha_for_risk(label):
    if "zelo_blizu" in label:
        return 102  # 40%
    elif "blizu" in label:
        return 51   # 20%
    else:
        return 25   # 10%

# === Distance model definicija ===
class DistanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
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
        return self.final_fc(combined).squeeze()

# === Naloži distance model ===
device = torch.device("cpu")
distance_model = DistanceModel().to(device)
distance_model.load_state_dict(torch.load("model_distance.pth", map_location=device))
distance_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Obdelaj in prikaži sliko ===
def process_and_display_image(frame):
    results = model.predict(source=frame, conf=0.5, iou=0.3, max_det=10, verbose=False)[0]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    objects = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].cpu().numpy())
        label = class_names[cls_id]
        objects.append(label)

        alpha = get_alpha_for_risk(label)
        color = (255, 0, 0, alpha)

        draw.rectangle([x1, y1, x2, y2], fill=color)

        # === Če je oseba, napovej razdaljo ===
        if "oseba" in label:
            try:
                # Pripravi podatke
                crop = pil_image.crop((x1, y1, x2, y2)).resize((224, 224)).convert("RGB")
                img_tensor = transform(crop).unsqueeze(0).to(device)            

                width, height = pil_image.size
                x_norm = x1 / width
                y_norm = y1 / height
                w_norm = (x2 - x1) / width
                h_norm = (y2 - y1) / height
                bbox_tensor = torch.tensor([[x_norm, y_norm, w_norm, h_norm]], dtype=torch.float32).to(device)

                # Napovej razdaljo
                with torch.no_grad():
                    prediction = distance_model(img_tensor, bbox_tensor)
                    distance = round(prediction.item(), 2)
                    label_text = f"{label} ({distance} m)"
            except Exception as e:
                label_text = f"{label} (?)"
                print("Napaka:", e)
        else:
            label_text = label

        draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(pil_image, overlay).convert("RGB")
    frame_tk = ImageTk.PhotoImage(combined)

    video_label.config(image=frame_tk, width=1280, height=720)
    video_label.image = frame_tk

    danger_text = ', '.join(set(objects)) if objects else "Ni nevarnosti"
    danger_label.config(text=f"Danger Level:\n{danger_text}")

# === MQTT obdelava ===
def on_message(client, userdata, msg):
    try:
        img_bytes = base64.b64decode(msg.payload)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (1280, 720))

        process_and_display_image(frame)
    except Exception as e:
        print("Napaka pri prejemu slike:", e)

def mqtt_thread():
    client = mqtt.Client()
    client.connect("localhost", 1883, 60)  # ali "mqtt" če si v Dockerju
    client.subscribe("camera/image")
    client.on_message = on_message
    client.loop_forever()

# === GUI ===
root = tk.Tk()
root.title("Live Detection")
root.configure(bg="#1c5285")

danger_label = tk.Label(root, text="Zaznano:\n-", width=25, height=5, bg="white", anchor='nw', justify='left')
danger_label.grid(row=0, column=0)

video_label = tk.Label(root, width=1280, height=720)
video_label.grid(row=0, column=1)

threading.Thread(target=mqtt_thread, daemon=True).start()
root.mainloop()
