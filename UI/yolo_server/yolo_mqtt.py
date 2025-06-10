from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
import paho.mqtt.client as mqtt
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io
from prometheus_client import start_http_server, Counter, Gauge
import time

# Prometheus metriki
processed_frames = Counter("processed_frames_total", "Skupno število obdelanih sličic")
recognized_people = Counter("recognized_people_total", "Skupno število razpoznanih oseb")
processing_time = Gauge("frame_processing_seconds", "Čas obdelave sličice (v sekundah)")
frames_per_second = Gauge("frames_per_second", "Sličice na sekundo")

# YOLO model
model = YOLO("best.pt")
start_http_server(8000)

# Spremenljivki za sledenje FPS
last_frame_time = time.time()
frame_count = 0

# Razdaljni model
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

# Naloži razdaljni model
device = torch.device("cpu")
distance_model = DistanceModel().to(device)
distance_model.load_state_dict(torch.load("model_distance.pth", map_location=device))
distance_model.eval()

# Transform za PIL sliko
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def estimate_distance(pil_img, x1, y1, x2, y2):
    width, height = pil_img.size
    x_norm = x1 / width
    y_norm = y1 / height
    w_norm = (x2 - x1) / width
    h_norm = (y2 - y1) / height

    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    bbox_tensor = torch.tensor([[x_norm, y_norm, w_norm, h_norm]], dtype=torch.float32).to(device)

    with torch.no_grad():
        prediction = distance_model(img_tensor, bbox_tensor)
    return round(prediction.item(), 2)

def calculate_fps():
    """Izračuna FPS na podlagi časa med sličicami"""
    global last_frame_time, frame_count
    
    current_time = time.time()
    time_diff = current_time - last_frame_time
    
    if time_diff > 1.0:  # Posodobi FPS vsako sekundo
        fps = frame_count / time_diff
        frames_per_second.set(fps)
        frame_count = 0
        last_frame_time = current_time
    else:
        frame_count += 1

# MQTT obdelava
def on_message(client, userdata, msg):
    start_time = time.time()
    try:
        # Poveča štetje obdelanih sličic
        processed_frames.inc()
        
        person_count = 0
        img_bytes = base64.b64decode(msg.payload)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        height, width = frame.shape[:2]

        results = model(frame)[0]
        detections = []

        # Pretvori OpenCV sliko v PIL za razdaljni model
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls]
            det_data = {
                "class": label,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            }

            if label.startswith("oseba"):
                try:
                    person_count += 1
                    distance = estimate_distance(pil_image, x1, y1, x2, y2)
                    det_data["distance_m"] = distance
                except Exception as e:
                    print("Napaka pri napovedi razdalje:", e)

            detections.append(det_data)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Poveča štetje razpoznanih oseb (POPRAVLJENO)
        if person_count > 0:
            recognized_people.inc(person_count)

        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        result_message = {
            "image": img_base64,
            "detections": detections,
            "person_count": person_count  # Dodaj tudi število oseb v sporočilo
        }

        client.publish("camera/results", json.dumps(result_message))

    except Exception as e:
        print("Napaka pri obdelavi slike:", e)
    finally:
        # Nastavi čas obdelave (POPRAVLJENO - premaknjeno v finally blok)
        end_time = time.time()
        processing_time.set(end_time - start_time)
        
        # Izračunaj in posodobi FPS
        calculate_fps()

# MQTT client setup
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Povezan z MQTT brokerjem")
        print(f"Prometheus metriki dostopne na http://localhost:8000")
    else:
        print(f"Napaka pri povezavi z MQTT: {rc}")

def on_disconnect(client, userdata, rc):
    print("Prekinjena povezava z MQTT brokerjem")

client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

try:
    client.connect("mqtt", 1883, 60)
    client.subscribe("camera/image")
    print("Začenjam poslušanje MQTT sporočil...")
    client.loop_forever()
except KeyboardInterrupt:
    print("Prekinitev s tipkovnico")
    client.disconnect()
except Exception as e:
    print(f"Napaka: {e}")
    client.disconnect()