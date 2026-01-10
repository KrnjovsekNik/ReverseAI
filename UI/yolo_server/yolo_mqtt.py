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
import requests
import threading
import time
from prometheus_client import start_http_server, Counter, Gauge
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


#######################################
# CONFIG — tukaj spremeni RPi endpoint
#######################################

RPI_ADDR = "http://nikkrasp:8080"   # <-- zamenjaj z IP ali hostname RPi
META_URL = f"{RPI_ADDR}/meta"
FRAME_URL = f"{RPI_ADDR}/frame"

#######################################
# MQTT SIGNALING STATE (NOVO!)
#######################################
streaming_active = True   # ali naj YOLO pobira frame-e
last_processed_id = -1     # prepreči podvajanje frame-ov


#######################################
# METRIKE
#######################################
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

processed_frames = Counter("processed_frames_total", "Skupno število obdelanih sličic")
recognized_people = Counter("recognized_people_total", "Skupno število razpoznanih oseb")
processing_time = Gauge("frame_processing_seconds", "Čas obdelave sličice (v sekundah)")
frames_per_second = Gauge("frames_per_second", "Sličice na sekundo")
recognized_vehicle = Counter("recognized_vehicle_total", "Skupno število razpoznanih vozil")
recognized_others = Counter("recognized_others_total", "Skupno število razpoznanih ostalih objektov")

last_frame_time = time.time()
frame_count = 0
conf_treshold = 0.2


#######################################
# MODELI (ne sprememb)
#######################################
model = YOLO("best_int8_openvino_model1", task="detect")
start_http_server(8000)

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

device = torch.device("cpu")
distance_model = DistanceModel().to(device)
distance_model.load_state_dict(torch.load("model_distance2.pth", map_location=device))
distance_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


#######################################
# FUNKCIJE ZA HTTP FRAME FETCH (NOVO!)
#######################################

def fetch_meta():
    """
    Prebere meta podatke s kamerine strani:
    { "id": int, "timestamp": int }
    """
    r = requests.get(META_URL, timeout=0.2)
    return r.json()

def fetch_frame():
    """
    Prebere JPEG s kamerine strani in ga pretvori v OpenCV sliko
    """
    r = requests.get(FRAME_URL, timeout=0.2)
    img = np.frombuffer(r.content, dtype=np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


#######################################
# RAZDELITE DISTANCE del (ne sprememb)
#######################################

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


#######################################
# FPS METRIKA (ne sprememb)
#######################################
def calculate_fps():
    global last_frame_time, frame_count
    current_time = time.time()
    time_diff = current_time - last_frame_time

    if time_diff > 1.0:
        fps = frame_count / time_diff
        frames_per_second.set(fps)
        frame_count = 0
        last_frame_time = current_time
    else:
        frame_count += 1


##########################################
# GLAVNA YOLO OBRATOVALNA FUNKCIJA (NOVO!)
##########################################

def handle_frame(frame):
    """ Tukaj je vnesena tvoja YOLO logika iz on_message() """

    start_time = time.time()
    processed_frames.inc()

    person_count = 0
    vehicle_count = 0
    others_count = 0

    results = model(frame)[0]
    detections = []

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    height, width = frame.shape[:2]

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < conf_treshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[cls]

        det_data = {
            "class": label,
            "confidence": round(conf, 2),
            "box": [x1, y1, x2, y2]
        }

        color = (0, 165, 255)
        opacity = 0.1

        if label.startswith("oseba"):
            try:
                person_count += 1
                dist = estimate_distance(pil_image, x1, y1, x2, y2)
                det_data["distance_m"] = dist
            except Exception:
                pass

        if label.startswith("oseba"):
            color = (0, 0, 255)
        elif label.startswith("vozilo"):
            vehicle_count += 1
            color = (0, 255, 255)
        elif label.startswith("ostalo"):
            others_count += 1

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        detections.append(det_data)

    if person_count > 0: recognized_people.inc(person_count)
    if vehicle_count > 0: recognized_vehicle.inc(vehicle_count)
    if others_count > 0: recognized_others.inc(others_count)

    # publish rezultat nazaj (kot prej)
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    result_message = {
        "image": img_base64,
        "detections": detections,
        "person_count": person_count
    }

    mqtt_client.publish("camera/results", json.dumps(result_message))

    processing_time.set(time.time() - start_time)
    calculate_fps()


#######################################
# NOV: REAL-TIME PROCESSING LOOP
#######################################

def process_loop():
    global last_processed_id, streaming_active

    while True:
        if not streaming_active:
            time.sleep(0.1)
            continue

        try:
            meta = fetch_meta()
            fid = meta["id"]

            if fid <= last_processed_id:
                time.sleep(0.01)
                continue

            frame = fetch_frame()
            if frame is None:
                continue

            last_processed_id = fid
            handle_frame(frame)

        except Exception:
            time.sleep(0.05)
            continue


#######################################
# MQTT SIGNALING HANDLERS (SPREMEMBA!)
#######################################

def on_message(client, userdata, msg):
    """ Sprejme: {"state":"streaming", "frame_id":..., "timestamp":...} """
    global streaming_active
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        state = data.get("state", "")
        streaming_active = (state == "streaming")
    except:
        pass

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT povezava OK")
        client.subscribe("camera/status")  # <-- NOVO!
    else:
        print("Napaka MQTT:", rc)

def on_disconnect(client, userdata, rc):
    print("MQTT prekinjen")


#######################################
# MQTT INIT
#######################################

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message = on_message

mqtt_client.connect("mqtt", 1883, 60)
mqtt_client.loop_start()   # <-- loop_forever NE SMEMO več ker blokira!


#######################################
# THREADI START
#######################################

threading.Thread(target=process_loop, daemon=True).start()

print("YOLO streaming server teče...")
while True:
    time.sleep(1)
