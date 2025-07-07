import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import base64
import time
import threading
import simpleaudio as sa
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
import os

# Globals
video_width, video_height = 1280, 720
frame_tk = None
img_id = None
last_pil_img = None
text_bg_img = None
text_bg_id = None
text_id = None
alarm_active = False
stop_threads = False
video_fps = 30

# UI Setup
root = tk.Tk()
root.title("ReverseAI")
root.configure(bg="#e6f0ef")
root.geometry(f"{video_width}x{video_height}")
canvas = tk.Canvas(root, bg="#e6f0ef", highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)
text_id = canvas.create_text(5, 5, text="", fill="black", font=("Helvetica", 16, "bold"), anchor="nw")

# Model loading
model = YOLO(os.path.join(os.path.dirname(__file__), "best.pt"))
device = torch.device(0)

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

distance_model = DistanceModel().to(device)
distance_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "model_distance2.pth"), map_location=device))
distance_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preanalyze_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    analysis = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (video_width, video_height))
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results = model(frame)[0]
        detections = []
        persons = []
        others = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[cls]
            color = (0, 165, 255)
            opacity = 0.1
            distance = None

            if label.startswith("oseba"):
                try:
                    distance = estimate_distance(pil_image, x1, y1, x2, y2)
                    label += classify_distance(distance)
                except Exception as e:
                    print("Napaka razdalje:", e)

            det = {"class": label, "box": [x1, y1, x2, y2]}
            if distance:
                det["distance_m"] = distance

            if label.startswith("oseba"):
                persons.append(det)
            else:
                others.append(det)

        detections = sorted(persons, key=lambda d: d.get("distance_m", float("inf"))) + others
        frames.append(frame)
        analysis.append(detections)

    cap.release()
    return frames, analysis

def play_preanalyzed_video(frames, analysis):
    global alarm_active
    delay = 1 / video_fps

    def loop():
        global alarm_active
        start_time = time.time()
        for idx, (frame, detections) in enumerate(zip(frames, analysis)):
            target_time = start_time + idx * delay
            now = time.time()
            wait_time = target_time - now
            if wait_time > 0:
                time.sleep(wait_time)

            overlay = frame.copy()
            text = "Ni nevarnosti"
            danger = False

            for det in detections:
                x1, y1, x2, y2 = det["box"]
                label = det["class"]
                distance = det.get("distance_m", None)

                color = (0, 165, 255)
                opacity = 0.1

                if label.startswith("oseba"):
                    color = (0, 0, 255)
                elif label.startswith("vozilo"):
                    color = (0, 255, 255)

                if label.endswith("zelo_blizu"):
                    opacity = 0.4
                elif label.endswith("blizu"):
                    opacity = 0.2
                elif label.endswith("dalec"):
                    opacity = 0.1

                ov = overlay.copy()
                cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(ov, opacity, overlay, 1 - opacity, 0, overlay)

            if detections:
                closest = detections[0]
                lbl = closest["class"]
                if lbl.startswith("oseba"):
                    text = f"Nevarnost: Oseba! {closest.get('distance_m', '?')}m"
                    if lbl.endswith("blizu"):
                        danger = True
                elif lbl.startswith("vozilo"):
                    text = "Nevarnost: Vozilo!"
                elif lbl.startswith("ostalo"):
                    text = "Nevarnost: Ostalo!"
                if lbl.endswith("zelo_blizu"):
                    danger = True

            alarm_active = danger
            pil_final = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            display_frame(pil_final, text, danger)

    threading.Thread(target=loop, daemon=True).start()



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

def create_rounded_rect_image(w, h, radius, fill_color, shadow_color, shadow_offset=2):
    img = Image.new("RGBA", (w + shadow_offset, h + shadow_offset), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((shadow_offset, shadow_offset, w + shadow_offset, h + shadow_offset), radius, fill=shadow_color)
    draw.rounded_rectangle((0, 0, w, h), radius, fill=fill_color)
    return ImageTk.PhotoImage(img)

def display_frame(pil_image, danger_text, is_danger=False):
    global frame_tk, img_id, last_pil_img, text_bg_img, text_bg_id
    last_pil_img = pil_image
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
    frame_tk = ImageTk.PhotoImage(pil_image)

    if img_id is None:
        img_id = canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
    else:
        canvas.itemconfig(img_id, image=frame_tk)

    color = "red" if is_danger else "black"
    canvas.itemconfig(text_id, text=danger_text, fill=color)
    canvas.coords(text_id, 32, 28)
    bbox = canvas.bbox(text_id)
    if bbox:
        x0, y0, x1, y1 = bbox
        text_w = x1 - x0
        text_h = y1 - y0
        text_bg_img = create_rounded_rect_image(text_w + 24, text_h + 16, 6, "#e6f0ef", "#cdd9d7", 2)
        if text_bg_id is None:
            text_bg_id = canvas.create_image(20, 20, anchor="nw", image=text_bg_img)
        else:
            canvas.itemconfig(text_bg_id, image=text_bg_img)
            canvas.coords(text_bg_id, 20, 20)
        canvas.tag_lower(text_bg_id, text_id)
        canvas.tag_lower(img_id, text_bg_id)
        canvas.image_ref = frame_tk
        canvas.bg_image_ref = text_bg_img

def analyze_and_display(frame):
    global alarm_active
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(frame)[0]
    detections = []
    persons = []
    others = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.5: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[cls]
        color = (0, 165, 255)
        opacity = 0.1
        distance = None

        if label.startswith("oseba"):
            try:
                distance = estimate_distance(pil_image, x1, y1, x2, y2)
                label += classify_distance(distance)
            except Exception as e:
                print("Napaka razdalje:", e)

        if label.startswith("oseba"):
            color = (0, 0, 255)
        elif label.startswith("vozilo"):
            color = (0, 255, 255)

        if label.endswith("zelo_blizu"):
            opacity = 0.4
        elif label.endswith("blizu"):
            opacity = 0.2
        elif label.endswith("dalec"):
            opacity = 0.1

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        det = {"class": label, "box": [x1, y1, x2, y2]}
        if distance: det["distance_m"] = distance

        if label.startswith("oseba"):
            persons.append(det)
        else:
            others.append(det)

    all_detections = sorted(persons, key=lambda d: d.get("distance_m", float("inf"))) + others
    text = "Ni nevarnosti"
    danger = False

    if all_detections:
        closest = all_detections[0]
        lbl = closest["class"]
        if lbl.startswith("oseba"):
            text = f"Nevarnost: Oseba! {closest.get('distance_m', '?')}m"
        elif lbl.startswith("vozilo"):
            text = "Nevarnost: Vozilo!"
        elif lbl.startswith("ostalo"):
            text = "Nevarnost: Ostalo!"
        if lbl.endswith("zelo_blizu"):
            danger = True

    alarm_active = danger
    pil_final = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    display_frame(pil_final, text, danger)

def classify_distance(dist):
    if dist < 2:
        return "_zelo_blizu"
    elif dist < 4:
        return "_blizu"
    else:
        return "_dalec"

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Media files", "*.mp4 *.png *.jpg")])
    if file_path.endswith(".mp4"):
        play_video(file_path)
    elif file_path.endswith((".jpg", ".png")):
        frame = cv2.imread(file_path)
        frame = cv2.resize(frame, (video_width, video_height))
        analyze_and_display(frame)

def play_video(path):
    def process_and_play():
        frames, analysis = preanalyze_video(path)
        play_preanalyzed_video(frames, analysis)

    threading.Thread(target=process_and_play, daemon=True).start()


def play_alarm():
    try:
        wave_obj = sa.WaveObject.from_wave_file(os.path.join(os.path.dirname(__file__), "alarm.wav"))
        while not stop_threads:
            if alarm_active:
                wave_obj.play()
                time.sleep(0.5)
            else:
                time.sleep(0.1)
    except Exception as e:
        print("Zvok napaka:", e)

def on_key(event):
    if event.char.lower() == "n":
        open_file()

def on_close():
    global stop_threads
    stop_threads = True
    root.destroy()

root.bind("<Configure>", lambda e: display_frame(last_pil_img, canvas.itemcget(text_id, "text"), alarm_active) if last_pil_img else None)
root.bind("<Key>", on_key)
root.protocol("WM_DELETE_WINDOW", on_close)
threading.Thread(target=play_alarm, daemon=True).start()
root.mainloop()
