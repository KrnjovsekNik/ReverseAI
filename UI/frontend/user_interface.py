import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import base64
import threading
import json
import paho.mqtt.client as mqtt

# === Prikaz slike in podatkov ===
def process_and_display_result(result):
    try:
        # Dekodiraj base64 JPEG sliko
        img_bytes = base64.b64decode(result["image"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (1280, 720))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(pil_image)

        video_label.config(image=frame_tk, width=1280, height=720)
        video_label.image = frame_tk

        # Prikaži zaznane objekte
        detections = result.get("detections", [])
        lines = []
        for det in detections:
            label = det.get("class", "unknown")
            dist = det.get("distance_m")
            if dist is not None:
                lines.append(f"{label}: {dist} m")
            else:
                lines.append(label)

        danger_text = "\n".join(lines) if lines else "Ni nevarnosti"
        danger_label.config(text=f"Zaznano:\n{danger_text}")

    except Exception as e:
        print("Napaka pri prikazu:", e)

# === MQTT obdelava ===
def on_message(client, userdata, msg):
    try:
        result = json.loads(msg.payload.decode("utf-8"))
        process_and_display_result(result)
    except Exception as e:
        print("Napaka pri sprejemu MQTT sporočila:", e)

def mqtt_thread():
    client = mqtt.Client()
    client.connect("localhost", 1883, 60)  # ali "mqtt" če si v Dockerju
    client.subscribe("camera/results")
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
