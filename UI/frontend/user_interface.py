import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import base64
import threading
import json
import paho.mqtt.client as mqtt
import time
import simpleaudio as sa

alarm_active = False
alarm_lock = threading.Lock()
stop_threads = False

def on_closing():
    global stop_threads
    stop_threads = True
    root.destroy()

def play_alarm_pattern():
    global alarm_active, stop_threads

    try:
        wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
        while not stop_threads:
            if alarm_active:
                play_obj = wave_obj.play()
                time.sleep(0.5)
            else:
                time.sleep(0.1)
    except Exception as e:
        print("Napaka pri predvajanju zvoka:", e)


min_padding = 100

def process_and_display_result(result):
    try:
        img_bytes = base64.b64decode(result["image"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (1280, 720))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(pil_image)

        video_label.config(image=frame_tk)
        video_label.image = frame_tk

        for widget in danger_label_frame.winfo_children():
            widget.destroy()

        detections = result.get("detections", [])

        persons = []
        others = []

        for det in detections:
            label = det.get("class", "")
            if label.startswith("oseba"):
                persons.append(det)
            else:
                others.append(det)

        def sort_by_danger(det):
            label = det.get("class", "")
            level = 3
            if label.endswith("zelo_blizu"):
                level = 1
            elif label.endswith("blizu"):
                level = 2
            return level

        persons.sort(key=lambda d: (sort_by_danger(d), d.get("distance_m", float("inf"))))
        others.sort(key=sort_by_danger)

        all_detections = persons + others

        if not all_detections:
            tk.Label(danger_label_frame, text="Ni nevarnosti", bg="#4b2994", fg="white",
                     font=("Helvetica", 12)).pack(anchor='w')
            global alarm_active
            alarm_active = False
            return
        
        danger_found = False

        for det in all_detections:
            label = det.get("class", "unknown")
            text = label

            if label.startswith("oseba") and "distance_m" in det:
                text += f": {det['distance_m']} m"

            if label.endswith("zelo_blizu"):
                fg = "red"
                font = ("Helvetica", 16, "bold")
                danger_found = True
            elif label.endswith("blizu"):
                fg = "white"
                font = ("Helvetica", 14)
            else:
                fg = "white"
                font = ("Helvetica", 12)

            tk.Label(danger_label_frame, text=text, bg="#4b2994", fg=fg, font=font).pack(anchor='w')

        if danger_found:
            print("very dangr")
            alarm_active = True
        else:
            print("no dengir")
            alarm_active = False
                
    except Exception as e:
        print("Napaka pri prikazu:", e)

def on_message(client, userdata, msg):
    try:
        result = json.loads(msg.payload.decode("utf-8"))
        process_and_display_result(result)
    except Exception as e:
        print("Napaka pri sprejemu MQTT sporočila:", e)

def mqtt_thread():
    client = mqtt.Client()
    client.connect("localhost", 1883, 60)
    client.subscribe("camera/results")
    client.on_message = on_message
    client.loop_forever()

root = tk.Tk()
root.title("Live Detection")
root.configure(bg="#4b2994")

danger_label_frame = tk.Frame(root, bg="#4b2994", width=300, height=720)
danger_label_frame.grid(row=0, column=0, sticky="ns")
danger_label_frame.pack_propagate(False)

video_label = tk.Label(root, width=1280, height=720)
video_label.grid(row=0, column=1)

threading.Thread(target=play_alarm_pattern, daemon=True).start()
threading.Thread(target=mqtt_thread, daemon=True).start()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
