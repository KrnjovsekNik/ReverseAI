import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
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

# === GUI ===
root = tk.Tk()
root.title("Live Detection")
root.configure(bg="#e4edec")
root.geometry("1280x720")

# Glavni canvas za video
canvas = tk.Canvas(root, bg="#e4edec", highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

# Label za najbližjo nevarnost
top_left_label = tk.Label(root, text="", bg="black", fg="white", font=("Helvetica", 16, "bold"))
top_left_label.place(x=10, y=10)

# Gumb za nastavitve
def toggle_sidebar():
    global sidebar_open
    sidebar_open = not sidebar_open
    update_overlay()

settings_button = tk.Button(root, text="⚙️", command=toggle_sidebar, font=("Helvetica", 14))
settings_button.place(x=200, y=10)

# Overlay za zatemnitev
overlay = tk.Label(root, bg="black")
overlay.place_forget()

# Sidebar frame
sidebar = tk.Frame(root, bg="#e4edec", width=300)
sidebar.place_forget()

sidebar_open = False

def update_overlay():
    if sidebar_open:
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()
        overlay.configure(bg="black")
        overlay.tkraise()

        canvas.lift()
        top_left_label.lift()
        settings_button.lift()

        sidebar.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")
    else:
        overlay.place_forget()
        sidebar.place_forget()

# Dinamično prilagajanje velikosti
def resize(event):
    canvas.config(width=event.width, height=event.height)

root.bind("<Configure>", resize)

# Video prikaz
frame_tk = None

def display_frame(pil_image):
    global frame_tk

    if sidebar_open:
        # Zatemni video, če je sidebar odprt
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(0.5)

    window_width = root.winfo_width()
    window_height = root.winfo_height()
    pil_image = pil_image.resize((window_width, window_height), Image.ANTIALIAS)

    frame_tk = ImageTk.PhotoImage(pil_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)

# ============ PROCESIRANJE ============
def process_and_display_result(result):
    try:
        img_bytes = base64.b64decode(result["image"])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

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

        global alarm_active
        danger_found = False

        if all_detections:
            closest = all_detections[0]
            label = closest.get("class", "unknown")
            text = label
            if label.startswith("oseba") and "distance_m" in closest:
                text += f": {closest['distance_m']} m"

            top_left_label.config(text=text)

            if label.endswith("zelo_blizu"):
                danger_found = True
        else:
            top_left_label.config(text="")

        alarm_active = danger_found
        display_frame(pil_image)

    except Exception as e:
        print("Napaka pri prikazu:", e)

# MQTT
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

threading.Thread(target=play_alarm_pattern, daemon=True).start()
threading.Thread(target=mqtt_thread, daemon=True).start()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
