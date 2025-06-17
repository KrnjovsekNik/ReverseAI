import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
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
video_width = 1280
video_height = 720

frame_tk = None
img_id = None
last_pil_img = None

root = tk.Tk()
root.title("ReverseAI")
root.configure(bg="#e6f0ef")
root.geometry(f"{video_width}x{video_height}")

canvas = tk.Canvas(root, bg="#e6f0ef", highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

text_id = canvas.create_text(5, 5, text="", fill="black", font=("Helvetica", 16, "bold"), anchor="nw")

def resize(event):
    canvas.config(width=event.width, height=event.height)
    if last_pil_img:
        danger_text = canvas.itemcget(text_id, "text")
        color = canvas.itemcget(text_id, "fill")
        is_danger = (color == "red")
        display_frame(last_pil_img, danger_text, is_danger)


root.bind("<Configure>", resize)

# Global za ozadje teksta
text_bg_img = None
text_bg_id = None
text_shadow_id = None

def create_rounded_rect_image(w, h, radius, fill_color, shadow_color, shadow_offset=2):
    img = Image.new("RGBA", (w + shadow_offset, h + shadow_offset), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Shadow
    shadow_box = (shadow_offset, shadow_offset, w + shadow_offset, h + shadow_offset)
    draw.rounded_rectangle(shadow_box, radius=radius, fill=shadow_color)

    # Main box
    main_box = (0, 0, w, h)
    draw.rounded_rectangle(main_box, radius=radius, fill=fill_color)

    return ImageTk.PhotoImage(img)

def display_frame(pil_image, danger_text, is_danger=False):
    global frame_tk, img_id, last_pil_img
    global text_id, text_bg_id, text_bg_img

    last_pil_img = pil_image
    width = canvas.winfo_width()
    height = canvas.winfo_height()

    pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
    frame_tk = ImageTk.PhotoImage(pil_image)

    if img_id is None:
        img_id = canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
    else:
        canvas.itemconfig(img_id, image=frame_tk)

    # Barva teksta
    text_color = "red" if is_danger else "black"
    canvas.itemconfig(text_id, text=danger_text, fill=text_color, anchor="nw")
    canvas.coords(text_id, 20 + 12, 20 + 8)

    bbox = canvas.bbox(text_id)
    if bbox:
        x0, y0, x1, y1 = bbox
        text_w = x1 - x0
        text_h = y1 - y0
        padding_x = 24
        padding_y = 16
        radius = 6  # manj zaobljeni robovi

        box_w = text_w + padding_x
        box_h = text_h + padding_y

        text_bg_img = create_rounded_rect_image(
            box_w, box_h,
            radius=radius,
            fill_color="#e6f0ef",
            shadow_color="#cdd9d7",
            shadow_offset=2
        )

        if text_bg_id is None:
            text_bg_id = canvas.create_image(20, 20, anchor="nw", image=text_bg_img)
        else:
            canvas.itemconfig(text_bg_id, image=text_bg_img)
            canvas.coords(text_bg_id, 20, 20)

        # -- PRAVILEN layering:
        canvas.tag_lower(text_bg_id, text_id)  # ozadje pod besedilo
        canvas.tag_lower(img_id, text_bg_id)   # slika še nižje

        canvas.image_ref = frame_tk
        canvas.bg_image_ref = text_bg_img


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
        #for widget in danger_label_frame.winfo_children():
        #    widget.destroy()

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

        text = ""
        if not all_detections:
            #tk.Label(danger_label_frame, text="Ni nevarnosti", bg="#4b2994", fg="white",
            #         font=("Helvetica", 12)).pack(anchor='w')
            global alarm_active
            alarm_active = False
            text = "Ni nevarnosti"
        
        danger_found = False
        if all_detections:
            closest = all_detections[0]
            label = closest.get("class", "unknown")
            if label.startswith("oseba"):
                text = f"Nevarnost: Oseba! {all_detections[0]['distance_m']}"
            elif label.startswith("vozilo"):
                text = "Nevarnost: Vozilo!"
            elif label.startswith("ostalo"):
                text = "Nevarnost: Ostalo!"
            
            if label.endswith("zelo_blizu"):
                danger_found = True


        alarm_active=danger_found
        display_frame(pil_image, text, danger_found)
                
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

threading.Thread(target=play_alarm_pattern, daemon=True).start()
threading.Thread(target=mqtt_thread, daemon=True).start()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
