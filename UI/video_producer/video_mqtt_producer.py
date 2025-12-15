import cv2
import base64
import time
import argparse
import paho.mqtt.client as mqtt
import os

VIDEO = os.getenv("VIDEO", "/videos/demo_video.mp4")
FPS = float(os.getenv("FPS", "30"))
BROKER = os.getenv("BROKER", "mqtt")

parser = argparse.ArgumentParser(description="Simulacija ESP kamere z video datoteko")
parser.add_argument("--video", required=True, help="Pot do video datoteke")
parser.add_argument("--broker", default="localhost", help="MQTT broker IP")
parser.add_argument("--topic", default="camera/image", help="MQTT topic")
parser.add_argument("--loop", action="store_true", help="Ponovi video v zanki")

args = parser.parse_args()

client = mqtt.Client()
client.connect(args.broker, 1883, 60)
client.loop_start()

cap = cv2.VideoCapture(args.video)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25  # fallback

frame_interval = 1.0 / fps
print(f"Video FPS: {fps:.2f}, interval: {frame_interval:.3f}s")

next_frame_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        if args.loop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            break

    # Encode frame
    _, buffer = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    client.publish(args.topic, img_base64)
    print(f"Video FPS: {fps:.2f}, Frame poslan")

    # --- FPS control ---
    next_frame_time += frame_interval
    sleep_time = next_frame_time - time.time()

    if sleep_time > 0:
        time.sleep(sleep_time)
    else:
        # We're behind schedule → reset timing
        next_frame_time = time.time()

cap.release()
client.loop_stop()
client.disconnect()
