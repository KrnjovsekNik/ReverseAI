import cv2
import paho.mqtt.client as mqtt
import base64
import time
import json

def on_message(client, userdata, msg):
    detections = json.loads(msg.payload.decode())
    print("Detekcije:", detections)

client = mqtt.Client()
client.connect("localhost", 1883, 60)
client.subscribe("camera/results")
client.on_message = on_message
client.loop_start()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer)
        client.publish("camera/image", img_base64)
    time.sleep(2)  
