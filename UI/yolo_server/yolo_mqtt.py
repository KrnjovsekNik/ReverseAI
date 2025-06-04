from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
import paho.mqtt.client as mqtt

model = YOLO("best.pt")

def on_message(client, userdata, msg):
    try:
        img_bytes = base64.b64decode(msg.payload)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": model.names[cls],
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, model.names[cls], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        result_message = {
            "image": img_base64,
            "detections": detections
        }

        client.publish("camera/results", json.dumps(result_message))

    except Exception as e:
        print("Napaka pri obdelavi slike:", e)

client = mqtt.Client()
client.connect("mqtt", 1883, 60)  
client.subscribe("camera/image")
client.on_message = on_message
client.loop_forever()
