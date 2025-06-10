FROM python:3.10-slim

WORKDIR /app

RUN apt update && apt install -y git ffmpeg libsm6 libxext6

RUN pip install --no-cache-dir ultralytics opencv-python paho-mqtt numpy
RUN pip install prometheus_client

COPY yolo_mqtt.py .
COPY best.pt .
COPY model_distance.pth .
COPY model_distance2.pth .


CMD ["python", "yolo_mqtt.py"]
