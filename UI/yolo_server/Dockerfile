FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY yolo_mqtt.py .
COPY best.pt .
COPY model_distance.pth .
COPY model_distance2.pth .
COPY best_int8_openvino_model/ ./best_int8_openvino_model/
COPY best_int8_openvino_model1/ ./best_int8_openvino_model1/
COPY best_int8_openvino_model_8n_416/ ./best_int8_openvino_model_8n_416/
COPY best_int8_openvino_model_11n_416/ ./best_int8_openvino_model_11n_416/


CMD ["python", "yolo_mqtt.py"]
