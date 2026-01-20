import sys
import base64
import json
import time

import numpy as np
import cv2
import paho.mqtt.client as mqtt

from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QImage, QPainter, QColor, QFont, QBrush, QPen
from PySide6.QtWidgets import QApplication, QWidget

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MQTT_HOST = "localhost"
MQTT_TOPIC_CAM = "camera/results"
MQTT_TOPIC_TOF = "sensor/tof"


# --------------------------------------------------
# MQTT WORKER
# --------------------------------------------------
class MqttWorker(QThread):
    frame_signal = Signal(dict)
    tof_signal = Signal(int)

    def __init__(self, host, topic_cam, topic_tof):
        super().__init__()
        self.host = host
        self.topic_cam = topic_cam
        self.topic_tof = topic_tof
        self.client = mqtt.Client()
        self.running = True

    def run(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("[MQTT] Connected")
                client.subscribe(self.topic_cam)
                client.subscribe(self.topic_tof)
            else:
                print("[MQTT] Connection failed:", rc)

        def on_message(client, userdata, msg):
            try:
                if msg.topic == self.topic_cam:
                    data = json.loads(msg.payload.decode("utf-8"))
                    self.frame_signal.emit(data)

                elif msg.topic == self.topic_tof:
                    tof_str = msg.payload.decode("utf-8").strip()
                    if tof_str.isdigit():
                        self.tof_signal.emit(int(tof_str))

            except Exception as e:
                print("[MQTT] Message error:", e)

        self.client.on_connect = on_connect
        self.client.on_message = on_message

        while self.running:
            try:
                print("[MQTT] Connecting to", self.host)
                self.client.connect(self.host, 1883, 60)
                self.client.loop_forever()
            except Exception as e:
                print("[MQTT] Error, retry in 2s:", e)
                time.sleep(2)

    def stop(self):
        self.running = False
        try:
            self.client.disconnect()
        except:
            pass


# --------------------------------------------------
# VIDEO WIDGET
# --------------------------------------------------
class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReverseAI")
        self.setStyleSheet("background-color: #e6f0ef;")
        self.resize(1280, 720)

        self.frame_rgb = None
        self.qimage = None
        self.text = "Ni nevarnosti"
        self.danger = False

        self.tof_mm = None

        self.font = QFont("Helvetica", 20, QFont.Bold)

    def update_from_result(self, result: dict):
        try:
            img_b64 = result.get("image", None)
            if not isinstance(img_b64, str):
                return

            img_bytes = base64.b64decode(img_b64)
            buf = np.frombuffer(img_bytes, np.uint8)
            if buf.size == 0:
                return

            frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.frame_rgb = frame_rgb

            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            self.qimage = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            detections = result.get("detections", [])
            if not isinstance(detections, list):
                detections = []

            if len(detections) == 0:
                self.text = "Ni nevarnosti"
                self.danger = False
            else:
                det = detections[0]
                label = det.get("class", "")
                dist = det.get("distance_m", None)

                if label.startswith("oseba"):
                    if isinstance(dist, (int, float)):
                        self.text = f"Oseba {dist:.2f} m"
                    else:
                        self.text = "Oseba"
                elif label.startswith("vozilo"):
                    self.text = "Vozilo"
                elif label.startswith("ostalo"):
                    self.text = "Nevarnost: Ostalo"
                else:
                    self.text = label or "Nevarnost"

                self.danger = label.endswith("zelo_blizu")

            self.update()

        except Exception as e:
            print("[UI] Error:", e)

    def update_tof(self, mm: int):
        cm = mm / 10

        if cm > 100:
            self.tof_mm = None
            self.tof_text = "N/A"
        else:
            self.tof_mm = cm
            self.tof_text = f"{cm:.1f} cm"

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#e6f0ef"))

        if self.qimage is not None:
            widget_w = self.width()
            widget_h = self.height()
            scaled = self.qimage.scaled(widget_w, widget_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (widget_w - scaled.width()) // 2
            y = (widget_h - scaled.height()) // 2
            painter.drawImage(x, y, scaled)

        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)

        metrics = painter.fontMetrics()

        line1 = self.text
        line2 = f"TOF: {self.tof_text}" if hasattr(self, "tof_text") else ""

        w1 = metrics.horizontalAdvance(line1)
        w2 = metrics.horizontalAdvance(line2) if line2 else 0

        tw = max(w1, w2)
        th = metrics.height()

        padding_x = 24
        padding_y = 16

        gap = 10 if line2 else 0

        box_w = tw + padding_x
        box_h = th + (th if line2 else 0) + gap + padding_y

        x0 = 20
        y0 = 20

        bg = QColor("#e6f0ef")
        shadow = QColor("#cdd9d7")

        painter.setBrush(QBrush(shadow))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(x0 + 2, y0 + 2, box_w, box_h), 6, 6)

        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(QRectF(x0, y0, box_w, box_h), 6, 6)

        text_x = x0 + padding_x / 2
        text_y = y0 + padding_y / 2 + th * 0.7
        painter.setPen(QPen(QColor("red") if self.danger else QColor("black")))
        painter.drawText(text_x, text_y, self.text)

        if hasattr(self, "tof_text"):
            tof_color = QColor("black")

            if isinstance(self.tof_mm, (int, float)) and self.tof_mm < 50:
                tof_color = QColor("red")

            painter.setPen(QPen(tof_color))
            painter.drawText(text_x, text_y + th + gap, f"TOF: {self.tof_text}")


# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
class ReverseAIApp:
    def __init__(self):
        self.qt_app = QApplication(sys.argv)
        self.win = VideoWidget()

        self.mqtt_worker = MqttWorker(MQTT_HOST, MQTT_TOPIC_CAM, MQTT_TOPIC_TOF)
        self.mqtt_worker.frame_signal.connect(self.win.update_from_result)
        self.mqtt_worker.tof_signal.connect(self.win.update_tof)
        self.mqtt_worker.start()

        self.win.show()
        exit_code = self.qt_app.exec()

        self.mqtt_worker.stop()
        sys.exit(exit_code)


if __name__ == "__main__":
    ReverseAIApp()
