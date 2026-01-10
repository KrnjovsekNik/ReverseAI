import sys
import base64
import json
import threading
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
MQTT_TOPIC = "camera/results"


# --------------------------------------------------
# MQTT WORKER (teče v threadu, UI dobi signal)
# --------------------------------------------------
class MqttWorker(QThread):
    frame_signal = Signal(dict)

    def __init__(self, host, topic):
        super().__init__()
        self.host = host
        self.topic = topic
        self.client = mqtt.Client()
        self.running = True

    def run(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("[MQTT] Povezan, subscribe na", self.topic)
                client.subscribe(self.topic)
            else:
                print("[MQTT] Napaka pri povezavi:", rc)

        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload.decode("utf-8"))
                # Pošlji UI-ju
                self.frame_signal.emit(data)
            except Exception as e:
                print("[MQTT] Napaka pri dekodiranju JSON:", e)

        self.client.on_connect = on_connect
        self.client.on_message = on_message

        while self.running:
            try:
                print("[MQTT] Povezujem na", self.host)
                self.client.connect(self.host, 1883, 60)
                self.client.loop_forever()
            except Exception as e:
                print("[MQTT] Napaka, ponovno povezovanje čez 2s:", e)
                time.sleep(2)

    def stop(self):
        self.running = False
        try:
            self.client.disconnect()
        except:
            pass


# --------------------------------------------------
# GLAVNI VIDEO WIDGET
# --------------------------------------------------
class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReverseAI")
        self.setStyleSheet("background-color: #e6f0ef;")
        self.resize(1280, 720)

        # slika + tekst stanje
        self.frame_rgb = None           # numpy array (H, W, 3)
        self.qimage = None              # QImage – držimo referenco!
        self.text = "Ni nevarnosti"
        self.danger = False

        # font in barve
        self.font = QFont("Helvetica", 20, QFont.Bold)

    def update_from_result(self, result: dict):
        """
        Slot, ki ga kliče MQTT worker (v UI threadu, preko signala)
        """
        try:
            img_b64 = result.get("image", None)
            if not isinstance(img_b64, str):
                print("[UI] Neveljaven ali manjkajoč 'image' field:", type(img_b64))
                return

            # base64 -> bytes
            try:
                img_bytes = base64.b64decode(img_b64)
            except Exception as e:
                print("[UI] base64 decode error:", e)
                return

            # bytes -> numpy -> cv2
            buf = np.frombuffer(img_bytes, np.uint8)
            if buf.size == 0:
                print("[UI] Prazen JPEG buffer")
                return

            frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                print("[UI] cv2.imdecode ni uspel (pokvarjen JPEG, len=", len(img_bytes), ")")
                return

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.frame_rgb = frame_rgb

            # pripravimo QImage in držimo referenco
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            self.qimage = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # detections / text logika
            detections = result.get("detections", [])
            if not isinstance(detections, list):
                detections = []

            if len(detections) == 0:
                self.text = "Ni nevarnosti"
                self.danger = False
            else:
                # podobna logika kot prej – vzamemo "najbolj nevarno"
                # privzeto prvi
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

            # zahtevaj repaint
            self.update()

        except Exception as e:
            print("[UI] Napaka v update_from_result:", e)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#e6f0ef"))

        if self.qimage is not None:
            # video slika
            widget_w = self.width()
            widget_h = self.height()
            scaled = self.qimage.scaled(widget_w, widget_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (widget_w - scaled.width()) // 2
            y = (widget_h - scaled.height()) // 2
            painter.drawImage(x, y, scaled)

        # tekst box
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self.font)

        # izračun širine/višine teksta
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(self.text)
        th = metrics.height()

        padding_x = 24
        padding_y = 16
        box_w = tw + padding_x
        box_h = th + padding_y

        x0 = 20
        y0 = 20

        bg = QColor("#e6f0ef")
        shadow = QColor("#cdd9d7")
        text_color = QColor("red") if self.danger else QColor("black")

        # shadow
        painter.setBrush(QBrush(shadow))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(x0 + 2, y0 + 2, box_w, box_h), 6, 6)

        # box
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(QRectF(x0, y0, box_w, box_h), 6, 6)

        # text
        painter.setPen(QPen(text_color))
        text_x = x0 + padding_x / 2
        text_y = y0 + padding_y / 2 + th * 0.7
        painter.drawText(text_x, text_y, self.text)


# --------------------------------------------------
# GLAVNI APP WRAPPER
# --------------------------------------------------
class ReverseAIApp:
    def __init__(self):
        self.qt_app = QApplication(sys.argv)
        self.win = VideoWidget()

        self.mqtt_worker = MqttWorker(MQTT_HOST, MQTT_TOPIC)
        self.mqtt_worker.frame_signal.connect(self.win.update_from_result)
        self.mqtt_worker.start()

        self.win.show()
        exit_code = self.qt_app.exec()

        # cleanup
        self.mqtt_worker.stop()
        sys.exit(exit_code)


if __name__ == "__main__":
    ReverseAIApp()
