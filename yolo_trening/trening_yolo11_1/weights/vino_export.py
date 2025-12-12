from ultralytics import YOLO

model = YOLO("best.pt", task="detect")
#model.export(format="openvino")

model.export(format="openvino", int8=True)