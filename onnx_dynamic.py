from ultralytics import YOLO
model = YOLO("./yolov8n-face.pt")
model.export(format="onnx",simplify=True,dynamic=True)
