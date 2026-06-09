from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train model on your dataset
model.train(
    data="Ram/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="ram_detector"
)