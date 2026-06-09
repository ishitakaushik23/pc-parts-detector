from ultralytics import YOLO


# Load trained model
model = YOLO("runs/detect/ram_detector-3/weights/best.pt")

# Predict on image
results = model("test.jpeg", show=True)

print(results)

input("Press Enter to close...")