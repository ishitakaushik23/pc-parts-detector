from ultralytics import YOLO

model = YOLO("trained_detector_v1.pt")

results = model(
    "test.jpeg",
    conf=0.25,
    save=True
)

for r in results:
    print("Detected objects:", len(r.boxes))