from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/ram_detector-3/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw boxes
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("RAM Detector", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()