from ultralytics import YOLO
import gradio as gr

model = YOLO("trained_detector_v1.pt")

def detect(image):

    results = model(image)

    annotated = results[0].plot()

    detections = []
    counts = {}

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        name = model.names[cls]

        detections.append(
            f"{name:<10} {conf:.2%}"
        )

        counts[name] = counts.get(name, 0) + 1

    summary = "Detected Components\n"
    summary += "─────────────────\n\n"

    if detections:
        summary += "\n".join(detections)

        summary += "\n\nTotal Count\n"

        for k, v in counts.items():
            summary += f"\n{k}: {v}"

    else:
        summary += "No components detected"

    return annotated, summary

    


app = gr.Interface(
    fn=detect,
    inputs=gr.Image(
    type="numpy",
    sources=["upload", "webcam"]
),
    outputs=[
    gr.Image(label="Detection Result"),
    gr.Textbox(
        label="Analysis",
        lines=12
    )
],
    title="PC Parts Detector"
)

app.launch()