from pathlib import Path
import json
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
DATA_YAML = "data/data.yaml"
EPOCHS = 5
IMAGE_SIZE = 640
BATCH_SIZE = 16
PROJECT_DIR = "models"
RUN_NAME = "ppe_yolov8"

def main():
    Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True
    )

    metrics = model.val(
        data=DATA_YAML,
        split="val"
    )

    metrics_dict = {
        "precision": float(getattr(metrics.box, "mp", 0.0)),
        "recall": float(getattr(metrics.box, "mr", 0.0)),
        "mAP50": float(getattr(metrics.box, "map50", 0.0)),
        "mAP50_95": float(getattr(metrics.box, "map", 0.0))
    }

    metrics_path = Path(PROJECT_DIR) / RUN_NAME / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    print("Training complete.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Best weights should be in: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")

if __name__ == "__main__":
    main()