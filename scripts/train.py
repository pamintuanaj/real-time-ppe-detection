from pathlib import Path
import json
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME  = "yolov8n.pt"
DATA_YAML   = "data/data.yaml"
EPOCHS      = 150
IMAGE_SIZE  = 640
BATCH_SIZE  = 16
PROJECT_DIR = "models"
RUN_NAME    = "ppe_yolov8"
PATIENCE    = 30
# ───────────────────────────────────────────────────────────────────────────────


def main():
    Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)

    print(f"Starting training: {EPOCHS} epochs, batch={BATCH_SIZE}, imgsz={IMAGE_SIZE}")
    print(f"Early stopping patience: {PATIENCE} epochs\n")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # ── Augmentation ──────────────────────────────────────────────────────
        mosaic=1.0,
        fliplr=0.5,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        shear=2.0,
    )

    print("\nRunning validation on best weights...")
    metrics = model.val(
        data=DATA_YAML,
        split="val"
    )

    metrics_dict = {
        "precision": float(getattr(metrics.box, "mp",    0.0)),
        "recall":    float(getattr(metrics.box, "mr",    0.0)),
        "mAP50":     float(getattr(metrics.box, "map50", 0.0)),
        "mAP50_95":  float(getattr(metrics.box, "map",   0.0))
    }

    metrics_path = Path(PROJECT_DIR) / RUN_NAME / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"  Precision  : {metrics_dict['precision']:.4f}")
    print(f"  Recall     : {metrics_dict['recall']:.4f}")
    print(f"  mAP50      : {metrics_dict['mAP50']:.4f}")
    print(f"  mAP50-95   : {metrics_dict['mAP50_95']:.4f}")
    print("="*50)

    map50 = metrics_dict["mAP50"]
    if map50 >= 0.80:
        print("✅ Target reached! Model is ready for live_cam.py testing.")
    elif map50 >= 0.75:
        print("🟡 Almost there. Add 300-500 more images and retrain.")
    elif map50 >= 0.65:
        print("⚠️  Decent. Add ~1000 more images focused on weak classes.")
    else:
        print("❌ Low accuracy. Add more data before deploying.")

    print(f"\nSaved metrics : {metrics_path}")
    print(f"Best weights  : {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    print(f"Plots         : {PROJECT_DIR}/{RUN_NAME}/")


if __name__ == "__main__":
    main()
