from pathlib import Path
import json
from ultralytics import YOLO

MODEL_NAME  = "yolov8n.pt"
DATA_YAML   = "data/data.yaml"
EPOCHS      = 100   # LAPID
IMAGE_SIZE  = 640
BATCH_SIZE  = 16
PROJECT_DIR = "models"
RUN_NAME    = "ppe_yolov8_baseline_dl"
PATIENCE    = 30  # LAPID

def main():
    Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)

# ----- ADDED LAPID
    print(f"Starting training: {EPOCHS} epochs, batch={BATCH_SIZE}, imgsz={IMAGE_SIZE}")
    print(f"Early stopping patience: {PATIENCE} epochs\n")
# ----
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE, # LAPID
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

# ---- LAPID
    )

    print("\nRunning validation on best weights...")
    metrics = model.val(
        data=DATA_YAML,
        split="val"
    )

    metrics_dict = {
        "precision":float(getattr(metrics.box, "mp",    0.0)),
        "recall":   float(getattr(metrics.box, "mr",    0.0)),
        "mAP50":    float(getattr(metrics.box, "map50", 0.0)),
        "mAP50_95": float(getattr(metrics.box, "map",   0.0))
    }

    metrics_path = Path(PROJECT_DIR) / RUN_NAME / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    # LAPID
    # ── Results Summary ────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"  Precision  : {metrics_dict['precision']:.4f}")
    print(f"  Recall     : {metrics_dict['recall']:.4f}")
    print(f"  mAP50      : {metrics_dict['mAP50']:.4f}")
    print(f"  mAP50-95   : {metrics_dict['mAP50_95']:.4f}")
    print("="*50)

    # ── Advice based on mAP50 ─────────────────────────────────────────────────
    map50 = metrics_dict["mAP50"]
    if map50 >= 0.80:
        print("✅ Target reached! Model is ready for live_cam.py testing.")
    elif map50 >= 0.75:
        print("🟡 Almost there. Add more images and retrain.")
    elif map50 >= 0.65:
        print("⚠️  Decent result. Consider adding more images and retraining.")
    else:
        print("❌ Low accuracy. Add more training data before deploying.")

    print(f"\nSaved metrics : {metrics_path}")
    print(f"Best weights  : {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    print(f"Confusion matrix and plots: {PROJECT_DIR}/{RUN_NAME}/")
    # LAPID

    '''****REMOVED******* # LAPID
    print("Training complete.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Best weights should be in: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")

    '''

if __name__ == "__main__":
    main()