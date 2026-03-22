import cv2
from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = "runs/detect/models/ppe_yolov8/weights/best.pt"  #MODEL_PATH = "yolov8n.pt" "models/ppe_yolov8/weights/best.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.25
WINDOW_NAME = "PPE Live Detection"
SAVE_VIDEO = False
OUTPUT_VIDEO = "models/live_output.mp4"


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    writer = None

    if SAVE_VIDEO:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("Live camera started.")
    print("Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break

        results = model(frame, conf=CONFIDENCE, verbose=False)
        annotated_frame = results[0].plot()

        if writer is not None:
            writer.write(annotated_frame)

        cv2.imshow(WINDOW_NAME, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()