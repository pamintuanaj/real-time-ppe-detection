import cv2
import json
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

IMG_SIZE = (64, 64)


def get_main_class(label_path: Path):
    if not label_path.exists():
        return None

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    first_line = text.splitlines()[0].strip()
    parts = first_line.split()

    if not parts:
        return None

    return int(parts[0])


def load_dataset(image_dir: str, label_dir: str):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    X, y = [], []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue

        label_path = label_dir / f"{img_path.stem}.txt"
        label = get_main_class(label_path)

        if label is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feat = gray.flatten()

        X.append(feat)
        y.append(label)

    return np.array(X), np.array(y)


def main():
    X_train, y_train = load_dataset("data/images/train", "data/labels/train")
    X_val, y_val = load_dataset("data/images/val", "data/labels/val")
    X_test, y_test = load_dataset("data/images/test", "data/labels/test")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Train, validation, or test dataset is empty.")

    print("Train samples:", len(X_train))
    print("Val samples:", len(X_val))
    print("Test samples:", len(X_test))

    print("Train classes:", np.unique(y_train))
    print("Val classes:", np.unique(y_val))
    print("Test classes:", np.unique(y_test))

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"Need at least 2 classes in training data, but found only: {np.unique(y_train)}"
        )

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(max_iter=5000, random_state=42)
    )

    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    test_preds = clf.predict(X_test)

    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)

    val_report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
    test_report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)

    output_dir = Path("models/baseline_ml")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "val_report": val_report,
        "test_report": test_report,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "train_classes": [int(x) for x in np.unique(y_train)],
        "val_classes": [int(x) for x in np.unique(y_val)],
        "test_classes": [int(x) for x in np.unique(y_test)]
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nBaseline ML complete.")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()