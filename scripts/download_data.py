from pathlib import Path


def main():
    base = Path("data")
    (base / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base / "images" / "val").mkdir(parents=True, exist_ok=True)
    (base / "images" / "test").mkdir(parents=True, exist_ok=True)

    (base / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "test").mkdir(parents=True, exist_ok=True)

    print("Dataset folders created.")
    print("Put your YOLO-format dataset inside the data folder.")


if __name__ == "__main__":
    main()