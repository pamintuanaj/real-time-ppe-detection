"""
prepare_dataset.py
------------------
Step 1 — Cleans your dataset (removes corrupt images, unmatched labels)
Step 2 — Splits into 70% train / 15% val / 15% test
Step 3 — Generates data.yaml automatically

Usage:
    python prepare_dataset.py

Expected input folder structure BEFORE running:
    data/
    ├── images/   (all your images in one flat folder)
    └── labels/   (all your YOLO .txt labels in one flat folder)

Output folder structure AFTER running:
    data/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_IMAGES_DIR = Path("data/images")
RAW_LABELS_DIR = Path("data/labels")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42  # same seed = same split every time (reproducible)

CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "Machinery",
    "Vehicle"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ───────────────────────────────────────────────────────────────────────────────


def clean_dataset(images_dir: Path, labels_dir: Path):
    print("\n" + "="*55)
    print("STEP 1 — CLEANING DATASET")
    print("="*55)

    image_files = {
        f.stem: f
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    }

    label_files = {
        f.stem: f
        for f in labels_dir.iterdir()
        if f.is_file() and f.suffix == ".txt"
    }

    print(f"  Found {len(image_files)} image(s)")
    print(f"  Found {len(label_files)} label(s)")

    removed_corrupt  = 0
    removed_no_label = 0
    removed_no_image = 0
    removed_empty    = 0

    # ── Remove corrupt images ─────────────────────────────────────────────────
    corrupt_stems = []
    for stem, img_path in image_files.items():
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            print(f"  [CORRUPT]    Removing: {img_path.name}")
            img_path.unlink(missing_ok=True)
            corrupt_stems.append(stem)
            removed_corrupt += 1

    for stem in corrupt_stems:
        image_files.pop(stem, None)
        lbl = labels_dir / f"{stem}.txt"
        if lbl.exists():
            lbl.unlink()

    # ── Remove images with no matching label ──────────────────────────────────
    no_label_stems = []
    for stem, img_path in image_files.items():
        if stem not in label_files:
            print(f"  [NO LABEL]   Removing: {img_path.name}")
            img_path.unlink(missing_ok=True)
            no_label_stems.append(stem)
            removed_no_label += 1

    for stem in no_label_stems:
        image_files.pop(stem, None)

    # ── Remove labels with no matching image ──────────────────────────────────
    no_image_stems = []
    for stem, lbl_path in label_files.items():
        if stem not in image_files:
            print(f"  [NO IMAGE]   Removing: {lbl_path.name}")
            lbl_path.unlink(missing_ok=True)
            no_image_stems.append(stem)
            removed_no_image += 1

    for stem in no_image_stems:
        label_files.pop(stem, None)

    # ── Remove empty label files ──────────────────────────────────────────────
    empty_stems = []
    for stem, lbl_path in label_files.items():
        if lbl_path.stat().st_size == 0:
            print(f"  [EMPTY LBL]  Removing: {lbl_path.name} and its image")
            lbl_path.unlink(missing_ok=True)
            img_path = image_files.get(stem)
            if img_path and img_path.exists():
                img_path.unlink(missing_ok=True)
            empty_stems.append(stem)
            removed_empty += 1

    for stem in empty_stems:
        image_files.pop(stem, None)
        label_files.pop(stem, None)

    valid_pairs = [
        (image_files[stem], label_files[stem])
        for stem in image_files
        if stem in label_files
    ]

    print(f"\n  Removed corrupt images  : {removed_corrupt}")
    print(f"  Removed no-label images : {removed_no_label}")
    print(f"  Removed no-image labels : {removed_no_image}")
    print(f"  Removed empty labels    : {removed_empty}")
    print(f"  Valid pairs remaining   : {len(valid_pairs)}")

    return valid_pairs


def split_and_copy(valid_pairs: list):
    print("\n" + "="*55)
    print("STEP 2 — SPLITTING  70 / 15 / 15")
    print("="*55)

    random.seed(SEED)
    random.shuffle(valid_pairs)

    total     = len(valid_pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end   = train_end + int(total * VAL_RATIO)

    splits = {
        "train": valid_pairs[:train_end],
        "val":   valid_pairs[train_end:val_end],
        "test":  valid_pairs[val_end:]
    }

    for split_name, pairs in splits.items():
        pct = len(pairs) / total * 100
        print(f"  {split_name:<6} : {len(pairs)} images ({pct:.1f}%)")

    print("\n" + "="*55)
    print("STEP 3 — COPYING FILES")
    print("="*55)

    for split_name, pairs in splits.items():
        img_out = Path(f"data/images/{split_name}")
        lbl_out = Path(f"data/labels/{split_name}")
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

        print(f"  Copied {len(pairs)} pairs → {img_out}")

    return splits


def generate_yaml():
    print("\n" + "="*55)
    print("STEP 4 — GENERATING data.yaml")
    print("="*55)

    yaml_content = f"""# Auto-generated by prepare_dataset.py
path: data

train: images/train
val:   images/val
test:  images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path = Path("data/data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"  Saved to: {yaml_path}")


def main():
    print("\n" + "="*55)
    print("PPE DATASET PREPARE SCRIPT")
    print("Split: 70% train / 15% val / 15% test")
    print("="*55)

    # Validate raw folders exist
    if not RAW_IMAGES_DIR.exists() or not RAW_LABELS_DIR.exists():
        print("\nERROR: Could not find data/images or data/labels folders.")
        print("Make sure all your images are in:  data/images/")
        print("And all your labels are in:        data/labels/")
        return

    # Step 1 — Clean
    valid_pairs = clean_dataset(RAW_IMAGES_DIR, RAW_LABELS_DIR)

    if len(valid_pairs) == 0:
        print("\nERROR: No valid image-label pairs found after cleaning.")
        return

    # Step 2 & 3 — Split and copy
    splits = split_and_copy(valid_pairs)

    # Step 4 — Generate YAML
    generate_yaml()

    # Final summary
    train_count = len(splits["train"])
    val_count   = len(splits["val"])
    test_count  = len(splits["test"])
    total       = train_count + val_count + test_count

    print("\n" + "="*55)
    print("DONE")
    print("="*55)
    print(f"  Total  : {total}")
    print(f"  Train  : {train_count}")
    print(f"  Val    : {val_count}")
    print(f"  Test   : {test_count}")
    print(f"\n  Next step: python train.py")
    print("="*55)


if __name__ == "__main__":
    main()
