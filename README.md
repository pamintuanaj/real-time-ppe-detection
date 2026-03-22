# Automated PPE Detection Using YOLOv8, NLP, and Reinforcement Learning

## Overview
This project develops an end-to-end AI system for detecting Personal Protective Equipment (PPE) in workplace environments using deep learning, natural language processing, and reinforcement learning.

The system combines:
* a **YOLOv8-based Convolutional Neural Network (CNN)** for real-time object detection  
* a **Natural Language Processing (NLP)** module for analyzing safety logs  
* a **Reinforcement Learning (RL)** component for adaptive threshold optimization  

The goal is to assist safety officers in monitoring PPE compliance, reducing workplace risks, and enabling real-time decision-making.

---

## Features
* YOLOv8-based real-time object detection  
* Detection of PPE classes (helmet, vest, mask, non-compliance)  
* EDA pipeline with dataset visualization  
* Real-time webcam detection using OpenCV  
* NLP prototype for safety log classification  
* RL-based threshold tuning (stub)  
* Multi-metric evaluation (Precision, Recall, mAP)  
* Modular and extensible system design  

---

## Project Structure
```

real-time-ppe-detection/
│
├── data/
│   ├── images/
│   └── labels/
│
├── notebook/
│   └── 01_eda.ipynb
│
├── runs/
│   └── detect/
│       └── models/
│           └── ppe_yolov8/
│
├── scripts/
│   ├── detection.py
│   ├── nlp_component_stub.py
│   └── rl_stub.py
│
├── models/
│
├── experiments/
│   └── results/
│
├── docs/
│   └── checkpoint_report.pdf
│
├── requirements.txt
└── README.md

````

---

## Installation
```bash
git clone https://github.com/pamintuanaj/real-time-ppe-detection.git
cd real-time-ppe-detection
````

```bash
pip install -r requirements.txt
```

---

## Dataset

### Source

[https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)

### Description

The dataset contains annotated images from construction and industrial environments with bounding box labels for PPE classes such as hard hats, safety vests, and face masks.

### Preprocessing

* Images resized to 640×640
* Labels normalized (YOLO format)
* Dataset split into train / validation / test

---

## Exploratory Data Analysis

Notebook:

```
notebook/01_eda.ipynb
```

Includes:

* class distribution
* sample image visualization
* bounding box analysis
* brightness distribution

Key findings:

* class imbalance
* small-object detection problem
* varying lighting conditions

---

## Model Training

Training using YOLOv8:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

---

## Evaluation Metrics (Week 2 Results)

| Metric       | Value  |
| ------------ | ------ |
| Precision    | 0.5455 |
| Recall       | 0.3967 |
| mAP@0.5      | 0.4239 |
| mAP@0.5:0.95 | 0.2315 |

### Interpretation

* moderate precision
* lower recall (missed detections)
* expected improvement with more training

---

## Real-Time Detection

Run:

```bash
python scripts/detection.py
```

Or:

```python
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("PPE Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## NLP Component

Script:

```
scripts/nlp_component_stub.py
```

Approach:

* text classification of safety logs
* keyword / TF-IDF based prototype

Purpose:

* identify risk patterns
* support safety reporting

---

## Reinforcement Learning Component

Script:

```
scripts/rl_stub.py
```

Approach:

* Q-learning (conceptual)
* threshold tuning for detection
* reward-based optimization

Status:

* prototype stage

---

## System Architecture

```
Image → YOLOv8 Detection
            ↓
   Bounding Boxes + Classes
            ↓
   RL Threshold Adjustment
            ↓
 Final Decision → NLP Analysis
```

---

## Ethical Considerations

* system is a decision-support tool
* sensitive to lighting and dataset bias
* not a replacement for human supervision
* no personally identifiable information stored

---

## Team Members

* Lapid, John Vincent Y.
* Pamintuan, Alexia John D.
* Santos, Jhanrelle L.
* Toribio, Dexter Christian C.

---

## Final Notes

This project demonstrates:

* real-time object detection using YOLOv8
* data analysis through EDA
* NLP for interpretability
* reinforcement learning for optimization

forming a modular AI-based safety monitoring system.

```
