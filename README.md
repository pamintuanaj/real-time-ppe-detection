# Automated PPE Detection Using CNN-Based Object Detection

## Overview
This project develops an intelligent safety monitoring system designed to automatically detect Personal Protective Equipment (PPE) in workplace environments. Using computer vision, the system identifies whether workers are wearing essential safety gear such as hard hats, safety vests, and face masks.

The system is built around a **YOLOv8-based Convolutional Neural Network (CNN)** for real-time object detection. To extend functionality, the project also includes a **Natural Language Processing (NLP) module** for analyzing safety logs and a **Reinforcement Learning (RL) component** for adaptive decision-making.

At the current stage (Week 2), the system has a working end-to-end pipeline including:
- Dataset preprocessing and structuring  
- Exploratory Data Analysis (EDA)  
- YOLOv8 model training  
- Initial evaluation metrics  
- Real-time webcam detection  

---

## System Architecture

- **Core Detection Model:** YOLOv8 (Ultralytics)
- **Backbone:** CSPDarknet (for feature extraction)
- **Detection Type:** Multi-class object detection
- **Deployment Mode:** Real-time (OpenCV webcam)

---

## Components

### 🔹 CNN Component (YOLOv8)
- Performs object detection and localization
- Detects PPE classes such as:
  - Hard hats
  - Safety vests
  - Face masks
  - Non-compliance cases
- Optimized for real-time inference

---

### 🔹 NLP Component (Prototype)
- Classifies safety-related textual logs
- Intended for:
  - Incident classification
  - Risk identification
- Currently in early-stage implementation

---

### 🔹 RL Component (Stub)
- Designed to dynamically adjust detection thresholds
- Uses reward-based tuning (e.g., penalizing missed detections)
- Not yet integrated into the main pipeline

---

## Dataset

- **Source:**  
  https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

- **Description:**  
  The dataset consists of annotated images from construction and industrial environments. Each image contains bounding box labels in YOLO format for multiple PPE classes.

- **Preprocessing:**
  - Images resized to 640×640
  - Labels normalized
  - Dataset split into train/val/test

- **Data Split:**
  - 70% Training  
  - 15% Validation  
  - 15% Testing  

- **Ethical Considerations:**
  - No personally identifiable information (PII) stored  
  - Dataset used under open academic license  

---

## Exploratory Data Analysis (EDA)

EDA revealed key dataset characteristics:

- **Class Imbalance:**  
  Some classes (e.g., hardhat) dominate others (e.g., face_mask)

- **Small Object Detection Problem:**  
  Many objects occupy only 3–12% of image area

- **Lighting Variability:**  
  Dataset includes both bright and low-light images

These factors directly impact detection performance and guide future improvements.

---

## Model Training

The model was trained using pretrained YOLOv8 weights:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
