# Automated PPE Detection Using YOLOv8, NLP, and Reinforcement Learning

## Overview
This project develops an edge-optimized, end-to-end AI system for detecting Personal Protective Equipment (PPE) in workplace environments using deep learning, natural language processing, and reinforcement learning.

The system is engineered as an advanced multi-agent pipeline combining:
* a **YOLOv8 Nano Convolutional Neural Network (CNN)** for high-speed spatial localization and real-time object detection  
* a **Natural Language Processing (NLP)** module for automated compliance reporting and translating bounding boxes into human-readable safety logs  
* a **Reinforcement Learning (RL)** Q-learning agent for dynamic confidence threshold adjustment based on environmental noise  

The goal is to provide a highly reliable, low-latency assistive tool for safety officers to monitor PPE compliance, reduce workplace risks, and enable real-time decision-making without invasive physical sensors.

---

## Features
* YOLOv8n-based real-time object detection optimized for edge computing 
* Precision detection of 3 critical PPE classes (Hard Hats, Safety Vests, Face Masks) utilizing a 10-class background awareness dataset 
* Aggressive data augmentation (HSV shift, rotation, flipping) for robust environmental adaptability 
* Real-time webcam inference using OpenCV  
* NLP text classifier for immediate safety log generation (e.g., "[HIGH RISK] Worker in Sector A missing Hard Hat") 
* RL-based dynamic threshold tuning to minimize false-negative safety misses during lighting changes 
* Multi-metric evaluation (mAP@0.5 = 0.821, Precision = 0.907, Recall = 0.747)
* Modular and extensible multi-agent system design  

---

## Project Structure
```text
real-time-ppe-detection/
│
├── data/
│   ├── images/
│   └── labels/
│
├── notebook/
│   └── 01_eda.ipynb
│
├── runs/
│   └── detect/
│       └── models/
│           └── ppe_yolov8/
│
├── scripts/
│   ├── train.py
│   ├── live_cam.py
│   ├── baseline_nondl.py
│   ├── nlp_component.py
│   └── rl_qlearning.py
│ 
├── models/
│
├── experiments/
│   └── results/
│
├── docs/
│   └── FinalReport.pdf
│
├── requirements.txt
├── run.bat
└── README.md
InstallationBashgit clone [https://github.com/pamintuanaj/real-time-ppe-detection.git](https://github.com/pamintuanaj/real-time-ppe-detection.git)
cd real-time-ppe-detection
Bashpython -m pip install --upgrade pip
pip install -r requirements.txt
pip install ultralytics torch
Dataset SourcesThis project utilizes publicly available datasets from Kaggle for the training and evaluation of PPE detection models, ensuring generalized global background awareness.Dataset CharacteristicsTotal Images: 3,088 images of real-world construction and industrial environmentsFormat: Normalized YOLO annotation format (.txt)Classes: 10 distinct environmental classes, with an operational focus on Hard Hats, Safety Vests, and Face Masks.Purpose: Designed for object detection tasks in workplace safety monitoring to combat severe class imbalance and occlusion.Privacy StatementThis dataset contains no high-resolution facial biometrics or Personally Identifiable Information (PII). Video feeds are processed entirely inferentially in volatile RAM.PreprocessingImages aggressively resized via interpolation to 640×640Labels mathematically converted to YOLO format (<object-class> <x-center> <y-center> <width> <height>)Dataset deterministically split (Seed 42): Train (2,059) / Validation (555) / Test (474)Exploratory Data AnalysisNotebook:notebook/01_eda.ipynb
Includes:class distributionsample image visualizationbounding box analysisbrightness distributionKey findings:severe statistical class imbalances (e.g., many hard hats, fewer face masks)varying lighting conditions requiring data augmentationvisual complexity and overlapping textures (occlusion)Model TrainingTraining using YOLOv8 Nano:Pythonfrom ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=150,
    patience=30,
    imgsz=640,
    batch=16,
    seed=42,
    workers=8
)
Evaluation Metrics (Final Test Set Results)Metric       Value Precision   0.9070Recall       0.7470mAP@0.5     0.8210InterpretationClassical ML Baseline (baseline_nondl.py) yielded an abysmal 0.4979 accuracy, proving deep learning's necessity.The model exhibits extreme statistical reliability for high-contrast items (Face Masks mAP = 0.920, Safety Vests mAP = 0.809).Boundary limitations primarily exist in background false negatives associated with "NO-Hardhat" instances due to severe physical occlusion or extreme distance.Real-Time DetectionTo autonomously execute the full multi-agent pipeline, use the master batch script:Bash.\run.bat
Or run the live camera detection module manually:Pythonimport cv2
from ultralytics import YOLO

model = YOLO("runs/detect/models/ppe_yolov8/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("PPE Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
NLP ComponentScript:scripts/nlp_component.py
Approach:rule-based tokenizer paired with a lightweight text classifierparses bounding box intersections to retrieve context-appropriate warningsPurpose:synthesize structural detection data into immediate, human-readable safety logsensure analytical outputs are immediately actionable for non-technical safety personnelReinforcement Learning ComponentScript:scripts/rl_qlearning.py
Approach:Q-learning agent utilizing an asymmetric reward functionautonomously adjusts the CNN's minimum confidence thresholdStatus:Fully functionalRapidly converges to prioritize safety recall over absolute precision in visually noisy frames (e.g., sudden shadows)System ArchitecturePlaintextImage Frame → YOLOv8 Nano Detection
            ↓
   Bounding Boxes + Confidence Tensors
            ↓
   RL Q-Learning Threshold Adjustment
            ↓
 Final Decision → NLP Text Classification
Ethical Considerationssystem is strictly an assistive decision-support tool, requiring human verificationno algorithmic capability for behavioral tracking or productivity micromanagementprocesses spatial frames strictly within volatile RAM; frames are immediately discardedno personally identifiable information (PII) captured, archived, or storedTeam MembersLapid, John Vincent Y. (lapidjohnvincent@gmail.com)Pamintuan, Alexia John D. (rose.pamintuan123@gmail.com)Santos, Jhanrelle L. (jhanrellelucero@gmail.com)Toribio, Dexter Christian C. (toribiodexterc@gmail.com)Holy Angel University, Angeles City, Pampanga, PhilippinesFinal NotesThis project demonstrates:real-time spatial object detection using a YOLOv8n convolutional backbonerigorous data augmentation and ablation studies for environmental robustnessNLP reporting agents for structural interpretabilityreinforcement learning Q-tables for dynamic threshold optimizationforming a meticulously engineered, edge-deployable multi-agent safety monitoring system. For the complete methodology, ablation studies, and comprehensive academic references, please refer to docs/FinalReport.pdf.
