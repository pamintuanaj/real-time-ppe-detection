# Model Card: Automated PPE Detection System

## 1. Model Overview
* **Architecture:** Multi-Agent Pipeline (YOLOv8 Nano CNN -> Q-Learning RL Agent -> TF-IDF/LogReg NLP Classifier).
* **Task:** Multi-class object detection and automated safety reporting.
* **Input:** 640x640 RGB image/video frames.
* **Classes (10):** Person, Hardhat, Safety Vest, Mask, Machinery, Safety Cone, Vehicle, NO-Hardhat, NO-Safety Vest, NO-Mask.

## 2. Intended Use
* **Primary Use Case:** Real-time compliance monitoring on construction and industrial worksites.
* **Prohibited Use:** Individual worker productivity tracking, biometric surveillance, or autonomous disciplinary enforcement without human review.

## 3. Training Data & Governance
* **Dataset:** 1,000+ open-source worksite images.
* **Split:** 70% Training / 15% Validation / 15% Test.
* **Governance:** Manual dataset audit confirmed the complete absence of Personally Identifiable Information (PII) such as recognizable faces or persistent facial IDs.

## 4. Quantitative Evaluation (Validation Split)
Evaluated using YOLOv8's native metrics, verified by `results.csv`:

| Metric | Value |
| :--- | :--- |
| **Overall mAP@0.5** | **0.821 (82.1%)** |
| **Precision (P)** | **0.907 (90.7%)** |
| **Recall (R)** | **0.747 (74.7%)** |

**Slice Analysis (Per-Class Performance):**
* **Macro-Reliability:** The model demonstrates near-perfect localization for large objects (**Machinery: 94.7% mAP**, **Person: 89.9% mAP**).
* **Micro-Feature Limitations:** While Masks achieved a high **92% mAP**, the **"NO-Mask"** compliance class dropped to **68.2%**. This highlights the difficulty of verifying the *absence* of gear in cluttered, dynamic scenes.

## 5. Error Analysis: Background False Negatives
A review of the confusion matrix identifies **Background Confusion** as a primary failure mode. Standard pooling layers in the Nano architecture lose fine-grained spatial resolution for tiny objects, occasionally causing the model to treat safety gear as background noise.
