# Model Card — Automated PPE Detection System

---

## 1. Model Overview

This project presents a Personal Protective Equipment (PPE) detection system built using **YOLOv8 Nano (yolov8n)** fine-tuned for object detection. The model classifies worksite images into 10 categories and is supported by auxiliary components to improve reporting and threshold logic.

The system integrates three AI components:
* **CNN (YOLOv8 Nano)** for real-time bounding box localization and classification.
* **NLP module (Logistic Regression + TF-IDF)** for generating human-readable safety incident logs.
* **Reinforcement Learning (Q-Learning)** for dynamic confidence threshold optimization.

**Input:**
* RGB worksite image or video frame.

**Output:**
* Localized bounding boxes with class labels.
* Confidence scores.
* Formatted NLP safety incident report.

**Classes:**
* Person, Hardhat, Safety Vest, Mask, Machinery
* NO-Hardhat, NO-Safety Vest, NO-Mask (and related compliance variants).

---

## 2. Intended Use

This system is designed as a **decision-support tool** for worksite safety monitoring.

It may be useful for:
* Construction site safety officers.
* Industrial zone compliance monitoring.
* Educational demonstrations of multi-agent ML systems.

⚠️ The system is **not intended for autonomous disciplinary enforcement**, biometric surveillance, or medical-grade particulate mask detection.

---

## 3. Training Data

The model was trained on a dataset of 1,000 worksite images sourced from open repositories.

### Preprocessing & Data Governance:
* Manual audit confirmed the absence of Personally Identifiable Information (PII) such as recognizable faces or nametags.
* Strict 70% Train, 15% Validation, 15% Test split to prevent data leakage.

### Addressing Imbalance:
* **Mosaic Augmentation** was utilized heavily during training to force the model to learn small-object features and address the spatial dominance of "Person" bounding boxes compared to "Mask" boxes.

---

## 4. Evaluation

### Overall Performance (Test Set)
| Metric | Value |
|---|---|
| mAP@0.5 | **0.551** |
| Max F1-Score | **0.52** (at threshold 0.297) |

### Slice Analysis (Per-Class Performance)
The model exhibits a stark contrast between macro and micro features:
* **High-Performing Classes (Macro-Features):**
  * Person: 0.835 mAP
  * Hardhat: 0.763 mAP
  * Safety Vest: 0.758 mAP
* **Low-Performing Classes (Micro-Features):**
  * Mask: 0.323 mAP
  * NO-Mask: 0.339 mAP

---

## 5. Error Analysis & Limitations

A granular review of the normalized confusion matrix identifies **Background False Negatives** as the primary failure mode.
* The model incorrectly classified **58% of actual "Mask" instances** as background.
* **Root Cause:** Standard YOLOv8 Nano pooling layers inherently lose fine-grained spatial resolution, causing micro-features to vanish before the detection head.

---

## 6. Ablation Study

An ablation study was conducted to evaluate the impact of spatial augmentations on resolving the micro-feature limitation.

| Experiment | Configuration | Observation |
|---|---|---|
| **Baseline** | Mosaic Augmentation ON | Smoother loss convergence; better generalization on occluded workers. |
| **Ablation** | Mosaic Augmentation OFF | Degraded ability to detect smaller, clustered objects. |

👉 This confirms that Mosaic augmentation is critical for worksite datasets containing extreme scale variations.

---

## 7. Reinforcement Learning Component

A **Q-learning agent** was deployed to optimize the YOLOv8 confidence threshold dynamically.

* **Purpose:** To adjust thresholds based on simulated environmental states (e.g., lowering the threshold in poor lighting).
* **Reward Design:** Heavily penalizes false negatives in hazardous conditions, ensuring the system prioritizes worker safety (detecting a potential missing helmet) over pure statistical precision.

---

## 8. NLP Explanation Module

An NLP component (Logistic Regression + TF-IDF) bridges the gap between raw arrays and actionable logs.
* **Behavior:** Ingests bounding box class data (e.g., "NO-Hardhat") and synthesizes it into standardized alerts.
* **Example Output:** `[HIGH_RISK REPORT]: Missing hardhat detected in active construction zone. Require immediate visual verification.`

---

## 9. Future Work

* Implement **SAHI (Slicing Aided Hyper Inference)** to crop large images before detection, resolving the small-object (Mask) failure mode.
* Increase input tensor resolution (`imgsz`).
* Expand dataset with localized, region-specific worksite data.
