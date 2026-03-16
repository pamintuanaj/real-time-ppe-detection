# Automated PPE Detection Using CNN-Based Object Detection

### Overview
This project develops an intelligent safety monitoring system designed to automatically detect Personal Protective Equipment (PPE) in workplace environments. By utilizing computer vision, the system identifies if workers are wearing essential safety gear like hard hats, safety vests, and face masks. 

To provide a comprehensive safety solution, the system integrates a **YOLOv8-based CNN** for visual detection, an **NLP module** to categorize safety logs, and a **Reinforcement Learning (RL) agent** that dynamically adjusts detection sensitivity. This multi-component approach helps safety officers maintain high compliance standards and mitigate occupational hazards in real-time.

### Components
* **Core Deep Learning Model:** YOLOv8 (You Only Look Once) optimized for real-time object detection and localization.
* **CNN Component:** A YOLOv8 CSPDarknet backbone used for multi-scale feature extraction from worksite imagery.
* **NLP Component:** A text classification module that analyzes written safety incident reports and logs to identify high-risk zones.
* **RL Component:** A Q-learning agent that dynamically tunes the model's confidence thresholds based on asymmetric costs (e.g., higher penalties for missing a hard hat vs. a false alarm).

### Dataset
* **Source:** [Roboflow Construction Site Safety Dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
* **Description:** This dataset contains annotated images from diverse industrial settings. It includes bounding box labels for multiple PPE classes, providing the variety needed for the model to generalize across different lighting and weather conditions.
* **Governance:** The data is used under an open-source academic license. All raw images are processed to ensure no personally identifiable information (PII) is stored in the repository.

### Evaluation Metrics
The system is assessed using the following metrics:
* **mAP@0.5 (Mean Average Precision):** Primary accuracy metric for object detection.
* **Precision-Recall (PR) Curves:** Used to evaluate the balance between detection accuracy and sensitivity.
* **Inference Latency:** Measured in frames per second (FPS) to ensure real-time viability.
* **RL Cumulative Reward:** Evaluates the efficiency of the threshold-tuning agent compared to a static baseline.
* **Macro-F1 Score:** Measures the performance of the auxiliary NLP log classifier.

### Team Roles
* **Project Lead / Integration (Pamintuan, Alexia):** Manages project scope, timeline, component integration, and final defense coordination.
* **Data & Ethics Lead (Lapid, John):** Oversees dataset sourcing, licensing, preprocessing, and ethical impact documentation.
* **Modeling Lead (Toribio, Dexter):** Responsible for model design (CNN, NLP, and RL), training pipelines, and error analysis.
* **Evaluation & MLOps Lead (Santos, Jhanrelle):** Manages performance metrics, reproducibility, environment configuration, and automation scripts.

### How to Run

#### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
