# Experiment Results & Methodology

## 1. Baseline Comparative Analysis
To establish a performance floor and justify the use of deep learning, a classical baseline was developed using traditional Machine Learning.
* **Methodology:** Support Vector Classifier (SVC) applied to flattened Histogram of Oriented Gradients (HOG) features.
* **Results:** The SVM achieved an accuracy of **0.3333 (33.3%)** and a Macro-F1 of **0.2500**.
* **Analysis:** Classical sliding-window techniques lack the spatial hierarchy necessary to process multi-object occlusion. The model effectively defaulted to majority-class predictions.

## 2. Deep Learning Core (YOLOv8 Nano)
The core architecture was upgraded to YOLOv8 Nano (`yolov8n.pt`) and trained for 150 epochs using SGD/AdamW.
* **Data Augmentation:** Mosaic and MixUp augmentations were utilized to improve small-object detection and variety.
* **Results:** The CNN achieved an overall **mAP@0.5 of 0.821 (82.1%)** and a Precision of **0.907**.
* **Analysis:** The CNN successfully mapped complex overlapping features, vastly outperforming the SVM baseline. Loss convergence was smooth, indicating that Mosaic augmentation prevented early overfitting.

## 3. Reinforcement Learning Optimization
A **Q-learning agent** was integrated to solve the "static threshold" problem.
* **Methodology:** The agent was trained over 100 episodes across 4 simulated worksite states.
* **Results:** The Q-table (`q_table.json`) converged with values between **7.58 and 9.88**.
* **Impact:** In simulated poor-visibility states, the agent learned to drop the confidence threshold (from 0.5 to ~0.3). This improved the system's **Safety Recall**, ensuring alerts are triggered even when environmental quality degrades.

## 4. Semantic Translation (NLP Classifier)
A TF-IDF based Logistic Regression model was trained to synthesize raw bounding box arrays into human-readable text.
* **Benefit:** This provides **Semantic Explainability**, allowing non-technical safety officers to understand the *intent* of the AI alert without needing to interpret raw coordinate data.
* **Accuracy:** 100% on synthetic test validations.
