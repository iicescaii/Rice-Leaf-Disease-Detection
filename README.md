# Rice Leaf Disease Detection System

This repository contains an AI-powered solution designed to help local farmers in the Philippines identify rice leaf diseases early. The project integrates **Object Detection** and **Image Classification** to provide accurate, real-time diagnostics, minimizing crop loss and ensuring food security.

---

## 🚀 Project Overview

Rice local farmers in the Philippines face significant challenges due to disease outbreaks that lead to yield loss and food insecurity. This project implements a two-stage pipeline to address these issues:

* **Object Detection:** Identifying and accurately localizing rice leaves in images using **YOLOv8n**.
* **Disease Classification:** Classifying detected leaf regions into 9 specific disease categories using a **DenseNet-121** Convolutional Neural Network (CNN).

---

## 📊 Performance Metrics

The system demonstrates high reliability across both stages of the pipeline:

Stage 1: Rice Leaf Detection (YOLOv8n) 

* **Training Accuracy:** 97.20% 
* **Test Accuracy:** 96.0% 
* **mAP50:** 0.993 (Overall)

Stage 2: Disease Classification (DenseNet-121) 

* **Training Accuracy:** 96%
* **Validation Accuracy:** 98%
* **Test Accuracy:** 99% 
* **Filtering:** Using a confidence threshold of **0.6**, the model filters out low-confidence predictions to ensure diagnostic precision.
---

## 🛠️ Technology Stack

* **Algorithms:** YOLOv8n (Object Detection) and DenseNet-121 (CNN for Classification).
* **Frameworks:** TensorFlow, Keras, Ultralytics.
* **Deployment:** User-friendly web-based application.
* **Data Source:** Kaggle Rice Leaf and Disease Datasets.

---

## 📈 Dataset Information

The models were trained on extensive datasets with heavy use of **Data Augmentation** (flipping, rotation, zooming, shearing) to ensure adaptability to varying field conditions.

| Model Stage | Total Images | Key Classes |
| --- | --- | --- |
| **Model 1 (Detection)** | 22,531 images | Rice Leaf vs. Non-Rice |
| **Model 2 (Classification)** | 12,242 images | 9 Disease Classes (e.g., Bacterial Leaf Blight, Brown Spot, Leaf Blast, etc.) |

---

## ⚙️ Installation & Usage

### 1. Environment Setup

```python
# Install YOLOv8 and Roboflow
!pip install ultralytics==8.2.103 -q
!pip install roboflow

```

### 2. Training Stage 1 (Detection)

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.train(data="data.yaml", epochs=50, imgsz=640)

```

### 3. Training Stage 2 (Classification)

The classification stage uses a pretrained **DenseNet-121** base with custom dense layers:

```python
from keras.applications import DenseNet121
conv_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
conv_base.trainable = False

```

---

## 💡 Recommendations for Future Work

* **Mobile Integration:** Deploying as a mobile app with offline functionality and multilingual support for accessibility in remote areas.
* **Treatment Advice:** Adding real-time detection features and instant treatment advice.
* **Diverse Data:** Further training on images captured under varying lighting, weather, and growth stages.

Would you like me to generate a specific `requirements.txt` file or a contribution guide for this repository?
