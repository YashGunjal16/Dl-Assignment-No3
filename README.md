# Dl-Assignment-No3
# 🚀 YOLOv11 Object Detection on COCO Dataset

This repository contains the code and documentation for a Google Colab lab assignment implementing YOLOv11 for real-time object detection using the COCO dataset. The project covers environment setup, dataset acquisition and preprocessing via Roboflow, model training, inference, and performance evaluation.

---


## 📌 Overview

This assignment focuses on implementing and evaluating a YOLOv11 model for object detection. The project is executed on Google Colab, where the required libraries (Roboflow, Ultralytics, PyTorch, OpenCV, etc.) are installed. The COCO dataset is acquired via Roboflow, preprocessed, and then used to train a YOLOv11 model. Finally, the trained model is tested on unseen data, and its performance is evaluated using metrics such as Mean Average Precision (mAP), Precision, Recall, and F1 Score.

---

## 📂 Dataset

### Dataset Acquisition

- **Source:** The COCO dataset, one of the most widely used datasets for object detection tasks.
- **Acquisition Method:**  
  The dataset is downloaded using the Roboflow API.  
  Example:
  ```python
  from roboflow import Roboflow
  rf = Roboflow(api_key="sLpQp9tNRxVlPd1zmIqo")
  project = rf.workspace("microsoft").project("coco")
  version = project.version(34)
  dataset = version.download("yolov11")
  ```

### Dataset Structure and Characteristics

- **Structure:**  
  After downloading, the dataset is organized in a YOLO-friendly format with separate directories for training, validation, and testing:
  - `train/`: Contains training images and YOLO-format labels.
  - `valid/`: Contains validation images and labels.
  - `test/`: Contains test images for inference.
- **Characteristics:**  
  - **Classes:** 80 object categories.
  - **Annotations:** Bounding boxes with class labels in YOLO format (normalized coordinates).
  - **Preprocessing:** Minimal preprocessing is required; however, verifying image sizes and annotations is crucial for the training pipeline.

---

## ⚙️ Methodology

1. **Environment Setup and Installation**
   - Set up your Google Colab environment by installing required libraries such as Roboflow, Ultralytics, PyTorch, and OpenCV.
   - Verify that the environment is correctly configured for YOLOv11.

2. **Dataset Preparation & Preprocessing**
   - Download the COCO dataset (version 34) in YOLOv11 format using the Roboflow API.
   - Verify the dataset structure (train, valid, test) and ensure the annotations are correctly formatted.
   - Conduct a data quality check by reviewing sample images and labels.

3. **Model Training**
   - Initialize the YOLOv11 model using the pre-trained weights (e.g., `yolo11n.pt`).
   - Configure training parameters, including the number of epochs (e.g., 100), batch size, learning rate, and input image size.
   - Monitor training progress via loss, mAP, precision, and recall.
   - Save the best-performing model based on validation metrics.

4. **Model Inference and Evaluation**
   - Load the best saved model weights.
   - Run inference on unseen test images and save the results.
   - Visualize the inference outputs (bounding boxes and confidence scores) using tools like matplotlib or PIL.
   - Evaluate performance using metrics such as mAP (at IoU 0.5 and 0.5–0.95), precision, recall, and F1 score.

---


## 📊 Results

The following performance metrics were obtained from the model evaluation:
- **mAP@50:** 0.1051
- **mAP@50-95:** 0.0839
- **Precision:** 0.6490
- **Recall:** 0.1107
- **F1 Score:** 0.1891

Inference results show bounding boxes on test images with confidence scores. Although the model demonstrates high precision, the low recall indicates that many objects are not detected.

---

## 🔍 Analysis

The evaluation metrics highlight the following key points:

1. **Precision vs. Recall:**
   - The model achieves a relatively high precision (~0.65), indicating that when it predicts an object, it is often correct.
   - However, the recall is notably low (~0.11), meaning that the model is missing a large number of objects present in the images. This imbalance suggests that while the model is cautious in its predictions, it is not sensitive enough to detect all relevant objects.

2. **Training Considerations:**
   - The current training setup, although a good starting point, appears to be insufficient for achieving robust detection performance on the COCO dataset.
   - Increasing training epochs, applying more extensive data augmentation, and further hyperparameter tuning (such as adjusting the learning rate schedule and modifying anchor boxes) are potential strategies to improve recall without compromising precision.

3. **Visual Inspection:**
   - The visualizations show that detected objects have correctly drawn bounding boxes and appropriate confidence scores. However, many objects are still missed, which is consistent with the low recall metric.
   - The visualization reinforces the notion that while the model is reliable when it makes a detection, its overall sensitivity is low.

---

## ✅ Conclusion

This project demonstrates the complete workflow for deploying YOLOv11 on the COCO dataset using Google Colab. While the model shows reliable detections when it does predict an object (high precision), the overall performance is limited by low recall and mAP. Future improvements will focus on extended training, enhanced data augmentation, and further hyperparameter tuning to develop a more robust object detection system.

---


## 📋 Requirements

- Python 3.7+
- Google Colab
- Roboflow API
- Ultralytics YOLOv11
- PyTorch
- OpenCV

---
