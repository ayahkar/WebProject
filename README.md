# Image-Based Age and Gender Detection System

A computer vision application built with Python and OpenCV that performs face detection, age estimation, and gender classification directly from input images.

---

## 📌 Project Overview
This system takes an input image, identifies facial regions using Haar Cascade Classifiers, and routes the extracted face blobs through pre-trained Caffe Deep Learning models to classify:
1. **Gender:** Male / Female
2. **Age Brackets:** (e.g., 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60+)

---

## 🛠️ Tech Stack & Requirements
- **Language:** Python 3.x
- **Core Libraries:**
  - `opencv-python` (Image processing & DNN inference module)
  - `numpy` (Array manipulation & blob preprocessing)
- **Model Framework:** Caffe Architecture (`.prototxt` for structure & `.caffemodel` for pre-trained weights)

---

## 📂 Repository Structure
```text
Age-Gender-Detection-Image/
│
├── data/                    # Model weights and XML cascade files
│   ├── haarcascade_frontalface_default.xml
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
│
├── main.py                  # Primary Python script for image inference
├── requirements.txt         # Required dependencies
└── README.md                # Project documentation# WebProject
Age-Gender-Detection-Image
