# 🧠 Brain Tumor MRI Classification using Deep Learning

A deep learning-based web application that classifies brain MRI scans into four categories using transfer learning with **EfficientNetV2B0**. The project includes model training, evaluation, and deployment using **Streamlit**.

---

## 📌 Overview

Brain tumors require timely diagnosis for effective treatment. This project applies deep learning to classify brain MRI images into one of four classes:

- No Tumor
- Pituitary Tumor
- Glioma Tumor
- Meningioma Tumor

The model is trained using transfer learning and deployed through an interactive Streamlit application for real-time inference.

---

## 🚀 Features

- Transfer Learning using EfficientNetV2B0
- Four-class MRI image classification
- Real-time prediction through Streamlit
- Confidence score for every prediction
- Probability distribution for all classes
- Clean and lightweight deployment
- Easy-to-use interface

---

## 🖥️ Demo

### Upload MRI Image

Upload a Brain MRI image (.jpg, .jpeg or .png).

### Model Prediction

The application predicts:

- Predicted Class
- Confidence Score
- Probability Distribution for all classes

---

## 🧠 Classes

| Class | Description |
|--------|-------------|
| No Tumor | Normal Brain MRI |
| Pituitary | Pituitary Tumor |
| Glioma | Glioma Tumor |
| Meningioma | Meningioma Tumor |

---

## 🏗️ Model Architecture

### Base Model

- EfficientNetV2B0
- ImageNet Pretrained Weights
- include_top=False

### Classification Head

- Global Average Pooling
- Dense Layer
- Batch Normalization
- ReLU Activation
- Dropout
- Softmax Output Layer

---

## 🗂 Dataset

Brain Tumor MRI Dataset containing four categories:

- Training Images
- Testing Images

Image Size:

```
224 × 224 × 3
```

---

## 📊 Model Performance

| Model | Test Accuracy |
|--------|--------------|
| EfficientNetV2B0 | ~95% |
| VGG16 | ~94% |

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- EfficientNetV2B0
- NumPy
- Pillow
- Pandas
- Streamlit

---

## 📁 Project Structure

```
Brain-Tumor-MRI-Classifier/
│
├── app.py
├── best_model_eff.keras
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── home.png
    ├── prediction.png
```

---


Run the application

```bash
streamlit run app.py
```

---

## 📈 Training Pipeline

1. Load MRI Dataset
2. Image Preprocessing
3. Transfer Learning using EfficientNetV2B0
4. Fine-tuning
5. Model Training
6. Evaluation
7. Model Saving
8. Streamlit Deployment

---

## 📸 Application Workflow

```
MRI Image
      │
      ▼
Image Preprocessing
      │
      ▼
EfficientNetV2B0
      │
      ▼
Softmax Prediction
      │
      ▼
Predicted Tumor Class
      │
      ▼
Confidence Score
```

---

## 📋 Requirements

```
streamlit==1.38.0
tensorflow==2.16.1
numpy<2
pandas
Pillow
scikit-learn
opencv-python-headless
```

---

---

## 👨‍💻 Author

**Anjani Tiwari**

Deep Learning | Machine Learning | Computer Vision


## ⭐ If you found this project useful, consider giving it a star.
