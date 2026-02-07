# Facial-Emotion-Detection-CNN-Gradio-WebApp
## 📌 Project Overview

This project focuses on facial mood classification using a Convolutional Neural Network (CNN).
The model learns visual patterns from facial images and predicts the emotional state (mood) of a person.
A user-friendly frontend is built using Gradio, allowing real-time image uploads and predictions.

## 🎯 Objective

To build an end-to-end deep learning pipeline for facial mood detection

To apply CNN architectures for image-based emotion recognition

To deploy the trained model with an interactive web interface

# 🗂️ Project Workflow (Step-by-Step)
## 1️⃣ Data Collection & Preprocessing

Facial image dataset organized into mood-based classes

Images resized and normalized for CNN compatibility

Data augmentation applied to improve generalization

Rotation

Zoom

Horizontal flip

## 2️⃣ Model Architecture (CNN)

Convolutional Layers for feature extraction

ReLU activation to introduce non-linearity

Max Pooling layers to reduce spatial dimensions

Flatten layer to convert feature maps into vectors

Dense (Fully Connected) layers for classification

Softmax output layer for multi-class mood prediction

## 3️⃣ Model Compilation & Training

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Training performed over multiple epochs with validation monitoring

## 4️⃣ Model Evaluation

Training vs validation accuracy analysis

Loss curves to check overfitting/underfitting

Final model performance evaluation on unseen data

## 5️⃣ Frontend Deployment (Gradio)

Simple and interactive web UI using Gradio

Users can upload facial images

Model instantly predicts the detected mood

Backend powered by Python + trained CNN model

## 🛠️ Technologies Used

Python

TensorFlow / Keras

OpenCV (image handling)

NumPy & Matplotlib

Gradio (frontend deployment)

Jupyter Notebook

## 📊 Key Learnings

Practical understanding of CNNs for image classification

Importance of data preprocessing in deep learning

Building deployable ML applications with minimal frontend code

Bridging model training and real-world usability


## 📁 Repository Contents

CNN_Face_classification.ipynb – Model training & evaluation

Gradio-based frontend implementation

Dataset structure (if applicable)

## ⭐ Conclusion

This project demonstrates a complete deep learning workflow, from data preprocessing and CNN modeling to deployment with an interactive UI.
It’s a strong foundation for anyone exploring computer vision, emotion AI, and applied deep learning.
