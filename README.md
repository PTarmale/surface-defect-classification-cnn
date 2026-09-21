# Surface Defect Classification Using Convolutional Neural Networks

A deep learning project for automatic classification of steel surface defects using Convolutional Neural Networks (CNNs).

This project experimentally compares a **Baseline CNN** with a **Final Modified CNN** on the NEU Surface Defect Database (NEU-DET).

The purpose of the study is to investigate how architectural modifications such as Batch Normalization and L2 regularization affect surface defect classification performance.

---

## 📌 Project Overview

Surface defects in industrial materials can affect product quality and manufacturing efficiency. Manual inspection can be time-consuming and may be inconsistent.

This project uses computer vision and deep learning to automatically classify steel surface images into six defect categories.

The implementation is based on the **NEU Surface Defect Database (NEU-DET)** and was developed independently as an experimental research implementation.

---

## 🎯 Objectives

- Develop a CNN-based surface defect classification system.
- Classify steel surface images into six defect categories.
- Establish a baseline CNN model.
- Develop a modified CNN architecture.
- Compare both models using standard evaluation metrics.
- Analyze classification performance using classification reports and confusion matrices.
- Study the effect of architectural and regularization changes on model performance.

---

## 🗂️ Dataset

The project uses the **NEU Surface Defect Database (NEU-DET)**.

The dataset contains six types of steel surface defects:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

### Dataset Distribution

| Dataset | Images |
|---|---:|
| Training | 1,440 |
| Validation | 360 |
| Total | 1,800 |

The images are processed as grayscale images and resized to **128 × 128 pixels**.

---

## 🔄 Data Preprocessing

The preprocessing pipeline includes:

- Image resizing to 128 × 128 pixels
- Grayscale image processing
- Pixel normalization to the range 0–1
- Data augmentation for training images

Training augmentation includes:

- Rotation
- Width shifting
- Height shifting
- Zooming
- Horizontal flipping

Validation images are normalized without augmentation.

---

## 🧠 Baseline CNN

The baseline model consists of:

- Convolutional layer — 32 filters
- Max Pooling
- Convolutional layer — 64 filters
- Max Pooling
- Convolutional layer — 128 filters
- Max Pooling
- Flatten layer
- Dense layer — 128 neurons
- Dropout — 0.5
- Softmax output layer — 6 classes

The model uses:

- Adam optimizer
- Categorical cross-entropy loss

---

## 🚀 Final Modified CNN

A modified CNN architecture was developed to investigate the effect of additional regularization and normalization techniques.

The final model includes:

- Three convolutional blocks
- Feature channels: 32 → 64 → 128
- Batch Normalization after each convolutional layer
- Max Pooling
- L2 kernel regularization
- Flatten layer
- Dense layer with 128 neurons
- Dropout — 0.5
- Six-class Softmax output
- Adam optimizer with learning rate 0.0005

### Architecture Comparison

| Component | Baseline CNN | Final Modified CNN |
|---|---|---|
| Convolutional blocks | 3 | 3 |
| Feature channels | 32 → 64 → 128 | 32 → 64 → 128 |
| Batch Normalization | No | Yes |
| L2 Regularization | No | Yes |
| Flatten | Yes | Yes |
| Dense layer | 128 | 128 |
| Dropout | 0.5 | 0.5 |
| Output classes | 6 | 6 |

---

## 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

The evaluation pipeline compares the Baseline CNN with the Final Modified CNN.

---

## 📈 Experimental Results

The experimental results obtained on the 360-image validation set are:

| Metric | Baseline CNN | Final Modified CNN |
|---|---:|---:|
| Accuracy | 78.89% | 68.33% |
| Precision | 81.79% | 77.38% |
| Recall | 78.89% | 68.33% |
| F1-Score | 78.98% | 66.50% |

The results show that the final modified architecture did not outperform the baseline CNN on this experimental setup.

This observation is useful for analyzing the effect of architectural modifications rather than assuming that additional layers or regularization techniques will always improve classification performance.

---

## 📋 Final Modified CNN Classification Results

| Defect Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Crazing | 0.70 | 1.00 | 0.82 |
| Inclusion | 0.80 | 0.67 | 0.73 |
| Patches | 1.00 | 0.58 | 0.74 |
| Pitted Surface | 0.64 | 0.58 | 0.61 |
| Rolled-in Scale | 0.51 | 1.00 | 0.67 |
| Scratches | 1.00 | 0.27 | 0.42 |

Overall validation accuracy: **68.33%**

---

## 📁 Project Structure

```text
surface-defect-classification-cnn/
│
├── Surface_Defect_Classification_CNN.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── final_accuracy.png
├── final_loss.png
├── final_confusion_matrix.png
├── final_classification_report.txt
├── final_model_comparison.csv
│
├── src/
│   ├── preprocessing.py
│   ├── baseline_model.py
│   ├── improved_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── results/
