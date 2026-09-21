# Surface Defect Classification Using CNN

A deep learning project for automatic classification of steel surface defects using Convolutional Neural Networks (CNNs).

The project compares a **Baseline CNN** with an **Improved CNN architecture** to study the effect of architectural improvements on surface defect classification.

## 📌 Project Overview

Surface defects in industrial materials can affect product quality and manufacturing efficiency. Manual inspection can be time-consuming and may be inconsistent.

This project uses computer vision and deep learning to automatically classify steel surface images into six defect categories.

The implementation is based on the **NEU Surface Defect Database (NEU-DET)** and is developed independently as a research-oriented implementation.

## 🎯 Objectives

* Develop a CNN-based surface defect classification system.
* Classify steel surface images into six defect categories.
* Establish a baseline CNN model.
* Develop an improved CNN architecture.
* Compare the performance of both models using standard evaluation metrics.
* Analyze classification performance using confusion matrices and classification reports.

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

| Dataset    | Images |
| ---------- | -----: |
| Training   |  1,440 |
| Validation |    360 |
| Total      |  1,800 |

The images are processed as grayscale images and resized to **128 × 128 pixels** for model training.

## 🔄 Data Preprocessing

The preprocessing pipeline includes:

* Image resizing to 128 × 128 pixels
* Conversion to grayscale
* Pixel normalization to the range 0–1
* Data augmentation for training images

Training augmentation includes:

* Rotation
* Width shifting
* Height shifting
* Zooming
* Horizontal flipping

Validation images are normalized without augmentation.

## 🧠 Baseline CNN

The baseline model consists of:

* Convolutional layer — 32 filters
* Max Pooling
* Convolutional layer — 64 filters
* Max Pooling
* Convolutional layer — 128 filters
* Max Pooling
* Flatten layer
* Dense layer — 128 neurons
* Dropout — 0.5
* Softmax output layer — 6 classes

The model is trained using the Adam optimizer and categorical cross-entropy loss.

## 🚀 Improved CNN

An improved CNN architecture was developed to investigate whether additional architectural components could improve classification performance.

The improved model includes:

* Four convolutional blocks
* Batch Normalization
* Max Pooling
* Increasing feature channels: 32 → 64 → 128 → 256
* Global Average Pooling
* Dense layer with 128 neurons
* Dropout — 0.5
* Six-class Softmax output

### Main architectural improvements

| Component              | Baseline CNN | Improved CNN |
| ---------------------- | ------------ | ------------ |
| Convolutional blocks   | 3            | 4            |
| Batch Normalization    | No           | Yes          |
| Global Average Pooling | No           | Yes          |
| Dropout                | Yes          | Yes          |
| Output classes         | 6            | 6            |

## 📊 Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

The evaluation pipeline generates a comparison between the Baseline CNN and Improved CNN.

Experimental results are available in the `results/` directory.

## 📁 Project Structure

```text
surface-defect-classification-cnn/
│
├── Surface_Defect_Classification_CNN.ipynb
├── README.md
├── requirements.txt
├── .gitignore
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
    ├── baseline_accuracy.png
    ├── baseline_loss.png
    ├── classification_report.txt
    ├── improved_classification_report.txt
    ├── improved_confusion_matrix.png
    └── model_comparison.csv
```

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Pillow
* Google Colab
* GitHub

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/PTarmale/surface-defect-classification-cnn.git
cd surface-defect-classification-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download the NEU-DET dataset and place it in:

```text
dataset/
└── NEU-DET/
    ├── train/
    │   └── images/
    └── validation/
        └── images/
```

The dataset itself is excluded from this repository through `.gitignore`.

### 4. Train the models

```bash
python src/train.py
```

This trains both the baseline and improved CNN models.

### 5. Evaluate the models

```bash
python src/evaluate.py
```

The evaluation script generates classification metrics and saves the model comparison results.

### 6. Predict a single image

Update the image path in `src/predict.py` and run:

```bash
python src/predict.py
```

## 🔬 Research Contribution

The project provides an independent experimental comparison between a basic CNN architecture and an improved CNN architecture incorporating:

* Batch Normalization
* An additional convolutional block
* Global Average Pooling
* Regularization through Dropout

The objective is to investigate how these architectural changes affect surface defect classification performance on the NEU-DET dataset.

## 📈 Results

The repository contains the generated experimental outputs, including:

* Training accuracy graph
* Training loss graph
* Classification reports
* Improved model confusion matrix
* Baseline vs Improved model comparison

Numerical performance values are intentionally reported directly from the generated experiment files rather than being manually entered here.

## 📚 Reference

Arikan, S., Varanasi, K., & Stricker, D. (2019). *Surface Defect Classification in Real-Time Using Convolutional Neural Networks*. arXiv:1904.04671.

Official paper:

https://arxiv.org/abs/1904.04671

## 👩‍💻 Author

**Prachi Tarmale**

AI & Data Science Student

## ⭐ Project Status

Research-oriented implementation and experimental comparison completed. Further work can explore transfer learning, lightweight architectures, additional augmentation strategies, and deployment for real-time industrial inspection.
