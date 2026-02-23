# Iris Flower Classification

## Project Overview
This is a **Python-based Machine Learning project** to classify Iris flowers based on their physical features.  
The dataset includes measurements of **sepal length, sepal width, petal length, and petal width**.  
Using the **K-Nearest Neighbors (KNN)** algorithm, the model predicts the species of the flower:

- `setosa`  
- `versicolor`  
- `virginica`  

This project is ideal for beginners to **practice ML concepts, model training, and prediction**.

---

## Dataset
- Built-in **Iris dataset** from `scikit-learn`
- Contains **150 samples** (50 per species)
- Features:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- Target: `species` (0=setosa, 1=versicolor, 2=virginica)

---

## Requirements

```bash
pip install numpy pandas scikit-learn matplotlib

## OUTPUT

========== IRIS DATASET PREVIEW ==========
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  species
0                5.1               3.5                1.4               0.2        0
1                4.9               3.0                1.4               0.2        0
2                4.7               3.2                1.3               0.2        0
3                4.6               3.1                1.5               0.2        0
4                5.0               3.6                1.4               0.2        0

Dataset Shape: (150, 5)
Species Mapping: {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

========== TRAIN/TEST SPLIT ==========
Training samples: 120
Testing samples: 30

========== MODEL PERFORMANCE ==========
Accuracy: 1.0

Classification Report:
              precision    recall  f1-score   support
setosa           1.00      1.00      1.00        10
versicolor       1.00      1.00      1.00         9
virginica        1.00      1.00      1.00        11

========== EXAMPLE PREDICTION ==========
Input features: [5.1, 3.5, 1.4, 0.2]
Predicted Flower Type: setosa
