# Telco Customer Churn Prediction – Deep Learning Approach

## Overview

This project leverages a supervised learning approach using a feedforward Artificial Neural Network (ANN) to predict customer churn in a telecom business. It follows a complete ML pipeline from data preprocessing, feature engineering, model building, and evaluation.

---

## Problem Statement

Customer churn directly impacts revenue. The objective is to predict the likelihood of a customer exiting the service using available structured data. A reliable churn prediction model enables proactive retention strategies.

---

## Dataset

- **Source**: IBM Telco Customer Churn  
- **Records**: ~7,000 rows, 20+ features  
- **Target Variable**: `Churn` (Binary – Yes/No)

Data fields include demographics, service subscription details, contract information, and billing metrics.

---

## Pipeline Summary

### 1. Data Preparation

- Dropped unique identifiers (`customerID`)
- Converted `TotalCharges` from object to numeric; handled coercion and removed nulls
- Applied `LabelEncoder` to categorical features
- Ensured numerical consistency across the dataset

### 2. Feature Selection

- Applied Recursive Feature Elimination (RFE) with a `DecisionTreeClassifier`
- Retained top 5 features based on importance ranking

### 3. Train-Test Split

- 80/20 split using `train_test_split` from `sklearn`
- Standardized inputs using `StandardScaler` for numerical stability

### 4. Model Architecture

- **Input Layer**: 16 neurons, ReLU
- **Hidden Layer**: 8 neurons, ReLU
- **Output Layer**: 1 neuron, Sigmoid (for binary classification)

Compiled with:
- Optimizer: `adam`
- Loss: `binary_crossentropy`
- Metrics: `accuracy`

### 5. Training Strategy

- Early stopping with patience of 3 epochs to prevent overfitting
- Validation split: 10% of training data
- Batch size: 10
- Epochs: up to 30 (early-stopped)

### 6. Evaluation

- Tracked and plotted training vs validation accuracy and loss
- Visual inspection of overfitting/underfitting trends

---

## Environment & Dependencies

```bash
python 3.8+
tensorflow >= 2.x
scikit-learn
pandas
numpy
matplotlib
