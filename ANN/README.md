# 🚀 Telco Customer Churn Prediction – Neural Network + Keras Tuner

## TL;DR

This project builds a production-grade neural network model to **predict telecom customer churn**, leveraging **feature selection**, **standardization**, and **hyperparameter tuning** with **Keras Tuner**. Final model achieves ~79% accuracy on the test set.

---

## 🔍 Problem

Churn is expensive. Retaining customers is cheaper than acquiring new ones. The objective: build an intelligent system that **predicts churn before it happens** so business teams can intervene early.

---

## 📊 Dataset

- **Source**: IBM Telco Churn Dataset  
- **Samples**: ~7,000  
- **Target**: `Churn` (Yes/No → 1/0)

Key features include contract type, monthly charges, tenure, service subscriptions, and more.

---

## 🧠 ML Stack

| Component        | Tool / Method                        |
|------------------|--------------------------------------|
| Feature Encoding | `LabelEncoder`                       |
| Scaling          | `StandardScaler`                     |
| Feature Selection| `RFE` with `DecisionTreeClassifier`  |
| Model Type       | Feedforward Neural Network (Keras)   |
| Tuning           | `Keras Tuner` – Random Search        |
| Optimization     | Adam + Binary Crossentropy           |
| Regularization   | `EarlyStopping` on `val_loss`        |

---

## 🏗️ Model Architecture (Tuned)

- `Input`: 5 features from RFE
- `Hidden Layer 1`: Tuned units (8–128, step=8), ReLU
- `Hidden Layer 2`: Tuned units (8–128, step=8), ReLU
- `Output Layer`: 1 neuron, Sigmoid

Best architecture is selected automatically via `kt.RandomSearch()` based on validation accuracy.

---

## 📈 Results

| Metric          | Value     |
|------------------|-----------|
| Test Accuracy    | ~78.9%    |
| Test Loss        | ~0.438    |

> Note: Actual performance may vary across runs due to randomness and search space.

---

## 📉 Training Curves

<img src="train_val_accuracy.png" width="400">  <img src="train_val_loss.png" width="400">

> Replace with actual plots if you save them via `plt.savefig()`.

---

## 🔁 Reproducible Pipeline

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib tensorflow keras-tuner

# Run your script / notebook:
python churn_model.py
