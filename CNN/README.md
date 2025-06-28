# 🧠 Handwritten Digit Recognition using CNN (MNIST)

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. It follows a structured deep learning workflow with data augmentation, regularization, and training optimizations to achieve high accuracy and generalization.

---

## ✅ Objectives

- Build a robust CNN model for digit recognition
- Apply data augmentation to reduce overfitting
- Use training callbacks for adaptive learning
- Evaluate and visualize model performance

---

## 📚 Dataset Overview

- Source: MNIST (60,000 training images, 10,000 test images)
- Grayscale 28×28 pixel images of digits 0–9
- Balanced across classes

---

## 🔄 Workflow Summary

1. **Data Loading & Preprocessing**
   - Normalize images
   - Reshape for channel compatibility
   - Convert labels to one-hot encoding

2. **Data Augmentation**
   - Rotation and zoom transformations applied during training

3. **Model Architecture**
   - 3 Convolutional + MaxPooling blocks
   - Batch Normalization for stability
   - Dropout for regularization
   - Dense classifier with softmax output

4. **Training Strategy**
   - Adam optimizer with categorical crossentropy
   - Early stopping to prevent overfitting
   - ReduceLROnPlateau for adaptive learning rate control

5. **Evaluation & Visualization**
   - Accuracy and loss curves
   - Final evaluation on the test set

---

## 📈 Performance

- Training converges with high validation accuracy
- Final test accuracy: ~98%
- Model generalizes well on unseen digits

---

## 🛠️ Tools Used

- Python 3.x
- TensorFlow / Keras
- NumPy & Matplotlib

---

## 🧠 Notes

This implementation is modular, scalable, and production-friendly. It incorporates best practices in computer vision modeling including augmentation, callbacks, and CNN design suited for small image datasets like MNIST.

---
```
