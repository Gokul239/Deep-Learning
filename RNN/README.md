# 🧠 Recurrent Neural Network (RNN) Model – Sequence Learning

## 📁 Project Overview

This project demonstrates the implementation of a **Recurrent Neural Network (RNN)** to learn and generate sequences. It focuses on processing sequential data using PyTorch, showcasing how an RNN can be trained on tokenized input and target sequences to predict the next elements in a sequence.

---

## 📌 Key Features

- 📚 Tokenization of input and output sequences
- 🧠 Custom RNN model with:
  - Embedding Layer
  - `nn.RNN` for sequential processing
  - Linear output layer for classification
- 🔁 Sequence-to-sequence learning using mini-batches
- 📈 Training loop with loss tracking and evaluation
- 🧪 Sample inference to evaluate learned patterns

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib

---

## 🧾 Notebook Structure

1. **Data Preparation**
   - Define input and output character sets
   - Create token mappings
   - Convert sequences to index tensors

2. **RNN Model Definition**
   - Embedding + RNN + Linear architecture
   - Forward pass through the sequence

3. **Training Loop**
   - Loss: Cross Entropy Loss
   - Optimizer: Adam
   - Epoch-wise training with periodic logging

4. **Evaluation**
   - Inference using trained model
   - Prediction comparison with ground truth

---
