# 🔁 Encoder-Decoder Architecture – Sequence-to-Sequence Model

## 📁 Project Overview

This project implements a **Sequence-to-Sequence (Seq2Seq)** model using an **Encoder-Decoder** architecture in PyTorch. It is designed to handle tasks involving input and output sequences of varying lengths, such as machine translation, text generation, and more.

---

## 📌 Key Features

- ✳️ Encoder and Decoder modules built with:
  - Embedding layers
  - GRU (Gated Recurrent Unit) for sequence modeling
- 🔄 Seq2Seq integration to connect encoder outputs to decoder inputs
- 🧮 End-to-end training loop using Cross Entropy Loss
- 🔍 Real-time prediction for sequence inference

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib (for optional visualization)

---

## 🧾 Notebook Structure

1. **Data Preparation**
   - Define input and output tokens
   - Create vocab mappings
   - Prepare input and target tensor sequences

2. **Model Architecture**
   - **Encoder**:
     - Embeds input sequence
     - Passes through GRU to capture hidden state
   - **Decoder**:
     - Embeds previous output
     - Uses GRU initialized with encoder’s final hidden state
     - Generates predictions step-by-step

3. **Seq2Seq Wrapper**
   - Connects encoder and decoder
   - Manages teacher forcing during training

4. **Training Process**
   - Uses `nn.CrossEntropyLoss`
   - Optimized using `torch.optim.Adam`
   - Loss printed per epoch

5. **Inference**
   - Predicts sequences from input tokens
   - Outputs final predicted string vs ground truth

---

