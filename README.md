# 🧠 Handwritten Digit Recognition — Neural Network from Scratch

> *"Just as the mind learns from experience, so does this network — layer by layer, iteration by iteration."*

A **pure NumPy** implementation of a 2-layer Neural Network trained on the [Neural Network By Samson Zhang](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/input) to classify handwritten digits (0–9). No TensorFlow. No PyTorch. Just math and code.

---

## 📊 Model Performance

| Iteration | Accuracy |
|-----------|----------|
| 0         | ~11.5%   |
| 1200      | ~88.5%   |
| 2400      | ~90.8%   |
| 3600      | ~91.9%   |
| 4800      | ~92.6%   |
| **Final (Train)** | **92.62%** ✅ |

---

## 🏗️ Architecture

```
Input Layer  →  784 neurons  (28×28 pixels flattened)
Hidden Layer →  10 neurons   (ReLU activation)
Output Layer →  10 neurons   (Softmax activation → digits 0–9)
```

---

## ⚙️ How It Works

### 1. Forward Propagation
```
Z1 = W1 · X + b1      → Linear transformation
A1 = ReLU(Z1)         → Non-linearity introduced
Z2 = W2 · A1 + b2     → Second linear transformation
A2 = Softmax(Z2)      → Probability distribution over 10 classes
```

### 2. Backward Propagation
Gradients are computed using the chain rule and weights are updated via **Gradient Descent**.

### 3. Parameter Update
```
W = W - α * dW
b = b - α * db
```

---

## 📁 Project Structure

```
📦 digit-recognition-nn
 ┣ 📄 neural_network.py     ← Main model code
 ┣ 📄 train.csv             ← MNIST training dataset (from Kaggle)
 ┗ 📄 README.md             ← You are here!
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas pillow matplotlib
```

### Dataset

Download `train.csv` from [Neural Network By Samson Zhang](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/input) and place it in the root directory.

### Run Training

```bash
python neural_network.py
```

Training runs for **5500 iterations** with a learning rate of `α = 0.1`.

### Predict on Custom Image

After training, the model will prompt you:

```
Enter Image Path: your_digit.png
Predicted Label: 7
Do you want to predict another image? (y/n):
```

> ⚠️ Make sure your image is a **grayscale** image (or it will be auto-converted). The model resizes it to **28×28** internally.

---

## 🔧 Hyperparameters

| Parameter       | Value  |
|-----------------|--------|
| Learning Rate α | 0.1    |
| Iterations      | 5500   |
| Hidden Units    | 10     |
| Train Split     | 41,000 |
| Dev Split       | 1,000  |

---

## 🧮 Key Functions

| Function | Purpose |
|----------|---------|
| `intial_W_b()` | Random weight & bias initialization |
| `ReLU(X)` | Rectified Linear Unit activation |
| `softmax(X)` | Output probability distribution |
| `froward_propagation()` | Full forward pass |
| `backward_propagation()` | Compute gradients |
| `update_parameters()` | Apply gradient descent step |
| `gradient_descent()` | Full training loop |

---

## 💡 Key Concepts Used

- **One-Hot Encoding** for labels
- **ReLU Derivative** for backprop through hidden layer
- **Numerically Stable Softmax** (`exp(X - max(X))`)
- **Vectorized operations** using NumPy — no Python loops in math

---

## 🙋 About

Built from scratch as a learning project to deeply understand how neural networks work **under the hood** — without any deep learning frameworks.

> *"Frameworks give you the car. This project teaches you how the engine works."*

---

## 📌 To-Do / Future Improvements

- [ ] Add validation accuracy tracking
- [ ] Add loss curve plotting with Matplotlib
- [ ] Try adding more hidden layers
- [ ] Experiment with different learning rates
- [ ] Save & load trained weights

---

## 📜 License

This project is open-source and free to use for educational purposes.

---

*Made with ❤️ and pure NumPy*
