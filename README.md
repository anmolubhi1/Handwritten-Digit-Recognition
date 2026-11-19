# 🧠 Multi-Layer Perceptron (MLP) from Scratch with NumPy on MNIST

A complete, end-to-end implementation of a **Multi-Layer Perceptron (MLP) neural network** trained using **explicit manual backpropagation and mini-batch gradient descent in NumPy** on the built-in **MNIST handwritten digit dataset** (loaded via Keras without manual file downloads).

This project covers data preprocessing, exploratory data analysis, custom forward and backward passes, training telemetry, unseen test evaluation, and clean visual diagnostics built with **Matplotlib** and **Seaborn**.

---

## 📌 Project Overview

High-level deep learning frameworks like TensorFlow and PyTorch often hide the calculus and linear algebra happening behind the scenes. This project builds a working neural network from the ground up using **NumPy matrix operations** so every stage of learning—from activation propagation to chain-rule gradient computation and parameter updates—is clear, inspectable, and easy to experiment with.

### ✨ Key Highlights
* 📦 **Zero Manual Dataset Downloads:** Loads the 60,000-image training set and 10,000-image test set directly via `tensorflow.keras.datasets.mnist`.
* ⚙️ **Manual Backpropagation Engine:** Implements matrix-based forward propagation, categorical cross-entropy loss, chain-rule gradients (`dZ2`, `dW2`, `db2`, `dZ1`, `dW1`, `db1`), and mini-batch gradient descent updates in pure NumPy.
* 🛡️ **Numerical Stability & Smart Initialization:** Uses **He initialization** for ReLU hidden layers, zero-initialized biases, and max-shifted **Softmax** with clipped log probabilities to avoid overflow or `NaN` loss.
* 📊 **Exploratory & Diagnostic Visuals:** Generates class balance plots, sample digit grids, loss and accuracy curves, a test-set confusion matrix heatmap, and a classification report.

---

## 🧮 Architecture & Mathematical Pipeline

The network is structured as a two-layer dense neural network (one hidden layer + one output layer):

1. **🔢 Input Representation:** Each $28 \times 28$ grayscale image is flattened into a $784$-dimensional vector and normalized from $[0, 255]$ to $[0.0, 1.0]$.
2. **⚡ Hidden Layer (128 Neurons, ReLU):**
   * Pre-activation: $Z^{[1]} = X W^{[1]} + b^{[1]}$
   * Activation: $A^{[1]} = \text{ReLU}(Z^{[1]}) = \max(0, Z^{[1]})$
3. **🎯 Output Layer (10 Neurons, Softmax):**
   * Pre-activation (logits): $Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}$
   * Class probabilities: $\hat{Y} = A^{[2]} = \text{Softmax}(Z^{[2]})$
4. **📉 Loss Function (Categorical Cross-Entropy):** Measures divergence between predicted probabilities $\hat{Y}$ and one-hot encoded ground truth labels $Y$ across $m$ batch samples.
5. **🔁 Backward Pass (Chain Rule):**
   * Output error term: $dZ^{[2]} = A^{[2]} - Y$
   * Output parameter gradients: $dW^{[2]} = \frac{1}{m} (A^{[1]})^T dZ^{[2]}$, $\quad db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$
   * Hidden error term: $dZ^{[1]} = (dZ^{[2]} (W^{[2]})^T) \odot \text{ReLU}'(Z^{[1]})$
   * Hidden parameter gradients: $dW^{[1]} = \frac{1}{m} X^T dZ^{[1]}$, $\quad db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$
6. **🛠️ Parameter Update (Mini-Batch Gradient Descent):**
   * $W^{[l]} \leftarrow W^{[l]} - \alpha \, dW^{[l]}$ and $b^{[l]} \leftarrow b^{[l]} - \alpha \, db^{[l]}$ with learning rate $\alpha = 0.05$ and batch size $128$.

---

## 🗂️ Project Workflow & Notebook Structure

* **Cell 1 — Library Imports:** Imports `numpy`, `matplotlib.pyplot`, `seaborn`, `tensorflow.keras.datasets.mnist`, and `sklearn.metrics` utilities, and sets random seeds for reproducibility.
* **Cell 2 — Data Loading & Preprocessing:** Loads MNIST, flattens $28 \times 28$ images to $784$ features, normalizes pixel values to $[0, 1]$, and one-hot encodes integer labels ($0$–$9$) into length-$10$ vectors.
* **Cell 3 — Exploratory Data Analysis (EDA):** Plots class balance across digits $0$–$9$ using a Seaborn countplot and renders a $2 \times 5$ grid of sample handwritten digits using Matplotlib.
* **Cell 4 — Activations, Derivatives & Initialization:** Defines `relu`, `relu_derivative`, and numerically stable `softmax`, configures hyperparameters (`hidden_size = 128`, `learning_rate = 0.05`, `epochs = 15`, `batch_size = 128`), and initializes weights and biases.
* **Cell 5 — Backpropagation Training Loop:** Shuffles training data each epoch, loops through mini-batches, runs forward propagation, computes cross-entropy loss, calculates backward gradients, and updates parameters.
* **Cell 6 — Training Curves Visualization:** Plots epoch-by-epoch training cross-entropy loss and training accuracy progression side by side using Seaborn and Matplotlib.
* **Cell 7 — Test Evaluation & Confusion Matrix:** Runs a forward pass on the unseen 10,000-image test set, prints test accuracy and precision/recall/F1 metrics, and displays a Seaborn confusion matrix heatmap.

---

## 📈 Visualizations Included

* 📊 **Class Distribution Chart:** Verifies balanced representation across all 10 digit classes in the training set.
* 🖼️ **Sample Digit Grid:** Displays sample handwritten digits alongside their ground-truth labels.
* 📉 **Loss & Accuracy Curves:** Tracks convergence and learning stability over training epochs.
* 🧩 **Confusion Matrix Heatmap:** Highlights exact counts of correct predictions and common digit misclassifications on unseen test data.

---

## 🛠️ Tech Stack

| Component | Technology | Role in Project |
| :--- | :--- | :--- |
| **Language** | Python 3.x | Core programming environment |
| **Math & Matrices** | NumPy | Forward pass, chain-rule backprop, weight updates |
| **Dataset Loader** | TensorFlow / Keras (`mnist`) | Built-in MNIST dataset access |
| **Visualizations** | Matplotlib, Seaborn | EDA plots, training curves, confusion matrix |
| **Metrics** | scikit-learn | Confusion matrix & classification report |

---

## 🚀 Future Improvements & Extensions

* 🧪 Add a dedicated validation split to monitor generalization and catch overfitting during training.
* 🛡️ Implement **L2 regularization (weight decay)** or **dropout** to improve generalization on unseen handwriting.
* ⚡ Experiment with additional optimizers (such as **Momentum** or **Adam**), learning rate schedules, or extra hidden layers.
* 💾 Save trained weights (`.npz`) and add an inference script to classify custom hand-drawn digit images.
