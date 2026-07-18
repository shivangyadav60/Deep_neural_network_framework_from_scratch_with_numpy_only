# DeepNN From Scratch

This is an exmaple of how Deep Neural Network framework built entirely from scratch using **NumPy**.

This project was created to understand how modern neural networks work internally without relying on machine learning libraries such as TensorFlow or PyTorch.

The framework implements the complete forward and backward propagation pipeline, including dense layers, activation functions, loss functions, gradient computation, and gradient descent optimization.

---

## Features

* Dense (Fully Connected) Layers
* Forward Propagation
* Backpropagation using the Chain Rule
* Gradient Descent Optimizer
* Multiple Activation Functions

  * ReLU
  * Sigmoid
  * Tanh
  * Softmax
* Multiple Loss Functions

  * Mean Squared Error (MSE)
  * Categorical Cross Entropy
* Multi-class Classification Support
* Object-Oriented Design
* Built entirely using NumPy

 Implemented Components

# Layers

* Dense Layer
* Weight Initialization
* Bias Initialization
* Forward Pass
* Backward Pass

# Activation Functions

* ReLU
* Sigmoid
* Tanh
* Softmax

# Loss Functions

* Mean Squared Error
* Categorical Cross Entropy

# Model Class

* Add Layers
* Compile Model
* Forward Propagation
* Backpropagation
* Weight Updates
* Training Loop
* Accuracy Calculation

---

## Training Flow

```
Input
   │
   ▼
Dense Layer
   │
   ▼
Activation
   │
   ▼
Dense Layer
   │
   ▼
Activation
   │
   ▼
Output Layer
   │
   ▼
Loss
   │
   ▼
Backpropagation
   │
   ▼
Gradient Descent
   │
   ▼
Updated Weights
```

---

## Example

```python
model = Model(train_X, train_Y)

model.add(8, "sigmoid")
model.add(3, "softmax")

model.compile(loss="categorical_crossentropy")

model.train(
    train_X,
    train_Y,
    epochs=50,
    lr=0.001
)
```

---

# Mathematics Used

The framework implements:

* Matrix Multiplication
* Chain Rule
* Jacobian Matrix (Softmax)
* Partial Derivatives
* Gradient Descent
* Dot Products
* Cross Entropy Loss

---

## Purpose

This project is intended for educational purposes to understand the mathematics behind deep learning rather than to replace mature frameworks such as TensorFlow or PyTorch.

---

## Future Improvements

* Mini-batch Gradient Descent
* SGD Optimizer
* Adam Optimizer
* Binary Cross Entropy
* Dropout
* Batch Normalization
* Regularization (L1/L2)
* Learning Rate Scheduling
* Model Saving and Loading
* GPU Support
* Convolutional Layers (CNN)
* Recurrent Layers (RNN/LSTM)

---

## Requirements

* Python 3.x
* NumPy
* Pandas

Install dependencies:

```bash
pip install numpy pandas
```

---

## Author

**Shivang Yadav**

Computer Science Engineering Student

Built from scratch to gain a deeper understanding of neural networks and backpropagation.
