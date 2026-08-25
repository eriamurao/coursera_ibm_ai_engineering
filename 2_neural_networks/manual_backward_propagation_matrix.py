#!/usr/bin/env python
# coding: utf-8

# continuation of manual_forward_propagation_matrix.py

import numpy as np

# Variables from the forward propagation
# Just for additional info:
# - input: 1 training example with 2 features
# - network: 2 feature inputs, 2 hidden layers with 2 nodes each, 1 output layer
# - activations used the network: sigmoid only
# - output: 1 ouput


# Values from forward propagation
X = np.array([[0.53, 0.81]])

W_1 = np.array([[0.62, 0.93], [0.77, 0.89]])
B_1 = np.array([[0.56, 0.09]])

Z_1 = np.array([[1.5123, 1.3038]])
A_1 = np.array([[0.81940182, 0.78647382]])

W_2 = np.array([[0.92, 0.15], [0.34, 0.83]])
B_2 = np.array([[0.07, 0.53]])

Z_2 = np.array([[1.09125077, 1.30568354]])
A_2 = np.array([[0.74861718, 0.78678996]])

W_3 = np.array([[0.12], [0.23]])
B_3 = np.array([[0.49]])

Z_3 = np.array([[0.76079575]])
A_3 = np.array([[0.68152648]])

# Additional Y for the expected value of the network
Y = np.array([[0.0]])

# ========= 1. Cost calculation =========
# cost calculation using squared error cost function
J = 1/2 * (Y - A_3) ** 2

# ========= 2. Backpropagation / gradient calculation =========
# compute the derivatives for output later
dA_3 = A_3 - Y
print(f"The derivative of A_3: {dA_3}")

dZ_3 = dA_3 * A_3 * (1 - A_3)
print(f"The derivative of Z_3: {dZ_3}")

dW_3 = A_2.T @ dZ_3
print(f"The derivative of W_3: {dW_3}")

dB_3 = dZ_3
print(f"The derivative of B_3: {dB_3}")

dA_2 = dZ_3 @ W_3.T
print(f"The derivative of A_2: {dA_2}")

# compute the derivaties for hidden layer 2
dZ_2 = dA_2 * A_2 * (1 - A_2)
print(f"The derivative of Z_2: {dZ_2}")

dW_2 = A_1.T @ dZ_2
print(f"The derivative of W_2: {dW_2}")

dB_2 = dZ_2
print(f"The derivative of B_2: {dB_2}")

dA_1 = dZ_2 @ W_2.T
print(f"The derivative of A_1: {dA_1}")

# compute the derivates for hidden layer 1
dZ_1 = dA_1 * A_1 * (1 - A_1)
print(f"The derivative of Z_1: {dZ_1}")

dW_1 = X.T @ dZ_1
print(f"The derivative of W_1: {dW_1}")

dB_1 = dZ_1
print(f"The derivative of B_1: {dB_1}")

# ========= 3. Update weights and biases =========
# update the weights and biases
learning_rate = 0.1

# Update output layer
W_3 = W_3 - learning_rate * dW_3
print(f"Updated W_3: {W_3}")

B_3 = B_3 - learning_rate * dB_3
print(f"Updated B_3: {B_3}")

# Update hidden layer 2
W_2 = W_2 - learning_rate * dW_2
print(f"Updated W_2: {W_2}")

B_2 = B_2 - learning_rate * dB_2
print(f"Updated B_2: {B_2}")

# Update hidden layer 1
W_1 = W_1 - learning_rate * dW_1
print(f"Updated W_1: {W_1}")

B_1 = B_1 - learning_rate * dB_1
print(f"Updated B_1: {B_1}")

# ========= 4. Repeat forward propagation and backward propagation with epochs =========
 