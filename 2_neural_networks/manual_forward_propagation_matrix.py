#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

# network has 2 features inputs, 2 hidden layers with 2 nodes each and 1 output
# same network and values in manual_forward_propagation

# input matrix
# feature per column
# training example per row
# [x1, x2] -> 1 training example with 2 features
X = np.array([[0.53, 0.81]])
print(f"Input matrix: {X}")
print(f"Input matrix shape: {X.shape}")

# weight matrix
# layer node weights per column
# feature per row
# [w1, w2]
# [w3, w4]
# x1 weights are w1, w2
# x2 weights are w3, w4
W_1 = np.array([[0.62, 0.93],
                [0.77, 0.89]])
print(f"Input weight matrix: {W_1}")
print(f"Input weight matrix shape: {W_1.shape}")

# bias matrix
# layer node biases per column
# [b1, b2]
B_1 = np.array([[0.56, 0.09]])
print(f"Input bias matrix: {B_1}")
print(f"Input bias matrix shape: {B_1.shape}")

def compute_weighted_sum(inputs, weights, bias):
    # dot matrix:
    # 1x2 (input) . 2x2 (weights) = 1x2
    # [x1*w1 + x2*w3] node1
    # [x1*w2 + x2*w4] node2
    return np.dot(inputs, weights) + bias

Z_1 = compute_weighted_sum(X, W_1, B_1)
print(f"The weighted sum of the first node in layer 1 is {Z_1[0][0]}")
print(f"The weighted sum of the second node in layer 1 is {Z_1[0][1]}")

def compute_node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-weighted_sum))

A_1 = compute_node_activation(Z_1)
print(f"The activation of the first node in layer 1 is {A_1[0][0]}")
print(f"The activation of the second node in layer 1 is {A_1[0][1]}")

# weight matrix
W_2 = np.array([[0.92, 0.15],
                [0.34, 0.83]])

# bias vector
B_2 = np.array([[0.07, 0.53]])

Z_2 = compute_weighted_sum(A_1, W_2, B_2)
print(f"The weighted sum of the first node in layer 2 is {Z_2[0][0]}")
print(f"The weighted sum of the second node in layer 2 is {Z_2[0][1]}")

A_2 = compute_node_activation(Z_2)
print(f"A_2 size: {A_2.shape}")
print(f"The activation of the first node in layer 2 is {A_2[0][0]}")
print(f"The activation of the second node in layer 2 is {A_2[0][1]}")

# weight matrix for output layer
W_3 = np.array([[0.12],
                [0.23]])
print(f"W_3 size: {W_3.shape}")

# bias vector for output layer
B_3 = np.array([[0.49]])
print(f"B_3 size: {B_3.shape}")

Z_3 = compute_weighted_sum(A_2, W_3, B_3)
print(f"The weighted sum of the first node in output layer is {Z_3[0][0]}")

A_3 = compute_node_activation(Z_3)
print(f"The activation of the first node in output layer is {A_3[0][0]}")
