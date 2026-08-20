#!/usr/bin/env python
# coding: utf-8

import numpy as np

# network has 2 inputs, 2 hidden layers with 2 nodes each and 1 output
# Input       Layer 1        Layer 2        Output

# x₁ ──────┐
#          ├──→ a₁₁ ──────┐
# x₂ ──────┘              ├──→ a₁₂ ───────┐
#          ┌──→ a₂₁ ──────┘               │
#          │                              ├──→ a₁₃
#          └────────────────→ a₂₂ ────────┘

# inputs
x_1 = 0.53
x_2 = 0.81

# layer 1 weights
w_1 = [0.62, 0.93] # weights of x_1
w_2 = [0.77, 0.89] # weights of x_2

# layer 1 biases
b_1 = [0.56, 0.09]

# compute weighted sum of the inputs for layer 1
z_11 = x_1 * w_1[0] + x_2 * w_2[0] + b_1[0]
print(f"The weighted sum of the first node in layer 1 is {z_11}")

z_21 = x_1 * w_1[1] + x_2 * w_2[1] + b_1[1]
print(f"The weighted sum of the second node in layer 1 is {z_21}")

# using the sigmoid activation function
# compute for the activations of the nodes
a_11 = 1.0 / (1.0 + np.exp(-z_11))
print(f"The activation of the first node in layer 1 is {a_11}")

a_21 = 1.0 / (1.0 + np.exp(-z_21))
print(f"The activation of the second node in layer 1 is {a_21}")

print("--------------------------------")
# layer 2 weights
w_11 = [0.92, 0.15] # weights of a11
w_21 = [0.34, 0.83] # weights of a21

# layer 2 biases
b_2 = [0.07, 0.53]

# compute weighted sum of the activation in layer 1 for layer 2
z_12 = a_11 * w_11[0] + a_21 * w_21[0] + b_2[0]
print(f"The weighted sum of the first node in layer 2 is {z_12}")

z_22 = a_11 * w_11[1] + a_21 * w_21[1] + b_2[1]
print(f"The weighted sum of the second node in layer 2 is {z_22}")

# using the sigmoid activation function
# compute for the activations of the nodes
a_12 = 1.0 / (1.0 + np.exp(-z_12))
print(f"The activation of the first node in layer 2 is {a_12}")

a_22 = 1.0 / (1.0 + np.exp(-z_22))
print(f"The activation of the second node in layer 2 is {a_22}")

print("--------------------------------")
# weights for output layer
w_12 = 0.12
w_22 = 0.23

# output layer bias
b_3 = 0.49

z_13 = a_12 * w_12 + a_22 * w_22 + b_3
print(f"The weighted sum of the output layer is {z_13}")

a_13 = 1.0 / (1.0 + np.exp(-z_13))
print(f"The output of the network is {a_13}")
