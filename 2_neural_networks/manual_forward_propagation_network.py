#!/usr/bin/env python
# coding: utf-8

import numpy as np

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    network = {}
    num_nodes_previous = num_inputs

    for layer in range(num_hidden_layers + 1):

        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]

        network[layer_name] = {}

        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)

            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=(num_nodes_previous, 1)), decimals=2),
                'bias': np.around(np.random.uniform(size=(1,1)), decimals=2)
            }

        num_nodes_previous = num_nodes

    return network

num_inputs = 2
num_hidden_layers = 2
num_nodes_hidden = [2, 2]
num_nodes_output = 1

network = initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output)
print(network)

num_inputs = 5
num_hidden_layers = 3
num_nodes_hidden = [2, 3, 4]
num_nodes_output = 2

network = initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output)
print(network)

def compute_weighted_sum(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

def compute_node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-weighted_sum))

# generate random inputs
np.random.seed(12)
inputs = np.around(np.random.uniform(size=(1,5)), decimals=2)
print(f"The inputs of the network are {inputs}")
print(network.keys())

def forward_propagate(network, inputs):
    layer_inputs = inputs
    for layer in network:
        layer_data = network[layer]
        layer_outputs = []

        for node in layer_data:
            node_data = layer_data[node]

            node_weights = node_data['weights']
            node_bias = node_data['bias']

            z = compute_weighted_sum(layer_inputs, node_weights, node_bias)
            a = compute_node_activation(z)

            layer_outputs.append(a.item())

        layer_inputs = layer_outputs

    return layer_outputs

print(f"The output of the network is {forward_propagate(network, inputs)}")
