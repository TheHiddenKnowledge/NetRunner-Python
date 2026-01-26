## @mainpage
# This project aims to create a neural net library based in python that can
# be used for various applications in other projects.
# @par Latest Release:
# V1.2 - 1/17/2026
# @par Created by: I. Finney
# @par Revision History:
# @version 1.0 Initial release.
# @version 1.1 Added gradient descent.
# @version 1.2

## @file netrunner.py
# @brief Implements a functional neural net library.

import numpy as np
import copy

## @class NetError
# @brief Exception class for custom error handling
class NetError(Exception):
    def __init__(self, value, message):
        self.value = value
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        # Override the default string representation
        return f'{self.value} -> {self.message}'

## @class NetLayer
# @brief Contains methods and attributes for a neural net layer
class NetLayer:
    ## @param neuron_count Neuron count in the layer
    # @param activation String declaring which activation function to use
    def __init__(self, neuron_count, activation):
        ## @brief Neuron count in the layer
        # @hideinitializer
        self.neuron_count = neuron_count
        ## @brief String declaring which activation function to use
        # @hideinitializer
        self.activation = activation
        ## @brief Layer values before the activation function is applied
        # @hideinitializer
        self.original_values = np.zeros((neuron_count, 1))
        ## @brief Layer values after the activation function is applied
        # @hideinitializer
        self.active_values = np.zeros((neuron_count, 1))
        ## @brief Derivative of the activation function for each layer value
        # @hideinitializer
        self.active_derivatives = np.zeros((neuron_count, 1))

    ## @brief Sets the layers values, applies the activation function,
    # and gets the applicable derivatives
    # @param layer_values Layer values to be assigned
    # @return None
    def set_layer_values(self, layer_values):
        self.original_values = layer_values
        self.use_active_functions()
        self.get_active_derivatives()

    ## @brief Applies the applicable activation function
    # @return None
    def use_active_functions(self):
        if self.activation == 'input':
            self.active_values = self.original_values
        elif self.activation == 'sigmoid':
            self.active_values = 1 / (1 + np.exp(-self.original_values))
        elif self.activation == 'tanh':
            self.active_values = np.tanh(self.original_values)
        elif self.activation == 'relu':
            self.active_values = np.where(self.original_values > 0,
                                          self.original_values,
                                          .1 * self.original_values)
        else:
            raise NetError(self.activation, "Invalid activation function.")

    ## @brief Gets the derivatives for the applicable activation function
    # @return None
    def get_active_derivatives(self):
        if self.activation == 'input':
            self.active_derivatives = np.zeros_like(self.original_values)
        elif self.activation == 'sigmoid':
            self.active_derivatives =  (self.active_values
                                        * (1 - self.active_values))
        elif self.activation == 'tanh':
            self.active_derivatives = 1 - np.power(self.active_values, 2)
        elif self.activation == 'relu':
            self.active_derivatives = np.where(self.active_values > 0, 1, .1)
        else:
            raise NetError(self.activation, "Invalid activation function.")

## @class NetRunner
# @brief Contains methods and attributes used for implementing and training a
# neural net.
class NetRunner:
    ## @param net_layers Array of NetLayer objects
    def __init__(self, net_layers):
        ## @brief Array of NetLayer objects
        # @hideinitializer
        self.net_layers = net_layers

        # Checks to see if there is an input layer present
        has_input = False
        for net_layer in self.net_layers:
            if net_layer.activation == 'input':
                has_input = True
        if not has_input:
            raise NetError(self, "Input layer not present.")

        ## @brief Weights used to transform the values from one layer into
        # another
        # @hideinitializer
        self.weights = []
        for a in range(len(self.net_layers) - 1):
            neuron_count = self.net_layers[a].neuron_count
            next_neuron_count = self.net_layers[a + 1].neuron_count
            self.weights.append(np.zeros((next_neuron_count, neuron_count)))
        ## @brief Biases applied to each layer output after transformation
        # @hideinitializer
        self.biases = []
        for a in range(len(self.net_layers) - 1):
            next_neuron_count = self.net_layers[a + 1].neuron_count
            self.biases.append(np.zeros((next_neuron_count, 1)))
        ## @brief Gradient for all weights
        # @hideinitializer
        self.weight_gradient = copy.deepcopy(self.weights)
        ## @brief Gradient for all biases
        # @hideinitializer
        self.bias_gradient = copy.deepcopy(self.biases)

        ## @brief Previous average gradient for all weights
        # @hideinitializer
        self.prev_weight_velocity = copy.deepcopy(self.weights)
        ## @brief Previous average gradient for all biases
        # @hideinitializer
        self.prev_bias_velocity = copy.deepcopy(self.biases)

        self.initialize_net()

    ## @brief Initialize the weights and biases of the neural net.
    # @return None
    def initialize_net(self):
        for weight in self.weights:
            variance = 1 / weight.shape[1]
            random_weight = np.random.uniform(-variance, variance, weight.shape)
            np.copyto(weight, random_weight)
        for bias in self.biases:
            bias.fill(0)

    ## @brief Runs the neural net given an input array
    # @param net_input Input for the neural net (inputted as a list)
    # @return Output values of neural net
    def run_net(self, net_input):
        self.net_layers[0].set_layer_values(np.array(net_input).reshape(-1, 1))
        for a in range(len(self.weights)):
            current_values = self.net_layers[a].active_values
            # Transforming the previous layer into the next layer
            next_values = (np.matmul(self.weights[a], current_values)
                          + self.biases[a])
            self.net_layers[a + 1].set_layer_values(next_values)
        # Returns the last layer (output)
        return self.net_layers[-1].active_values

    ## @brief Gets the gradients for the weights and biases
    # @param expected_output The expected output for the training example
    # (inputted as a list)
    # @return None
    def get_gradients(self, expected_output):
        expected_array = np.array(expected_output).reshape(-1, 1)
        # List of layer derivatives with respect to cost
        # (used for backpropagation)
        layer_derivatives = []
        for net_layer in self.net_layers:
            layer_derivatives.append(np.zeros_like(net_layer.original_values))
        for a in reversed(range(len(self.weights))):
            active_values = self.net_layers[a].active_values
            next_active_values = self.net_layers[a + 1].active_values
            next_active_derivatives = self.net_layers[a + 1].active_derivatives
            for b in range(self.weights[a].shape[0]):
                # Getting bias gradient for last hidden layer
                if a == len(self.weights) - 1:
                    self.bias_gradient[a][b] = (
                        next_active_derivatives[b][0]
                        * 2 * (next_active_values[b][0]
                               - expected_array[b][0])
                    )
                # Getting bias gradient used backpropagation
                else:
                    self.bias_gradient[a][b] = (
                        next_active_derivatives[b][0]
                        * layer_derivatives[a + 1][b][0]
                    )
                for c in range(self.weights[a].shape[1]):
                    if a == len(self.weights) - 1:
                        # Getting weight gradient for last hidden layer
                        self.weight_gradient[a][b][c] = (
                            active_values[c][0]
                            * next_active_derivatives[b][0]
                            * 2 * (next_active_values[b][0]
                                   - expected_array[b][0])
                        )
                        # Calculating layer derivatives with respect to cost
                        layer_derivatives[a][c][0] += (
                            self.weights[a][b][c]
                            * next_active_derivatives[b][0]
                            * 2 * (next_active_values[b][0]
                                   - expected_array[b][0])
                        )
                    else:
                        # Getting weight gradient using backpropagation
                        self.weight_gradient[a][b][c] = (
                            active_values[c][0]
                            * next_active_derivatives[b][0]
                            * layer_derivatives[a + 1][b][0]
                        )
                        # Calculating layer derivatives with respect to cost
                        layer_derivatives[a][c][0] += (
                            self.weights[a][b][c]
                            * next_active_derivatives[b][0]
                            * layer_derivatives[a + 1][b][0]
                        )

    ## @brief Performs gradient descent on the neural net
    # @param input_set Set of input values
    # @param expected_set Set of expected output values
    # @param rate Learning rate for the neural net
    # @param beta Velocity utilization constant
    # @return None
    def step_gradient_descent(self, input_set, expected_set, rate, beta):
        if beta > 1 or beta < 0:
            raise NetError(beta, 'Beta must be between 0 and 1')
        avg_weight_gradient = copy.deepcopy(self.weights)
        avg_bias_gradient = copy.deepcopy(self.biases)
        for a in range(len(self.weights)):
            avg_weight_gradient[a].fill(0)
            avg_bias_gradient[a].fill(0)
        avg_cost = 0
        for a in range(len(input_set)):
            self.run_net(input_set[a])
            self.get_gradients(expected_set[a])
            for b in range(len(self.weights)):
                avg_weight_gradient[b] += (self.weight_gradient[b]
                                           / len(expected_set))
                avg_bias_gradient[b] += (self.bias_gradient[b]
                                         / len(expected_set))
            cost = 0
            for b in range(len(expected_set[a])):
                output_values = self.net_layers[-1].active_values
                cost += (output_values[b][0] - expected_set[a][b]) ** 2
            avg_cost += cost / len(expected_set)
        # Descending the gradient using the momentum theorem for neural nets
        for a in range(len(self.weights)):
            weight_velocity = (beta * self.prev_weight_velocity[a]
                               + (1 - beta) * avg_weight_gradient[a])
            bias_veloctiy = (beta * self.prev_bias_velocity[a]
                             + (1 - beta) * avg_bias_gradient[a])
            self.weights[a] -= rate * weight_velocity
            self.biases[a] -= rate * bias_veloctiy
            np.copyto(self.prev_weight_velocity[a], weight_velocity)
            np.copyto(self.prev_bias_velocity[a], bias_veloctiy)
        return avg_cost

    ## @brief Loads neural net from a npz file
    # @param file File name
    # @return None
    def loadnet(self, file):
        if len(np.load(file + '.npz', allow_pickle=True).files) > 1:
            self.weights = np.load(file + '.npz', allow_pickle=True)['arr_0']
            self.biases = np.load(file + '.npz', allow_pickle=True)['arr_1']

    ## @brief Save neural net to a npz file
    # @param file File name
    # @return None
    def savenet(self, file):
        np.savez(file, self.weights, self.biases, fmt='%s')