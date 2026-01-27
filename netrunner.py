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
import random as rnd
import math

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
        self.original_values = layer_values.copy()
        self.use_active_functions()
        self.get_active_derivatives()

    ## @brief Applies the applicable activation function
    # @return None
    def use_active_functions(self):
        if self.activation == 'input':
            self.active_values = self.original_values.copy()
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
    ## @param net_layers List of NetLayer objects
    def __init__(self, net_layers):
        ## @brief List of NetLayer objects
        # @hideinitializer
        self.__net_layers = copy.deepcopy(net_layers)
        self.__check_net()

        ## @brief Weights used to transform the values from one layer into
        # another
        # @hideinitializer
        self.__weights = []
        for a in range(len(self.__net_layers) - 1):
            neuron_count = self.__net_layers[a].neuron_count
            next_neuron_count = self.__net_layers[a + 1].neuron_count
            self.__weights.append(np.zeros((next_neuron_count, neuron_count)))
        ## @brief Biases applied to each layer output after transformation
        # @hideinitializer
        self.__biases = []
        for a in range(len(self.__net_layers) - 1):
            next_neuron_count = self.__net_layers[a + 1].neuron_count
            self.__biases.append(np.zeros((next_neuron_count, 1)))
        ## @brief Gradient for all weights
        # @hideinitializer
        self.__weight_gradient = copy.deepcopy(self.__weights)
        ## @brief Gradient for all biases
        # @hideinitializer
        self.__bias_gradient = copy.deepcopy(self.__biases)

        ## @brief Previous average gradient velocity for all weights
        # @hideinitializer
        self.__prev_weight_velocity = copy.deepcopy(self.__weights)
        ## @brief Previous average gradient velocity for all biases
        # @hideinitializer
        self.__prev_bias_velocity = copy.deepcopy(self.__biases)

        ## @brief Learning rate used for gradient descent
        # @hideinitializer
        self.__rate = 0
        ## @brief Velocity utilization factor for gradient descent
        # @hideinitializer
        self.__beta = 0

        ## @brief Size of the mini batch
        # @hideinitializer
        self.__mini_batch_size = 0
        ## @brief Maximum index used for looping through the mini batches
        # @hideinitializer
        self.__mini_batch_max_index = 0
        ## @brief Current epoch count
        # @hideinitializer
        self.__epoch = 0

    ## @brief Checks the layers of the neural net to see if the net is valid.
    # @return None
    def __check_net(self):
        # The minimum layer count is 2
        if len(self.__net_layers) < 2:
            raise NetError(self, "Layer count is less than 2.")
        # Checking for the input layer
        input_index = -1
        for a in range(len(self.__net_layers)):
            net_layer = self.__net_layers[a]
            if net_layer.activation == 'input':
                input_index = a
        if input_index < 0:
            raise NetError(self, "Input layer not present.")
        elif input_index > 0:
            raise NetError(self, "Input layer is not at the right position.")

    ## @brief Resizes the neural net given the new layers list.
    # @param net_layers New list of NetLayer objects
    # @return None
    def resize_net(self, net_layers):
        has_resized = False
        # Checking if the neural net has resized
        if len(net_layers) != len(self.__net_layers):
            has_resized = True
        else:
            for a in range(len(net_layers)):
                new_neuron_count = net_layers[a].neuron_count
                old_neuron_count = self.__net_layers[a].neuron_count
                if new_neuron_count != old_neuron_count:
                    has_resized = True
                    break
        # Resizing neural net parameters
        if has_resized:
            self.__net_layers = copy.deepcopy(net_layers)
            self.__check_net()

            self.__weights = []
            for a in range(len(self.__net_layers) - 1):
                neuron_count = self.__net_layers[a].neuron_count
                next_neuron_count = self.__net_layers[a + 1].neuron_count
                self.__weights.append(
                    np.zeros((next_neuron_count, neuron_count)))
            self.__biases = []
            for a in range(len(self.__net_layers) - 1):
                next_neuron_count = self.__net_layers[a + 1].neuron_count
                self.__biases.append(np.zeros((next_neuron_count, 1)))
            self.__weight_gradient = copy.deepcopy(self.__weights)
            self.__bias_gradient = copy.deepcopy(self.__biases)
            self.__prev_weight_velocity = copy.deepcopy(self.__weights)
            self.__prev_bias_velocity = copy.deepcopy(self.__biases)

    ## @brief Initializes the weights and biases of the neural net.
    # @return None
    def init_net(self):
        for weight in self.__weights:
            variance = 1 / weight.shape[1]
            random_weight = np.random.uniform(-variance, variance, weight.shape)
            np.copyto(weight, random_weight)
        for bias in self.__biases:
            bias.fill(0)

    ## @brief Forward passes the neural net given an input array
    # @param net_input Input for the neural net
    # @return Output values of neural net
    def forward_pass(self, net_input):
        self.__net_layers[0].set_layer_values(net_input)
        for a in range(len(self.__weights)):
            current_values = self.__net_layers[a].active_values
            # Transforming the previous layer into the next layer
            next_values = (np.matmul(self.__weights[a], current_values)
                          + self.__biases[a])
            self.__net_layers[a + 1].set_layer_values(next_values)
        # Returns the last layer (output)
        return self.__net_layers[-1].active_values

    ## @brief Gets the gradients for the weights and biases
    # @param expected_output The expected output for the training example
    # @return None
    def __get_gradients(self, expected_output):
        # List of layer derivatives with respect to cost
        # (used for backpropagation)
        layer_derivatives = []
        for net_layer in self.__net_layers:
            layer_derivatives.append(np.zeros_like(net_layer.original_values))
        for a in reversed(range(len(self.__weights))):
            active_values = self.__net_layers[a].active_values
            next_active_values = self.__net_layers[a + 1].active_values
            next_active_derivatives = self.__net_layers[a + 1].active_derivatives
            for b in range(self.__weights[a].shape[0]):
                # Getting bias gradient for last hidden layer
                if a == len(self.__weights) - 1:
                    self.__bias_gradient[a][b] = (
                        next_active_derivatives[b][0]
                        * 2 * (next_active_values[b][0]
                               - expected_output[b][0])
                    )
                # Getting bias gradient used backpropagation
                else:
                    self.__bias_gradient[a][b] = (
                        next_active_derivatives[b][0]
                        * layer_derivatives[a + 1][b][0]
                    )
                for c in range(self.__weights[a].shape[1]):
                    if a == len(self.__weights) - 1:
                        # Getting weight gradient for last hidden layer
                        self.__weight_gradient[a][b][c] = (
                            active_values[c][0]
                            * next_active_derivatives[b][0]
                            * 2 * (next_active_values[b][0]
                                   - expected_output[b][0])
                        )
                        # Calculating layer derivatives with respect to cost
                        layer_derivatives[a][c][0] += (
                            self.__weights[a][b][c]
                            * next_active_derivatives[b][0]
                            * 2 * (next_active_values[b][0]
                                   - expected_output[b][0])
                        )
                    else:
                        # Getting weight gradient using backpropagation
                        self.__weight_gradient[a][b][c] = (
                            active_values[c][0]
                            * next_active_derivatives[b][0]
                            * layer_derivatives[a + 1][b][0]
                        )
                        # Calculating layer derivatives with respect to cost
                        layer_derivatives[a][c][0] += (
                            self.__weights[a][b][c]
                            * next_active_derivatives[b][0]
                            * layer_derivatives[a + 1][b][0]
                        )

    ## @brief Initializes the neural net runner.
    # @param rate Learning rate
    # @param beta Velocity utilization factor
    # @return None
    def init_runner(self, rate, beta):
        self.__rate = rate
        if beta > 1 or beta < 0:
            raise NetError(self.__beta, 'Beta must be between 0 and 1.')
        self.__beta = beta
        self.__epoch = 0

    ## @brief Initializes mini batch parameters.
    # @param batch The whole batch of training examples
    # @param mini_batch_size The size of a mini batch
    # @param is_shuffled Shuffles the batch if true
    # @return None
    def init_mini_batch(self, batch, mini_batch_size, is_shuffled):
        batch_size = len(batch)
        if mini_batch_size > batch_size:
            raise NetError(self.__mini_batch_size, 'Mini batch size must be '
                                                   'less than the batch size.')
        if is_shuffled:
            rnd.shuffle(batch)
        self.__mini_batch_size = mini_batch_size
        self.__mini_batch_max_index = math.ceil(batch_size
                                                / self.__mini_batch_size)

    ## @brief Performs gradient descent on the mini batch.
    # @param mini_batch The mini batch of training examples
    # @return The average cost value of the mini batch
    def __step_mini_batch(self, mini_batch):
        avg_weight_gradient = copy.deepcopy(self.__weights)
        avg_bias_gradient = copy.deepcopy(self.__biases)
        for a in range(len(self.__weights)):
            avg_weight_gradient[a].fill(0)
            avg_bias_gradient[a].fill(0)
        avg_cost = 0
        for a in range(len(mini_batch)):
            self.forward_pass(mini_batch[a][0])
            self.__get_gradients(mini_batch[a][1])
            for b in range(len(self.__weights)):
                avg_weight_gradient[b] += (self.__weight_gradient[b]
                                           / len(mini_batch))
                avg_bias_gradient[b] += (self.__bias_gradient[b]
                                         / len(mini_batch))
            cost = 0
            for b in range(len(mini_batch[a][1])):
                output_values = self.__net_layers[-1].active_values
                cost += (output_values[b][0] - mini_batch[a][1][b][0]) ** 2
            avg_cost += cost / len(mini_batch)
        # Descending the gradient using the momentum theorem for neural nets
        for a in range(len(self.__weights)):
            weight_velocity = (self.__beta * self.__prev_weight_velocity[a]
                               + (1 - self.__beta) * avg_weight_gradient[a])
            bias_veloctiy = (self.__beta * self.__prev_bias_velocity[a]
                             + (1 - self.__beta) * avg_bias_gradient[a])
            self.__weights[a] -= self.__rate * weight_velocity
            self.__biases[a] -= self.__rate * bias_veloctiy
            np.copyto(self.__prev_weight_velocity[a], weight_velocity)
            np.copyto(self.__prev_bias_velocity[a], bias_veloctiy)
        return avg_cost

    ## @brief Performs gradient descent on all mini batches in the epoch.
    # @param batch The whole batch of training examples
    # @return The final average cost and the epoch count
    def step_epoch(self, batch):
        avg_cost = 0
        for a in range(self.__mini_batch_max_index):
            start_idx = a * self.__mini_batch_size
            end_idx = (a + 1) * self.__mini_batch_size - 1
            mini_batch = batch[start_idx:end_idx]
            avg_cost = self.__step_mini_batch(mini_batch)
        self.__epoch += 1
        return avg_cost, self.__epoch

    ## @brief Loads neural net from a npz file
    # @param file File name
    # @return None
    def loadnet(self, file):
        if len(np.load(file + '.npz', allow_pickle=True).files) > 1:
            self.__weights = np.load(file + '.npz', allow_pickle=True)['arr_0']
            self.__biases = np.load(file + '.npz', allow_pickle=True)['arr_1']

    ## @brief Save neural net to a npz file
    # @param file File name
    # @return None
    def savenet(self, file):
        np.savez(file, self.__weights, self.__biases, fmt='%s')