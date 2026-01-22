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
import pygame

## @class NetRunner
# @brief Contains methods and attributes used for implementing and training a
# neural net.
class NetRunner:
    ## @param neuron_count Neuron count in each layer
    # @param activation String declaring which activation function to use
    # @param alpha Alpha value using in momentum based gradient descent
    # @param beta Beta value using in momentum based gradient descent
    def __init__(self, neuron_counts, activation, alpha, beta):
        ## @brief Neuron count in each layer
        # @hideinitializer
        self.neuron_counts = neuron_counts
        ## @brief Layer values before activation
        # @hideinitializer
        self.pre_active_values = []
        ## @brief Output values for each layer
        # @hideinitializer
        self.neuron_values = []
        ## @brief Acitvation function derivatives
        # @hideinitializer
        self.active_derivatives = []
        for neuron_count in self.neuron_counts:
            self.pre_active_values.append(np.zeros((neuron_count, 1)))
            self.neuron_values.append(np.zeros((neuron_count, 1)))
            self.active_derivatives.append(np.zeros((neuron_count, 1)))
        ## @brief String declaring which activation function to use
        # @hideinitializer
        self.activation = activation

        ## @brief Weights used to transform the values from one layer into
        # another
        # @hideinitializer
        self.weights = []
        for a in range(len(self.neuron_counts) - 1):
            self.weights.append(np.zeros((self.neuron_counts[a + 1],
                                          self.neuron_counts[a])))
        ## @brief Biases applied to each layer output after transformation
        # @hideinitializer
        self.biases = []
        for a in range(len(self.neuron_counts) - 1):
            self.biases.append(np.zeros((self.neuron_counts[a + 1], 1)))
        ## @brief Gradient for all weights
        # @hideinitializer
        self.weight_gradient = copy.deepcopy(self.weights)
        ## @brief Gradient for all biases
        # @hideinitializer
        self.bias_gradient = copy.deepcopy(self.biases)

        ## @brief Alpha value using in momentum based gradient descent
        # @hideinitializer
        self.alpha = alpha
        ## @brief Beta value using in momentum based gradient descent
        # @hideinitializer
        self.beta = beta
        ## @brief Previous average gradient for all weights
        # @hideinitializer
        self.prev_avg_weight_gradient = copy.deepcopy(self.weights)
        ## @brief Previous average gradient for all biases
        # @hideinitializer
        self.prev_avg_bias_gradient = copy.deepcopy(self.biases)

    ## @brief Randomizes the weights and biases of the neural net.
    # @param min_value Minimum allowed value
    # @param max_value Maximum allowed value
    # @return None
    def randomize_net(self, min_value, max_value):
        for weight in self.weights:
            random_weight = np.random.uniform(min_value, max_value,
                                                        weight.shape)
            np.copyto(weight, random_weight)
        for bias in self.biases:
            random_bias = np.random.uniform(min_value, max_value,
                                                      bias.shape)
            np.copyto(bias, random_bias)

    ## @brief Runs the neural net given an input array
    # @param net_input Input for the neural net (inputted as a list)
    # @return Output values of neural net
    def run_net(self, net_input):
        np.copyto(self.neuron_values[0], np.array(net_input).reshape(-1, 1))
        for a in range(len(self.weights)):
            current_layer = self.neuron_values[a]
            # Transforming the previous layer into the next layer
            next_layer = (np.matmul(self.weights[a], current_layer)
                          + self.biases[a])
            np.copyto(self.pre_active_values[a + 1], next_layer)
            # Applying the activation function
            active_layer = self.use_active_function(next_layer)
            np.copyto(self.neuron_values[a + 1], active_layer)
            # Getting the activation function derivatives
            active_derivatives = self.get_active_derivative(next_layer)
            np.copyto(self.active_derivatives[a + 1], active_derivatives)
        # Returns the last layer (output)
        return self.neuron_values[-1]

    ## @brief Applies the applicable activation function
    # @param layer_values Array of values for the layer
    # @return Array of layer values with the activation function applied
    def use_active_function(self, layer_values):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-layer_values))
        elif self.activation == 'tanh':
            return np.tanh(layer_values)
        elif self.activation == 'relu':
            return np.where(layer_values > 0, layer_values, 0)
        else:
            return layer_values

    ## @brief Gets the derivatives for the applicable activation function
    # @param layer_values Array of values for the layer
    # @return Array of activation function derivatives
    def get_active_derivative(self, layer_values):
        active_values = self.use_active_function(layer_values)
        if self.activation == 'sigmoid':
            return active_values * (1 - active_values)
        elif self.activation == 'tanh':
            return 1 - np.power(active_values, 2)
        elif self.activation == 'relu':
            return np.where(active_values > 0, 1, 0)
        else:
            return active_values

    ## @brief Gets the gradients for the weights and biases
    # @param expected_output The expected output for the training example
    # (inputted as a list)
    # @return None
    def get_gradients(self, expected_output):
        expected_array = np.array(expected_output).reshape(-1, 1)
        # List of layer derivatives with respect to cost
        # (used for backpropagation)
        layer_derivatives = copy.deepcopy(self.neuron_values)
        for matrix in layer_derivatives:
            matrix.fill(0)
        for a in reversed(range(len(self.weights))):
            for b in range(self.weights[a].shape[0]):
                # Getting bias gradient for last hidden layer
                if a == len(self.weights) - 1:
                    self.bias_gradient[a][b] = (
                        self.active_derivatives[a + 1][b][0]
                        * 2 * (self.neuron_values[a + 1][b][0]
                               - expected_array[b][0])
                    )
                # Getting bias gradient used backpropagation
                else:
                    self.bias_gradient[a][b] = (
                        self.active_derivatives[a + 1][b][0]
                        * layer_derivatives[a + 1][b][0]
                    )
                for c in range(self.weights[a].shape[1]):
                    if a == len(self.weights) - 1:
                        # Getting weight gradient for last hidden layer
                        self.weight_gradient[a][b][c] = (
                            self.neuron_values[a][c][0]
                            * self.active_derivatives[a + 1][b][0]
                            * 2 * (self.neuron_values[a + 1][b][0]
                                   - expected_array[b][0])
                        )
                        # Calculating layer derivatives with respect to cost
                        layer_derivatives[a][c][0] += (
                            self.weights[a][b][c]
                            * self.active_derivatives[a + 1][b][0]
                            * 2 * (self.neuron_values[a + 1][b][0]
                                   - expected_array[b][0])
                        )
                    else:
                        # Getting weight gradient using backpropagation
                        self.weight_gradient[a][b][c] = (
                            self.neuron_values[a][c][0]
                            * self.active_derivatives[a + 1][b][0]
                            * layer_derivatives[a + 1][b][0]
                        )
                        # Calculating layer derivatives with respect to cost
                        layer_derivatives[a + 1][c][0] += (
                            self.weights[a][b][c]
                            * self.active_derivatives[a + 1][b][0]
                            * layer_derivatives[a + 1][b][0]
                        )

    ## @brief Performs gradient descent on the nerual net
    # @param input_set Set of input values
    # @param expected_set Set of expected output values
    # @return None
    def use_gradient_descent(self, input_set, expected_set):
        # Average gradients for the set
        avg_weight_gradient = copy.deepcopy(self.weights)
        avg_bias_gradient = copy.deepcopy(self.biases)
        for a in range(len(self.weights)):
            avg_weight_gradient[a].fill(0)
            avg_bias_gradient[a].fill(0)
        # Average cost function value
        avg_cost = 0
        for a in range(len(input_set)):
            self.run_net(input_set[a])
            self.get_gradients(expected_set[a])
            for b in range(len(self.weights)):
                avg_weight_gradient[b] += (self.weight_gradient[b]
                                           / len(expected_set))
                avg_bias_gradient[b] += (self.bias_gradient[b]
                                         / len(expected_set))
            # Cost function value
            cost = 0
            for b in range(len(expected_set[a])):
                cost += (self.neuron_values[-1][b][0] - expected_set[a][b]) ** 2
            avg_cost += cost / len(expected_set)
        # Descending the gradient using the momentum theorem for neural nets
        for a in range(len(self.weights)):
            self.weights[a] -= (self.beta * self.prev_avg_weight_gradient[a]
                                + self.alpha * avg_weight_gradient[a])
            self.biases[a] -= (self.beta * self.prev_avg_bias_gradient[a]
                               + self.alpha * avg_bias_gradient[a])
            np.copyto(self.prev_avg_weight_gradient[a], avg_weight_gradient[a])
            np.copyto(self.prev_avg_bias_gradient[a], avg_bias_gradient[a])
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


    # # Draws the net using pygame
    # def drawnet(self, screen, width, height, offsetx, offsety):
    #     points = []
    #     tempp = []
    #     # Draws the neurons as circles within the bounds provided
    #     for i in range(len(self.neuron_counts)):
    #         for j in range(self.neuron_counts[i]):
    #             x = round(i * (width / (len(self.neuron_counts) - 1)) + offsetx)
    #             y = round(j * (height / (max(self.neuron_counts) - 1)) + offsety)
    #             tempp.append([x, y])
    #             # If the neurons are inputs they are green, outputs they are red, hidden they are grey
    #             if i == 0:
    #                 pygame.draw.circle(screen, (0, 255, 0), (x, y), 5)
    #             elif i == len(self.neuron_counts) - 1:
    #                 pygame.draw.circle(screen, (255, 0, 0), (x, y), 5)
    #             else:
    #                 pygame.draw.circle(screen, (211, 211, 211), (x, y), 5)
    #         # Adds the point of the neuron to a list
    #         points.append(tempp)
    #         tempp = []
    #     # Draws the connections between neurons using the points list generated before
    #     for i in range(len(self.weights)):
    #         for j in range(self.weights[i].shape[0]):
    #             for k in range(self.weights[i].shape[1]):
    #                 point1 = points[i][k]
    #                 point2 = points[i + 1][j]
    #                 if self.weights[i][j][k] > 0:
    #                     pygame.draw.line(screen, (
    #                         0, round(255 * abs(self.weights[i][j][k] / np.amax(abs(self.weights[i])))), 0), point1,
    #                                      point2)
    #                 if self.weights[i][j][k] < 0:
    #                     pygame.draw.line(screen, (
    #                         round(255 * abs(self.weights[i][j][k] / np.amax(abs(self.weights[i])))), 0, 0), point1,
    #                                      point2)
    #     font = pygame.font.Font('freesansbold.ttf', 16)
    #     stext = font.render('Successes: ' + str(self.successes), True, (255, 255, 255), (0, 0, 0))
    #     textRect = stext.get_rect()
    #     w, h = stext.get_size()
    #     textRect.center = (width + 2 * offsetx + w / 2, 2 * offsety)
    #     screen.blit(stext, textRect)
    #     ftext = font.render('Fails: ' + str(self.fails), True, (255, 255, 255), (0, 0, 0))
    #     textRect = ftext.get_rect()
    #     w, h = ftext.get_size()
    #     textRect.center = (width + 2 * offsetx + w / 2, 4 * offsety)
    #     screen.blit(ftext, textRect)
    #
    # # Applies the sigmoid function to a given input
    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x / self.squish))
    #
    # # Gets the derivative of the sigmoid function at a given input
    # def sigderiv(self, x):
    #     return (np.exp(x / self.squish) / (np.exp(x / self.squish) + 1) ** 2) / self.squish
    #
