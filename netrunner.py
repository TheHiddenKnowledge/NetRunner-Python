import numpy as np
import pygame


# NetRunner for Python
# Created by Isaiah Finney
# This module provides a means to create, execute, and optimize a neural net
# The gradient descent uses momentum to increase optimization speed

class net:
    # Initializes the net object with the desired parameters
    def __init__(self, inputs, outputs, layers, alpha, beta, squish):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        # The output of each layer before the sigmoid is applied
        self.layerout = []
        # The output of each layer after the sigmoid is applied
        self.layersquish = []
        # The gradient with respect to the weights
        self.gradientw = []
        # The gradient with respect to the biases
        self.gradientb = []
        self.maxInput = inputs
        self.maxOutput = outputs
        self.layers = layers
        # Parameters used for gradient descent
        self.alpha = alpha
        self.beta = beta
        # Factor used in sigmoid to determine how much the output is "squished"
        self.squish = squish
        # List of number of neurons per layer
        self.totalNeurons = []
        self.totalNeurons.append(self.maxInput)
        for i in range(len(layers)):
            self.totalNeurons.append(self.layers[i])
        self.totalNeurons.append(self.maxOutput)
        self.successes = 0
        self.fails = 0

    # Randomly generates weights and biases for the nets, must be called after creating net object
    def randomGenes(self, minweight, maxweight, minbias, maxbias):
        # Initializing weights and biases for net
        for i in range(len(self.totalNeurons) - 1):
            tempw = (maxweight - minweight) * np.random.random_sample(
                (self.totalNeurons[i + 1], self.totalNeurons[i])) + minweight
            self.weights.append(tempw)
            tempb = (maxbias - minbias) * np.random.random_sample((self.totalNeurons[i + 1])) + minbias
            self.biases.append(tempb)
            self.gradientw.append(np.zeros((self.totalNeurons[i + 1], self.totalNeurons[i])))
            self.gradientb.append(np.zeros((self.totalNeurons[i + 1])))
        # Previous gradient used for the momentum of the gradient
        self.prevgradientw = self.gradientw
        self.prevgradientb = self.gradientb

    # Draws the net using pygame
    def drawnet(self, screen, width, height, offsetx, offsety):
        points = []
        tempp = []
        # Draws the neurons as circles within the bounds provided
        for i in range(len(self.totalNeurons)):
            for j in range(self.totalNeurons[i]):
                x = round(i * (width / (len(self.totalNeurons) - 1)) + offsetx)
                y = round(j * (height / (max(self.totalNeurons) - 1)) + offsety)
                tempp.append([x, y])
                # If the neurons are inputs they are green, outputs they are red, hidden they are grey
                if i == 0:
                    pygame.draw.circle(screen, (0, 255, 0), (x, y), 5)
                elif i == len(self.totalNeurons) - 1:
                    pygame.draw.circle(screen, (255, 0, 0), (x, y), 5)
                else:
                    pygame.draw.circle(screen, (211, 211, 211), (x, y), 5)
            # Adds the point of the neuron to a list
            points.append(tempp)
            tempp = []
        # Draws the connections between neurons using the points list generated before
        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    point1 = points[i][k]
                    point2 = points[i + 1][j]
                    if self.weights[i][j][k] > 0:
                        pygame.draw.line(screen, (
                            0, round(255 * abs(self.weights[i][j][k] / np.amax(abs(self.weights[i])))), 0), point1,
                                         point2)
                    if self.weights[i][j][k] < 0:
                        pygame.draw.line(screen, (
                            round(255 * abs(self.weights[i][j][k] / np.amax(abs(self.weights[i])))), 0, 0), point1,
                                         point2)
        font = pygame.font.Font('freesansbold.ttf', 16)
        stext = font.render('Successes: ' + str(self.successes), True, (255, 255, 255), (0, 0, 0))
        textRect = stext.get_rect()
        w, h = stext.get_size()
        textRect.center = (width + 2 * offsetx + w / 2, 2 * offsety)
        screen.blit(stext, textRect)
        ftext = font.render('Fails: ' + str(self.fails), True, (255, 255, 255), (0, 0, 0))
        textRect = ftext.get_rect()
        w, h = ftext.get_size()
        textRect.center = (width + 2 * offsetx + w / 2, 4 * offsety)
        screen.blit(ftext, textRect)

    # Applies the sigmoid function to a given input
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x / self.squish))

    # Gets the derivative of the sigmoid function at a given input
    def sigderiv(self, x):
        return (np.exp(x / self.squish) / (np.exp(x / self.squish) + 1) ** 2) / self.squish

    # Given that the input is defined this function will evaluate the net to get the output
    def in2out(self):
        self.layerout = []
        self.layersquish = []
        current = np.array(self.inputs)
        self.layerout.append(current)
        self.layersquish.append(current)
        for i in range(len(self.weights)):
            current = np.matmul(self.weights[i], current) + self.biases[i]
            self.layerout.append(current)
            for j in range(current.shape[0]):
                current[j] = self.sigmoid(current[j])
            self.layersquish.append(current)
        return current

    # The output is typically between 0 and 1, so this function will adjust the output to the desired values
    def adjustoutput(self, minOut, maxOut):
        return (maxOut - minOut) * self.outputs + minOut

    # Loads net from an npz file
    def loadnet(self, filename):
        if len(np.load(filename + '.npz', allow_pickle=True).files) > 1:
            self.weights = np.load(filename + '.npz', allow_pickle=True)['arr_0']
            self.biases = np.load(filename + '.npz', allow_pickle=True)['arr_1']

    # Saves net to an npz file
    def savenet(self, filename):
        np.savez(filename, self.weights, self.biases, fmt='%s')

    # Parses a set of data from a text file to be used for training
    def parsesets(self, filename):
        sets = []
        f = open(filename, 'r')
        for line in f.readlines():
            sets.append(eval(line.strip()))
        return sets

    # Gets the gradient of the weights and biases from the expected
    def getgradient(self, expected):
        tempg = []
        for i in range(len(self.weights)):
            idx = len(self.weights) - i - 1
            tempgg = []
            for j in range(self.weights[idx].shape[0]):
                # Calculates gradient of the biases using chain rule
                if i == 0:
                    self.gradientb[idx][j] = self.sigderiv(self.layerout[-1][j]) * 2 * (
                            self.layersquish[idx + 1][j] - expected[j])
                else:
                    self.gradientb[idx][j] = self.sigderiv(self.layerout[idx + 1][j]) * \
                                             tempg[i - 1][j]
                for k in range(self.weights[idx].shape[1]):
                    # Calculates gradient of the weights using chain rule
                    if i == 0:
                        self.gradientw[idx][j][k] = self.layerout[idx][k] * self.sigderiv(
                            self.layerout[-1][j]) * 2 * (
                                                            self.layersquish[-1][j] - expected[j])
                        # Uses backpropagation to find the gradients of the previous layers
                        if j == 0:
                            tempgg.append(self.weights[idx][j][k] * self.sigderiv(self.layerout[idx + 1][j]) * 2 * (
                                    self.layersquish[idx + 1][j] - expected[j]))
                        else:
                            tempgg[k] += self.weights[idx][j][k] * self.sigderiv(self.layerout[idx + 1][j]) * 2 * (
                                    self.layersquish[idx + 1][j] - expected[j])
                    else:
                        self.gradientw[idx][j][k] = self.layerout[idx][k] * self.sigderiv(self.layerout[idx + 1][j]) * \
                                                    tempg[i - 1][j]
                        if j == 0:
                            tempgg.append(
                                self.weights[idx][j][k] * self.sigderiv(self.layerout[idx + 1][j]) * tempg[i - 1][j])
                        else:
                            tempgg[k] += self.weights[idx][j][k] * self.sigderiv(self.layerout[idx + 1][j]) * \
                                         tempg[i - 1][j]
            tempg.append(tempgg)

    # Gets the average gradient for a set of data and applies the changes to the net
    def getset(self, inputset, expectedset):
        # Average gradients for the set
        avg_gw = []
        avg_gb = []
        # Avg error for the set
        avg_error = 0
        for i in range(len(inputset)):
            self.inputs = inputset[i]
            self.in2out()
            self.getgradient(expectedset[i])
            for j in range(len(self.weights)):
                if i == 0:
                    avg_gw.append(self.gradientw[j])
                    avg_gb.append(self.gradientb[j])
                else:
                    avg_gw[j] += self.gradientw[j]
                    avg_gb[j] += self.gradientb[j]
            # Calculated the avg error for the set
            avg_expected = 0
            for j in range(len(expectedset[i])):
                avg_expected += (self.layersquish[-1][j] - expectedset[i][j]) ** 2
            avg_expected /= len(expectedset[i])
            avg_error += avg_expected
        avg_error /= len(expectedset)
        # Descending the gradient using the momentum theorem for neural nets
        for i in range(len(self.weights)):
            self.weights[i] -= (self.beta * self.prevgradientw[i] + self.alpha * avg_gw[i]) / len(inputset)
            self.biases[i] -= (self.beta * self.prevgradientb[i] + self.alpha * avg_gb[i]) / len(inputset)
        self.prevgradientw = avg_gw
        self.prevgradientb = avg_gb
        return avg_error
