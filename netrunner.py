import numpy as np
import pygame


class net:
    def __init__(self, inputs, outputs, layers, minOut, maxOut, squish):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.layerout = []
        self.layersquish = []
        self.gradientw = []
        self.gradientb = []
        self.generation = 0
        self.maxInput = inputs
        self.maxOutput = outputs
        self.layers = layers
        self.minOut = minOut
        self.maxOut = maxOut
        self.squish = squish
        self.totalNeurons = []
        self.totalNeurons.append(self.maxInput)
        for i in range(len(layers)):
            self.totalNeurons.append(self.layers[i])
        self.totalNeurons.append(self.maxOutput)
        for i in range(len(self.totalNeurons) - 1):
            tempw = 1 * np.random.random_sample((self.totalNeurons[i + 1], self.totalNeurons[i])) - .5
            self.weights.append(tempw)
            tempb = np.random.random_sample((self.totalNeurons[i + 1]))
            self.biases.append(tempb)
            self.gradientw.append(np.zeros((self.totalNeurons[i + 1], self.totalNeurons[i])))
            self.gradientb.append(np.zeros((self.totalNeurons[i + 1])))

        self.successes = 0
        self.fails = 0

    def randomGenes(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.totalNeurons) - 1):
            tempw = (self.maxOut - self.minOut) * np.random.random_sample(
                (self.totalNeurons[i + 1], self.totalNeurons[i])) - self.minOut
            self.weights.append(tempw)
            tempb = (self.maxOut - self.minOut) * np.random.random_sample((self.totalNeurons[i + 1])) - self.minOut
            self.biases.append(tempb)

    def drawnet(self, screen, width, height, offsetx, offsety):
        points = []
        tempp = []
        for i in range(len(self.totalNeurons)):
            for j in range(self.totalNeurons[i]):
                x = round(i * (width / (len(self.totalNeurons) - 1)) + offsetx)
                y = round(j * (height / (max(self.totalNeurons) - 1)) + offsety)
                tempp.append([x, y])
                if i == 0:
                    pygame.draw.circle(screen, (0, 255, 0), (x, y), 5)
                elif i == len(self.totalNeurons) - 1:
                    pygame.draw.circle(screen, (255, 0, 0), (x, y), 5)
                else:
                    pygame.draw.circle(screen, (211, 211, 211), (x, y), 5)
            points.append(tempp)
            tempp = []
        # print(points)
        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    point1 = points[i][k]
                    point2 = points[i + 1][j]
                    if self.weights[i][j][k] > 0:
                        pygame.draw.line(screen, (
                        0, round(255 * abs(self.weights[i][j][k] / np.amax(abs(self.weights[i])))), 0), point1, point2)
                    if self.weights[i][j][k] < 0:
                        pygame.draw.line(screen, (
                        round(255 * abs(self.weights[i][j][k] / np.amax(abs(self.weights[i])))), 0, 0), point1, point2)
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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x / self.squish))

    def sigderiv(self, x):
        return (np.exp(x/self.squish)/(np.exp(x/self.squish)+1)**2)/self.squish

    def in2out(self):
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

    def adjustoutput(self):
        return (self.maxOut - self.minOut) * self.outputs + self.minOut

    def loadnet(self, filename):
        if len(np.load(filename + '.npz', allow_pickle=True).files) != 0:
            self.weights = np.load(filename + '.npz', allow_pickle=True)['arr_0']
            self.biases = np.load(filename + '.npz', allow_pickle=True)['arr_1']

    def savenet(self, filename):
        np.savez(filename, self.weights, self.biases, fmt='%s')

    def getgradient(self, expected):
        tempg = []
        for i in range(len(self.weights)):
            idx = len(self.weights) - i - 1
            tempgg = []
            for j in range(self.weights[idx].shape[0]):
                if i == 0:
                    self.gradientb[idx][j] = self.sigderiv(self.layerout[idx + 1][j]) * 2 * (
                            self.layersquish[idx][j] - expected[j])
                else:
                    self.gradientb[idx][j] = self.sigderiv(self.layerout[idx + 1][j]) * \
                                                tempg[i - 1][j]
                for k in range(self.weights[idx].shape[1]):
                    if i == 0:
                        self.gradientw[idx][j][k] = self.layerout[idx][k] * self.sigderiv(self.layerout[idx+1][j]) * 2 * (
                                    self.layersquish[idx][j] - expected[j])
                        if j == 0:
                            tempgg.append(self.weights[idx][j][k] * self.sigderiv(self.layerout[idx+1][j]) * 2 * (
                                        self.layersquish[idx+1][j] - expected[j]))
                        else:
                            tempgg[k] += self.weights[idx][j][k] * self.sigderiv(self.layerout[idx+1][j]) * 2 * (
                                    self.layersquish[idx+1][j] - expected[j])
                    else:
                        self.gradientw[idx][j][k] = self.layerout[idx][k] * self.sigderiv(self.layerout[idx+1][j]) * tempg[i-1][j]
                        if j == 0:
                            tempgg.append(self.weights[idx][j][k] * self.sigderiv(self.layerout[idx+1][j]) * tempg[i - 1][j])
                        else:
                            tempgg[k] += self.weights[idx][j][k] * self.sigderiv(self.layerout[idx+1][j]) * tempg[i - 1][j]
            tempg.append(tempgg)
            #print(tempg)
        print(self.weights)
        print(self.gradientw)
        print(self.biases)
        print(self.gradientb)



