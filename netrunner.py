import random
import pygame

pygame.init()


class net:
    def __init__(self, inputs, outputs, layers):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = []
        self.maxweights = []
        self.synapses = []
        self.maxsynapses = []
        self.fitness = 0
        self.maxfitness = 0
        self.neuronrate = .5
        self.synapserate = .9
        self.weightrate = .9
        self.generation = 0
        self.maxInput = len(inputs)
        self.maxOutput = len(outputs)
        self.maxNeuron = self.maxInput
        self.layers = layers
        self.totalNeurons = []
        self.totalNeurons.append(self.maxInput)
        for i in range(layers):
            self.totalNeurons.append(self.maxNeuron)
        self.totalNeurons.append(self.maxOutput)
        self.actualNeurons = []

    def randomGenes(self):
        self.weights = []
        self.synapses = []
        self.actualNeurons = []
        temps = []
        tempw = []
        tempn = []
        tempd = []
        prevneurons = [[*range(self.totalNeurons[0])]]
        prevrand = []
        deadneurons = []
        for i in range(len(self.totalNeurons) - 1):
            temps = []
            tempw = []
            tempn = []
            tempd = []
            for j in range(self.totalNeurons[i]):
                if j in prevneurons[i]:
                    num = random.randint(1, self.totalNeurons[i + 1])
                    prevrand.clear()
                    for k in range(num):
                        newneuron = 0
                        while True:
                            newneuron = random.randint(0, self.totalNeurons[i + 1] - 1)
                            if newneuron not in prevrand:
                                break
                        tempn.append(newneuron)
                        prevrand.append(newneuron)
                        temps.append([j, newneuron])
                        tempw.append(2 * (random.random() - .5))
                else:
                    tempd.append(j + 1)
            deadneurons.append(tempd)
            prevneurons.append(tempn)
            self.synapses.append(temps)
            self.weights.append(tempw)
        # print(self.synapses)
        # print(self.weights)
        self.actualNeurons.append(self.maxInput)
        for i in range(self.layers):
            self.actualNeurons.append(self.totalNeurons[i + 1] - len(deadneurons[i]))
        self.actualNeurons.append(self.maxOutput)

    def drawnet(self, screen, width, height, offsetx, offsety):
        points = []
        tempp = []
        for i in range(len(self.actualNeurons)):
            for j in range(self.actualNeurons[i]):
                x = round(i * (width / (len(self.actualNeurons) - 1)) + offsetx)
                y = round(j * (height / (max(self.actualNeurons) - 1)) + offsety)
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
        for i in range(len(self.synapses)):
            for j in range(len(self.synapses[i])):
                point1 = points[i][self.synapses[i][j][0]]
                point2 = points[i + 1][self.synapses[i][j][1]]
                if self.weights[i][j] > 0:
                    pygame.draw.line(screen, (0, 255, 0), point1, point2)
                else:
                    pygame.draw.line(screen, (255, 0, 0), point1, point2)
        pygame.display.flip()

    def in2out(self):
        layerout = [self.inputs]
        templ = []
        prevneurons = []
        for i in range(len(self.synapses)):
            prevneurons = []
            templ = [*range(self.actualNeurons[i + 1])]
            for j in range(len(self.synapses[i])):
                if self.synapses[i][j][1] not in prevneurons:
                    templ[self.synapses[i][j][1]] = layerout[i][self.synapses[i][j][0]] * self.weights[i][j]
                else:
                    templ[self.synapses[i][j][1]] *= layerout[i][self.synapses[i][j][0]] * self.weights[i][j]
                prevneurons.append(self.synapses[i][j][1])
            layerout.append(templ)
        for i in range(self.maxOutput):
            self.outputs[i] = layerout[-1][i]
        # print(self.outputs)

    def mutatesynapse(self):
        num = random.randint(1, len(self.synapses))
        for i in range(num):
            idx = random.randint(0, len(self.synapses[i]) - 1)
            newneuron = random.randint(0, self.actualNeurons[i] - 1)
            if i == len(self.synapses) - 1:
                if newneuron is not self.synapses[i][idx][1]:
                    self.synapses[i][idx][0] = random.randint(0, self.actualNeurons[i] - 1)
            else:
                if newneuron is not self.synapses[i][idx][1]:
                    self.synapses[i][idx][1] = random.randint(0, self.actualNeurons[i] - 1)

    def mutateweight(self):
        num = random.randint(1, len(self.synapses))
        for i in range(num):
            idx = random.randint(0, len(self.synapses[i]) - 1)
            newneuron = random.randint(0, self.actualNeurons[i] - 1)
            if newneuron is not self.synapses[i][idx][1]:
                self.weights[i][idx] = 2 * (random.random() - .5)

    def mutate(self):
        p1 = random.random()
        p2 = random.random()
        if p1 < self.synapserate:
            self.mutatesynapse()
        if p2 < self.weightrate:
            self.mutateweight()


