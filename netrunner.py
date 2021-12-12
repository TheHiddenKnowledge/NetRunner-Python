import numpy as np
import pygame

class net:
    def __init__(self, inputs, outputs, layers, minOut, maxOut, error):
        self.inputs = []
        self.outputs = []
        self.weights = []
        self.biases = []
        self.generation = 0
        self.maxInput = inputs
        self.maxOutput = outputs
        self.layers = layers
        self.minOut = minOut
        self.maxOut = maxOut
        self.error = error
        self.totalNeurons = []
        self.totalNeurons.append(self.maxInput)
        for i in range(len(layers)):
            self.totalNeurons.append(self.layers[i])
        self.totalNeurons.append(self.maxOutput)
        for i in range(len(self.totalNeurons)-1):
            tempw = 1*np.random.random_sample((self.totalNeurons[i+1],self.totalNeurons[i]))-.5
            self.weights.append(tempw)
            tempb = (self.maxOut-self.minOut)*np.random.random_sample((self.totalNeurons[i+1]))+self.minOut
            self.biases.append(tempb)
        self.successes = 0
        self.fails = 0

    def randomGenes(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.totalNeurons)-1):
            tempw = (self.maxOut-self.minOut)*np.random.random_sample((self.totalNeurons[i+1],self.totalNeurons[i]))-self.minOut
            self.weights.append(tempw)
            tempb = (self.maxOut-self.minOut)*np.random.random_sample((self.totalNeurons[i+1]))-self.minOut
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
                        pygame.draw.line(screen, (0, round(255*abs(self.weights[i][j][k]/np.amax(abs(self.weights[i])))), 0), point1, point2)
                    if self.weights[i][j][k] < 0:
                        pygame.draw.line(screen, (round(255*abs(self.weights[i][j][k]/np.amax(abs(self.weights[i])))), 0, 0), point1, point2)
        font = pygame.font.Font('freesansbold.ttf', 16)
        stext = font.render('Successes: '+str(self.successes), True, (255, 255, 255), (0, 0, 0))
        textRect = stext.get_rect()
        w,h = stext.get_size()
        textRect.center = (width+2*offsetx+w/2, 2*offsety)
        screen.blit(stext, textRect)
        ftext = font.render('Fails: ' + str(self.fails), True, (255, 255, 255), (0, 0, 0))
        textRect = ftext.get_rect()
        w, h = ftext.get_size()
        textRect.center = (width + 2 * offsetx + w / 2, 4 * offsety)
        screen.blit(ftext, textRect)

    def sigmoid(self, x):
        return (self.maxOut-self.minOut)*(1/(1+np.exp(-x/25)))+self.minOut

    def invsigmoid(self, x):
        if x > 0:
            return -25*np.log(((x-self.minOut)/(self.maxOut-self.minOut))**-1-1)
        else:
            return 25*np.log(((abs(x) - self.minOut) / (self.maxOut - self.minOut)) ** -1 - 1)

    def in2out(self, weights, biases, isOut):
        layerout = np.array(self.inputs)
        for i in range(len(weights)):
            layerout = np.matmul(weights[i], layerout)+biases[i]
        if isOut:
            self.outputs = layerout
        else:
            return layerout

    def saturateoutput(self):
        for i in range(self.outputs.shape[0]):
            self.outputs[i] = self.sigmoid(self.outputs[i])

    def loadnet(self):
        if len(np.load('bestnet.npz', allow_pickle=True).files) != 0:
            self.weights = np.load('bestnet.npz', allow_pickle=True)['arr_0']
            self.biases = np.load('bestnet.npz', allow_pickle=True)['arr_1']

    def savenet(self):
        np.savez('bestnet', self.weights, self.biases, fmt='%s')

    def gradientDescent(self, expected, step):
        cost1 = np.square(expected-self.outputs)
        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                neg = False
                db = self.biases
                db[i][j] += step
                cost2 = np.square(expected - self.in2out(self.weights, db, False))
                difference = cost2 - cost1
                #print("Bias")
                #print(cost1)
                #print(cost2)
                for m in range(difference.shape[0]):
                    if difference[m]/step > 0:
                        self.biases[i][j] -= step
                        neg = True
                        break
                if not neg:
                    self.biases[i][j] += step
                for k in range(self.weights[i].shape[1]):
                    neg = False
                    dw = self.weights
                    dw[i][j][k] += step
                    cost2 = np.square(expected - self.in2out(dw, self.biases, False))
                    difference = cost2 - cost1
                    #print("Weight")
                    #print(cost1)
                    #print(cost2)
                    for m in range(difference.shape[0]):
                        if difference[m]/step > 0:
                            self.weights[i][j][k] -= step
                            neg = True
                            break
                    if not neg:
                        self.weights[i][j][k] += step
        #return s

    def trainNet(self, expectlist, inputlist):
        for i in range(len(expectlist)):
            self.inputs = inputlist[i]
            sum = 0
            step = 10
            counter = 0
            counter1 = 0
            for m in range(expectlist[i].shape[0]):
                expectlist[i][m] = self.invsigmoid(expectlist[i][m])
            while sum < expectlist[i].shape[0]:
                counter += 1
                counter1 += 1
                self.in2out(self.weights, self.biases, True)
                self.gradientDescent(expectlist[i], step)
                sum = 0
                for j in range(expectlist[i].shape[0]):
                    if (expectlist[i][j] - self.outputs[j]) ** 2 < (expectlist[i][j]*self.error)**2:
                        sum += 1
                # if counter > 1100:
                #     counter = 0
                #     step *= .1
                # if counter1 > 5000:
                #     counter1 = 0
                #     step = np.random.randint(1,100)
                #print("A")
                #print(np.square(expectlist[i]-self.outputs))
                #print("B")
                #print(np.square(expectlist[i]*self.error))
                #print(step)
            print(self.outputs)
            print("Completed")


