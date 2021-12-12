import numpy as np
import netrunner
import pygame

finished = False
#pygame.init()
#field = pygame.display.set_mode((600, 400))
n = netrunner.net(3, 2, [4,3,3], -10, 10, .25)
inputs = [[-10, 30, 4],[10,30,4],[100,100,8]]
expected = [np.array([5, -5]),np.array([-5, -5]),np.array([1, 5])]
done = False
#n.loadnet()
n.trainNet(expected, inputs)
n.inputs = inputs[0]
n.in2out(n.weights, n.biases, True)
n.saturateoutput()
print(n.outputs)
n.savenet()

