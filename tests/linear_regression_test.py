import netrunner
import random as rnd
import math
import numpy as np

SAMPLE_COUNT = 20

INPUT_LAYER = netrunner.NetLayer(2 * SAMPLE_COUNT, 'input')
INNER_LAYER = netrunner.NetLayer(SAMPLE_COUNT, 'sigmoid')
OUTPUT_LAYER = netrunner.NetLayer(2, 'sigmoid')

NET_LAYERS = [INPUT_LAYER, INNER_LAYER, OUTPUT_LAYER]

NET = netrunner.NetRunner(NET_LAYERS)
NET.init_net()
NET.init_runner(.1, .9)

COORDINATE_COUNT = 500
M_ACTUAL = 1
B_ACTUAL = 0
X_MAX = 10
COORDINATES = []

# Creating random sample points from the true line function
for a in range(COORDINATE_COUNT):
    x_actual = X_MAX * a / COORDINATE_COUNT
    y_actual = M_ACTUAL * x_actual + B_ACTUAL
    x_random = x_actual + rnd.gauss(0, 1)
    y_random = y_actual + rnd.gauss(0, 1)
    COORDINATES.append([x_random, y_random])

COORDINATES.sort()

TRAINING_SET = []
SAMPLE_IDX_MAX = math.ceil(COORDINATE_COUNT / SAMPLE_COUNT)
EXPECTED_SET = np.array([[M_ACTUAL], [B_ACTUAL]])

# Generating training set
for a in range(SAMPLE_IDX_MAX):
    sample_list = []
    for b in range(SAMPLE_COUNT):
        sample_list.append([COORDINATES[a * SAMPLE_COUNT + b][0]])
        sample_list.append([COORDINATES[a * SAMPLE_COUNT + b][1]])
    TRAINING_SET.append([np.array(sample_list), EXPECTED_SET])

COST = 1
EPOCH = 0
while EPOCH < 100:
    NET.init_mini_batch(TRAINING_SET, 5, False)
    COST, EPOCH = NET.step_epoch(TRAINING_SET)
print("The error after " + str(EPOCH) + " EPOCHs is " + str(COST))

for a in range(1):
    out = NET.forward_pass(TRAINING_SET[a][0])
    print(TRAINING_SET[a][0])
    print('')
    print(out)