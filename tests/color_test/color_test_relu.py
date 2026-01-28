import netrunner
import numpy as np
import random

INPUT_LAYER = netrunner.NetLayer(3, 'input')
INNER_LAYER_1 = netrunner.NetLayer(2, 'relu')
INNER_LAYER_2 = netrunner.NetLayer(2, 'relu')
OUTPUT_LAYER = netrunner.NetLayer(1, 'relu')

NET_LAYERS = [INPUT_LAYER, INNER_LAYER_1, INNER_LAYER_2, OUTPUT_LAYER]

NET = netrunner.NetRunner(NET_LAYERS)
NET.init_net()
NET.init_runner(.1, .9)

TRAINING_SET = [[np.array([[1], [0], [0]])], [np.array([[0], [1], [0]])],
               [np.array([[0], [0], [1]])], [np.array([[1], [1], [0]])],
               [np.array([[1], [0], [1]])], [np.array([[0], [1], [1]])],
               [np.array([[1], [1], [1]])], [np.array([[0], [0], [0]])]]
COLORS = ["Red", "Green", "Blue", "Yellow",
          "Purple", "Cyan", "White", "Black"]

# Generating training set expected values
for a in range(len(TRAINING_SET)):
    rgb = TRAINING_SET[a][0]
    if rgb[0] == 1 and rgb[1] < 1 and rgb[2] < 1:
        TRAINING_SET[a].append(np.array([[1]]))
    elif rgb[0] < 1 and rgb[1] == 1 and rgb[2] < 1:
        TRAINING_SET[a].append(np.array([[2]]))
    elif rgb[0] < 1 and rgb[1] < 1 and rgb[2] == 1:
        TRAINING_SET[a].append(np.array([[3]]))
    elif rgb[0] == 1 and rgb[1] == 1 and rgb[2] < 1:
        TRAINING_SET[a].append(np.array([[4]]))
    elif rgb[0] == 1 and rgb[1] < 1 and rgb[2] == 1:
        TRAINING_SET[a].append(np.array([[5]]))
    elif rgb[0] < 1 and rgb[1] == 1 and rgb[2] == 1:
        TRAINING_SET[a].append(np.array([[6]]))
    elif rgb[0] == 1 and rgb[1] == 1 and rgb[2] == 1:
        TRAINING_SET[a].append(np.array([[7]]))
    elif rgb[0] < 1 and rgb[1] < 1 and rgb[2] < 1:
        TRAINING_SET[a].append(np.array([[8]]))

COST = 1
EPOCH = 0
while EPOCH < 10000:
    NET.init_mini_batch(TRAINING_SET, 6, True)
    COST, EPOCH = NET.step_epoch(TRAINING_SET)
    if EPOCH % 1000 == 0:
        print("The error after " + str(EPOCH)
              + " EPOCHs is " + str(COST))
print("The error after " + str(EPOCH) + " EPOCHs is " + str(COST))

for value in TRAINING_SET:
    out = NET.forward_pass(value[0]).tolist()
    print(value[0], COLORS[round(out[0][0]) - 1])