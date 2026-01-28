import netrunner
import numpy as np
import pandas

INPUT_LAYER = netrunner.NetLayer(3, 'input')
INNER_LAYER = netrunner.NetLayer(6, 'sigmoid')
OUTPUT_LAYER = netrunner.NetLayer(8, 'sigmoid')

NET = netrunner.NetRunner([INPUT_LAYER, INNER_LAYER, OUTPUT_LAYER])

TRAINING_NET = [[np.array([[1], [0], [0]])], [np.array([[0], [1], [0]])],
               [np.array([[0], [0], [1]])], [np.array([[1], [1], [0]])],
               [np.array([[1], [0], [1]])], [np.array([[0], [1], [1]])],
               [np.array([[1], [1], [1]])], [np.array([[0], [0], [0]])]]
COLORS = ["Red", "Green", "Blue", "Yellow",
          "Purple", "Cyan", "White", "Black"]

for a in range(len(TRAINING_NET)):
    rgb = TRAINING_NET[a][0]
    if rgb[0] == 1 and rgb[1] < 1 and rgb[2] < 1:
        TRAINING_NET[a].append(np.array([[1], [0], [0], [0],
                                        [0], [0], [0], [0]]))
    elif rgb[0] < 1 and rgb[1] == 1 and rgb[2] < 1:
        TRAINING_NET[a].append(np.array([[0], [1], [0], [0],
                                        [0], [0], [0], [0]]))
    elif rgb[0] < 1 and rgb[1] < 1 and rgb[2] == 1:
        TRAINING_NET[a].append(np.array([[0], [0], [1], [0],
                                        [0], [0], [0], [0]]))
    elif rgb[0] == 1 and rgb[1] == 1 and rgb[2] < 1:
        TRAINING_NET[a].append(np.array([[0], [0], [0], [1],
                                        [0], [0], [0], [0]]))
    elif rgb[0] == 1 and rgb[1] < 1 and rgb[2] == 1:
        TRAINING_NET[a].append(np.array([[0], [0], [0], [0],
                                        [1], [0], [0], [0]]))
    elif rgb[0] < 1 and rgb[1] == 1 and rgb[2] == 1:
        TRAINING_NET[a].append(np.array([[0], [0], [0], [0],
                                        [0], [1], [0], [0]]))
    elif rgb[0] == 1 and rgb[1] == 1 and rgb[2] == 1:
        TRAINING_NET[a].append(np.array([[0], [0], [0], [0],
                                        [0], [0], [1], [0]]))
    elif rgb[0] < 1 and rgb[1] < 1 and rgb[2] < 1:
        TRAINING_NET[a].append(np.array([[0], [0], [0], [0],
                                        [0], [0], [0], [1]]))

def run_color_net(rate, beta):
    NET.init_net()
    NET.init_runner(rate, beta)
    cost = 1
    epoch = 0
    while cost > .1:
        NET.init_mini_batch(TRAINING_NET, 4, True)
        cost, epoch = NET.step_epoch(TRAINING_NET)
        if epoch > 5000:
            return 0
    return epoch

RATES = [.25, .5, .75, 1, 2, 2.5, 5, 10]
BETAS = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
DATA = []
for rate in RATES:
    DATA_row = []
    for beta in BETAS:
        iteration = run_color_net(rate, beta)
        DATA_row.append(iteration)
        print('Rate: {}, Beta: {}, Epoch: {}'.format(rate, beta,
                                                         iteration))
    DATA.append(DATA_row)

DF = pandas.DataFrame(DATA)
DF.to_excel('color_performance.xlsx', index=False, header=False)