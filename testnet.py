import random

import netrunner

# This example trains the neural net to recognize colors based on boolean RGB values
# Initializing the net and net constitutes
n = netrunner.net(3, 8, [6, 8], .0001, .9, 2.5)
n.randomGenes(-5, 5, -5, 5)
inputs = []
expected = []
colors = ["Red", "Green", "Blue", "Yellow", "Purple", "Cyan", "White", "Black"]
error = 1
i = 0
# Continues gradient descent until the error is below the threshold
while error > .001:
    tempe = []
    tempi = []
    i += 1
    # Generating the sets of inputs and expected values for gradient descent
    for j in range(5):
        color = [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]
        tempi.append(color)
        if color[0] == 1 and color[1] < 1 and color[2] < 1:
            tempe.append([1, 0, 0, 0, 0, 0, 0, 0])
        elif color[0] < 1 and color[1] == 1 and color[2] < 1:
            tempe.append([0, 1, 0, 0, 0, 0, 0, 0])
        elif color[0] < 1 and color[1] < 1 and color[2] == 1:
            tempe.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif color[0] == 1 and color[1] == 1 and color[2] < 1:
            tempe.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif color[0] == 1 and color[1] < 1 and color[2] == 1:
            tempe.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif color[0] < 1 and color[1] == 1 and color[2] == 1:
            tempe.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif color[0] == 1 and color[1] == 1 and color[2] == 1:
            tempe.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif color[0] < 1 and color[1] < 1 and color[2] < 1:
            tempe.append([0, 0, 0, 0, 0, 0, 0, 1])
    error = n.getset(tempi, tempe)
    if i % 100 == 0:
        print("The error after " + str(i) + " iterations is " + str(error))
print("The error after " + str(i) + " iterations is " + str(error))
# Inputting to the neural net and testing each input
n.inputs = [1, 0, 0]
out = n.in2out().tolist()
print(colors[out.index(max(out))])
n.inputs = [0, 1, 0]
out = n.in2out().tolist()
print(colors[out.index(max(out))])
n.inputs = [0, 0, 1]
out = n.in2out().tolist()
print(colors[out.index(max(out))])
n.inputs = [1, 1, 0]
out = n.in2out().tolist()
print(colors[out.index(max(out))])
n.inputs = [1, 0, 1]
out = n.in2out().tolist()
print(colors[out.index(max(out))])
n.inputs = [0, 1, 1]
out = n.in2out().tolist()
print(colors[out.index(max(out))])
