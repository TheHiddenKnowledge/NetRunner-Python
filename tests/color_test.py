import netrunner

net = netrunner.NetRunner([3, 6, 8], 'sigmoid', .01, .9)
net.randomize_net(-1.0, 1.0)

# RGB values list
rgb = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
       [1, 1, 1], [0, 0, 0]]
# Color output list
colors = ["Red", "Green", "Blue", "Yellow", "Purple", "Cyan", "White", "Black"]

# Cost of the neural net
cost = 1
# Iteration count for the training algorithm
iteration = 0

# Uses gradient descent until the error is below the threshold
while cost > .001:
    expected_set = []
    iteration += 1
    # Generating the sets of inputs and expected values for gradient descent
    for j in range(len(rgb)):
        color = rgb[j]
        if color[0] == 1 and color[1] < 1 and color[2] < 1:
            expected_set.append([1, 0, 0, 0, 0, 0, 0, 0])
        elif color[0] < 1 and color[1] == 1 and color[2] < 1:
            expected_set.append([0, 1, 0, 0, 0, 0, 0, 0])
        elif color[0] < 1 and color[1] < 1 and color[2] == 1:
            expected_set.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif color[0] == 1 and color[1] == 1 and color[2] < 1:
            expected_set.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif color[0] == 1 and color[1] < 1 and color[2] == 1:
            expected_set.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif color[0] < 1 and color[1] == 1 and color[2] == 1:
            expected_set.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif color[0] == 1 and color[1] == 1 and color[2] == 1:
            expected_set.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif color[0] < 1 and color[1] < 1 and color[2] < 1:
            expected_set.append([0, 0, 0, 0, 0, 0, 0, 1])
    cost = net.use_gradient_descent(rgb, expected_set)
    if iteration % 1000 == 0:
        print("The error after " + str(iteration)
              + " iterations is " + str(cost))
print("The error after " + str(iteration) + " iterations is " + str(cost))

# Running the neural net and testing each input
for value in rgb:
    out = net.run_net(value).tolist()
    print(colors[out.index(max(out))])