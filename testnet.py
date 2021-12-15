import netrunner

n = netrunner.net(2, 2, [4], -10, 10, 5, 1)
n.loadnet("bestnet")
inputs = n.parsesets("inputs.txt")
expected = n.parsesets("expected.txt")
print(n.weights)
n.trainnet(inputs, expected)
print(n.weights)
n.inputs = [30, 30]
print(n.in2out())
n.savenet("bestnet")
