import numpy as np
import pygame
import random
import netrunner

# Creates an object for the net
n = netrunner.net(4, 2, [4, 4, 4], -5, 5, 5)
pygame.init()
field = pygame.display.set_mode((800, 800))
finished = False


# Class for the player object
class Player:
    # Initializes the player position and size
    def __init__(self, x, y, w, h):
        # Position, velocity, and size of the player
        self.x = x
        self.y = y
        self.xorg = x
        self.yorg = y
        self.w = w
        self.h = h
        self.xc = self.x + round(self.w / 2)
        self.yc = self.y + round(self.h / 2)
        self.moveY = 10
        self.moveX = 0
        self.success = False
        # Variables used to define the field of view
        self.vision = 50
        self.angle = np.pi / 3
        self.step = 5
        # Sets used for gradient descent
        self.expectedlist = []
        self.inputlist = []

    # Draws the players with pygame
    def load(self):
        self.xc = self.x + round(self.w / 2)
        self.yc = self.y + round(self.h / 2)
        pygame.draw.rect(field, (255, 0, 200), pygame.Rect(self.x, self.y, self.w, self.h))

    # Detects the object to use as an input to the net
    def detect(self, obstacles):
        # Magnitude of the players velocity
        mag = (self.moveX ** 2 + self.moveY ** 2) ** .5
        # Output of the detection algorithm
        data = [0, 0, 0, 0]
        # This algorithm uses raytracing to find the objects within the player's field of view
        for i in range(self.step):
            px = self.vision * self.moveX / mag
            py = self.vision * self.moveY / mag
            # Rotating the original vector by increments of the view angle
            px0 = px * np.cos(self.angle * (i - self.step / 2) / self.step) - py * np.sin(
                self.angle * (i - self.step / 2) / self.step)
            py0 = px * np.sin(self.angle * (i - self.step / 2) / self.step) + py * np.cos(
                self.angle * (i - self.step / 2) / self.step)
            for j in range(self.vision):
                for k in range(len(obstacles)):
                    px1 = j * px0 / self.vision
                    py1 = j * py0 / self.vision
                    if obstacles[k].yc - obstacles[k].h / 2 < -py1 + self.yc < obstacles[k].yc + obstacles[k].h / 2 and \
                            obstacles[
                                k].xc - obstacles[k].w / 2 < -px1 + self.xc < obstacles[k].xc + obstacles[k].w / 2:
                        obstacles[k].color = (0, 255, 0)
                        px0 = px1
                        py0 = py1
                        data = [obstacles[k].xc, obstacles[k].yc, obstacles[k].w, obstacles[k].h]
            pygame.draw.line(field, (0, 255, 255), (self.xc, self.yc), (-px0 + self.xc, -py0 + self.yc))
        return data

    # Determines screen and goal collisions
    def bounds(self, goal):
        width, height = field.get_size()
        # If the player is within the bounds of the goal it has succeeded and then the player's position is reset
        if abs(p.yc - goal.yc) < p.h / 2 + goal.h / 2 and abs(p.xc - goal.xc) < p.w / 2 + goal.w / 2:
            self.y = self.yorg
            self.x = random.randint(200, width - 200)
            self.xorg = self.x
            self.success = True
            n.successes += 1
        # If the player is outside the bounds of the pygame screen then is has failed, and it's position is reset
        if p.yc >= height or p.yc <= 0 or p.xc >= width or p.xc <= 0:
            self.y = self.yorg
            self.x = self.xorg
            n.fails += 1

    # Evaluates and trains the net based on the given obstacle and goal positions
    def updatenet(self, obstacles, goal):
        # Gets the obstacle that best suits the net
        chosen = self.detect(obstacles)
        # Assigns the inputs to the net as relative to the players position
        inputs = [self.xc - chosen[0], self.yc - chosen[1], self.xc - goal.xc, self.yc - goal.yc]
        n.inputs = inputs
        # Evaluates the net using the provided inputs
        n.outputs = n.in2out()
        # Fails the player if it has collided with an obstacle
        if chosen[1] - chosen[3] / 2 < self.yc < chosen[1] + chosen[3] / 2 and chosen[0] - chosen[
            2] / 2 < self.xc < chosen[0] + chosen[2] / 2:
            self.y = self.yorg
            self.x = self.xorg
            n.fails += 1
        self.inputlist.append(inputs)
        # The following if-elif-else statements provide the net with the expected behavior based on the inputs
        # Expected behavior if the player is near an obstacle
        if inputs[0] != self.xc and inputs[1] != self.yc:
            if n.inputs[1] > self.h / 2 + chosen[3] / 2:
                if inputs[0] > 0:
                    self.expectedlist.append([0, 0])
                else:
                    self.expectedlist.append([1, 0])
            else:
                if inputs[0] > 0:
                    self.expectedlist.append([0, 1])
                else:
                    self.expectedlist.append([1, 1])
        # Expected behavior if the player is near the goal
        else:
            if n.inputs[2] < 0 and n.inputs[3] > 0:
                self.expectedlist.append([.25, .75])
            elif n.inputs[2] > 0 and n.inputs[3] > 0:
                self.expectedlist.append([.75, .75])
            elif n.inputs[2] < 0 and n.inputs[3] < 0:
                self.expectedlist.append([.25, .25])
            elif n.inputs[2] > 0 and n.inputs[3] < 0:
                self.expectedlist.append([.75, .25])
            elif n.inputs[2] > 0 and abs(n.inputs[3]) < self.h / 2 + goal.h / 2:
                self.expectedlist.append([.75, .5])
            elif n.inputs[2] < 0 and abs(n.inputs[3]) < self.h / 2 + goal.h / 2:
                self.expectedlist.append([.25, .5])
        # Gets and applies the gradients of both the weights and biases
        if len(self.expectedlist) == 10:
            n.getset(self.inputlist, self.expectedlist)
            self.inputlist = []
            self.expectedlist = []

    # Applies the output of the net to the player to create the desired movement
    def move(self):
        out = n.adjustoutput()
        self.moveX = out[0]
        self.moveY = out[1]
        p.y -= self.moveY
        p.x -= self.moveX


# Class for the obstacle object
class Obstacle:
    # Initializes the obstacles position, size, and initial color
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xc = self.x + round(self.w / 2)
        self.yc = self.y + round(self.h / 2)
        self.color = (255, 0, 0)

    # Draws the obstacle using pygame
    def load(self, color):
        self.xc = self.x + round(self.w / 2)
        self.yc = self.y + round(self.h / 2)
        pygame.draw.rect(field, self.color, pygame.Rect(self.x, self.y, self.w, self.h))
        self.color = color


# Generates the obstacle course in such a way that there is enough space for the player to reach the goal
def generateCourse(num, xmin, xmax, ymin, ymax, smin, smax):
    factor = 2 * smax
    points = []
    for i in range(round((ymax - ymin) / factor)):
        off = random.randint(-100, 100)
        for j in range(round((xmax - xmin) / factor)):
            points.append([j * factor + xmin + off, i * factor + ymin])
    obstacles = []
    idxs = random.sample(range(len(points)), num)
    for i in range(num):
        o = Obstacle(round(points[idxs[i]][0]), round(points[idxs[i]][1]), random.randint(smin, smax),
                     random.randint(smin, smax))
        obstacles.append(o)
    return obstacles


# Given that the obstacle course has already been initialized, this will reassign the course positions
def updateCourse(obstacles, xmin, xmax, ymin, ymax, smin, smax):
    factor = 2 * smax
    points = []
    for i in range(round((ymax - ymin) / factor)):
        off = random.randint(-100, 100)
        for j in range(round((xmax - xmin) / factor)):
            points.append([j * factor + xmin + off, i * factor + ymin])
    idxs = random.sample(range(len(points)), len(obstacles))
    for i in range(len(obstacles)):
        obstacles[i].x = round(points[idxs[i]][0])
        obstacles[i].y = round(points[idxs[i]][1])


# Draws the player, obstacles, goal, and net onto the pygame window
def updateGUI(player, course, goal):
    field.fill((0, 0, 0))
    player.load()
    for i in range(len(course)):
        course[i].load((255, 0, 0))
    goal.load((255, 211, 0))
    n.drawnet(field, 100, 100, 10, 10)


# Creating all necessary simulation objects
p = Player(400, 780, 10, 10)
obstacles = generateCourse(60, 25, 775, 100, 700, 25, 40)
goal = Obstacle(380, 50, 40, 40)
clock = pygame.time.Clock()
# Loop that keeps the pygame window constantly updated
while not finished:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            finished = True
    updateGUI(p, obstacles, goal)
    p.updatenet(obstacles, goal)
    p.bounds(goal)
    p.move()
    # If the player has succeeded the course positions are reassigned to create diverse results
    if p.success:
        updateCourse(obstacles, 25, 775, 100, 700, 25, 40)
        p.success = False
    pygame.display.flip()
    # The clock speed can be increased to make the simulation run faster. This could have consequences if it is too high
    clock.tick(100)
