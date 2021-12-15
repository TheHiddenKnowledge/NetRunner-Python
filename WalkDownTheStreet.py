import pygame
import random
import netrunner

# Creates an object for the net
n = netrunner.net(4, 2, [4, 4], -5, 5, 5, .1)
pygame.init()
field = pygame.display.set_mode((800, 800))
finished = False


# Class for the player object
class Player:
    # Initializes the player position and size
    def __init__(self, x, y, w, h):
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
        self.mind = 1
        self.idx = 0

    # Draws the players with pygame
    def load(self):
        self.xc = self.x + round(self.w / 2)
        self.yc = self.y + round(self.h / 2)
        pygame.draw.rect(field, (255, 0, 200), pygame.Rect(self.x, self.y, self.w, self.h))

    # Detects the object to use as an input to the net
    def detect(self, obstacles):
        # Gets the x offset, y offset, and distance between the player and all obstacles
        dx = []
        dy = []
        distances = []
        temp = []
        for i in range(len(obstacles)):
            DX = self.xc - obstacles[i].xc
            DY = self.yc - obstacles[i].yc
            dist = (((self.xc - obstacles[i].xc) ** 2) + ((self.yc - obstacles[i].yc) ** 2)) ** .5
            dx.append(DX)
            dy.append(DY)
            distances.append(dist)
            temp.append(dist)
        temp.sort()
        # This algorithm selects the best fit obstacle for the net to use
        mind = dx[distances.index(temp[0])]
        for i in range(len(temp)):
            if abs(dy[i]) < 50:
                if abs(dx[i]) < mind:
                    mind = dx[i]
        if abs(mind - self.mind) > self.mind / 4:
            self.mind = mind
            self.idx = dx.index(mind)
        obstacles[self.idx].color = (0, 255, 0)
        return [obstacles[self.idx]]

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
        inputs = [self.xc - chosen[0].xc, self.yc - chosen[0].yc, self.xc - goal.xc, self.yc - goal.yc]
        n.inputs = inputs
        # Evaluates the net using the provided inputs
        n.outputs = n.in2out()
        # Fails the player if it has collided with an obstacle
        if abs(inputs[0]) < .9 * (self.w / 2 + chosen[0].w / 2) and abs(inputs[1]) < .9 * (
                self.h / 2 + chosen[0].h / 2):
            self.y = self.yorg
            self.x = self.xorg
            n.fails += 1
        # The following if-elif-else statements provide the net with the expected behavior based on the inputs
        if abs(n.inputs[0]) < 1.5 * (self.w / 2 + chosen[0].w) / 2 and n.inputs[1] < 4 * (self.h / 2 + chosen[0].h / 2):
            if n.inputs[1] > self.h / 2 + chosen[0].h / 2:
                if inputs[0] > 0:
                    expected = [0, 0]
                else:
                    expected = [1, 0]
            else:
                if inputs[0] > 0:
                    expected = [0, 1]
                else:
                    expected = [1, 1]
        else:
            if n.inputs[2] < 0 and n.inputs[3] > 0:
                expected = [.4, .6]
            elif n.inputs[2] > 0 and n.inputs[3] > 0:
                expected = [.6, .6]
            elif n.inputs[2] < 0 and n.inputs[3] < 0:
                expected = [.4, .4]
            elif n.inputs[2] > 0 and n.inputs[3] < 0:
                expected = [.6, .4]
            elif n.inputs[2] > 0 and abs(n.inputs[3]) < self.h / 2 + goal.h / 2:
                expected = [.6, .5]
            elif n.inputs[2] < 0 and abs(n.inputs[3]) < self.h / 2 + goal.h / 2:
                expected = [.4, .5]
        # Gets and applies the gradients of both the weights and biases
        n.getset([n.inputs], [expected])

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
obstacles = generateCourse(60, 25, 775, 25, 725, 25, 40)
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
        updateCourse(obstacles, 25, 775, 25, 725, 25, 40)
        p.success = False
    pygame.display.flip()
    # The clock speed can be increased to make the simulation run faster. This could have consequences if it is too high
    clock.tick(100)
