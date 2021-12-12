import pygame
import random
import netrunner
import numpy as np
n = netrunner.net(3, 2, [4,3,2], -10, 10, .1)
pygame.init()
field = pygame.display.set_mode((800, 800))
finished = False
n.weights = np.load('bestnet.npz',allow_pickle=True)['arr_0']
n.biases = np.load('bestnet.npz',allow_pickle=True)['arr_1']
class Player:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.xorg = x
        self.yorg = y
        self.w = w
        self.h = h
        self.xc = self.x + round(self.w / 2)
        self.yc = self.y + round(self.h / 2)
        self.maxV = 5
        self.moveY = 10
        self.moveX = 0
        self.pmoveY = self.maxV
        self.pmoveX = 0
        self.fitness = 0
        self.timeout = 0
        self.noinput = 0
        self.success = False
        self.mind = 1
        self.idx = 0
        self.top = 800

    def load(self):
        self.xc = self.x + round(self.w / 2)
        self.yc = self.y + round(self.h / 2)
        pygame.draw.rect(field, (255, 0, 200), pygame.Rect(self.x, self.y, self.w, self.h))

    def detect(self, obstacles):
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
        mind = temp[0]
        switch = False
        if abs(mind - self.mind) > self.mind / 4:
            self.mind = mind
            self.idx = distances.index(mind)
        minx = dx[self.idx]
        miny = dy[self.idx]
        if abs(minx) / 1.5 < self.w / 2 + obstacles[self.idx].w / 2 and abs(miny) / 1.5 < self.h / 2 + obstacles[self.idx].h / 2:
            self.noinput = False
            obstacles[self.idx].color = (0, 255, 0)
        else:
            self.noinput = True
        return [obstacles[self.idx]]

    def bounds(self):
        width, height = field.get_size()
        if p.yc <= 0:
            self.timeout = 0
            self.fitness = 0
            self.y = self.yorg
            self.x = random.randint(200, width - 200)
            self.xorg = self.x
            self.success = True
            self.pmoveY = self.maxV
            self.pmoveX = 0
            self.top = 800
            n.successes += 1
        if p.yc >= height or p.xc >= width or p.xc <= 0:
            self.timeout = 0
            self.fitness = 0
            self.y = self.yorg
            self.x = self.xorg
            self.pmoveY = self.maxV
            self.pmoveX = 0
            self.top = 800
            n.fails += 1

    def timecheck(self, maxtime):
        if self.timeout > maxtime:
            self.y = self.yorg
            self.x = self.xorg
            self.pmoveY = self.maxV
            self.pmoveX = 0
            self.top = 800
            self.timeout = 0
            self.fitness = 0

    def updatenet(self, obstacles):
        chosen = self.detect(obstacles)
        inputs = [self.xc - chosen[0].xc, self.yc - chosen[0].yc, self.yc/100]
        n.inputs = inputs
        n.in2out(n.weights, n.biases, True)
        n.saturateoutput()
        if abs(inputs[0]) < .9 * (self.w / 2 + chosen[0].w / 2) and abs(inputs[1]) < .9 * (
                self.h / 2 + chosen[0].h / 2):
            self.timeout = 0
            n.fitness = self.fitness
            self.fitness = 0
            self.y = self.yorg
            self.x = self.xorg
            self.pmoveY = self.maxV
            self.pmoveX = 0
            self.top = 800
            n.fails += 1

    def move(self):
        self.moveX = n.outputs[0]
        self.moveY = n.outputs[1]
        p.y -= self.moveY
        p.x -= self.moveX


class Obstacle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xc = self.x + round(self.w / 2)
        self.yc = self.y + round(self.h / 2)
        self.color = (255, 0, 0)

    def load(self):
        self.xc = self.x + round(self.w / 2)
        self.yc = self.y + round(self.h / 2)
        pygame.draw.rect(field, self.color, pygame.Rect(self.x, self.y, self.w, self.h))
        self.color = (255, 0, 0)


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


def updateGUI(player, course):
    field.fill((0, 0, 0))
    player.load()
    for i in range(len(course)):
        course[i].load()
    n.drawnet(field, 100, 100, 10, 10)


p = Player(400, 780, 10, 10)
obstacles = generateCourse(60, 25, 775, 25, 725, 25, 40)
clock = pygame.time.Clock()
millis = 300
maxtime = 900 / millis
while not finished:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            finished = True
    updateGUI(p, obstacles)
    p.fitness = p.top - p.yc
    p.timeout += 1
    p.bounds()
    p.timecheck(250)
    p.updatenet(obstacles)
    p.move()
    if p.success:
        updateCourse(obstacles, 25, 775, 25, 725, 25, 40)
        p.success = False
    pygame.display.flip()
    clock.tick(50)
