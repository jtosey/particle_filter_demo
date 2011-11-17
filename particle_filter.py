# ------------------------------------------------------------------------
# coding=utf-8
# ------------------------------------------------------------------------
#
#  Created by Martin J. Laubach on 2011-11-15
#  Modified by Joseph Tosey on 2011-11-16
#
# ------------------------------------------------------------------------

from __future__ import absolute_import

import random
import math

from draw import Maze

# 0 - empty square
# 1 - occupied square
# 2 - occupied square with a beacon at each corner, detectable by the robot

maze_data = ( ( 1, 1, 0, 0, 1, 2, 0, 1, 0, 0 ),
              ( 1, 2, 0, 1, 1, 1, 0, 0, 0, 0 ),
              ( 0, 1, 0, 0, 0, 0, 0, 1, 0, 1 ),
              ( 0, 0, 0, 0, 1, 0, 0, 1, 1, 2 ),
              ( 1, 1, 1, 1, 1, 2, 0, 0, 0, 0 ),
              ( 1, 1, 1, 0, 1, 1, 1, 0, 1, 0 ),
              ( 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 ),
              ( 1, 2, 0, 1, 1, 0, 0, 2, 0, 0 ),
              ( 0, 0, 0, 0, 1, 1, 0, 0, 1, 0 ),
              ( 0, 1, 0, 0, 2, 1, 0, 0, 1, 0 ))

N = 2000    # Total number of particles

# ------------------------------------------------------------------------
# Some utility functions

def add_noise(level, *coords):
    return [x + random.uniform(-level, level) for x in coords]

def add_little_noise(*coords):
    return add_noise(0.2, *coords)

def add_some_noise(*coords):
    return add_noise(0.1, *coords)

def weightedPick(particles):
    r = random.uniform(0, 1)
    s = 0.0
    for p in particles:
        s += p.w
        if r < s:
            return p
    return p

# ------------------------------------------------------------------------
class Particle(object):
    def __init__(self, coord, heading=270, w=1):
        self.x = coord[0]
        self.y = coord[1]
        self.h = heading
        self.w = w

    def __repr__(self):
        return "(%f, %f, w=%f)" % (self.x, self.y, self.w)

    @property
    def xy(self):
        return self.x, self.y

    @property
    def xyh(self):
        return self.x, self.y, self.h

    @classmethod
    def create_random(cls, count, maze):
        return [cls(maze.random_free_place(), w=1.0 / count) for _ in range(0, count)]

    def read_sensor(self, maze):
        """
        Find distance to nearest beacon.
        """
        return maze.distance_to_nearest_beacon(*self.xy)

# ------------------------------------------------------------------------
class Robot(Particle):
    def __init__(self, maze):
        super(Robot, self).__init__(maze.random_free_place(), heading=90)
        self.chose_random_direction()
        self.step_count = 0

    def chose_random_direction(self):
        self.dx, self.dy = add_noise(0.1, 0, 0)

    def read_sensor(self, maze):
        """
        Poor robot, it's sensors are noisy and pretty strange,
        it only can measure the distance to the nearest beacon(!)
        and is not very accurate at that too!
        """
        return add_little_noise(super(Robot, self).read_sensor(maze))[0]

    def move(self, maze):
        """
        Move the robot. Note that the movement is stochastic too.
        """
        while True:
            self.step_count += 1
            xx, yy = add_noise(0.02, self.x + self.dx, self.y + self.dy)
            if maze.is_free(xx, yy) and self.step_count % 30 != 0:
                self.x, self.y = xx, yy
                break
            # Bumped into something or too long in same direction,
            # chose random new direction
            self.chose_random_direction()

# ------------------------------------------------------------------------

world = Maze(maze_data)
world.draw()

# initial distribution assigns each particle an equal probability
robot = Robot(world)
state = Particle.create_random(N, world)

while True:

    # draw the state
    world.show_particles(state)
    world.show_robot(robot)

    # move randomly
    robot.move(world)

    # take measurement
    z = robot.read_sensor(world)

    # resample
    state_prime = []
    eta = 0
    for _ in range(0, N):

        # j ~ {w} with replacement
        sj = weightedPick(state)

        # x' ~ P( x' | U, sj )
        x_prime = Particle(add_some_noise(sj.x + robot.dx, sj.y + robot.dy))

        # w' = P( z | x' ); 1 is close to the robot's measurement, 0 is farther away
        error = z - x_prime.read_sensor(world)
        w_prime = math.e ** -(error ** 2 * 11) if world.is_free(*x_prime.xy) else 0
        x_prime.w = w_prime

        # accumulate normalizer
        eta += w_prime

        # add to new state
        state_prime.append(x_prime)

    # normalize weights
    for x in state_prime:
        x.w *= 1 / eta

    state = state_prime
