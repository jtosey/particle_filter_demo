# ------------------------------------------------------------------------
# coding=utf-8
# ------------------------------------------------------------------------
#
#  Created by Martin J. Laubach on 2011-11-15
#  Modified by Joseph Tosey on 2011-11-16
#    - use algorithm presented in ai-class.org 11-20
#    - reduced beacons from 4 to 3
#    - added barriers to create more opportunities for eliminating wrong
#      localization conclusions
#  Modified by Joseph Tosey on 2011-11-16
#    - make pick operation O(nlog(n)); previously O(n^2)
#    - more disperse error calculation to reduce wrong convergence
#    - extract StdDev so it is easier to manipulate
#    - don't create new particles in occupied cells
#    - double speed of turtle
#  Modified by Joseph Tosey on 2011-11-20
#    - added cross-check to verify that particles converged correctly,
#      and reset to random state when they do not
#
# ------------------------------------------------------------------------

from __future__ import absolute_import

import random
import math
import bisect

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

N = 500    # Total number of particles


# ------------------------------------------------------------------------
# Some utility functions

def add_noise(level, *coords):
    return [x + random.uniform(-level, level) for x in coords]

def add_little_noise(*coords):
    return add_noise(0.2, *coords)

def add_some_noise(*coords):
    return add_noise(0.1, *coords)

# ------------------------------------------------------------------------
class WeightedDistribution(object):
    def __init__(self, state):
      accum = 0.0
      self.state = state
      self.distribution = []
      for x in state:
          accum += x.w
          self.distribution.append(accum)

    def pick(self):
        return self.state[bisect.bisect_left(self.distribution, random.uniform(0.01, 0.99))]

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
        self.dx, self.dy = add_noise(0.3, 0, 0)

    def read_sensor(self, maze):
        """
        Poor robot, it's sensors are noisy and pretty strange,
        it only can measure the distance to the nearest beacon(!)
        and is not very accurate at that too!
        """
        return abs(add_little_noise(super(Robot, self).read_sensor(maze))[0])

    def move(self, maze):
        """
        Move the robot. Note that the movement is stochastic too.
        """
        while True:
            self.step_count += 1
            xx, yy = add_noise(0.02, self.x + self.dx, self.y + self.dy)
            if maze.is_free(xx, yy) and self.step_count % 60 != 0:
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

StdDev = 1.4
estimate = False

while True:

    # draw the state
    world.clear()
    world.show_particles(state)
    world.show_robot(robot)
    if estimate:
        world.show_estimate(estimate)
    world.update()

    # move randomly
    robot.move(world)

    # take measurement
    z = robot.read_sensor(world)

    # This is the weighted average of the particle estimates, and is used to determine if the model
    # converged to the correct location.  This extension was not presented in class.
    z_estimate = 0

    # create a weighted distribution, for fast picking
    dist = WeightedDistribution(state)

    # resample
    state_prime = []
    eta  = 0
    for _ in range(0, N):

        while True:

          # j ~ {w} with replacement
          sj = dist.pick()

          # x' ~ P( x' | U, sj )
          x_prime = Particle(add_some_noise(sj.x + robot.dx, sj.y + robot.dy))

          # loop to discard particles in impossible places (occupied cells) because
          # our probability distribution distribution doesn't know about occupancy
          if world.is_free(*x_prime.xy):
              break

        # w' = P( z | x' ); 1 is close to the robot's measurement, 0 is farther away
        particle_z = x_prime.read_sensor(world)
        error = z - particle_z
        w_prime = math.e ** -(error ** 2 / (2 * StdDev ** 2))
        x_prime.w = w_prime
        z_estimate += particle_z * w_prime

        # accumulate normalizer
        eta += w_prime

        # add to new state
        state_prime.append(x_prime)

    # normalize weights
    for x in state_prime:
        x.w *= 1 / eta

    z_estimate /= eta

    state = state_prime


    # This is an extension to what was presented in class.  With a single sensor and a highly regular world,
    # this model sometimes converges to the wrong location.  The algorithm below detects that the particle
    # filter has converged, and then verifies that it has converged to the correct location by comparing its
    # estimates to the measurements of the robot.  If it converged but has the wrong measurement, it managed
    # to converge to the wrong point.  If that happens, we just start all over again with a new set of particles.
    x = [p.x for p in state]
    x_min, x_max = min(x), max(x)
    y = [p.y for p in state]
    y_min, y_max = min(y), max(y)
    particle_area = (x_max - x_min) * (y_max - y_min)
    converged = particle_area < 3
    z_error = abs(z_estimate - z)
    estimate = False
    print "particle area (roughly): %4.1f; weighted z error: %3.2f" %  (particle_area, z_error),
    if converged:
        if z_error < 1.5:
            print " converged with highly correlated measurements"
            estimate = Particle( (sum([p.w * p.x for p in state]), sum([p.w * p.y for p in state])) )
        else:
            print " converged, but detected poorly correlated measurements - RESETTING"
            state = Particle.create_random(N, world)
    else:
        print " converging ..."
