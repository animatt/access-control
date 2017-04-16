#!/usr/local/bin/python3

import RLtoolkit.tiles as tiles
import numpy as np
import matplotlib.pyplot as plt


# from scipy.ndimage.interpolation import shift
# print(shift(queue, 2))
# queue = 2 ** np.random.randint(4, size = 10)


# misc functions
def argmax(func, S_tiles, f_domain, weights) :
    largest = float('-inf')
    for action in f_domain :
        if func(S_tiles, action, weights) > largest :
            largest = func(S_tiles, action, weights)
            A = action
    return A


# initialize env
k = 16 # number of servers, chosen so number of states will be a power of 2
servers_busy = 0
p = 0.06 # probability of a server becoming available


# initialize agent
tile_width = 1
num_tilings = 8
memsize = 512

epsilon = .1
alpha = 0.01
beta = 0.01
gamma = 1
R_ = 0


theta = [np.zeros(memsize), np.zeros(memsize)]


def q(S_tiles, A, weights) :
    val = 0
    for index in S_tiles :
        val += weights[A][index]
    return val


def policy(S_tiles, weights, epsilon) :
    rand = np.random.randint(2)
    return np.random.choice((argmax(q, S_tiles, range(2), weights), rand), \
        p = (1 - epsilon, epsilon))


# begin simulation
queue = np.random.randint(4)
servers_busy -= np.random.choice((0, 1), servers_busy, p = (1 - p, p)).sum()

S2 = tiles.tiles(num_tilings, memsize, (np.random.randint(4), servers_busy))
A2 = policy(S2, theta, epsilon) if servers_busy < k else 0

servers_busy = servers_busy + A2

R_ = A2 * 2 ** queue

count = 1
converging = True
while converging :

    S1 = S2
    A1 = A2

    queue = np.random.randint(4)
    servers_busy -= np.random.choice((0, 1), servers_busy, p = (1 - p, p)).sum()

    S2 = tiles.tiles(num_tilings, memsize, (queue, servers_busy))
    A2 = policy(S2, theta, epsilon) if servers_busy < k else 0

    servers_busy = servers_busy + A2 if servers_busy < k else servers_busy

    delta = 2 ** queue * A1 - R_ + gamma * q(S2, A2, theta) - q(S1, A1, theta)
    R_ = 2 ** queue * A1 + beta * delta

    theta[A1][S1] += alpha * delta

    count += 1
    if count % 10000 == 0 :
        epsilon /= 1.025
        print('count:', count)
        print('avg reward:', R_)
        print('servers_busy:', servers_busy, 'queue:', 2 ** queue, 'q(...):', \
            q(S1, A1, theta))
