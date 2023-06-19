from __future__ import annotations

import numpy as np


class Neuron(object):
    def __init__(self, x, y, neighbour_left=None, neighbour_right=None, neighbour_top=None, neighbour_bottom=None):
        self.x = x
        self.y = y

        self.neighbour_left = neighbour_left
        self.neighbour_right = neighbour_right
        self.neighbour_top = neighbour_top
        self.neighbour_bottom = neighbour_bottom

    def get_neighbours(self):
        return [x for x in
                [self.neighbour_left, self.neighbour_right, self.neighbour_top, self.neighbour_bottom] if
                x is not None]

    def get_euclidian_distance(self, other: Neuron):
        return np.linalg.norm(np.array((self.x, self.y)) - np.array((other.x, other.y)))

    def get_2d_distance(self, other:Neuron):
        return np.sum((np.array([self.x, self.y]) - np.array((other.x, other.y))) ** 2)

    def get_np_array(self):
        return np.array((self.x, self.y))

    def __repr__(self):
        return f"Point ({self.x}, {self.y})"
