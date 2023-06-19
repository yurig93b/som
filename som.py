import math
import os
import random
import time
import uuid

import imageio
import matplotlib.pyplot as plt
import numpy as np

from neuron import Neuron

def data_supplier_non_uniform_boxes(i):
    if i % 3 == 0:
        return SOM.get_rand_point(0.0, 0.0, 0.3, 0.3)
    elif i % 4 == 0:
        return SOM.get_rand_point(0.7, 0.7, 1.0, 1.0)

    return SOM.get_rand_point(0.7, 0, 1.0, 0.3)


def data_supplier_non_uniform_scatter(i):
    return SOM.get_rand_point(0.0, 0.0, 0.5, 1.0)

def data_supplier_uniform_box(i):
    return SOM.get_rand_point(0.0, 0.0, 1.0, 1.0)

def data_supplier_uniform_donut(i):

    ang = math.radians(random.uniform(0, 360))
    r = random.uniform(4, 16)
    x = r * math.cos(ang)
    y = r * math.sin(ang)
    return Neuron(x, y)

def data_supplier_non_uniform_donut(i):

    ang = math.radians(random.uniform(-90, 90))
    r = random.uniform(4, 16)
    x = r * math.cos(ang)
    y = r * math.sin(ang)
    return Neuron(x, y)


class SOM(object):
    def __init__(self, initial_radius_factor = 4, initial_learning_rate=0.3, max_iterations=500, neurons_count=20):
        self._neurons = []
        self._train_points = []
        self._initial_learning_rate = initial_learning_rate
        self._initial_radius_factor = initial_radius_factor
        self._max_iterations = max_iterations
        self._init_radius = None
        self._neurons_count = neurons_count
        self.uuid = f'radius-factor--{self._initial_radius_factor}-learning-rate--{self._initial_learning_rate}-iterations--{self._max_iterations}-neurons--{self._neurons_count}'#str(uuid.uuid4())
        self.prep_images()

    @property
    def initial_radius(self):
        if self._init_radius is None:
            raise Exception("Init radius was not initialized")
        return self._init_radius

    @property
    def time_constant(self):
        return self._max_iterations / np.log(self.initial_radius)

    def init_line_of_neurons(self, min_x: float, min_y: float, max_x: float, max_y: float):
        if self._neurons:
            raise Exception("Neurons already initialized.")

        neurons = []

        step_x = abs(max_x - min_x) / (self._neurons_count - 1)
        step_y = abs(max_y - min_y) / (self._neurons_count - 1)

        self._init_radius = max(abs(max_x - min_x), abs(max_y - min_y)) / self._initial_radius_factor

        # Add first
        neurons.append(Neuron(min_x, min_y))

        for i in range(1, self._neurons_count - 1):
            neurons.append(Neuron(i * step_x, i * step_y, neighbour_left=neurons[-1]))
            neurons[-2].neighbour_right = neurons[-1]

        # Add last
        neurons.append(Neuron(max_x, max_y, neighbour_left=neurons[-1]))
        neurons[-2].neighbour_right = neurons[-1]

        self._neurons = neurons

    def init_circle_of_neurons(self, max_x:float, max_y:float):
        if self._neurons:
            raise Exception("Neurons already initialized.")

        step_ang = math.radians(abs(360 / self._neurons_count))
        neurons = []
        self._init_radius = max(max_y, max_x) / self._initial_radius_factor

        for i in range(0, self._neurons_count):
            x = math.cos(i * step_ang) * max_x
            y = math.sin(i * step_ang) * max_y
            neurons.append(Neuron(x, y))

        self._neurons = neurons
        return self

    def get_decayed_radius(self, iteration):
        return self.initial_radius * np.exp(iteration / self.time_constant) if self.initial_radius< 1 else self.initial_radius * np.exp(-iteration / self.time_constant)

    def get_decayed_learning_rate(self, iteration):
        return self._initial_learning_rate * np.exp(-iteration / self._max_iterations)

    def get_neighbour_influence(self, distance, r):
        return np.exp(-distance / (2 * (r ** 2)))

    def get_bmu(self, input_point: Neuron) -> Neuron:
        best_d = math.inf
        best_p = None

        for p in self._neurons:
            d = input_point.get_euclidian_distance(p)
            if d < best_d:
                best_d = d
                best_p = p

        return best_p

    @staticmethod
    def get_rand_point(min_x: float, min_y: float, max_x: float, max_y: float):
        return Neuron(random.uniform(min_x, max_x), random.uniform(min_y, max_y))

    def train(self, f_data_supplier):
        for i in range(self._max_iterations):

            t = f_data_supplier(i)
            self._train_points.append(t)

            # Find best node
            bmu_point: Neuron = self.get_bmu(t)

            r = self.get_decayed_radius(i)
            a = self.get_decayed_learning_rate(i)
            for n in self._neurons:
                # Get the 2-D distance (not Euclidean as no sqrt)
                w_dist = bmu_point.get_2d_distance(n)

                # If the distance is within the current neighbourhood radius
                if w_dist <= r ** 2:
                    # Calculate the degree of influence (based on the 2-D distance)
                    influence = self.get_neighbour_influence(w_dist, r)
                    # Update weight:
                    new_w = n.get_np_array() + (a * influence * (t.get_np_array() - n.get_np_array()))
                    n.x = new_w[0]
                    n.y = new_w[1]

            self.plot_frame_to_png(i)

    def prep_images(self):
        try:
            os.mkdir('./imgs')
        except:
            pass

        try:
            os.mkdir('./imgs/' + self.uuid)
        except:
            pass

    def plot_frame_to_png(self, i):
        fig = plt.figure(figsize=(8, 8))

        fig.suptitle(
            f'Neurons={len(self._neurons)} Iteration={i} Radius={round(self.get_decayed_radius(i), 4)} Alpha={round(self.get_decayed_learning_rate(i), 4)}',
            fontsize=16)

        plt.scatter([i.x for i in self._neurons], [i.y for i in self._neurons], marker='.', color='red')
        plt.scatter([i.x for i in self._train_points], [i.y for i in self._train_points], marker='x', color='blue')

        # Show the boundary between the regions:
        for n in self._neurons:
            x_neigh = [i.x for i in n.get_neighbours()]
            y_neigh = [i.y for i in n.get_neighbours()]

            plt.plot([n.x] + x_neigh, [n.y] + y_neigh, linestyle='solid', color='black', linewidth=0.1)

        plt.savefig(f'./imgs/{self.uuid}/img_{i}.png',
                    transparent=False,
                    facecolor='white'
                    )
        plt.close(fig)

    def create_gif(self):
        frames = []
        for i in range(self._max_iterations):
            image = imageio.v2.imread(f'./imgs/{self.uuid}/img_{i}.png')
            frames.append(image)

        imageio.mimsave(f'./imgs/{self.uuid}.gif',  # output gif
                        frames,  # array of input frames
                        duration=1000 * 1 / 15)  # optional: frames per second
