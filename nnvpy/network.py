from jax_engine.star import Star
import numpy as np
from dataclasses import dataclass
import heapq
from functools import total_ordering


@dataclass(order=True)
class ProbStar():
    def __init__(self, star, prob, layer_posn, remaining_relu_steps):
        """relu_steps lists the remaining stepRelus before the next affine"""
        self.negprob = -1. * prob
        self.star = star
        self.layer_posn = layer_posn
        self.relu_steps = remaining_relu_steps

    def next_op(self, mu, sigma, affine_transformations):
        if self.relu_steps:
            children = self.star.step_relu(self.relu_steps[0])
            children = [
                ProbStar(child,
                         child.trunc_gaussian_cdf(mu, sigma)[0],
                         self.layer_posn, self.relu_steps[1:])
                for child in children
            ]
        else:
            self.star.affine_map(*affine_transformations[self.layer_posn])
            self.layer_posn += 1
            if self.layer_posn < len(affine_transformations):
                self.relu_steps = list(
                    range(len(affine_transformations[self.layer_posn - 1][1])))
            children = [self]
        return children

    @property
    def prob(self):
        return -1 * self.negprob


class Layer(object):
    'a layer object for feedforward neural network'

    def __init__(self, W, b, f):
        # L: y = f(Wx +b), f is an activation function
        #    W: is weight matrix, b is bias vector
        if W.shape[0] == b.shape[0]:
            self.is_right_mul = False
        elif W.shape[1] == b.shape[0]:
            self.side = True
        else:
            raise Exception(
                'inconsistency between weight matrix and bias vector')

        self.W = W
        self.b = b

        assert f == 'ReLU', 'error: unknown or unsupported activation function'
        self.f = f

    @property
    def num_neurons(self):
        return self.W.shape[1]

    @property
    def input_shape(self):
        return self.W.shape[0]

    def flatten(self):
        'flatten a layer into a sequence of operations'
        ops = [lambda x: x.affine_map(self.W, self.b)]
        for i in range(self.num_neurons):
            if self.f == 'ReLU':
                ops.append(lambda x: x.step_relu(i))
        return ops


class Network():
    def __init__(self, layers=None):
        self.layers = []
        if layers:
            for W, b in layers:
                self.layers.append(Layer(W, b, 'ReLU'))

    @property
    def input_shape(self):
        return self.layers[0].input_shape

    def add_dense_layer(self, W, b):
        """W and b are TF variables and it's assumed the layer is relu activated"""
        self.layers.append(Layer(W, b, 'ReLU'))

    def gaussian_astar(self, mu, sigma):
        """
        Args:
            mu - 1-D numpy array
            sigma - 2-D np.array covariance matrix
        """
        from queue import PriorityQueue
        layers = [(l.W, l.b) for l in self.layers]

        output_stars = []
        frontier = PriorityQueue()

        input_star = Star(basis=np.eye(mu.shape[0]), center=np.zeros_like(mu))
        frontier.put_nowait((-1., ProbStar(input_star, 1., 0, None)))

        done = False
        while not done and not frontier.empty():
            _, current_star = frontier.get_nowait()
            # do a relu if there are steps left
            children = current_star.next_op(mu, sigma, layers)
            for child in children:
                if child.layer_posn < len(layers):
                    frontier.put_nowait((child.negprob, child))
                else:
                    output_stars.append(child)
        return output_stars