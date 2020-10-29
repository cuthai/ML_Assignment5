import numpy as np
import math


class Neuron:
    def __init__(self, dimensions, classes, class_names, step_size):
        self.dimensions = dimensions
        self.classes = classes
        self.class_names = class_names

        # Tune Variables
        self.step_size = step_size

        self.weights = np.random.uniform(low=-.01, high=.01, size=(1, self.dimensions + 1))

        self.current_x = None
        self.current_output = 0

    def predict(self, data):
        # Assign X and Y
        x = data[:-1].astype(float)

        output = np.matmul(x[:, None].T, self.weights.T)
        sigmoid = 1 / (1 + np.exp(-output))[0][0]

        self.current_x = x
        self.current_output = sigmoid

        return sigmoid

    def update(self, backpropagation):
        weights_delta = backpropagation * self.current_output * (1 - self.current_output) * self.current_x

        self.weights = self.weights + (self.step_size * weights_delta)
