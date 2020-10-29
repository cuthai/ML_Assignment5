import numpy as np
import math
from neural_network.neuron import Neuron


class HiddenLayer:
    def __init__(self, dimensions, classes, class_names, step_size, node_count):
        self.dimensions = dimensions
        self.classes = classes
        self.class_names = class_names
        self.node_count = node_count

        # Tune Variables
        self.step_size = step_size

        self.nodes = [Neuron(dimensions, classes, class_names, step_size) for index in range(node_count)]

    def predict(self, data):
        y = data[-1]

        output = np.empty(self.node_count + 1).astype('O')

        for index in range(self.node_count):
            output[index] = self.nodes[index].predict(data)

        output[index + 1] = y

        return output

    def update(self, backpropagation):
        for index in range(self.node_count):
            self.nodes[index].update(backpropagation)
