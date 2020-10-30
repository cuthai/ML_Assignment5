import numpy as np
from neural_network.neuron import Neuron


class HiddenLayer:
    def __init__(self, dimensions, classes, class_names, hl_index, step_size, node_count):
        # Data Variables
        self.dimensions = dimensions
        self.classes = classes
        self.class_names = class_names

        # Layer Variables
        self.hl_index = hl_index

        # Tune Variables
        self.step_size = step_size
        self.node_count = node_count

        if hl_index == 0:
            input_count = dimensions
        else:
            input_count = node_count

        kwargs = {
            'dimensions': dimensions,
            'class_names': class_names,
            'input_count': input_count,
            'step_size': step_size
        }

        self.nodes = [Neuron(**kwargs) for index in range(node_count)]

    def predict(self, data):
        y = data[-1]

        output = np.empty(self.node_count + 1).astype('O')

        for index in range(self.node_count):
            output[index] = self.nodes[index].predict(data)

        output[index + 1] = y

        return output

    def update(self, backpropagation):
        new_backpropagation = np.zeros(self.node_count)

        for index in range(self.node_count):
            backpropagation_update = self.nodes[index].update(backpropagation[index])

        if self.hl_index > 0:
            new_backpropagation += backpropagation_update

        return new_backpropagation
