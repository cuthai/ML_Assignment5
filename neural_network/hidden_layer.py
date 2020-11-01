import numpy as np
from neural_network.neuron import Neuron


class HiddenLayer:
    """
    Class HiddenLayer

    This class implements an update and predict function. It creates neurons based on the node_count
    """
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

        # For the first layer, inputs are equal to the dimensions, otherwise it is the node count of the previous layer
        if hl_index == 0:
            input_count = dimensions
        else:
            input_count = node_count

        # Initialize nodes
        kwargs = {
            'dimensions': dimensions,
            'class_names': class_names,
            'input_count': input_count,
            'step_size': step_size
        }
        self.nodes = [Neuron(**kwargs) for index in range(node_count)]

    def predict(self, data):
        """
        Predict function

        This function returns the outputs of each of the nodes in the layer as an array
        """
        # Set y
        y = data[-1]

        # Initialize output vector
        output = np.empty(self.node_count + 1).astype('O')

        # Send data to node and add to output
        for index in range(self.node_count):
            output[index] = self.nodes[index].predict(data)

        # Add y to output
        output[index + 1] = y

        return output

    def update(self, backpropagation):
        """
        Update function

        This function updates weights from backpropagation. A new backpropagation is calculated and returned in case
            there are other hidden nodes underneath
        """
        # Initial array for new backpropagation
        new_backpropagation = np.zeros(self.node_count)

        # Update the nodes using backpropagation for that node
        for index in range(self.node_count):
            backpropagation_update = self.nodes[index].update(backpropagation[index])

        # Assign the new backpropagation from the node to send further back, new is an array that needs to be added up
        if self.hl_index > 0:
            new_backpropagation += backpropagation_update

        return new_backpropagation
