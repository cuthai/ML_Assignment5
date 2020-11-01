import numpy as np
import math


class Neuron:
    """
    Class Neuron

    Implements a simple neuron that outputs the Sigmoid function of the inputs and node weights
    """
    def __init__(self, dimensions, class_names, input_count, step_size):
        # Data Variables
        self.dimensions = dimensions
        self.class_names = class_names

        # Neuron Variables
        self.input_count = input_count

        # Tune Variables
        self.step_size = step_size

        # Initialize weights
        self.weights = np.random.uniform(low=-.01, high=.01, size=(1, self.input_count + 1))

        # Variables for holding current values, used for updating
        self.current_x = None
        self.current_output = 0

    def predict(self, data):
        # Assign X and Y
        x = data[:-1].astype(float)

        # Calculate output and sigmoid output
        output = np.matmul(x[:, None].T, self.weights.T)
        sigmoid = 1 / (1 + np.exp(-output))[0][0]

        # Assign currents to object
        self.current_x = x
        self.current_output = sigmoid

        return sigmoid

    def update(self, backpropagation):
        """
        Update function

        Takes a float and updates the weights using that backpropagation. New backpropagation is calculated using
            that split among the weights of this nodes inputs. This is calculated against the current output and data
        """
        # Calculate change of weights
        weights_delta = backpropagation * self.current_output * (1 - self.current_output) * self.current_x

        # Calculate new backpropagation by splitting among weights
        new_backpropagation = self.weights[:, 1:] * backpropagation

        # Update weights
        self.weights = self.weights + (self.step_size * weights_delta)

        return new_backpropagation[0]
