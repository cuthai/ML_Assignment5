import numpy as np


class OutputLayerRegressor:
    """
    Class OutputLayer

    This class receives the inputs and calculates the output
    """
    def __init__(self, dimensions, classes, class_names, input_count, step_size, convergence_threshold):
        # Data Variables
        self.dimensions = dimensions
        self.classes = classes
        self.class_names = class_names

        # Layer Variables
        self.input_count = input_count

        # Tune Variables
        self.step_size = step_size
        self.convergence_threshold = convergence_threshold

        # Initialize weights
        self.weights = np.random.uniform(low=-.01, high=.01, size=(1, self.input_count + 1))

    def fit(self, data):
        """
        Fit function

        This function calculates mse, updates weights, and backpropagation. The function used is a linear model
        """
        # Assign X and Y
        x = data[:-1].astype(float)
        y = data[-1]

        # Calculate output
        output = np.matmul(x[:, None].T, self.weights.T)[0][0]

        # Initial delta variables, all set to 0
        weights_delta = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        mse = 0

        # Initial backpropagation
        backpropagation = np.zeros(self.input_count)

        # Calculate the difference between softmax and target
        difference = (y - output)

        # Add up for backpropagation
        for input_index in range(self.input_count):
            backpropagation[input_index] += difference * self.weights[0, input_index + 1]

            # Update the deltas for this class
            weights_delta[0, :] = x * difference

            # Calculate MSE
            mse = (difference ** 2)

        # Check for convergence
        if mse < self.convergence_threshold:
            return True, 0

        # If not converged, update weights and return backpropagation
        else:
            self.weights = self.weights + (self.step_size * weights_delta)

            return False, backpropagation

    def predict(self, data):
        """
        Predict function

        Return output
        """
        # Assign X
        x = data[:-1].astype(float)

        # Output function
        output = np.matmul(x[:, None].T, self.weights.T)[0][0]

        return output
