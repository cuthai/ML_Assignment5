import numpy as np
import math


class OutputLayer:
    def __init__(self, dimensions, classes, class_names, step_size):
        self.dimensions = dimensions
        self.classes = classes
        self.class_names = class_names

        # Tune Variables
        self.step_size = step_size
        self.convergence = .001

        self.weights = np.random.uniform(low=-.01, high=.01, size=(self.classes, self.dimensions + 1))

    def fit(self, data):
        # Assign X and Y
        x = data[:-1].astype(float)
        y = data[-1]

        output = np.matmul(x[:, None].T, self.weights.T)
        likelihood = (np.exp(output) / np.sum(np.exp(output), axis=1)[:, None])

        # Initial delta variables, all set to 0
        weights_delta = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        entropy = 0

        # For each class, we will calculate the changes to delta
        for index in range(self.classes):
            current_class = self.class_names[index]
            if current_class == y:
                actual = 1
            else:
                actual = 0

            difference = (actual - likelihood[:, index])

            # Update the deltas for this class
            weights_delta[index, :] = x * difference

            entropy -= actual * math.log(likelihood[:, index])

        self.weights = self.weights + (self.step_size * weights_delta)

        print(entropy)

        if entropy < self.convergence:
            return True
        else:
            return False