import numpy as np
import math


class OutputLayer:
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

        self.weights = np.random.uniform(low=-.01, high=.01, size=(self.classes, self.input_count + 1))

    def fit(self, data):
        # Assign X and Y
        x = data[:-1].astype(float)
        y = data[-1]

        output = np.matmul(x[:, None].T, self.weights.T)
        likelihood = (np.exp(output) / np.sum(np.exp(output), axis=1)[:, None])

        # Initial delta variables, all set to 0
        weights_delta = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        entropy = 0

        backpropagation = np.zeros(self.input_count)

        # For each class, we will calculate the changes to delta
        for index in range(self.classes):
            current_class = self.class_names[index]
            if current_class == y:
                actual = 1
            else:
                actual = 0

            difference = (actual - likelihood[:, index])

            for input_index in range(self.input_count):
                backpropagation[input_index] += sum(difference * self.weights[index, input_index + 1])

            # Update the deltas for this class
            weights_delta[index, :] = x * difference

            entropy -= actual * math.log(likelihood[:, index])

        print(entropy)

        if entropy < self.convergence_threshold:
            return True, 0
        else:
            self.weights = self.weights + (self.step_size * weights_delta)

            return False, backpropagation

    def predict(self, data):
        # Assign X
        x = data[:-1].astype(float)

        output = np.matmul(x[:, None].T, self.weights.T)
        likelihood = (np.exp(output) / np.sum(np.exp(output), axis=1)[:, None])
        prediction_index = np.argmax(likelihood, axis=1).astype('O')[0]

        prediction = self.class_names[prediction_index]

        return prediction
