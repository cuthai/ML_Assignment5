import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


class LogisticRegressor:
    """
    Class LogisticRegressor

    This class implements a logistic regression which splits the classes using a logistic discriminant and a softmax
        function. Weights for determining class splits are learned using gradient descent. The class with the highest
        softmax is the predicted class. Functions implemented are Tune, Fit, Predict, and Summarize. The tune function
        is optional and outputs results for various step sizes. To run a Fit on a desired step size add the command line
        argument <-s float> when running main.
    """
    def __init__(self, etl, step_size=.01):
        """
        Init function

        Sets main variables and array conversion

        :param etl: etl, etl object with transformed and split data
        :param step_size: float, desired step_size to use during training
        """
        # Meta Variables
        self.etl = etl
        self.data_name = self.etl.data_name
        self.class_names = etl.class_names
        self.classes = etl.classes

        # Tune Variables
        self.step_size = step_size

        # Data Variables
        self.tune_data = etl.tune_data
        self.test_split = etl.test_split
        self.train_split = etl.train_split

        # Array Variables
        self.tune_array = self.tune_data.to_numpy()
        self.test_array_split = {key: self.test_split[key].to_numpy() for key in self.test_split.keys()}
        self.train_array_split = {key: self.train_split[key].to_numpy() for key in self.train_split.keys()}

        # Train Models
        self.train_models = {index: {} for index in range(5)}

        # Tune Results
        self.tune_results = {
            round(step_size, 2): None for step_size in np.linspace(.01, .25, 25)
        }

        # Test Results
        self.test_results = {index: {} for index in range(5)}

        # Summary
        self.summary = {}
        self.summary_classification = None

    def tune(self):
        """
        Tune function.

        This function runs through a couple of step_sizes. The model is then trained for that step_size and then tested
            against the tune data set. The resulting misclassification is stored and then visualized as a chart. The
            step_size for the entire model is not set here. That should be done by adding a step_size argument
        """
        # Gather default step_sizes
        step_sizes = self.tune_results.keys()

        # Loop through step_sizes to test
        for step_size in step_sizes:
            misclassification = 0

            # Train over the 5 CV splits
            for index in range(5):
                # Gather variables, weights and intercepts come from the training
                data = self.train_array_split[index]
                weights, intercepts = self.train(data, step_size)

                # Classify using our model and tune data set
                model = {
                    'weights': weights,
                    'intercepts': intercepts
                }
                predictions = self.classify(self.tune_array, model)

                # Compare results
                results = pd.DataFrame.copy(self.tune_data)
                results['Prediction'] = predictions

                # Calculate misclassification
                misclassification += len(results[results['Class'] != results['Prediction']]) / len(results)

            # Save results
            self.tune_results.update({step_size: misclassification / 5})

        # Trigger visualization
        self.visualize()

    def visualize(self):
        """
        Tune visualization function

        This function uses the results of the tune function to create a plot graph

        :return: matplotlib saved jpg in output folder
        """
        # Figure / axis set up
        fig, ax = plt.subplots()

        # We'll plot the list of params and their accuracy
        ax.plot(self.tune_results.keys(), self.tune_results.values())

        # Title
        ax.set_title(rf'{self.data_name} Tune Results')

        # X axis
        ax.set_xlabel('Step_Size')
        ax.set_xlim(0, .25)
        ax.set_xticks(list(self.tune_results.keys()))
        ax.set_xticklabels(list(self.tune_results.keys()), rotation=45, fontsize=6)

        # Y axis
        ax.set_ylabel('Misclassification')

        # Saving
        plt.savefig(f'output_{self.data_name}\\logistic_{self.data_name}_tune.jpg')

    def fit(self):
        """
        Fit Function

        This function loops through the 5 CV splits and then calls to the train function. It receives the results of the
            train function and sets the model for that split
        """
        # Loop through the 5 CV splits
        for index in range(5):
            # Assign data
            data = self.train_array_split[index]

            # Train
            weights, intercepts = self.train(data)

            # Set model
            self.train_models[index].update({
                'weights': weights,
                'intercepts': intercepts
            })

    def train(self, data, step_size=None):
        """
        Train Function

        This function takes data and then performs gradient descent on it to discover the optimal weights. Gradient
            descent occurs until there are no further changes to misclassification. The step_size is used as a scalar
            for the deltas in gradient descent

        :param data: np.array, data to train on
        :param step_size: float, step_size for gradient descent
        :return weights: np.array, floats of the weights by class and dimensions (rows = classes, columns = dimensions)
        :return intercepts: np.array, floats of the intercepts by class (rows = classes, columns = only 1)
        """
        # Grab step_size
        if not step_size:
            step_size = self.step_size

        # Assign X and Y
        x = data[:, :-1].astype(float)
        y = data[:, -1]

        # Initial weight and intercept variables. Rows = classes, columns = dimensions
        weights = np.random.uniform(low=-.01, high=.01, size=(self.classes, x.shape[1]))
        intercepts = np.zeros((self.classes, 1))

        # Initial misclassification
        misclassification = 1

        # Use a while loop, we'll break out with a comparison later
        while True:
            # Initial delta variables, all set to 0
            weights_delta = np.zeros((self.classes, x.shape[1]))
            intercepts_delta = np.zeros((self.classes, 1))

            # Grab outputs, likelihoods, and make predictions on the current weights and intercepts
            # The likelihood calculation uses a softmax
            outputs = np.matmul(x, weights.T) + intercepts.T
            likelihood = (np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None])
            predictions = np.argmax(likelihood, axis=1).astype('O')

            # For each class, we will calculate the changes to delta
            for index in range(self.classes):
                # For this current class, calculate the difference between actual and the softmax likelihood
                current_class = self.class_names[index]
                actuals = (y == current_class).astype(int)
                difference = (actuals - likelihood[:, index])

                # Update the deltas for this class
                weights_delta[index, :] = np.matmul(difference, x)
                intercepts_delta[index, :] += sum(difference)

                # Update the predictions with class names
                predictions[predictions == index] = self.class_names[index]

            # Calculate new misclassification
            new_misclassification = sum(predictions != y) / len(y)

            # Stop condition, if no convergence, set misclassification to new and continue while
            if new_misclassification < misclassification:
                misclassification = new_misclassification

                # Update weights and intercepts with deltas
                weights = weights + (step_size * weights_delta)
                intercepts = intercepts + (step_size * intercepts_delta)

            # If misclassification did not change between new and old, stop and break from while loop
            else:
                break

        # Return final weights and intercepts
        return weights, intercepts

    def predict(self):
        """
        Predict Function

        This function loops through the 5 CV splits and then calls to the classify function. It receives the results of
            the classification function and sets the results for that split
        """
        # Loop through the 5 CV splits
        for index in range(5):
            # Assign data
            data = self.test_array_split[index]
            model = self.train_models[index]

            # Classify, get results
            predictions = self.classify(data, model)

            # Compare results
            results = pd.DataFrame.copy(self.test_split[index])
            results['Prediction'] = predictions

            # Calculate misclassification
            misclassification = len(results[results['Class'] != results['Prediction']]) / len(results)

            # Save results
            self.test_results[index].update({
                'results': results,
                'misclassification': misclassification
            })

    def classify(self, data, model):
        """
        Classify Function

        This function takes data and then calculates the softmax likelihood using the weights of the model. The highest
            likelihood is the assigned class

        :param data: np.array, data to train on
        :param model: dict, dictionary of out model with weights and intercepts
        :return predictions: np.array, assigned class, ordered in the order of the data param
        """
        # Assign data
        x = data[:, :-1].astype(float)

        # Assign model variables
        weights = model['weights']
        intercepts = model['intercepts']

        # Grab outputs, likelihoods, and make predictions on the current weights and intercepts
        # The likelihood calculation uses a softmax
        outputs = np.matmul(x, weights.T) + intercepts.T
        likelihood = (np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None])
        predictions = np.argmax(likelihood, axis=1).astype('O')

        # For each class, make a prediction using the index of the highest softmax likelihood
        for index in range(self.classes):
            predictions[predictions == index] = self.class_names[index]

        # Return the predictions
        return predictions

    def summarize(self):
        """
        Summarize Function

        This function outputs a CSV and a JSON file for the results of the predict. The CSV is the combined results of
            all 5 CV splits and their prediction. The JSON file is the step_size used and the misclassification.

        :return csv: csv, output of predictions of all CV splits to output folder
        :return json: json, dictionary of the results of all CV splits to output folder
        """
        # Calculate misclassification
        misclassification = sum([self.test_results[index]['misclassification'] for index in range(5)])

        # Summary JSON
        self.summary = {
            'tune': {
                'step_size': self.step_size
            },
            'test': {
                'misclassification': misclassification / 5
            }
        }

        # Output JSON
        with open(f'output_{self.data_name}\\logistic_{self.data_name}_summary.json', 'w') as file:
            json.dump(self.summary, file)

        # Summary CSV
        summary_classification = pd.DataFrame()

        # Loop through each test data set and add the results
        for index in range(5):
            summary_classification = summary_classification.append(self.test_results[index]['results'])

        # Dump CSV and save
        summary_classification.to_csv(f'output_{self.data_name}\\logistic_{self.data_name}_classification.csv')
        self.summary_classification = summary_classification
