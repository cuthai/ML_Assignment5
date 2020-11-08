import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from neural_network.output_layer import OutputLayer
from neural_network.output_layer_regression import OutputLayerRegressor
from neural_network.hidden_layer import HiddenLayer


class NeuralNetwork:
    """
    Class NeuralNetwork

    Creates a NeuralNetwork object that implements hidden layers and output layers. The output layer uses a perceptron.
        Hidden layers are activated using a sigmoid function.
    """
    def __init__(self, etl, hidden_layers_count=0, step_size=.01, node_count=1, convergence_threshold=.01,
                 random_state=1):
        """
        Init function

        Sets main variables and array conversion

        :param etl: etl, etl object with transformed and split data
        :param step_size: float, desired step_size to use during training
        :param node_count: int, number of nodes in the hidden layer
        :param convergence_threshold: float, convergence threshold for stopping, MSE or misclassification %
        """
        # Meta Variables
        self.etl = etl
        self.data_name = self.etl.data_name
        self.dimensions = len(etl.transformed_data.columns) - 1
        self.classes = etl.classes
        self.class_names = etl.class_names
        self.random_state = random_state
        self.squared_average_target = etl.squared_average_target

        # Model Variables
        self.hidden_layers_count = hidden_layers_count
        self.type = 'c'
        if self.classes == 0:
            self.type = 'r'

        # Tune Variables
        self.step_size = step_size
        self.node_count = node_count
        self.convergence_threshold = convergence_threshold

        # Data Variables
        self.tune_data = etl.tune_data
        self.test_split = etl.test_split
        self.train_split = etl.train_split

        # Array Variables
        self.tune_array = self.tune_data.to_numpy()
        self.test_array_split = {key: self.test_split[key].to_numpy() for key in self.test_split.keys()}
        self.train_array_split = {key: self.train_split[key].to_numpy() for key in self.train_split.keys()}

        # Train Models
        # Hidden Layers
        # If no hidden layers were set, skip
        if self.hidden_layers_count == 0:
            input_count = self.dimensions
            self.hidden_layers = None

        # Otherwise set up hidden layers, the output layer's input will be the number of nodes in the hidden layers
        else:
            input_count = self.node_count
            hl_kwargs = {
                'dimensions': self.dimensions,
                'classes': self.classes,
                'class_names': self.class_names,
                'step_size': self.step_size,
                'node_count': self.node_count,
            }
            self.hidden_layers = {
                index: {
                    hl_index: HiddenLayer(**hl_kwargs, hl_index=hl_index)
                    for hl_index in range(self.hidden_layers_count)
                } for index in range(5)
            }

        # Output Layer
        ol_kwargs = {
            'dimensions': self.dimensions,
            'classes': self.classes,
            'class_names': self.class_names,
            'step_size': self.step_size,
            'convergence_threshold': self.convergence_threshold,
            'input_count': input_count
        }
        if self.type == 'c':
            self.output_layer = {
                index: OutputLayer(**ol_kwargs)
                for index in range(5)
            }
        else:
            ol_kwargs.update({'convergence_threshold': self.convergence_threshold * self.squared_average_target})
            self.output_layer = {
                index: OutputLayerRegressor(**ol_kwargs)
                for index in range(5)
            }

        # Train Results
        self.epochs = {index: 0 for index in range(5)}

        # Tune Results
        self.tune_results = {
            'Step_Size': {round(step_size, 2): None for step_size in np.linspace(.01, .25, 25)},
            'Node_Count': {node_count: None for node_count in range(1, 11, 1)},
            'Convergence_Threshold': {
                round(convergence_threshold, 2): None for convergence_threshold in np.linspace(.01, .25, 25)
            },
        }

        # For regression, adjust the tune parameters lower for step size and convergence threshold
        if self.type == 'r':
            self.tune_results.update({
                'Step_Size': {round(step_size, 3): None for step_size in np.linspace(.001, .025, 25)},
                'Convergence_Threshold': {
                    round(convergence_threshold, 5): None for convergence_threshold in np.linspace(.00001, .00025, 25)
                }
            })

        # Test Results
        self.test_results = {index: {} for index in range(5)}

        # Summary
        self.summary = {}
        self.summary_prediction = None

    def tune(self, param):
        """
        Tune function

        This function takes a char and tunes for the corresponding parameter. The chars are:
            step_size (s), node_count (n), convergence_threshold(c)
            This can be passed at the command args level. The other params are set to the defaults or whatever was
            passed. The tune function resets the current model, and then calls to fit on the params before testing
        """
        # Determine param being tuned
        if param == 's':
            param_name = 'Step_Size'
        elif param == 'n':
            param_name = 'Node_Count'
        else:
            param_name = 'Convergence_Threshold'

        # Grab param range
        param_range = self.tune_results[param_name].keys()

        # Calculate original classes for misclassification
        original = self.tune_data.iloc[:, -1].to_list()

        # Loop through step_sizes to test
        for param_value in param_range:
            # Set param being tested
            if param == 's':
                self.step_size = param_value
            elif param == 'n':
                self.node_count = param_value
            else:
                self.convergence_threshold = param_value

            # Reset model
            self.reset()
            target = 0

            # Fit model
            self.fit()

            # Loop through the 5 CV splits
            for index in range(5):
                predictions = []

                # Assign data
                data = self.tune_array
                data = np.insert(data, 0, 1, axis=1)

                # Predict on tune dataset
                for row in data:
                    current_data = row

                    # If there are hidden layers, the data needs to go through there first
                    if self.hidden_layers_count > 0:
                        for hl_index in range(self.hidden_layers_count):
                            current_data = self.hidden_layers[index][hl_index].predict(current_data)
                            current_data = np.insert(current_data, 0, 1)

                    # After hidden layers, call to output layer and grab predictions
                    predictions.append(self.output_layer[index].predict(current_data))

                if self.type == 'c':
                    # Calculate misclassification
                    target += sum(np.array(original) != np.array(predictions)) / len(original)
                else:
                    target += sum((np.array(original) - np.array(predictions)) ** 2) / len(original)

            # Save results
            self.tune_results[param_name].update({param_value: target / 5})

        # Trigger visualization
        self.visualize(self.tune_results[param_name], param_name)

    def reset(self):
        """
        Reset function

        This function resets the model by clearing out any trained layers and reinitializing them with the model's
            params
        """
        # If no hidden layers were set, skip
        if self.hidden_layers_count == 0:
            input_count = self.dimensions
            self.hidden_layers = None

        # Otherwise set up hidden layers, the output layer's input will be the number of nodes in the hidden layers
        else:
            input_count = self.node_count
            hl_kwargs = {
                'dimensions': self.dimensions,
                'classes': self.classes,
                'class_names': self.class_names,
                'step_size': self.step_size,
                'node_count': self.node_count,
            }
            self.hidden_layers = {
                index: {
                    hl_index: HiddenLayer(**hl_kwargs, hl_index=hl_index)
                    for hl_index in range(self.hidden_layers_count)
                } for index in range(5)
            }

        # Output Layers
        ol_kwargs = {
            'dimensions': self.dimensions,
            'classes': self.classes,
            'class_names': self.class_names,
            'step_size': self.step_size,
            'convergence_threshold': self.convergence_threshold,
            'input_count': input_count
        }
        if self.type == 'c':
            self.output_layer = {
                index: OutputLayer(**ol_kwargs)
                for index in range(5)
            }
        else:
            ol_kwargs.update({'convergence_threshold': self.convergence_threshold * self.squared_average_target})
            self.output_layer = {
                index: OutputLayerRegressor(**ol_kwargs)
                for index in range(5)
            }

        # Train Results
        self.epochs = {index: 0 for index in range(5)}

    def visualize(self, data, name):
        """
        Tune visualization function

        This function uses the results of the tune function to create a plot graph

        :return: matplotlib saved jpg in output folder
        """
        # Figure / axis set up
        fig, ax = plt.subplots()

        # We'll plot the list of params and their accuracy
        ax.plot(data.keys(), data.values())

        # Title
        ax.set_title(rf'{self.data_name} {name} Tune Results')

        # X axis
        ax.set_xlabel(name)
        ax.set_xlim(min(data.keys()), max(data.keys()))
        ax.set_xticks(list(data.keys()))
        ax.set_xticklabels(list(data.keys()), rotation=45, fontsize=6)

        # Y axis
        ax.set_ylabel('Target')

        # Saving
        plt.savefig(f'output_{self.data_name}\\{self.data_name}_{self.hidden_layers_count}_layers_tune_{name}.jpg')

    def fit(self):
        """
        Fit Function

        This function loops through the 5 CV splits and calls the individual layers to fit based on the data being
            passed. Everytime the dataset is read, an epoch is added. Fit function terminates when the convergence is
            met or 1000 epochs.
        """
        # Set seed
        np.random.seed(self.random_state)

        # Loop through the 5 CV splits
        for index in range(5):
            convergence = False

            # While loop, continue until convergence or epochs are met
            while not convergence:
                # Assign data
                data = self.train_array_split[index]
                data = np.insert(data, 0, 1, axis=1)

                # Shuffle data
                np.random.shuffle(data)

                # Increment epochs
                self.epochs[index] += 1

                # Train
                for row in data:
                    # Grab current row
                    current_data = row

                    # Train hidden layers
                    if self.hidden_layers_count > 0:
                        for hl_index in range(self.hidden_layers_count):
                            current_data = self.hidden_layers[index][hl_index].predict(current_data)
                            current_data = np.insert(current_data, 0, 1)

                    # Train output, retrieve convergence status and backpropagation
                    convergence, backpropagation = self.output_layer[index].fit(current_data)

                    # Check for convergence
                    if convergence:
                        break

                    # Backpropagate through hidden layers, if multiple backpropagation must be updated after layer
                    if self.hidden_layers_count > 0:
                        for hl_index in reversed(range(self.hidden_layers_count)):
                            backpropagation = self.hidden_layers[index][hl_index].update(backpropagation)

                # Check for epoch stopper
                if self.epochs[index] >= 1000:
                    convergence = True

    def predict(self):
        """
        Predict function

        Loops through the CV splits and tests against trained models
        """
        # Loop through the 5 CV splits
        for index in range(5):
            predictions = []

            # Assign data
            data = self.test_array_split[index]
            data = np.insert(data, 0, 1, axis=1)

            # Test
            for row in data:
                current_data = row

                # If hidden layers, transform data through them and update current_data
                if self.hidden_layers_count > 0:
                    for hl_index in range(self.hidden_layers_count):
                        current_data = self.hidden_layers[index][hl_index].predict(current_data)
                        current_data = np.insert(current_data, 0, 1)

                # Get prediction from output_layer
                predictions.append(self.output_layer[index].predict(current_data))

            # Compare results
            results = pd.DataFrame.copy(self.test_split[index])
            results['Prediction'] = predictions

            # Calculate misclassification
            if self.type == 'c':
                misclassification = len(results[results['Class'] != results['Prediction']]) / len(results)

                # Save results
                self.test_results[index].update({
                    'results': results,
                    'misclassification': misclassification
                })

            # Calculate MSE
            else:
                mse = sum((results.iloc[:, -2] - results.iloc[:, -1]) ** 2) / len(results)

                # Save results
                self.test_results[index].update({
                    'results': results,
                    'mse': mse
                })

    def summarize(self):
        """
        Summarize Function

        This function outputs a CSV and a JSON file for the results of the predict. The CSV is the combined results of
            all 5 CV splits and their prediction. The JSON file is the step_size used and the misclassification.

        :return csv: csv, output of predictions of all CV splits to output folder
        :return json: json, dictionary of the results of all CV splits to output folder
        """
        if self.type == 'c':
            # Calculate misclassification
            misclassification = sum([self.test_results[index]['misclassification'] for index in range(5)])

            # Summary JSON
            self.summary = {
                'tune': {
                    'hidden_layers_count': self.hidden_layers_count,
                    'step_size': self.step_size,
                    'node_count': self.node_count,
                    'convergence_threshold': self.convergence_threshold
                },
                'train': {
                    'epochs': self.epochs
                },
                'test': {
                    'misclassification': misclassification / 5
                }
            }

        else:
            # Calculate mse
            mse = sum([self.test_results[index]['mse'] for index in range(5)])

            # Summary JSON
            self.summary = {
                'tune': {
                    'hidden_layers_count': self.hidden_layers_count,
                    'step_size': self.step_size,
                    'node_count': self.node_count,
                    'convergence_threshold': self.convergence_threshold
                },
                'train': {
                    'epochs': self.epochs
                },
                'test': {
                    'mse': mse / 5
                }
            }

        # Output JSON
        with open(f'output_{self.data_name}\\'
                  f'{self.data_name}_{self.hidden_layers_count}_layers_summary.json', 'w') as file:
            json.dump(self.summary, file)

        # Summary CSV
        summary_prediction = pd.DataFrame()

        # Loop through each test data set and add the results
        for index in range(5):
            summary_prediction = summary_prediction.append(self.test_results[index]['results'])

        # Dump CSV and save
        summary_prediction.to_csv(f'output_{self.data_name}\\'
                                  f'{self.data_name}_{self.hidden_layers_count}_layers_predictions.csv')
        self.summary_prediction = summary_prediction
