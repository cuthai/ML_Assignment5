import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from neural_network.output_layer import OutputLayer
from neural_network.hidden_layer import HiddenLayer


class NeuralNetwork:
    def __init__(self, etl, hidden_layers_count=0, step_size=.01, node_count=1, convergence_threshold=.01):
        """
        Init function

        Sets main variables and array conversion

        :param etl: etl, etl object with transformed and split data
        :param step_size: float, desired step_size to use during training
        """
        # Meta Variables
        self.etl = etl
        self.data_name = self.etl.data_name
        self.dimensions = len(etl.transformed_data.columns) - 1
        self.classes = etl.classes
        self.class_names = etl.class_names

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
        if self.hidden_layers_count == 0:
            input_count = self.dimensions
            self.hidden_layers = None
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
        self.output_layer = {
            index: OutputLayer(**ol_kwargs)
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

        # Test Results
        self.test_results = {index: {} for index in range(5)}

        # Summary
        self.summary = {}
        self.summary_classification = None

    def tune(self, param):
        """
        Tune function.

        This function runs through a couple of step_sizes. The model is then trained for that step_size and then tested
            against the tune data set. The resulting misclassification is stored and then visualized as a chart. The
            step_size for the entire model is not set here. That should be done by adding a step_size argument
        """
        if param == 's':
            param_name = 'Step_Size'
        elif param == 'n':
            param_name = 'Node_Count'
        else:
            param_name = 'Convergence_Threshold'

        param_range = self.tune_results[param_name].keys()

        original = self.tune_data['Class'].to_list()

        # Loop through step_sizes to test
        for param_value in param_range:
            if param == 's':
                self.step_size = param_value
            elif param == 'n':
                self.node_count = param_value
            else:
                self.convergence_threshold = param_value

            misclassification = 0

            self.fit()

            # Loop through the 5 CV splits
            for index in range(5):
                predictions = []

                # Assign data
                data = self.tune_array
                data = np.insert(data, 0, 1, axis=1)

                # Train
                for row in data:
                    current_data = row

                    if self.hidden_layers_count > 0:
                        for hl_index in range(self.hidden_layers_count):
                            current_data = self.hidden_layers[index][hl_index].predict(current_data)
                            current_data = np.insert(current_data, 0, 1)

                    predictions.append(self.output_layer[index].predict(current_data))

                # Calculate misclassification
                misclassification += sum(np.array(original) != np.array(predictions)) / len(original)

            # Save results
            self.tune_results[param_name].update({param_value: misclassification / 5})

            self.reset()

        # Trigger visualization
        self.visualize(self.tune_results[param_name], param_name)

    def reset(self):
        if self.hidden_layers_count == 0:
            input_count = self.dimensions
            self.hidden_layers = None
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
        self.output_layer = {
            index: OutputLayer(**ol_kwargs)
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
        ax.set_ylabel('Misclassification')

        # Saving
        plt.savefig(f'output_{self.data_name}\\{self.data_name}_{self.hidden_layers_count}_layers_tune_{name}.jpg')

    def fit(self):
        """
        Fit Function

        This function loops through the 5 CV splits and then calls to the train function. It receives the results of the
            train function and sets the model for that split
        """
        # Loop through the 5 CV splits
        for index in range(5):
            convergence = False
            i = 0

            while not convergence:
                # Assign data
                data = self.train_array_split[index]
                data = np.insert(data, 0, 1, axis=1)
                np.random.shuffle(data)
                self.epochs[index] += 1

                # Train
                for row in data:
                    current_data = row

                    if self.hidden_layers_count > 0:
                        for hl_index in range(self.hidden_layers_count):
                            current_data = self.hidden_layers[index][hl_index].predict(current_data)
                            current_data = np.insert(current_data, 0, 1)

                    convergence, backpropagation = self.output_layer[index].fit(current_data)

                    if convergence:
                        break

                    if self.hidden_layers_count > 0:
                        for hl_index in reversed(range(self.hidden_layers_count)):
                            backpropagation = self.hidden_layers[index][hl_index].update(backpropagation)

                    print(i)
                    i += 1

                if self.epochs[index] >= 1000:
                    convergence = True

    def predict(self):
        """
        """
        # Loop through the 5 CV splits
        for index in range(5):
            predictions = []

            # Assign data
            data = self.test_array_split[index]
            data = np.insert(data, 0, 1, axis=1)

            # Train
            for row in data:
                current_data = row

                if self.hidden_layers_count > 0:
                    for hl_index in range(self.hidden_layers_count):
                        current_data = self.hidden_layers[index][hl_index].predict(current_data)
                        current_data = np.insert(current_data, 0, 1)

                predictions.append(self.output_layer[index].predict(current_data))

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

        # Output JSON
        with open(f'output_{self.data_name}\\'
                  f'{self.data_name}_{self.hidden_layers_count}_layers_summary.json', 'w') as file:
            json.dump(self.summary, file)

        # Summary CSV
        summary_classification = pd.DataFrame()

        # Loop through each test data set and add the results
        for index in range(5):
            summary_classification = summary_classification.append(self.test_results[index]['results'])

        # Dump CSV and save
        summary_classification.to_csv(f'output_{self.data_name}\\'
                                      f'{self.data_name}_{self.hidden_layers_count}_layers_predictions.csv')
        self.summary_classification = summary_classification
