import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from neural_network.output_layer import OutputLayer
from neural_network.hidden_layer import HiddenLayer


class NeuralNetwork:
    def __init__(self, etl, step_size=.01, hidden_layers_count=0, node_count=2):
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
        self.dimensions = len(etl.transformed_data.columns) - 1

        self.type = 'c'
        if self.classes == 0:
            self.type = 'r'

        # Tune Variables
        self.step_size = step_size
        self.hidden_layers_count = hidden_layers_count
        self.node_count = node_count
        self.convergence_threshold = .01

        # Data Variables
        self.tune_data = etl.tune_data
        self.test_split = etl.test_split
        self.train_split = etl.train_split

        # Array Variables
        self.tune_array = self.tune_data.to_numpy()
        self.test_array_split = {key: self.test_split[key].to_numpy() for key in self.test_split.keys()}
        self.train_array_split = {key: self.train_split[key].to_numpy() for key in self.train_split.keys()}

        # Train Models
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

        ol_kwargs = {
            'dimensions': self.dimensions,
            'classes': self.classes,
            'class_names': self.class_names,
            'step_size': self.step_size,
            'input_count': input_count,
            'convergence_threshold': self.convergence_threshold
        }
        self.output_layer = {
            index: OutputLayer(**ol_kwargs)
            for index in range(5)
        }

        # Train Results
        self.epochs = {index: 0 for index in range(5)}

        # Tune Results
        self.tune_results = {
            round(step_size, 2): None for step_size in np.linspace(.01, .25, 25)
        }

        # Test Results
        self.test_results = {index: {} for index in range(5)}

        # Summary
        self.summary = {}
        self.summary_classification = None

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
                'step_size': self.step_size,
                'hidden_layers_count': self.hidden_layers_count,
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
