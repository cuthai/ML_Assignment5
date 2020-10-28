import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from neural_network.output_layer import OutputLayer


class NeuralNetwork:
    def __init__(self, etl, step_size=.01, hidden_layers=0):
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
        self.hidden_layers = hidden_layers

        # Data Variables
        self.tune_data = etl.tune_data
        self.test_split = etl.test_split
        self.train_split = etl.train_split

        # Array Variables
        self.tune_array = self.tune_data.to_numpy()
        self.test_array_split = {key: self.test_split[key].to_numpy() for key in self.test_split.keys()}
        self.train_array_split = {key: self.train_split[key].to_numpy() for key in self.train_split.keys()}

        # Train Models
        self.hidden_layers = {index: {} for index in range(5)}
        self.output_layer = {
            index: OutputLayer(self.dimensions, self.classes, self.class_names, self.step_size) for index in range(5)
        }

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

                # Train
                for row in data:
                    convergence = self.output_layer[index].fit(row)

                    if convergence:
                        break

                    print(i)
                    i += 1
