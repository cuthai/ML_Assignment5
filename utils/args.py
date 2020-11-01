import argparse


def args():
    """
    Function to create command line arguments

    Arguments:
        -dn <str> (data_name) name of the data to import form the data folder
            they are: breast-cancer, car, segmentation, abalone, machine, forest-fires
        -rs <int> (random_seed) seed used for data split. Defaults to 1. All submitted output uses random_seed 1
        -t <str> (tune) Trigger tune. Pass a character to denote the param being tuned
            they are: step_size (s), node_count (n), convergence_threshold(c)
        -hl <int> (hidden_layers_count) Number of hidden layers, 0 is a simple perceptron
        -s <float> (step_size) Step size to control gradient descent
        -n <int> (node_count) number of nodes per hidden layer. If multiple layers all layers have the same count
        -c <float> (convergence_threshold) Float to control convergence, refers to the error function
    """
    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-dn', '--data_name', help='Specify data name to extract and process')
    parser.add_argument('-rs', '--random_state', default=1, type=int,
                        help='Specify a seed to pass to the data splitter')
    parser.add_argument('-t', '--tune', type=str,
                        help='Define a parameter to tune,'
                             'pass a single character in s, n, or c to tune the corresponding arg')
    parser.add_argument('-hl', '--hidden_layers_count', default=0, type=int, help='Number of hidden layers to create')
    parser.add_argument('-s', '--step_size', default=.01, type=float,
                        help='Step_size to pass to logistic model gradient descent')
    parser.add_argument('-n', '--node_count', default=2, type=int, help='Number of nodes in each hidden layer')
    parser.add_argument('-c', '--convergence_threshold', default=.01, type=float,
                        help='Convergence Threshold of error function')

    # Parse arguments
    command_args = parser.parse_args()

    # Return the parsed arguments
    return command_args
