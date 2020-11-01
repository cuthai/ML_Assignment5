from utils.args import args
from etl.etl import ETL
from neural_network.neural_network import NeuralNetwork


def main():
    """
    Main function to run Logistic Regression/Adaline Regression
    """
    # Parse arguments
    arguments = args()

    # Set up kwargs for ETL
    kwargs = {
        'data_name': arguments.data_name,
        'random_state': arguments.random_state
    }
    etl = ETL(**kwargs)

    # Set up kwargs and create object
    kwargs = {
        'etl': etl,
        'hidden_layers_count': arguments.hidden_layers_count,
        'step_size': arguments.step_size,
        'node_count': arguments.node_count,
        'convergence_threshold': arguments.convergence_threshold,
        'random_state': arguments.random_state
    }
    model = NeuralNetwork(**kwargs)

    # Tune
    if arguments.tune:
        if arguments.tune not in ('s', 'n', 'c'):
            raise ValueError('Please pass s, n, or c to tune the corresponding parameter')
        model.tune(arguments.tune)

    else:
        # Fit
        model.fit()

        # Predict
        model.predict()

        # Summarize
        model.summarize()

    pass


if __name__ == '__main__':
    main()
