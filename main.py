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
        'step_size': arguments.step_size
    }
    model = NeuralNetwork(**kwargs)

    # # Tune
    # if arguments.tune:
    #     model.tune()

    # Fit
    model.fit()

    # # Predict
    # model.predict()
    #
    # # Summarize
    # model.summarize()

    pass


if __name__ == '__main__':
    main()
