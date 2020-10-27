Usage
	python main.py -dn <str> [-rs <int>] [-s <float>] [-t] [-a]

Args:
	-dn <str>
	Required, specifies the name of the data. Please use:
		breast-cancer
		glass
		iris
        soybean
		vote

	-rs <int>
	Optional, specifies the random_state of the data for splitting. Defaults to 1

	-s
	Optional, specifies step_size to use in gradient descent for both regressions

    -t
	Optional, specifies tuning, does not set the step_size but outputs a graph, use -s to specify step_size

	-a
	Optional, specifies adaline regression. default: logistic
