Usage
	python main.py -dn <str> [-rs <int>] [-s <float>] [-t] [-a]

Args:
	-dn <str>
	Required, specifies the name of the data. Please use:
		breast-cancer
		glass
        soybean
		abalone
		forest-ires
		machine

	-rs <int>
	Optional, specifies the random_state of the data for splitting. Defaults to 1

	-t <str>
	Optional, triggers tune on the specified param. The params are:
	    t
	    n
	    c

    -hl <int>
	Optional, number of hidden layers to create

	-s <float>
	Optional, specified step_size to use for gradient descent

	-n <int>
	Optional, specifies nodes to use in the hidden layer

	-c <float>
	Optional, specified convergence threshold to use
