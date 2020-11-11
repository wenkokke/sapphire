from functools import partial
from z3 import *

from .activation import *
from .linearalgebra import *

def NN(model):
	# Here 2 lists are defined, one for the inputs, and the other for all 
	# of the hidden layers of the network. This includes the output later
	# which can be taken from H and returned at the end.
    X = None
    H = []
    
    # Next, looping over each layer in the model, get all of the 
    # information that is stored in the Keras model of the 
    # neural network, the weights.
    for layer in model.layers:
        # print(layer)
        params = layer.get_weights()
        # print(params)

        # Ignore layers without weights:
        if len(params) <= 0: continue
		
		# For any other layer get its 
		# configuration, shape, weights, and biases
        config = layer.get_config()
        rows, cols = params[0].shape
        weights = params[0].tolist()
        biases = params[1].tolist()

        # Initialise input vector with a bunch of Real variables for Z3.
        # These will serve as the inputs to the network:
        if X is None:
            X = [ Real('x%s' % i) for i in range(rows) ]
            H.append(X)

        # Compute output from input and weights:
          
        I = H[-1] # I refers to the inputs of each of the hidden layers
        
        # The weight matrix, is translated into Z3 values by applying 
        # the RealVal function to each of the values in the matrix. 
        # And the same is done for the biases.
        W = map(partial(map, RealVal), weights)
        B = map(RealVal, biases)
        
        # Next the output is computed using vector addition and the 
		# vector-matrix product, followed by applying the activation 
		# function on its sum. Once this is completed it is added to 
		# the list of hidden layer outputs.
        O = vecadd(vecmatprod(I, W), B)
        O = activation(config['activation'], O)
        H.append(O)

	# Finally, after looping through all of the layers in the model, 
	# take the last element of the hidden layer list, which is the 
	# output of the whole neural network model and return it and the 
	# input vector list from the function.
    Y = H.pop()
    # These 2 lists, X (the NN inputs) and Y (the NN output), can then 
    # be used with the Z3 Solver() in order to verify the neural network
    return (X, Y) 
