from functools import partial
from z3 import *

from .activation import *
from .linearalgebra import *

def NN(model):
    X = None
    H = []
    for layer in model.layers:
        params = layer.get_weights()

        # Ignore layers without weights:
        if len(params) <= 0: break

        config = layer.get_config()
        rows, cols = params[0].shape
        weights = params[0].tolist()
        biases = params[1].tolist()

        # Initialise input vector:
        if X is None:
            X = [ Real('x%s' % i) for i in range(rows) ]
            H.append(X)

        # Compute output from input and weights:
        I = H[-1]
        W = map(partial(map, RealVal), weights)
        B = map(RealVal, biases)
        O = vecadd(vecmatprod(I, W), B)
        O = activation(config['activation'], O)
        H.append(O)

    Y = H.pop()
    return (X, Y)
