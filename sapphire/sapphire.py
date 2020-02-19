from functools import reduce
from operator  import mul
from z3        import *

def LinExp(x):
    return If(x <= -1, 0.00001, If(x >= 1, 5.898 * x - 3.898, x + 1))

def Norm(X):
    return map(lambda x: x / sum(X), X)

def LinSoftmax(X):
    return Norm(map(LinExp, X))

def LinSigmoid(x):
    return If(x < 0, 0, If(x > 1, 1, 0.25 * x + 0.5))

def ReLU(x):
    return If(x > 0, x, 0)

def Activation(activation, X):
    return list({
        'linear'  : X,
        'relu'    : map(ReLU, X),
        'sigmoid' : map(LinSigmoid, X),
        'softmax' : LinSoftmax(X)
    }[activation])

def Dot(X, W):
    return sum(map(mul, X, W))

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
        O = [ Dot(I, [ RealVal(r) for r in row ]) + bias
              for row, bias in zip(weights, biases) ]
        O = Activation(config['activation'], O)
        H.append(O)

    Y = H.pop()
    return (X, Y)
