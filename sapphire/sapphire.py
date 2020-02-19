from functools import reduce
from itertools import product
from operator import add, mul
from z3 import Real, RealVar, If

def exp(x):
    return If(x <= -1, 0.00001, If(x >= 1, 5.898 * x - 3.898, x + 1))

def norm(X):
    return map(lambda x: x / sum(X), X)

def softmax(X):
    return norm(map(exp, X))

def sigmoid(x):
    return If(x < 0, 0, If(x > 1, 1, 0.25 * x + 0.5))

def relu(x):
    return If(x > 0, x, 0)

def run_activation(activation, X):
    return list({
        'linear'  : X,
        'relu'    : map(relu, X),
        'sigmoid' : map(sigmoid, X),
        'softmax' : softmax(X)
    }[activation])

def dot(X, W):
    return sum(map(mul, X, W))

def NN(model):
    X = None
    H = []
    for layer in model.layers:
        params = layer.get_weights()

        # Ignore layers without weights:
        if len(params) <= 0: break

        config = layer.get_config()
        activation = config['activation']
        rows, cols = params[0].shape
        weights = params[0].tolist()
        biases = params[1].tolist()

        # Initialise input vector:
        if X is None:
            X = [ Real('x%s' % i) for i in range(rows) ]
            H.append(X)

        # Compute output from input and weights:
        I = H[-1]
        O = [ dot(I, [ RealVal(r) for r in row ]) + bias
              for row, bias in zip(weights, biases) ]
        O = run_activation(activation, O)
        H.append(O)

    Y = H.pop()
    return (X, Y)
