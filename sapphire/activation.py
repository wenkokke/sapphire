from z3 import *

def linexp(x):
    return If(x <= -1, 0.00001, If(x >= 1, 5.898 * x - 3.898, x + 1))

def norm(X):
    return map(lambda x: x / sum(X), X)

def linsoftmax(X):
    return norm(map(linexp, X))

def linsigmoid(x):
    return If(x < 0, 0, If(x > 1, 1, 0.25 * x + 0.5))

def relu(x):
    return If(x > 0, x, 0)

def activation(a, X):
    return list({
        'linear'  : X,
        'relu'    : map(relu, X),
        'sigmoid' : map(linsigmoid, X),
        'softmax' : linsoftmax(X)
    }[a])
