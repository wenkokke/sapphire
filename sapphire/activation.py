from functools import partial, reduce
import numpy as np
from z3 import *

def activation(a, X):
    return list({
        'linear'  : X,
        'relu'    : map(relu, X),
        'sigmoid' : map(linsigmoid, X),
        'softmax' : linsoftmax(X)
    }[a])

def linexp(x):
    """Linear approximation of the exponential function."""
    return If(x <= -1, 0.00001, If(x >= 1, 5.898 * x - 3.898, x + 1))

def norm(X):
    """Normalisation."""
    return map(lambda x: x / sum(X), X)

def linsoftmax(X):
    """Linear approximation of the Softmax function."""
    return norm(map(linexp, X))

def linsigmoid(x):
    """Linear approximation of the Sigmoid function."""
    return If(x < 0, 0, If(x > 1, 1, 0.25 * x + 0.5))

def relu(x):
    """Rectified linear unit."""
    return If(x > 0, x, 0)

def lin(f, x_min, x_max, num=3):
    """Approximates the function 'f' between 'x_min' and 'x_max' using 'num' line segments."""
    X = np.linspace(start=x_min, stop=x_max, num=num+1)
    Y = tuple(map(f, X))
    M = tuple(map(_slope, X, Y, X[1:], Y[1:]))
    N = tuple(map(_intercept, X[1:], Y[1:], M))
    *M, last_m = M
    *N, last_n = N
    return lambda x:\
        reduce(_revapp,
               reversed(tuple(map(partial(_seg,x), X[1:], M, N))),
               x * last_m + last_n)

def _seg(x, x_max, m, n):
    """Computes a line segment up to 'x_max', abstracted over the remainder 'rest'."""
    return lambda rest: If(x <= x_max, x * m + n, rest)

def _revapp(x, f):
    """Reversed function application."""
    return f(x)

def _slope(x1, y1, x2, y2):
    """Computes the slope for a linear function connecting 'p1' and 'p2'."""
    return (y2 - y1) / (x2 - x1)

def _intercept(x, y, m):
    """Computes the intercept for a linear function with slope 'm' given a point 'p' on the function."""
    return y - m * x
