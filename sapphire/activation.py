import operator
import functools
import numpy as np
import z3

def activation(a, X):
    return list({
        'linear'  : X,
        'relu'    : map(relu, X),
        'sigmoid' : map(linsigmoid, X),
        'softmax' : linsoftmax(X)
    }[a])

def linexp(x):
    """Linear approximation of the exponential function."""
    return z3.If(x <= -1, 0.00001, z3.If(x >= 1, 5.898 * x - 3.898, x + 1))

def norm(X):
    """Normalisation."""
    return map(lambda x: x / sum(X), X)

def linsoftmax(X):
    """Linear approximation of the Softmax function."""
    return norm(map(linexp, X))

def linsigmoid(x):
    """Linear approximation of the Sigmoid function."""
    return z3.If(x < 0, 0, z3.If(x > 1, 1, 0.25 * x + 0.5))

def relu(x):
    """Rectified linear unit."""
    return z3.z3.If(x > 0, x, 0)

def lin(f, x_min, x_max, num=3):
    """Approximates the function 'f' between 'x_min' and 'x_max' using 'num' line segments."""
    X = np.linspace(start=x_min, stop=x_max, num=num+1)
    Y = [f(x) for x in X]
    M = [_slope(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(X, Y, X[1:], Y[1:])]
    N = [_intercept(x, y, m) for x, y, m in zip(X[1:], Y[1:], M)]
    *M, last_m = M
    *N, last_n = N
    return lambda x:\
        functools.reduce(_revapp,
                         [lambda rest: z3.If(x <= x_max, x * m + n, rest)
                          for x_max, m, n
                          in reversed(list(zip(X[1:], M, N)))],
                         x * last_m + last_n)

def _revapp(x, f):
    """Reversed function application."""
    return f(x)

def _slope(x1, y1, x2, y2):
    """Computes the slope for a linear function connecting 'p1' and 'p2'."""
    return (y2 - y1) / (x2 - x1)

def _intercept(x, y, m):
    """Computes the intercept for a linear function with slope 'm' given a point 'p' on the function."""
    return y - m * x

def eval_lin(f,x):
    """Evaluates a linear approximation, created by 'lin', at a specific input."""
    return _rational(str(z3.simplify(f(z3.RealVal(x)))))

def _rational(s):
    """Parses a rational number, e.g., '12/3'."""
    return operator.truediv(*map(float,s.split('/')))
