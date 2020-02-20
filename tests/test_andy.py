import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from z3 import *
from sapphire import *

def andy(X, epsilon):
    """Boolean AND gate lifted to floating point numbers."""
    return float(all(1-epsilon <= x and x <= 1+epsilon for x in X))

def mk_data(epsilon):
    """Create data for training an AND gate."""
    T = np.linspace(1-epsilon, 1+epsilon, num=100)
    F = np.linspace(0-epsilon, 0+epsilon, num=100)
    D = np.append(T, F)
    X = np.array([ (x1, x2) for x1 in D for x2 in D ])
    Y = np.array(list(map(partial(andy, epsilon=epsilon), X)))
    return (X, Y)

def plot_db(model, epsilon):
    """Plot the decision boundary for an AND gate."""
    x_train, y_train = mk_data(epsilon=epsilon)
    x1_span = np.linspace(-1-epsilon, 2+epsilon, num=100)
    x2_span = np.linspace(-1-epsilon, 2+epsilon, num=100)
    x1_span, x2_span = np.meshgrid(x1_span, x2_span)
    labels = model.predict(np.c_[x1_span.ravel(), x2_span.ravel()])
    y_span = labels[:,0].reshape(x1_span.shape)
    y_span = y_span > 0.5

    plt.figure()
    plt.contour(x1_span, x2_span, y_span, cmap='RdBu', alpha=0.5)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.ravel(), cmap='RdBu', alpha=0.5)
    plt.show()


def test_andy():

    # Train network
    x_train, y_train = mk_data(epsilon=0.25)
    model = keras.Sequential([
        keras.layers.Dense(1, activation='sigmoid', input_shape=(2,)),
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20)

    # Verify network
    Epsilon = RealVal(0.2)

    def Truthy(x): return And([1 - Epsilon <= x, x <= 1 + Epsilon])
    def Falsey(x): return And([0 - Epsilon <= x, x <= 0 + Epsilon])
    def Andy(x1, x2): return If(And([Truthy(x1), Truthy(x2)]), 1, 0)

    X, Y = NN(model)

    s = Solver()
    s.add(ForAll(X, Implies(And([Truthy(X[0]), Truthy(X[1])]), Y[0] > 0.5)))
    s.add(ForAll(X, Implies(And([Falsey(X[0]), Truthy(X[1])]), Y[0] < 0.5)))
    s.add(ForAll(X, Implies(And([Truthy(X[0]), Falsey(X[1])]), Y[0] < 0.5)))
    s.add(ForAll(X, Implies(And([Falsey(X[0]), Falsey(X[1])]), Y[0] < 0.5)))

    assert s.check() == sat

if __name__ == '__main__':
    test_andy()
