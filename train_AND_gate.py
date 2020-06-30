#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import tensorflow as tf
from z3 import *
from sapphire import *
from itertools import repeat

def and_gate_correctness_constraint(s, X, Y, inputs):
    s.add(Implies(
        And(list(map(lambda x,b: x == float(b), X, inputs))),
        Y[0] == float(all(inputs))))

def and_gate(n, total):

    # Compute weights and biases
    weight  = total / float(n)
    weights = np.full((n,1), weight)
    bias    = -(total - (0.5 * weight))
    biases  = np.full((1,), bias)

    # Initialise Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(n,))
    ])
    model.layers[0].set_weights([weights, biases])

    # Export Keras model to Z3
    X, Y = NN(model)

    s = SolverFor('LRA')
    and_gate_correctness_constraint(s, X, Y, list(repeat(False, n))),
    and_gate_correctness_constraint(s, X, Y, [False] + list(repeat(True, n - 1))),
    and_gate_correctness_constraint(s, X, Y, list(repeat(False, n - 1)) + [True]),
    and_gate_correctness_constraint(s, X, Y, list(repeat(True, n)))

    return s.sexpr()

if __name__ == "__main__":
    filename_tpl = 'benchmarks/AND_Gate_{n}_Sigmoid_1.z3'
    for n in range(100,1000,10):
        with open(filename_tpl.format(n=n), 'w') as fp:
            fp.write(and_gate(n, total=1E9))
            fp.write('(check-sat)\n')
