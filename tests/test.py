import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from z3 import *
from sapphire import *


# Create data
epsilon = 0.25

def is_truthy(x): return 1-epsilon < x and x < 1+epsilon
def is_falsy(x): return 0-epsilon < x and x < 0+epsilon
def andy(x1, x2): return 1.0 if is_truthy(x1) and is_truthy(x2) else 0.0

truthy  = np.linspace(1-epsilon, 1+epsilon, num=100)
falsy   = np.linspace(0-epsilon, 0+epsilon, num=100)
domain  = np.append(truthy, falsy)
x       = np.array([ (x1, x2) for x1 in domain for x2 in domain ])
y       = np.array([ andy(x1, x2) for x1, x2 in x ])


# Train network
model = keras.Sequential([
    keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=10)
model.summary()


# Test network
Epsilon = RealVal(0.1)

def Truthy(x): return And([1 - Epsilon < x, x < 1 + Epsilon])
def Falsy(x): return And([0 - Epsilon < x, x < 0 + Epsilon])
def Andy(x1, x2): return If(And([Truthy(x1), Truthy(x2)]), 1, 0)

X, Y = NN(model)

s = Solver()
s.add(ForAll(X, Implies(And([Truthy(X[0]), Truthy(X[1])]), Y[0] == 1)))
s.add(ForAll(X, Implies(And([Falsy(X[0]),  Truthy(X[1])]), Y[0] == 0)))
s.add(ForAll(X, Implies(And([Truthy(X[0]), Falsy(X[1])]),  Y[0] == 0)))
s.add(ForAll(X, Implies(And([Falsy(X[0]),  Falsy(X[1])]),  Y[0] == 0)))

print(s.check())
print(s.sexpr())
