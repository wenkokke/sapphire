import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from z3 import *
from sapphire import *


# Load the spirals data set from the 'data' directory.
data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
x_train = np.load(file=os.path.join(data, 'spirals_x_9.0.npy'))
y_train = np.load(file=os.path.join(data, 'spirals_y_9.0.npy')).astype(int)

# Train network
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50)

# Sample input/output
def sample_spiral(label):
    """Select a random sample with a particular label."""
    label_mask = (y_train == label).flatten()
    label_count = np.count_nonzero(label_mask)
    return x_train[label_mask][np.random.randint(0, label_count, 1)][0]

X_sample = sample_spiral(0)
Y_sample = model.predict(np.array([X_sample]))[0]

# Verify network
X, Y = NN(model)

X_sample = list(RealVal(x) for x in X_sample)
Y_sample = list(RealVal(y) for y in Y_sample)

Epsilon = RealVal(100.0)

def Sq(x):
    return x * x

def SqEuclidianDist(X1, X2):
    return sum(Sq(x1 - x2) for x1, x2 in zip(X1, X2))

def Eq(X1, X2):
    return And([x1 == x2 for x1, x2 in zip(X1, X2)])

s = SolverFor('NRA')
s.add(Implies(Eq(X, X_sample), SqEuclidianDist(Y, Y_sample) < Epsilon))
print(s.check())
