
"""
invert_network.py
"""

from functools import partial
from itertools import product
import numpy as np
import operator
from z3 import *

pre_dims = 6
dims = 5

# x = np.random.uniform(low=-1, high=1, size=(dims, 1))
# x_prime = map(partial(map, RealVal), x)
# list(list(x_prime)[0])

# weights = np.random.normal(loc=0, scale=1.0, size=(pre_dims, dims))
# print(weights.shape)
# weights_prime = map(partial(map, RealVal), weights)
# print(list(list(weights_prime)[0]))
# print(list(list(weights_prime)))

# weights = [[1, 2], [3, 4], [5, 6]]
weights = [
    [1, 2], 
    [3, 4], 
    [5, 6]]
print(weights)
print(np.array(weights))
print(np.array(weights).T)

len(weights)

[[item[i] for item in weights] for i in range(len(weights[0]))]

print(next(iter(weights)))
print(next(iter(weights)))
print(next(iter(weights)))

def mattranspose(X):
    return [[item[i] for item in weights] for i in range(len(X[0]))]

print(mattranspose(weights))

# x = np.random.uniform(low=-1, high=1, size=(dims, 1))
# weights = np.random.normal(loc=0, scale=1.0, size=(pre_dims, dims))

# x_prime = np.matmul(weights, x)

# print(x)
# print(x_prime)

# weights_inv = np.matmul(np.linalg.inv(np.matmul(weights.T, weights)), weights.T) 
# # weights_inv = np.linalg.inv(weights)
# x_inv = np.matmul(weights_inv, x_prime)
# print(x_inv)


