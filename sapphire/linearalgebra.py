from functools import reduce
from itertools import repeat
from operator  import add, mul

def dot(X, Y):
    return sum(map(mul, X, Y))

def scale(x, Y):
    return list(map(mul, repeat(x), Y))

def vecadd(X, Y):
    return list(map(add, X, Y))

def vecmatprod(X, M):
    return reduce(vecadd, map(scale, X, M))
