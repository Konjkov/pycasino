
import numpy as np
import numba as nb


@nb.jit(nopython=True, cache=True)
def einsum(subscripts, x, y):
    result = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(y.shape[1]):
                result[i, k] += x[i, j] * y[j, k]
    return result


@nb.jit(nopython=True, cache=True)
def factorial(n):
    result = 1.0
    for i in range(n, 0, -1):
        result *= i
    return result


@nb.jit(nopython=True, cache=True)
def uniform(low, high, size):
    return np.dstack((np.random.uniform(low[0], high[0], size=size[0]), np.random.uniform(low[1], high[1], size=size[0]), np.random.uniform(low[2], high[2], size=size[0])))[0]
