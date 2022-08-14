import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True, cache=True)
def random_step(dr, ne):
    """Random N-dim square distributed step"""
    return dr * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))


@nb.jit(nopython=True, nogil=True, parallel=False)
def subtract_outer(x, y):
    """Outer subtract two 1-D array."""

    res = np.empty((x.shape[0], y.shape[0], 3))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            res[i, j] = x[i] - y[j]
    return res
