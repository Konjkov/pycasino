import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def random_step(dr, ne):
    """Random N-dim square distributed step"""
    return dr * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))


@nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def subtract_outer(x, y):
    """Outer subtract two 2-D array."""
    res = np.empty(shape=(x.shape[0], y.shape[0], 3))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            res[i, j] = x[i] - y[j]
    return res


# @nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
# def subtract_outer(x, y):
#     """Outer subtract two 2-D array.
#     [vladimir-Kubuntu:25560] *** Process received signal ***
#     [vladimir-Kubuntu:25560] Signal: Segmentation fault (11)
#     [vladimir-Kubuntu:25560] Signal code:  (128)
#     [vladimir-Kubuntu:25560] Failing at address: (nil)
#     """
#     return np.expand_dims(x, 0) - np.expand_dims(y, 1)
