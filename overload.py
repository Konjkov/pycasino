#!/usr/bin/env python3

import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True, parallel=False)
def subtract_outer(x, y):
    """Outer subtract two 1-D array."""

    res = np.empty((x.shape[0], y.shape[0], 3))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            res[i, j] = x[i] - y[j]
    return res


@nb.jit(nopython=True, nogil=True, parallel=False)
def polyval(x, c):
    y = 0.0
    for i in range(c.shape[0], 0, -1):
        y = x * y + c[i - 1]
    return y
