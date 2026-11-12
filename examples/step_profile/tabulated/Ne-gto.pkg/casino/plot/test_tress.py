#!/usr/bin/env python3

import numpy as np
import numba as nb
import pytest
import time

c = 1

def tress_numpy(tr_grad, partial_hess):
    ne3 = tr_grad.shape[0]
    tress = np.zeros((ne3, ne3, ne3), dtype=np.float64)
    tress += c * (
        tr_grad * np.expand_dims(partial_hess, 2)
        + np.expand_dims(tr_grad, 1) * np.expand_dims(partial_hess, 1)
        + np.expand_dims(np.expand_dims(tr_grad, 1), 2) * partial_hess
    )
    return tress


@nb.njit(fastmath=True)
def tress_numba_loops(tr_grad, partial_hess):
    ne3 = tr_grad.shape[0]
    tress = np.zeros((ne3, ne3, ne3), dtype=np.float64)
    for e1 in range(ne3):
        for e2 in range(ne3):
            for e3 in range(ne3):
                tress[e1, e2, e3] += c * (
                    tr_grad[e3] * partial_hess[e1, e2]
                    + tr_grad[e2] * partial_hess[e1, e3]
                    + tr_grad[e1] * partial_hess[e2, e3]
                )
    return tress


if __name__ == '__main__':
    np.random.seed(42)
    ne = 36  # попробуйте увеличить, например до 20–40
    tr_grad = np.random.randn(ne * 3)
    partial_hess = np.random.randn(ne * 3, ne * 3)

    assert tress_numpy(tr_grad, partial_hess) == pytest.approx(tress_numba_loops(tr_grad, partial_hess))
