import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def random_step(dr, ne):
    """Random N-dim square distributed step"""
    return dr * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))


@nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def block_diag(res_list):
    shape_0_list = np.cumsum(np.array([r.shape[0] for r in res_list]))
    shape_1_list = np.cumsum(np.array([r.shape[1] for r in res_list]))
    res = np.zeros(shape=(shape_0_list[-1], shape_1_list[-1]))
    for r_part, p0, p1 in zip(res_list, shape_0_list, shape_1_list):
        res[p0 - r_part.shape[0]:p0, p1 - r_part.shape[1]:p1] = r_part
    return res
