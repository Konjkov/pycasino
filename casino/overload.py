import numpy as np
import numba as nb


@nb.njit(nogil=True, parallel=False, cache=True)
def random_step(dr, ne):
    """Random N-dim square distributed step"""
    return dr * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))


@nb.njit(nogil=True, parallel=False, cache=True)
def block_diag(res_list):
    """Near equivalent of scipy.linalg.block_diag"""
    if res_list:
        shape_0_list = np.cumsum(np.array([r.shape[0] for r in res_list]))
        shape_1_list = np.cumsum(np.array([r.shape[1] for r in res_list]))
        res = np.zeros(shape=(shape_0_list[-1], shape_1_list[-1]))
        for r_part, p0, p1 in zip(res_list, shape_0_list, shape_1_list):
            res[p0 - r_part.shape[0]:p0, p1 - r_part.shape[1]:p1] = r_part
        return res
    else:
        return np.zeros(shape=(0, 0))


@nb.njit(nogil=True, parallel=False, cache=True)
def rref(a: np.ndarray, tol=1e-12):
    """
    Construct RREF matrix which:
    1. All the rows consisting entirely of zeros are at the bottom
    2. In each non-zero row, the leftmost non-zero entry is a 1. These are called the leading ones.
    3. Each leading one is further to the right than the leading ones of previous rows.
    4. The column of each leading one is “clean”, that is all other entries in the column are 0.

    https://stackoverflow.com/questions/7664246/python-built-in-function-to-do-matrix-reduction

    definitions
    https://www.statlect.com/matrix-algebra/row-echelon-form
    """
    m, n = a.shape
    i = j = 0
    pivot_positions = []

    while i < m and j < n:
        # Find value and index of the largest element in the remainder of column j
        k = np.argmax(np.abs(a[i:m, j])) + i
        if np.abs(a[k, j]) <= tol:
            # The column is negligible, zero it out
            a[i:m, j] = 0.0
            j += 1
        else:
            # Remember the column index
            pivot_positions.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                a[np.array([i, k]), j:n] = a[np.array([k, i]), j:n]
            # Divide the pivot row i by the pivot element A[i, j]
            a[i, j:n] = a[i, j:n] / a[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for k in range(m):
                if k != i:
                    a[k, j:n] -= a[k, j] * a[i, j:n]
            i += 1
            j += 1
    # Finished
    return a, np.array(pivot_positions)
