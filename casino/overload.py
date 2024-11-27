import math

import numba as nb
import numpy as np

# import scipy as sp
from numpy.polynomial.polynomial import polyval


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
            res[p0 - r_part.shape[0] : p0, p1 - r_part.shape[1] : p1] = r_part
        return res
    else:
        return np.zeros(shape=(0, 0))


@nb.njit(nogil=True, parallel=False, cache=True)
def fact2(n):
    """n!! = 1 * 3 * 5 ...n."""
    res = 1
    for i in range(n, 0, -2):
        res *= i
    return res


@nb.njit(nogil=True, parallel=False, cache=True)
def comb(n, k) -> float:
    """(n, k)"""
    # return math.comb(n, k)
    return math.gamma(n + 1) / math.gamma(k + 1) / math.gamma(n - k + 1)


@nb.njit(nogil=True, parallel=False, cache=True)
def boys(n, x):
    """Boys function.

    F_n(x) = \int_0^1 t^{2n}e^{-xt^2} dt
    https://github.com/berquist/boys?tab=readme-ov-file
    """
    if x == 0:
        return 1 / (2 * n + 1)
    elif n == 0:
        return math.sqrt(math.pi / 4 / x) * math.erf(math.sqrt(x))
    # else:
    #     return math.gamma(n + 1/2) * sp.special.gammaincc(n + 1/2, x) / (2 * x ** (n + 1/2))


@nb.njit(nogil=True, parallel=False, cache=True)
def polyval2d(x, y, c):
    """Evaluate a 2-D polynomial on the Cartesian product of x and y."""
    x, y = [np.asarray(a) for a in (x, y)]
    if x.shape != y.shape:
        raise ValueError('x, y are incompatible')
    return polyval(y, polyval(x, c, tensor=True), tensor=False)


@nb.njit(nogil=True, parallel=False, cache=True)
def polyval3d(x, y, z, c):
    """Evaluate a 3-D polynomial at points (x, y, z)."""
    x, y, z = [np.asarray(a) for a in (x, y, z)]
    if x.shape != y.shape != z.shape:
        raise ValueError('x, y, z are incompatible')
    return polyval(z, polyval(y, polyval(x, c, tensor=True), tensor=False), tensor=False)


@nb.njit(nogil=True, parallel=False, cache=True)
def repeat(a, repeats):
    """Repeat each element of an array after themselves along 0-axis."""
    res = np.empty(shape=repeats.shape + a.shape[1:], dtype=a.dtype)
    pos = 0
    for i in range(repeats.size):
        for _ in repeats[i]:
            res[pos] = a[i]
            pos += 1
    return res


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
