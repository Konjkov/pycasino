import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def rref(a, tol=1e-12):
    """
    Construct RREF matrix which:
    1. All the rows consisting entirely of zeros are at the bottom
    2. In each non-zero row, the leftmost non-zero entry is a 1. These are called the leading ones.
    3. Each leading one is further to the right than the leading ones of previous rows.
    3. The column of each leading one is “clean”, that is all other entries in the column are 0.

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
