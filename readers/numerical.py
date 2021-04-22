import numpy as np


def rref(A, tol=1.0e-12):
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
    m, n = A.shape
    i, j = 0, 0
    jb = []

    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
        if p <= tol:
            # The column is negligible, zero it out
            A[i:m, j] = 0.0
            j += 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
            # Divide the pivot row i by the pivot element A[i, j]
            A[i, j:n] = A[i, j:n] / A[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for k in range(m):
                if k != i:
                    A[k, j:n] -= A[k, j] * A[i, j:n]
            i += 1
            j += 1
    # Finished
    return A, jb
