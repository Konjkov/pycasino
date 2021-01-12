import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import numba as nb

# np.show_config()


@nb.jit(nopython=True, nogil=True, parallel=False)
def nuclear_repulsion(atom_positions, atom_charges) -> float:
    """nuclear-nuclear repulsion"""
    res = 0.0
    for i in range(atom_positions.shape[0] - 1):
        for j in range(i + 1, atom_positions.shape[0]):
            res += atom_charges[i] * atom_charges[j]/np.linalg.norm(atom_positions[i] - atom_positions[j])
    return res


@nb.jit(nopython=True)
def coulomb(e_vectors, n_vectors, atom_charges) -> float:
    """Coulomb attraction between the electron and nucleus."""
    res = 0.0
    for i in range(n_vectors.shape[0]):
        for j in range(n_vectors.shape[1]):
            res -= atom_charges[j] / np.linalg.norm(n_vectors[i, j])

    for i in range(e_vectors.shape[0] - 1):
        for j in range(i + 1, e_vectors.shape[1]):
            res += 1 / np.linalg.norm(e_vectors[i, j])

    return res
