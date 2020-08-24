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
def nuclear_repulsion(atom_positions, atom_charges):
    """nuclear-nuclear repulsion"""
    result = 0.0
    for i in range(atom_positions.shape[0]):
        for j in range(atom_positions.shape[0]):
            if i > j:
                result += atom_charges[i] * atom_charges[j]/np.linalg.norm(atom_positions[i] - atom_positions[j])
    return result


@nb.jit(nopython=True)
def coulomb(r_e, atom_positions, atom_charges):
    """Coulomb attraction between the electron and nucleus."""
    res = 0.0
    for i in range(atom_positions.shape[0]):
        for j in range(r_e.shape[0]):
            res -= atom_charges[i] / np.linalg.norm(r_e[j] - atom_positions[i])

    for i in range(r_e.shape[0] - 1):
        for j in range(i + 1, r_e.shape[0]):
            res += 1 / np.linalg.norm(r_e[i] - r_e[j])

    return res
