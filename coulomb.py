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
def nuclear_repulsion(atoms):
    """nuclear-nuclear repulsion"""
    result = 0.0
    for atom1 in range(atoms.shape[0]):
        for atom2 in range(atoms.shape[0]):
            if atom1 > atom2:
                r = atoms[atom1].position - atoms[atom2].position
                result += atoms[atom1].charge * atoms[atom2].charge/np.linalg.norm(r)
    return result


@nb.jit(nopython=True)
def coulomb(r_e, atoms):
    """Coulomb attraction between the electron and nucleus."""
    res = 0.0
    for atom in range(atoms.shape[0]):
        charge = atoms[atom].charge
        for i in range(r_e.shape[0]):
            res -= charge / np.linalg.norm(r_e[i] - atoms[atom].position)

    for i in range(r_e.shape[0] - 1):
        for j in range(i + 1, r_e.shape[0]):
            res += 1 / np.linalg.norm(r_e[i] - r_e[j])

    return res


# @nb.jit(nopython=True)
# def coulomb(r_e, r_eI, atoms):
#     """Coulomb attraction between the electron and nucleus."""
#     res = 0.0
#     for atom in range(atoms.shape[0]):
#         charge = atoms[atom].charge
#         for i in range(r_eI.shape[0]):
#             res -= charge / np.linalg.norm(r_eI[i, atom])
#
#     for i in range(r_e.shape[0] - 1):
#         for j in range(i + 1, r_e.shape[0]):
#             res += 1 / np.linalg.norm(r_e[i] - r_e[j])
#
#     return res
