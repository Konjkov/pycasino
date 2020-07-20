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
    natoms = atoms.shape[0]
    for natom1 in range(natoms):
        for natom2 in range(natoms):
            if natom1 > natom2:
                r = atoms[natom1].position - atoms[natom2].position
                result += atoms[natom1].charge * atoms[natom2].charge/np.linalg.norm(r)
    return result


@nb.jit(nopython=True)
def coulomb(r_u, r_d, r_uI, r_dI, atoms):
    """Coulomb attraction between the electron and nucleus."""
    res = 0.0
    for atom in range(atoms.shape[0]):
        charge = atoms[atom].charge
        for i in range(r_uI.shape[0]):
            res -= charge / np.linalg.norm(r_uI[i, atom])
        for i in range(r_dI.shape[0]):
            res -= charge / np.linalg.norm(r_dI[i, atom])

    for i in range(r_u.shape[0]):
        for j in range(i + 1, r_u.shape[0]):
            res += 1 / np.linalg.norm(r_u[i] - r_u[j])

    for i in range(r_d.shape[0]):
        for j in range(i + 1, r_d.shape[0]):
            res += 1 / np.linalg.norm(r_d[i] - r_d[j])

    for i in range(r_u.shape[0]):
        for j in range(r_d.shape[0]):
            res += 1 / np.linalg.norm(r_u[i] - r_d[j])

    return res
