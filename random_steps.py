import numpy as np
import numba as nb


@nb.jit(forceobj=True)
def initial_position(ne, atom_positions, atom_charges):
    """Initial positions of electrons."""
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
    return r_e


@nb.jit(nopython=True)
def random_square_step(dX, ne):
    """Random N-dim square distributed step"""
    return np.random.uniform(-dX, dX, ne*3).reshape((ne, 3))


@nb.jit(nopython=True)
def random_normal_step(dX, ne):
    """Random normal distributed step"""
    return np.random.normal(0.0, dX/np.sqrt(3), ne*3).reshape((ne, 3))


random_step = random_normal_step
