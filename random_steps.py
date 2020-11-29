import numpy as np
import numba as nb


@nb.jit(nopython=True)
def initial_position(ne, atom_positions):
    """Initial positions of electrons."""
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        r_e[i] = atom_positions[np.random.randint(natoms)]
    return r_e


@nb.jit(nopython=True)
def random_laplace_step(dX, ne):
    """Random N-dim laplace distributed step"""
    return np.random.laplace(0.0, dX/(3*np.pi/4), ne*3).reshape((ne, 3))


@nb.jit(nopython=True)
def random_triangular_step(dX, ne):
    """Random N-dim triangular distributed step"""
    return np.random.triangular(-1.5*dX, 0, 1.5*dX, ne*3).reshape((ne, 3))


@nb.jit(nopython=True)
def random_square_step(dX, ne):
    """Random N-dim square distributed step"""
    return np.random.uniform(-dX, dX, ne*3).reshape((ne, 3))


@nb.jit(nopython=True)
def random_normal_step(dX, ne):
    """Random normal distributed step"""
    return np.random.normal(0.0, dX/np.sqrt(3), ne*3).reshape((ne, 3))


@nb.jit(nopython=True)
def random_on_sphere_step(dX, ne):
    """Random on a sphere distributed step"""
    result = []
    for i in range(ne):
        x = np.random.normal(0.0, 1, 3)
        res = dX * x / np.linalg.norm(x)
        result.append(res[0])
        result.append(res[1])
        result.append(res[2])
    return np.array(result).reshape((ne, 3))


random_step = random_normal_step
