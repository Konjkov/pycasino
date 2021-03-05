#!/usr/bin/env python3

import os
from typing import Tuple
from math import gamma
from timeit import default_timer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import numba as nb

# np.show_config()

from decorators import pool, thread
from overload import subtract_outer
from readers.wfn import GAUSSIAN_TYPE, SLATER_TYPE
from readers.casino import Casino


@nb.jit(nopython=True, nogil=True, parallel=False)
def angular_part(r):
    """Angular part of gaussian WFN.
    :return:
    """
    x, y, z = r
    x2 = x**2
    y2 = y**2
    z2 = z**2
    r2 = x2 + y2 + z2
    return np.array([
        1.0,
        x,
        y,
        z,
        (3.0 * z2 - r2) / 2.0,
        3.0 * x*z,
        3.0 * y*z,
        3.0 * (x2 - y2),
        6.0 * x*y,
        z * (5.0 * z2 - 3.0 * r2) / 2.0,
        1.5 * x * (5 * z2 - r2),
        1.5 * y * (5 * z2 - r2),
        15.0 * z * (x2 - y2),
        30.0 * x * y*z,
        15.0 * x * (x2 - 3 * y2),
        15.0 * y * (3 * x2 - y2),
        (35.0 * z**4 - 30.0 * z2 * r2 + 3.0 * r2**2) / 8.0,
        2.5 * x*z * (7 * z2 - 3 * r2),
        2.5 * y*z * (7 * z2 - 3 * r2),
        7.5 * (x2 - y2) * (7 * z2 - r2),
        15.0 * x*y * (7 * z2 - r2),
        105.0 * x*z * (x2 - 3 * y2),
        105.0 * y*z * (3 * x2 - y2),
        105.0 * (x2**2 - 6 * x2 * y2 + y2**2),
        420.0 * x*y * (x2 - y2)
    ])


@nb.jit(nopython=True, nogil=True, parallel=False)
def gradient_angular_part(r):
    """Angular part of gaussian WFN gradient.
    :return:
    """
    x, y, z = r
    x2 = x**2
    y2 = y**2
    z2 = z**2
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-x, -y, 2.0*z],
        [3.0*z, 0.0, 3.0*x],
        [0.0, 3.0*z, 3.0*y],
        [6.0*x, -6.0*y, 0.0],
        [6.0*y, 6.0*x, 0.0],
        [-3.0*x*z, -3.0*y*z, -1.5*x2 - 1.5*y2 + 3.0*z2],
        [-4.5*x2 - 1.5*y2 + 6.0*z2, -3.0*x*y, 12.0*x*z],
        [-3.0*x*y, -1.5*x2 - 4.5*y2 + 6.0*z2, 12.0*y*z],
        [30.0*x*z, -30.0*y*z, 15.0*x2 - 15.0*y2],
        [30.0*y*z, 30.0*x*z, 30.0*x*y],
        [45.0*x2 - 45.0*y2, -90.0*x*y, 0],
        [90.0*x*y, 45.0*x2 - 45.0*y2, 0],
        [x*(1.5*x2 + 1.5*y2 - 6.0*z2), y*(1.5*x2 + 1.5*y2 - 6.0*z2), z*(-6.0*x2 - 6.0*y2 + 4.0*z2)],
        [z*(-22.5*x2 - 7.5*y2 + 10.0*z2), -15.0*x*y*z, x*(-7.5*x2 - 7.5*y2 + 30.0*z2)],
        [-15.0*x*y*z, z*(-7.5*x2 - 22.5*y2 + 10.0*z2), y*(-7.5*x2 - 7.5*y2 + 30.0*z2)],
        [x*(-30.0*x2 + 90.0*z2), y*(30.0*y2 - 90.0*z2), 90.0*z*(x2 - y2)],
        [y*(-45.0*x2 - 15.0*y2 + 90.0*z2), x*(-15.0*x2 - 45.0*y2 + 90.0*z2), 180.0*x*y*z],
        [315.0*z*(x2 - y2), -630.0*x*y*z, x*(105.0*x2 - 315.0*y2)],
        [630.0*x*y*z, 315.0*z*(x2 - y2), y*(315.0*x2 - 105.0*y2)],
        [x*(420.0*x2 - 1260.0*y2), y*(-1260.0*x2 + 420.0*y2), 0],
        [y*(1260.0*x2 - 420.0*y2), x*(420.0*x2 - 1260.0*y2), 0]
    ])


spec = [
    ('nbasis_functions', nb.int64),
    ('first_shells', nb.int64[:]),
    ('orbital_types', nb.int64[:]),
    ('shell_moments', nb.int64[:]),
    ('slater_orders', nb.int64[:]),
    ('primitives', nb.int64[:]),
    ('coefficients', nb.float64[:]),
    ('exponents', nb.float64[:]),
    ('mo_up', nb.float64[:, :, :]),
    ('mo_down', nb.float64[:, :, :]),
    ('coeff', nb.float64[:]),
]


@nb.experimental.jitclass(spec)
class Slater:

    def __init__(self, nbasis_functions, first_shells, orbital_types, shell_moments, slater_orders, primitives, coefficients, exponents, mo_up, mo_down, coeff):
        self.nbasis_functions = nbasis_functions
        self.first_shells = first_shells
        self.orbital_types = orbital_types
        self.shell_moments = shell_moments
        self.slater_orders = slater_orders
        self.primitives = primitives
        self.coefficients = coefficients
        self.exponents = exponents
        self.mo_up = mo_up
        self.mo_down = mo_down
        self.coeff = coeff

    def AO_wfn(self, n_vectors: np.ndarray) -> np.ndarray:
        """
        Atomic orbitals for every electron
        :param n_vectors: electron-nuclei vectors shape = (nelec, natom, 3)
        :return: AO matrix shape = (nelec, nbasis_functions)
        """
        orbital = np.zeros((n_vectors.shape[0], self.nbasis_functions))
        for i in range(n_vectors.shape[0]):
            p = 0
            ao = 0
            for atom in range(n_vectors.shape[1]):
                rI = n_vectors[i, atom]
                r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
                angular_part_data = angular_part(rI)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    l = self.shell_moments[nshell]
                    radial_part = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            radial_part += self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r2)
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        for primitive in range(self.primitives[nshell]):
                            radial_part += r**self.slater_orders[nshell] * self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r)
                    p += self.primitives[nshell]
                    for j in range(2 * l + 1):
                        orbital[i, ao+j] = radial_part * angular_part_data[l*l+j]
                    ao += 2*l+1
        return orbital

    def AO_gradient(self, n_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gradient matrix.
        :param n_vectors: electron-nuclei vectors shape = (nelec, natom, 3)
        """
        orbital_x = np.zeros((n_vectors.shape[0], self.nbasis_functions))
        orbital_y = np.zeros((n_vectors.shape[0], self.nbasis_functions))
        orbital_z = np.zeros((n_vectors.shape[0], self.nbasis_functions))
        for i in range(n_vectors.shape[0]):
            p = 0
            ao = 0
            for atom in range(n_vectors.shape[1]):
                rI = n_vectors[i, atom]
                r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
                angular_part_data = angular_part(rI)
                gradient_angular_part_data = gradient_angular_part(rI)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    radial_part_1 = 0.0
                    radial_part_2 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            exponent = self.coefficients[p + primitive] * np.exp(-alpha * r2)
                            radial_part_1 -= 2 * alpha * exponent
                            radial_part_2 += exponent
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            exponent = self.coefficients[p + primitive] * np.exp(-alpha * r)
                            radial_part_1 -= alpha/r * exponent
                            radial_part_2 += exponent
                    p += self.primitives[nshell]
                    l = self.shell_moments[nshell]
                    for j in range(2 * l + 1):
                        orbital_x[i, ao+j] = radial_part_1 * rI[0] * angular_part_data[l*l+j] + radial_part_2 * gradient_angular_part_data[l*l+j, 0]
                        orbital_y[i, ao+j] = radial_part_1 * rI[1] * angular_part_data[l*l+j] + radial_part_2 * gradient_angular_part_data[l*l+j, 1]
                        orbital_z[i, ao+j] = radial_part_1 * rI[2] * angular_part_data[l*l+j] + radial_part_2 * gradient_angular_part_data[l*l+j, 2]
                    ao += 2*l+1
        return orbital_x, orbital_y, orbital_z

    def AO_laplacian(self, n_vectors: np.ndarray) -> np.ndarray:
        """Laplacian matrix.
        :param n_vectors: electron-nuclei vectors shape = (nelec, natom, 3)
        """
        orbital = np.zeros((n_vectors.shape[0], self.nbasis_functions))
        for i in range(n_vectors.shape[0]):
            p = 0
            ao = 0
            for atom in range(n_vectors.shape[1]):
                rI = n_vectors[i, atom]
                r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
                angular_part_data = angular_part(rI)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    l = self.shell_moments[nshell]
                    radial_part = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            radial_part += 2 * alpha * (2 * alpha * r2 - 2 * l - 3) * self.coefficients[p + primitive] * np.exp(-alpha * r2)
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            radial_part += alpha * (alpha - 2*(l+1)/r) * self.coefficients[p + primitive] * np.exp(-alpha * r)
                    p += self.primitives[nshell]
                    for j in range(2 * l + 1):
                        orbital[i, ao+j] = radial_part * angular_part_data[l*l+j]
                    ao += 2*l+1
        return orbital

    def value(self, n_vectors: np.ndarray, neu: int) -> float:
        """Multideterminant wave function value.
        :param n_vectors: electron-nuclei vectors shape = (nelec, natom, 3)
        :param neu: number of up-electrons
        """
        ao = self.AO_wfn(n_vectors)

        res = 0.0
        for i in range(self.coeff.shape[0]):
            res += self.coeff[i] * np.linalg.det(np.dot(self.mo_up[i], ao[:neu].T)) * np.linalg.det(np.dot(self.mo_down[i], ao[neu:].T))
        return res

    def numerical_gradient(self, n_vectors: np.ndarray, neu: int, ned: int) -> float:
        """Numerical gradient
        :param n_vectors: electron-nuclei vectors shape = (nelec, natom, 3)
        :param neu: number of up-electrons
        :param ned: number of down-electrons
        """
        delta = 0.00001

        res = np.zeros((n_vectors.shape[0], 3))
        for i in range(n_vectors.shape[0]):
            for j in range(3):
                n_vectors[i, :, j] -= delta
                res[i, j] -= self.value(n_vectors, neu)
                n_vectors[i, :, j] += 2 * delta
                res[i, j] += self.value(n_vectors, neu)
                n_vectors[i, :, j] -= delta

        return res / delta / 2

    def numerical_laplacian(self, n_vectors: np.ndarray, neu: int, ned: int) -> float:
        """Numerical laplacian
        :param n_vectors: electron-nuclei vectors shape = (nelec, natom, 3)
        :param neu: number of up-electrons
        :param ned: number of down-electrons
        """
        delta = 0.00001

        res = 0.0
        for i in range(n_vectors.shape[0]):
            for j in range(3):
                n_vectors[i, :, j] -= delta
                res += self.value(n_vectors, neu)
                n_vectors[i, :, j] += 2 * delta
                res += self.value(n_vectors, neu)
                n_vectors[i, :, j] -= delta
        res -= 6 * n_vectors.shape[0] * self.value(n_vectors, neu)

        return res / delta / delta

    def gradient(self, n_vectors: np.ndarray, neu: int, ned: int) -> np.ndarray:
        """∇(phi).
        :param n_vectors: electron-nuclei vectors shape = (nelec, natom, 3)
        :param neu: number of up-electrons
        :param ned: number of down-electrons

        As numba overloaded function 'dot' only supported on 1-D and 2-D arrays,
        so I use list of 2-D arrays (gradient_x, gradient_y, gradient_z) to represent gradient.
        """
        ao = self.AO_wfn(n_vectors)
        gradient_x, gradient_y, gradient_z = self.AO_gradient(n_vectors)
        cond_u = np.arange(neu) * np.ones((neu, neu))
        cond_d = np.arange(ned) * np.ones((ned, ned))

        res = np.zeros((neu + ned, 3))
        for i in range(self.coeff.shape[0]):

            wfn_u = np.dot(self.mo_up[i], ao[:neu].T)
            grad_x, grad_y, grad_z = np.dot(self.mo_up[i], gradient_x[:neu].T), np.dot(self.mo_up[i], gradient_y[:neu].T), np.dot(self.mo_up[i], gradient_z[:neu].T)

            res_u = np.zeros((neu, 3))
            for j in range(neu):
                res_u[j, 0] = np.linalg.det(np.where(cond_u == j, grad_x, wfn_u))
                res_u[j, 1] = np.linalg.det(np.where(cond_u == j, grad_y, wfn_u))
                res_u[j, 2] = np.linalg.det(np.where(cond_u == j, grad_z, wfn_u))

            wfn_d = np.dot(self.mo_down[i], ao[neu:].T)
            grad_x, grad_y, grad_z = np.dot(self.mo_down[i], gradient_x[neu:].T), np.dot(self.mo_down[i], gradient_y[neu:].T), np.dot(self.mo_down[i], gradient_z[neu:].T)

            res_d = np.zeros((ned, 3))
            for j in range(ned):
                res_d[j, 0] = np.linalg.det(np.where(cond_d == j, grad_x, wfn_d))
                res_d[j, 1] = np.linalg.det(np.where(cond_d == j, grad_y, wfn_d))
                res_d[j, 2] = np.linalg.det(np.where(cond_d == j, grad_z, wfn_d))

            res += self.coeff[i] * np.concatenate((res_u * np.linalg.det(wfn_d), res_d * np.linalg.det(wfn_u)))

        return res

    def laplacian(self, n_vectors: np.ndarray, neu: int, ned: int) -> float:
        """∇²(phi).
        :param n_vectors: electron-nuclei vectors shape = (nelec, natom, 3)
        :param neu: number of up-electrons
        :param ned: number of down-electrons
        """
        ao = self.AO_wfn(n_vectors)
        ao_laplacian = self.AO_laplacian(n_vectors)
        cond_u = np.arange(neu) * np.ones((neu, neu))
        cond_d = np.arange(ned) * np.ones((ned, ned))

        res = 0
        for i in range(self.coeff.shape[0]):

            wfn_u = np.dot(self.mo_up[i], ao[:neu].T)
            lap_u = np.dot(self.mo_up[i], ao_laplacian[:neu].T)

            res_u = 0
            for j in range(neu):
                res_u += np.linalg.det(np.where(cond_u == j, lap_u, wfn_u))

            wfn_d = np.dot(self.mo_down[i], ao[neu:].T)
            lap_d = np.dot(self.mo_down[i], ao_laplacian[neu:].T)

            res_d = 0
            for j in range(ned):
                res_d += np.linalg.det(np.where(cond_d == j, lap_d, wfn_d))

            res += self.coeff[i] * (res_u * np.linalg.det(wfn_d) + res_d * np.linalg.det(wfn_u))

        return res


@nb.jit(nopython=True)
def random_step(dx, ne):
    """Random N-dim square distributed step"""
    return np.random.uniform(-dx, dx, ne * 3).reshape((ne, 3))


# @pool
@nb.jit(nopython=True, nogil=True)
def integral(dX, neu, ned, steps, atom_positions, slater, r_initial):
    """https://en.wikipedia.org/wiki/Monte_Carlo_integration"""
    v = (2 * dX) ** (3 * (neu + ned))  # integration volume
    slater_determinant_normalization_factor = np.sqrt(1 / gamma(neu+1) / gamma(ned+1))

    result = 0.0
    for i in range(steps):
        r_e = r_initial + random_step(dX, neu + ned)
        n_vectors = subtract_outer(r_e, atom_positions)
        result += (slater_determinant_normalization_factor * slater.value(n_vectors, neu)) ** 2

    return result * v / steps


@nb.jit(forceobj=True)
def initial_position(ne, atom_positions, atom_charges):
    """Initial positions of electrons."""
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
    return r_e


def main(casino):
    dX = 3.0

    slater = Slater(
        casino.wfn.nbasis_functions, casino.wfn.first_shells, casino.wfn.orbital_types, casino.wfn.shell_moments,
        casino.wfn.slater_orders, casino.wfn.primitives, casino.wfn.coefficients, casino.wfn.exponents,
        casino.mdet.mo_up, casino.mdet.mo_down, casino.mdet.coeff
    )

    r_initial = initial_position(casino.input.neu + casino.input.ned, casino.wfn.atom_positions, casino.wfn.atom_charges)
    return integral(dX, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, slater, r_initial)


if __name__ == '__main__':
    """
    be HF/cc-pVQZ

    steps = 5 * 1000 * 1000 * 1000
    offset = 4.5

    0.925763438273841

    real    464m11,715s
    user    464m11,194s
    sys     0m0,488s
    """

    # path = 'test/gwfn/h/HF/cc-pVQZ/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/'
    # path = 'test/gwfn/be/HF-CASSCF(2.4)/def2-QZVP/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/'
    # path = 'test/gwfn/acetic/HF/cc-pVQZ/'
    # path = 'test/gwfn/acetaldehyde/HF/cc-pVQZ/'
    # path = 'test/gwfn/si2h6/HF/cc-pVQZ/'
    # path = 'test/gwfn/alcl3/HF/cc-pVQZ/'
    # path = 'test/gwfn/s4-c2v/HF/cc-pVQZ/'

    # path = 'test/stowfn/he/HF/QZ4P/'
    path = 'test/stowfn/be/HF/QZ4P/'

    start = default_timer()
    res = main(Casino(path))
    print(np.mean(res), '+/-', np.var(res))
    end = default_timer()
    print(f'total time {end-start}')
