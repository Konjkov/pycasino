#!/usr/bin/env python3

import os
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
        3.0 * z2 - r2,
        x*z,
        y*z,
        x2 - y2,
        x*y,
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
        15 * x*y * (7 * z2 - r2),
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
        [-2.0*x, -2.0*y, 4.0*z],
        [z, 0.0, x],
        [0.0, z, y],
        [2.0*x, -2.0*y, 0.0],
        [y, x, 0.0],
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


@nb.jit(nopython=True, nogil=True, parallel=False)
def wfn(r_e, mo, atoms, shells):
    """
    Slater matrix
    :param r_e: electrons coordinates shape = (nelec, 3)
    :param mo: MO-coefficients shape = (nelec, nbasis_functions)
    :param atoms - list (natom, ) of struct of
        number: atomic number shape = (1,)
        charge: charge shape = (1,)
        positions: centered position of the shell shape = (3,)
        shells:
    :param shells - list (nshell, ) of struct of
        moment: l-number of the shell shape = (1,)
        primitives: number of primitives on each shell shape = (1,)
        coefficients: contraction coefficients of a primitives shape = (nprimitives,)
        exponents: exponents of a primitives shape = (nprimitives,)
    :return: slater matrix shape = (nelec, nelec)
    """
    orbital = np.zeros(mo.shape)
    for i in range(r_e.shape[0]):
        ao = 0
        for atom in range(atoms.shape[0]):
            rI = r_e[i] - atoms[atom].position
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            angular_part_data = angular_part(rI)
            for nshell in range(atoms[atom].shells[0], atoms[atom].shells[1]):
                l = shells[nshell].moment
                radial_part = 0.0
                if shells[nshell].type == GAUSSIAN_TYPE:
                    for primitive in range(shells[nshell].primitives):
                        radial_part += shells[nshell].coefficients[primitive] * np.exp(-shells[nshell].exponents[primitive] * r2)
                elif shells[nshell].type == SLATER_TYPE:
                    r = np.sqrt(r2)
                    for primitive in range(shells[nshell].primitives):
                        radial_part += r**shells[nshell].order * shells[nshell].coefficients[primitive] * np.exp(-shells[nshell].exponents[primitive] * r)
                for j in range(2 * l + 1):
                    orbital[i, ao+j] = radial_part * angular_part_data[l*l+j]
                ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True)
def wfn_gradient(r_e, mo, atoms, shells):
    """Gradient matrix."""
    orbital_x = np.zeros(mo.shape)
    orbital_y = np.zeros(mo.shape)
    orbital_z = np.zeros(mo.shape)
    for i in range(r_e.shape[0]):
        ao = 0
        for atom in range(atoms.shape[0]):
            rI = r_e[i] - atoms[atom].position
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            angular_part_data = angular_part(rI)
            gradient_angular_part_data = gradient_angular_part(rI)
            for nshell in range(atoms[atom].shells[0], atoms[atom].shells[1]):
                radial_part_1 = 0.0
                radial_part_2 = 0.0
                if shells[nshell].type == GAUSSIAN_TYPE:
                    for primitive in range(shells[nshell].primitives):
                        alpha = shells[nshell].exponents[primitive]
                        exponent = shells[nshell].coefficients[primitive] * np.exp(-alpha * r2)
                        radial_part_1 -= 2 * alpha * exponent
                        radial_part_2 += exponent
                elif shells[nshell].type == SLATER_TYPE:
                    r = np.sqrt(r2)
                    for primitive in range(shells[nshell].primitives):
                        alpha = shells[nshell].exponents[primitive]
                        exponent = shells[nshell].coefficients[primitive] * np.exp(-alpha * r)
                        radial_part_1 -= alpha/r * exponent
                        radial_part_2 += exponent
                l = shells[nshell].moment
                for j in range(2 * l + 1):
                    orbital_x[i, ao+j] = radial_part_1 * rI[0] * angular_part_data[l*l+j] + radial_part_2 * gradient_angular_part_data[l*l+j, 0]
                    orbital_y[i, ao+j] = radial_part_1 * rI[1] * angular_part_data[l*l+j] + radial_part_2 * gradient_angular_part_data[l*l+j, 1]
                    orbital_z[i, ao+j] = radial_part_1 * rI[2] * angular_part_data[l*l+j] + radial_part_2 * gradient_angular_part_data[l*l+j, 2]
                ao += 2*l+1
    return np.stack((np.dot(mo, orbital_x.T), np.dot(mo, orbital_y.T), np.dot(mo, orbital_z.T)))


@nb.jit(nopython=True)
def wfn_laplacian(r_e, mo, atoms, shells):
    """Laplacian matrix."""
    orbital = np.zeros(mo.shape)
    for i in range(r_e.shape[0]):
        ao = 0
        for atom in range(atoms.shape[0]):
            rI = r_e[i] - atoms[atom].position
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            angular_part_data = angular_part(rI)
            for nshell in range(atoms[atom].shells[0], atoms[atom].shells[1]):
                l = shells[nshell].moment
                radial_part = 0.0
                if shells[nshell].type == GAUSSIAN_TYPE:
                    for primitive in range(shells[nshell].primitives):
                        alpha = shells[nshell].exponents[primitive]
                        radial_part += 2 * alpha * (2 * alpha * r2 - 2 * l - 3) * shells[nshell].coefficients[primitive] * np.exp(-alpha * r2)
                elif shells[nshell].type == SLATER_TYPE:
                    r = np.sqrt(r2)
                    for primitive in range(shells[nshell].primitives):
                        alpha = shells[nshell].exponents[primitive]
                        radial_part += alpha * (alpha - 2*(l+1)/r) * shells[nshell].coefficients[primitive] * np.exp(-alpha * r)
                for j in range(2 * l + 1):
                    orbital[i, ao+j] = radial_part * angular_part_data[l*l+j]
                ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True)
def wfn_numerical_gradient(r_e, mo, atoms, shells):
    """Numerical gradient
    :param r_e: up/down electrons coordinates shape = (nelec, 3)
    """
    delta = 0.00001

    res = np.zeros((3, r_e.shape[0], r_e.shape[0]))
    for j in range(3):
        r_e[:, j] -= delta
        res[j, :, :] -= wfn(r_e, mo, atoms, shells)
        r_e[:, j] += 2 * delta
        res[j, :, :] += wfn(r_e, mo, atoms, shells)
        r_e[:, j] -= delta

    return res / delta / 2


@nb.jit(nopython=True)
def wfn_numerical_laplacian(r_e, mo, atoms, shells):
    """Numerical laplacian
    :param r_e: up/down electrons coordinates shape = (nelec, 3)
    """
    delta = 0.00001

    res = -6 * wfn(r_e, mo, atoms, shells)
    for j in range(3):
        r_e[:, j] -= delta
        res += wfn(r_e, mo, atoms, shells)
        r_e[:, j] += 2 * delta
        res += wfn(r_e, mo, atoms, shells)
        r_e[:, j] -= delta

    return res / delta / delta


@nb.jit(nopython=True)
def wfn_gradient_log(r_e, mo, atoms, shells):
    """∇(phi)/phi.
    """
    orb = wfn(r_e, mo, atoms, shells)
    grad = wfn_gradient(r_e, mo, atoms, shells)
    cond = np.arange(r_e.shape[0]) * np.ones(orb.shape)

    res = np.zeros(r_e.shape)
    for i in range(r_e.shape[0]):
        for j in range(3):
            res[i, j] = np.linalg.det(np.where(cond == i, grad[j], orb))

    return res / np.linalg.det(orb)


@nb.jit(nopython=True)
def wfn_laplacian_log(r_e, mo, atoms, shells):
    """∇²(phi)/phi.
    """
    orb = wfn(r_e, mo, atoms, shells)
    lap = wfn_laplacian(r_e, mo, atoms, shells)
    cond = np.arange(r_e.shape[0]) * np.ones(orb.shape)

    res = 0
    for i in range(r_e.shape[0]):
        res += np.linalg.det(np.where(cond == i, lap, orb))

    return res / np.linalg.det(orb)


@nb.jit(nopython=True, nogil=True, parallel=False)
def integral(low, high, neu, ned, steps, mo_u, mo_d, atoms, shells, atomic_positions):
    """https://en.wikipedia.org/wiki/Monte_Carlo_integration"""
    dV = np.prod(high - low) ** (neu + ned) / steps

    def random_position(low, high, ne):
        """
        The size argument is not supported.
        https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#random

        https://numba.pydata.org/numba-doc/dev/extending/overloading-guide.html
        """
        return np.dstack((
            np.random.uniform(low[0], high[0], size=ne),
            np.random.uniform(low[1], high[1], size=ne),
            np.random.uniform(low[2], high[2], size=ne)
        ))[0]

    result = 0.0
    for i in range(steps):
        r_e = random_position(low, high, neu + ned)
        result += (np.linalg.det(wfn(r_e[:neu], mo_u, atoms, shells)) * np.linalg.det(wfn(r_e[neu:], mo_d, atoms, shells))) ** 2

    return result * dV / gamma(neu+1) / gamma(ned+1)


@nb.jit(nopython=True, nogil=True, parallel=True)
def p_integral(low, high, neu, ned, steps, mo_u, mo_d, atoms, shells, atomic_positions):
    res = 0.0
    for i in nb.prange(4):
        res += integral(low, high, neu, ned, steps, mo_u, mo_d, atoms, shells, atomic_positions)
    return res / 4


def main(mo_up, mo_down, neu, ned, atoms, shells):
    steps = 10 * 1024 * 1024
    offset = 3.0

    low = np.min(atoms['position'], axis=0) - offset
    high = np.max(atoms['position'], axis=0) + offset

    atomic_positions = atoms['position']

    mo_u = mo_up[:neu]
    mo_d = mo_down[:ned]

    return integral(low, high, neu, ned, steps, mo_u, mo_d, atoms, shells, atomic_positions)


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

    # casino = Casino('test/gwfn/h/HF/cc-pVQZ/')
    casino = Casino('test/gwfn/be/HF/cc-pVQZ/')
    # casino = Casino('test/gwfn/be2/HF/cc-pVQZ/')
    # casino = Casino('test/gwfn/acetic/HF/cc-pVQZ/')
    # casino = Casino('test/gwfn/acetaldehyde/HF/cc-pVQZ/')
    # casino = Casino('test/gwfn/si2h6/HF/cc-pVQZ/')
    # casino = Casino('test/gwfn/alcl3/HF/cc-pVQZ/')
    # casino = Casino('test/gwfn/s4-c2v/HF/cc-pVQZ/')

    # casino = Casino('test/stowfn/he/HF/DZ/')
    # casino = Casino('test/stowfn/be/HF/QZ4P/')

    start = default_timer()
    res = main(casino.wfn.mo_up, casino.wfn.mo_down, casino.input.neu, casino.input.ned, casino.wfn.atoms, casino.wfn.shells)
    print(res)
    end = default_timer()
    print(f'total time {end-start}')
