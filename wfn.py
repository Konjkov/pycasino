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

from decorators import multi_process
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
def AO_wfn(r_e, nbasis_functions, atoms, shells):
    """
    Atomic orbitals for every electron
    :param r_e: electrons coordinates shape = (nelec, 3)
    :param nbasis_functions:
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
    :return: AO matrix shape = (nelec, nbasis_functions)
    """
    orbital = np.zeros((r_e.shape[0], nbasis_functions))
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
    return orbital


@nb.jit(nopython=True)
def AO_gradient(r_e, nbasis_functions, atoms, shells):
    """Gradient matrix."""
    orbital_x = np.zeros((r_e.shape[0], nbasis_functions))
    orbital_y = np.zeros((r_e.shape[0], nbasis_functions))
    orbital_z = np.zeros((r_e.shape[0], nbasis_functions))
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
    return orbital_x, orbital_y, orbital_z


@nb.jit(nopython=True)
def AO_laplacian(r_e, nbasis_functions, atoms, shells):
    """Laplacian matrix."""
    orbital = np.zeros((r_e.shape[0], nbasis_functions))
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
    return orbital


@nb.jit(nopython=True)
def wfn(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, atoms, shells):
    ao = AO_wfn(r_e, nbasis_functions, atoms, shells)

    res = 0.0
    for i in range(coeff.shape[0]):
        res += coeff[i] * np.linalg.det(np.dot(mo_u[i], ao[:neu].T)) * np.linalg.det(np.dot(mo_d[i], ao[neu:].T))
    return res


@nb.jit(nopython=True)
def wfn_numerical_gradient(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, ned, atoms, shells):
    """Numerical gradient
    :param r_e: electrons coordinates shape = (nelec, 3)
    """
    delta = 0.00001

    res = np.zeros(r_e.shape)
    for i in range(r_e.shape[0]):
        for j in range(r_e.shape[1]):
            r_e[i, j] -= delta
            res[i, j] -= wfn(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, atoms, shells)
            r_e[i, j] += 2 * delta
            res[i, j] += wfn(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, atoms, shells)
            r_e[i, j] -= delta

    return res / delta / 2


@nb.jit(nopython=True)
def wfn_numerical_laplacian(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, ned, atoms, shells):
    """Numerical laplacian
    :param r_e: electrons coordinates shape = (nelec, 3)
    """
    delta = 0.00001

    res = 0.0
    for i in range(r_e.shape[0]):
        for j in range(r_e.shape[1]):
            r_e[i, j] -= delta
            res += wfn(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, atoms, shells)
            r_e[i, j] += 2 * delta
            res += wfn(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, atoms, shells)
            r_e[i, j] -= delta
    res -= 2 * r_e.size * wfn(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, atoms, shells)

    return res / delta / delta


@nb.jit(nopython=True)
def wfn_gradient(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, ned, atoms, shells):
    """∇(phi).
    """
    ao = AO_wfn(r_e, nbasis_functions, atoms, shells)
    gradient_x, gradient_y, gradient_z = AO_gradient(r_e, nbasis_functions, atoms, shells)
    cond_u = np.arange(neu) * np.ones((neu, neu))
    cond_d = np.arange(ned) * np.ones((ned, ned))

    res = np.zeros((neu + ned, 3))
    for i in range(coeff.shape[0]):

        wfn_u = np.dot(mo_u[i], ao[:neu].T)
        grad_x, grad_y, grad_z = np.dot(mo_u[i], gradient_x[:neu].T), np.dot(mo_u[i], gradient_y[:neu].T), np.dot(mo_u[i], gradient_z[:neu].T)

        res_u = np.zeros((neu, 3))
        for j in range(neu):
            res_u[j, 0] = np.linalg.det(np.where(cond_u == j, grad_x, wfn_u))
            res_u[j, 1] = np.linalg.det(np.where(cond_u == j, grad_y, wfn_u))
            res_u[j, 2] = np.linalg.det(np.where(cond_u == j, grad_z, wfn_u))

        wfn_d = np.dot(mo_d[i], ao[neu:].T)
        grad_x, grad_y, grad_z = np.dot(mo_d[i], gradient_x[neu:].T), np.dot(mo_d[i], gradient_y[neu:].T), np.dot(mo_d[i], gradient_z[neu:].T)

        res_d = np.zeros((ned, 3))
        for j in range(ned):
            res_d[j, 0] = np.linalg.det(np.where(cond_d == j, grad_x, wfn_d))
            res_d[j, 1] = np.linalg.det(np.where(cond_d == j, grad_y, wfn_d))
            res_d[j, 2] = np.linalg.det(np.where(cond_d == j, grad_z, wfn_d))

        res += np.concatenate((res_u * np.linalg.det(wfn_d), res_d * np.linalg.det(wfn_u)))

    return res


@nb.jit(nopython=True)
def wfn_laplacian(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, ned, atoms, shells):
    """∇²(phi).
    """
    ao = AO_wfn(r_e, nbasis_functions, atoms, shells)
    ao_laplacian = AO_laplacian(r_e, nbasis_functions, atoms, shells)
    cond_u = np.arange(neu) * np.ones((neu, neu))
    cond_d = np.arange(ned) * np.ones((ned, ned))

    res = 0
    for i in range(coeff.shape[0]):

        wfn_u = np.dot(mo_u[i], ao[:neu].T)
        lap_u = np.dot(mo_u[i], ao_laplacian[:neu].T)

        res_u = 0
        for j in range(neu):
            res_u += np.linalg.det(np.where(cond_u == j, lap_u, wfn_u))

        wfn_d = np.dot(mo_d[i], ao[neu:].T)
        lap_d = np.dot(mo_d[i], ao_laplacian[neu:].T)

        res_d = 0
        for j in range(ned):
            res_d += np.linalg.det(np.where(cond_d == j, lap_d, wfn_d))

        res += coeff[i] * (res_u * np.linalg.det(wfn_d) + res_d * np.linalg.det(wfn_u))

    return res


@multi_process
@nb.jit(nopython=True, nogil=True, parallel=False)
def integral(low, high, neu, ned, steps, nbasis_functions, mo_u, mo_d, coeff, atoms, shells):
    """https://en.wikipedia.org/wiki/Monte_Carlo_integration"""
    dV = np.prod(high - low) ** (neu + ned) / steps

    def random_position(low, high, ne):
        """
        The size argument is not supported.
        https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#random

        https://numba.pydata.org/numba-doc/dev/extending/overloading-guide.html
        """
        return np.stack((
            np.random.uniform(low[0], high[0], size=ne),
            np.random.uniform(low[1], high[1], size=ne),
            np.random.uniform(low[2], high[2], size=ne)
        )).T

    result = 0.0
    for i in range(steps):
        r_e = random_position(low, high, neu + ned)
        result += wfn(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, atoms, shells) ** 2

    return result * dV / gamma(neu+1) / gamma(ned+1)


def main(casino):
    offset = 3.0

    low = np.min(casino.wfn.atoms['position'], axis=0) - offset
    high = np.max(casino.wfn.atoms['position'], axis=0) + offset

    return integral(
        low, high, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.nbasis_functions,
        casino.mdet.mo_up, casino.mdet.mo_down, casino.mdet.coeff, casino.wfn.atoms, casino.wfn.shells
    )


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
    path = 'test/gwfn/be/HF/cc-pVQZ/'
    # path = 'test/gwfn/be/HF-CASSCF(2.4)/def2-QZVP/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/'
    # path = 'test/gwfn/acetic/HF/cc-pVQZ/'
    # path = 'test/gwfn/acetaldehyde/HF/cc-pVQZ/'
    # path = 'test/gwfn/si2h6/HF/cc-pVQZ/'
    # path = 'test/gwfn/alcl3/HF/cc-pVQZ/'
    # path = 'test/gwfn/s4-c2v/HF/cc-pVQZ/'

    # casino = Casino('test/stowfn/he/HF/DZ/')
    # casino = Casino('test/stowfn/be/HF/QZ4P/')

    start = default_timer()
    res = main(Casino(path))
    print(np.mean(res), '+/-', np.var(res))
    end = default_timer()
    print(f'total time {end-start}')
