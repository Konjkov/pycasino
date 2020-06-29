#!/usr/bin/env python3

import os
from math import exp, sqrt, gamma
from timeit import default_timer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import numba as nb

# np.show_config()

from readers.wfn import Gwfn, Stowfn, GAUSSIAN_TYPE, SLATER_TYPE
from readers.input import Input


@nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
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


@nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def angular_part(r, l, result, radial):
    """Angular part of gaussian WFN.
    :return:
    """
    if l == 0:
        result[0] += radial
    elif l == 1:
        x, y, z = r
        result[0] += radial * x
        result[1] += radial * y
        result[2] += radial * z
    elif l == 2:
        x, y, z = r
        result[0] += radial * (2 * z*z - x*x - y*y)
        result[1] += radial * x*z
        result[2] += radial * y*z
        result[3] += radial * (x*x - y*y)
        result[4] += radial * x*y
    elif l == 3:
        x, y, z = r
        result[0] += radial * z * (2 * z*z - 3 * x*x - 3 * y*y) / 2
        result[1] += radial * 3 * x * (4 * z*z - x*x - y*y) / 2
        result[2] += radial * 3 * y * (4 * z*z - x*x - y*y) / 2
        result[3] += radial * 15 * z * (x*x - y*y)
        result[4] += radial * 30 * x*y*z
        result[5] += radial * 15 * x * (x*x - 3 * y*y)
        result[6] += radial * 15 * y * (3 * x*x - y*y)
    elif l == 4:
        x, y, z = r
        r2 = x * x + y * y + z * z
        result[0] += radial * (35 * z*z*z*z - 30 * z*z * r2 + 3 * r2 * r2) / 8
        result[1] += radial * 5 * x*z * (7 * z*z - 3 * r2) / 2
        result[2] += radial * 5 * y*z * (7 * z*z - 3 * r2) / 2
        result[3] += radial * 15 * (x*x - y*y) * (7 * z*z - r2) / 2
        result[4] += radial * 30 * x*y * (7 * z*z - r2) / 2
        result[5] += radial * 105 * x*z * (x*x - 3 * y*y)
        result[6] += radial * 105 * y*z * (3 * x*x - y*y)
        result[7] += radial * 105 * (x*x*x*x - 6 * x*x*y*y + y*y*y*y)
        result[8] += radial * 420 * x*y * (x*x - y*y)


@nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def gradient_angular_part(r, l, result, radial):
    """Angular part of gaussian WFN.
    :return:
    """
    if l == 1:
        result[0] += radial
        result[1] += radial
        result[2] += radial
    if l == 2:
        x, y, z = r
        result[0] += radial * 2 * (2 * z - x - y)
        result[1] += radial * (x + z)
        result[2] += radial * (y + z)
        result[3] += radial * 2 * (x - y)
        result[4] += radial * (x + y)
    if l == 3:
        x, y, z = r
        result[0] += radial * (-1.5*x**2 - 3.0*x*z - 1.5*y**2 - 3.0*y*z + 3.0*z**2)
        result[1] += radial * (-4.5*x**2 - 3.0*x*y + 12.0*x*z - 1.5*y**2 + 6.0*z**2)
        result[2] += radial * (-1.5*x**2 - 3.0*x*y - 4.5*y**2 + 12.0*y*z + 6.0*z**2)
        result[3] += radial * (15.0*x**2 + 30.0*x*z - 15.0*y**2 - 30.0*y*z)
        result[4] += radial * 30 * (x*y + x*z + y*z)
        result[5] += radial * 45 * (x**2 - 2*x*y - y**2)
        result[6] += radial * 45 * (x**2 + 2*x*y - y**2)
    if l == 4:
        x, y, z = r
        result[0] += radial * (-7.5*x*z**2 + 1.5*x*(x**2 + y**2 + z**2) - 7.5*y*z**2 + 1.5*y*(x**2 + y**2 + z**2) + 10.0*z**3 - 6.0*z*(x**2 + y**2 + z**2))
        result[1] += radial * (-15.0*x**2*z - 15.0*x*y*z + 37.5*x*z**2 - 7.5*x*(x**2 + y**2 + z**2) + 17.5*z**3 - 7.5*z*(x**2 + y**2 + z**2))
        result[2] += radial * (-15.0*x*y*z - 15.0*y**2*z + 37.5*y*z**2 - 7.5*y*(x**2 + y**2 + z**2) + 17.5*z**3 - 7.5*z*(x**2 + y**2 + z**2))
        result[3] += radial * (-30.0*x**3 + 90.0*x**2*z + 90.0*x*z**2 + 30.0*y**3 - 90.0*y**2*z - 90.0*y*z**2)
        result[4] += radial * (-30.0*x**2*y - 30.0*x*y**2 + 180.0*x*y*z + 105.0*x*z**2 - 15.0*x*(x**2 + y**2 + z**2) + 105.0*y*z**2 - 15.0*y*(x**2 + y**2 + z**2))
        result[5] += radial * (105.0*x**3 + 315.0*x**2*z - 315.0*x*y**2 - 630.0*x*y*z - 315.0*y**2*z)
        result[6] += radial * (315.0*x**2*y + 315.0*x**2*z + 630.0*x*y*z - 105.0*y**3 - 315.0*y**2*z)
        result[7] += radial * (420.0*x**3 - 1260.0*x**2*y - 1260.0*x*y**2 + 420.0*y**3)
        result[8] += radial * (420.0*x**3 + 1260.0*x**2*y - 1260.0*x*y**2 - 420.0*y**3)


@nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def wfn(re, mo, atoms, shells):
    """
    Slater matrix
    :param re: electrons coordinates shape = (nelec, 3)
    :param mo: MO-coefficients shape = (nbasis_functions, nelec)
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
    for i in range(mo.shape[0]):
        ao = 0
        for natom in range(atoms.shape[0]):
            rI = re[i] - atoms[natom].position
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            for nshell in range(atoms[natom].shells[0], atoms[natom].shells[1]):
                # radial part
                radial_part = 0.0
                if shells[nshell].type == GAUSSIAN_TYPE:
                    for primitive in range(shells[nshell].primitives):
                        radial_part += shells[nshell].coefficients[primitive] * np.exp(-shells[nshell].exponents[primitive] * r2)  # 20s from 60s
                elif shells[nshell].type == SLATER_TYPE:
                    r = np.sqrt(r2)
                    for primitive in range(shells[nshell].primitives):
                        radial_part += r**shells[nshell].order * shells[nshell].coefficients[primitive] * np.exp(-shells[nshell].exponents[primitive] * r)
                # angular part
                l = shells[nshell].moment
                angular_part(rI, l, orbital[i, ao: ao+2*l+1], radial_part)  # 10s from 60s
                ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True, cache=True)
def gradient(re, mo, atoms, shells):
    """Gradient matrix."""
    orbital = np.zeros(mo.shape)
    for i in range(mo.shape[0]):
        ao = 0
        for natom in range(atoms.shape[0]):
            atom = atoms[natom]
            rI = re[i] - atom.position
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            grad_r = rI[0] + rI[1] + rI[2]
            for nshell in range(atom.shells[0], atom.shells[1]):
                # radial part
                radial_part_1 = 0.0
                radial_part_2 = 0.0
                if shells[nshell].type == GAUSSIAN_TYPE:
                    for primitive in range(shells[nshell].primitives):
                        alpha = shells[nshell].exponents[primitive]
                        exponent = np.exp(-alpha * r2)
                        radial_part_1 -= 2 * alpha * grad_r * shells[nshell].coefficients[primitive] * exponent   # 20s from 60s
                        radial_part_2 += exponent
                elif shells[nshell].type == SLATER_TYPE:
                    r = np.sqrt(r2)
                    for primitive in range(shells[nshell].primitives):
                        alpha = shells[nshell].exponents[primitive]
                        exponent = np.exp(-alpha * r)
                        radial_part_1 -= (alpha * grad_r)/r * shells[nshell].coefficients[primitive] * exponent   # 20s from 60s
                        radial_part_2 += exponent
                    return
                # angular part
                l = shells[nshell].moment
                angular_part(rI, l, orbital[i, ao: ao+2*l+1], radial_part_1)  # 10s from 60s
                gradient_angular_part(rI, l, orbital[i, ao: ao+2*l+1], radial_part_2)  # 10s from 60s
                ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True, cache=True)
def laplacian(re, mo, atoms, shells):
    """Laplacian matrix."""
    orbital = np.zeros(mo.shape)
    for i in range(mo.shape[0]):
        ao = 0
        for natom in range(atoms.shape[0]):
            atom = atoms[natom]
            rI = re[i] - atom.position
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            for nshell in range(atom.shells[0], atom.shells[1]):
                l = shells[nshell].moment
                # radial part
                radial_part = 0.0
                if shells[nshell].type == GAUSSIAN_TYPE:
                    for primitive in range(shells[nshell].primitives):
                        alpha = shells[nshell].exponents[primitive]
                        radial_part += 2 * alpha * (2 * alpha * r2 - 2 * l - 3) * shells[nshell].coefficients[primitive] * np.exp(-alpha * r2)  # 20s from 60s
                elif shells[nshell].type == SLATER_TYPE:
                    r = np.sqrt(r2)
                    for primitive in range(shells[nshell].primitives):
                        alpha = shells[nshell].exponents[primitive]
                        radial_part += alpha * (alpha - 2*(l+1)/r) * shells[nshell].coefficients[primitive] * np.exp(-alpha * r)
                # angular part
                angular_part(rI, l, orbital[i, ao: ao+2*l+1], radial_part)  # 10s from 60s
                ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def wfn_det(r_u, r_d, mo_u, mo_d, atoms, shells):
    """Slater determinant without norm factor 1/sqrt(N!).
    """
    u_orb = wfn(r_u, mo_u, atoms, shells)
    d_orb = wfn(r_d, mo_d, atoms, shells)
    return np.linalg.det(u_orb) * np.linalg.det(d_orb)


@nb.jit(nopython=True, cache=True)
def gradient_log(r_u, r_d, mo_u, mo_d, atoms, shells):
    """∇(phi)/phi.
    """
    u_orb = wfn(r_u, mo_u, atoms, shells)
    u_grad = gradient(r_u, mo_u, atoms, shells)
    cond = np.arange(r_u.shape[0]) * np.ones(u_orb.shape)

    res_u = 0
    for i in range(r_u.shape[0]):
        res_u += np.linalg.det(np.where(cond == i, u_grad, u_orb))

    d_orb = wfn(r_d, mo_d, atoms, shells)
    d_grad = gradient(r_d, mo_d, atoms, shells)
    cond = np.arange(r_d.shape[0]) * np.ones(d_orb.shape)

    res_d = 0
    for i in range(r_d.shape[0]):
        res_d += np.linalg.det(np.where(cond == i, d_grad, d_orb))

    return res_u / np.linalg.det(u_orb) + res_d / np.linalg.det(d_orb)


@nb.jit(nopython=True, cache=True)
def laplacian_log(r_u, r_d, mo_u, mo_d, atoms, shells):
    """∇²(phi)/phi.
    """
    u_orb = wfn(r_u, mo_u, atoms, shells)
    u_lap = laplacian(r_u, mo_u, atoms, shells)
    cond = np.arange(r_u.shape[0]) * np.ones(u_orb.shape)

    res_u = 0
    for i in range(r_u.shape[0]):
        res_u += np.linalg.det(np.where(cond == i, u_lap, u_orb))

    d_orb = wfn(r_d, mo_d, atoms, shells)
    d_lap = laplacian(r_d, mo_d, atoms, shells)
    cond = np.arange(r_d.shape[0]) * np.ones(d_orb.shape)

    res_d = 0
    for i in range(r_d.shape[0]):
        res_d += np.linalg.det(np.where(cond == i, d_lap, d_orb))

    return res_u / np.linalg.det(u_orb) + res_d / np.linalg.det(d_orb)


@nb.jit(nopython=True, cache=True)
def numerical_gradient_log(r_u, r_d, mo_u, mo_d, atoms, shells):
    """Numerical gradient
    :param r_u: up electrons coordinates shape = (nelec, 3)
    :param r_d: down electrons coordinates shape = (nelec, 3)
    """
    delta = 0.00001

    u_det = np.linalg.det(wfn(r_u, mo_u, atoms, shells))
    res_u = 0
    for i in range(r_u.shape[0]):
        for j in range(r_u.shape[1]):
            r_u[i, j] -= delta
            res_u -= np.linalg.det(wfn(r_u, mo_u, atoms, shells))
            r_u[i, j] += 2 * delta
            res_u += np.linalg.det(wfn(r_u, mo_u, atoms, shells))
            r_u[i, j] -= delta

    d_det = np.linalg.det(wfn(r_d, mo_d, atoms, shells))
    res_d = 0
    for i in range(r_d.shape[0]):
        for j in range(r_d.shape[1]):
            r_d[i, j] -= delta
            res_d -= np.linalg.det(wfn(r_d, mo_d, atoms, shells))
            r_d[i, j] += 2 * delta
            res_d += np.linalg.det(wfn(r_d, mo_d, atoms, shells))
            r_d[i, j] -= delta

    return (res_u / u_det + res_d / d_det) / delta / 2


@nb.jit(nopython=True, cache=True)
def numerical_laplacian_log(r_u, r_d, mo_u, mo_d, atoms, shells):
    """Numerical laplacian
    :param r_u: up electrons coordinates shape = (nelec, 3)
    :param r_d: down electrons coordinates shape = (nelec, 3)
    """
    delta = 0.00001

    u_det = np.linalg.det(wfn(r_u, mo_u, atoms, shells))
    res_u = 0
    for i in range(r_u.shape[0]):
        for j in range(r_u.shape[1]):
            r_u[i, j] -= delta
            res_u += np.linalg.det(wfn(r_u, mo_u, atoms, shells))
            r_u[i, j] += 2 * delta
            res_u += np.linalg.det(wfn(r_u, mo_u, atoms, shells))
            r_u[i, j] -= delta

    d_det = np.linalg.det(wfn(r_d, mo_d, atoms, shells))
    res_d = 0
    for i in range(r_d.shape[0]):
        for j in range(r_d.shape[1]):
            r_d[i, j] -= delta
            res_d += np.linalg.det(wfn(r_d, mo_d, atoms, shells))
            r_d[i, j] += 2 * delta
            res_d += np.linalg.det(wfn(r_d, mo_d, atoms, shells))
            r_d[i, j] -= delta

    return (res_u / u_det - 2 * r_u.size + res_d / d_det - 2 * r_d.size) / delta / delta


@nb.jit(nopython=True, cache=True)
def local_kinetic(r_u, r_d, mo_u, mo_d, atoms, shells):
    """local kinetic energy on the point.
    -1/2 * laplacian(phi) / phi
    """
    return -laplacian_log(r_u, r_d, mo_u, mo_d, atoms, shells) / 2
    # return -numerical_laplacian_log(r_u, r_d, mo_u, mo_d, atoms, shells) / 2


@nb.jit(nopython=True, cache=True)
def coulomb(r_u, r_d, atoms):
    """Coulomb attraction between the electron and nucleus."""
    res = 0.0
    for atom in range(atoms.shape[0]):
        I = atoms[atom].position
        charge = atoms[atom].charge
        for i in range(r_u.shape[0]):
            x = r_u[i][0] - I[0]
            y = r_u[i][1] - I[1]
            z = r_u[i][2] - I[2]
            r2 = x * x + y * y + z * z
            res -= charge / sqrt(r2)

        for i in range(r_d.shape[0]):
            x = r_d[i][0] - I[0]
            y = r_d[i][1] - I[1]
            z = r_d[i][2] - I[2]
            r2 = x * x + y * y + z * z
            res -= charge / sqrt(r2)

    for i in range(r_u.shape[0]):
        for j in range(i + 1, r_u.shape[0]):
            x = r_u[i][0] - r_u[j][0]
            y = r_u[i][1] - r_u[j][1]
            z = r_u[i][2] - r_u[j][2]
            r2 = x * x + y * y + z * z
            res += 1 / sqrt(r2)

    for i in range(r_d.shape[0]):
        for j in range(i + 1, r_d.shape[0]):
            x = r_d[i][0] - r_d[j][0]
            y = r_d[i][1] - r_d[j][1]
            z = r_d[i][2] - r_d[j][2]
            r2 = x * x + y * y + z * z
            res += 1 / sqrt(r2)

    for i in range(r_u.shape[0]):
        for j in range(r_d.shape[0]):
            x = r_u[i][0] - r_d[j][0]
            y = r_u[i][1] - r_d[j][1]
            z = r_u[i][2] - r_d[j][2]
            r2 = x * x + y * y + z * z
            res += 1 / sqrt(r2)

    return res


@nb.jit(nopython=True, cache=True)
def local_energy(r_u, r_d, mo_u, mo_d, atoms, shells):
    return coulomb(r_u, r_d, atoms) + local_kinetic(r_u, r_d, mo_u, mo_d, atoms, shells)


@nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def integral(low, high, neu, ned, steps, mo_u, mo_d, atoms, shells):
    """"""
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
        X_u = random_position(low, high, neu)
        X_d = random_position(low, high, ned)
        result += wfn_det(X_u, X_d, mo_u, mo_d, atoms, shells) ** 2

    return result * dV / gamma(neu+1) / gamma(ned+1)


def main(mo_up, mo_down, neu, ned, atoms, shells):
    steps = 10 * 1024 * 1024
    offset = 3.0

    low = np.min(atoms['position'], axis=0) - offset
    high = np.max(atoms['position'], axis=0) + offset

    mo_u = mo_up[:neu]
    mo_d = mo_down[:ned]
    return integral(low, high, neu, ned, steps, mo_u, mo_d, atoms, shells)


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

    # wfn_data = Gwfn('test/gwfn/h/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/h/HF/cc-pVQZ/input')
    wfn_data = Gwfn('test/gwfn/be/HF/cc-pVQZ/gwfn.data')
    input_data = Input('test/gwfn/be/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/be2/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/be2/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/acetic/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/acetic/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/acetaldehyde/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/acetaldehyde/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/si2h6/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/si2h6/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/alcl3/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/alcl3/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/s4-c2v/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/s4-c2v/HF/cc-pVQZ/input')

    # wfn_data = Stowfn('test/stowfn/he/HF/DZ/stowfn.data')
    # input_data = Input('test/stowfn/he/HF/DZ/input')
    # wfn_data = Stowfn('test/stowfn/be/HF/QZ4P/stowfn.data')
    # input_data = Input('test/stowfn/be/HF/QZ4P/input')

    start = default_timer()
    print(main(wfn_data.mo_up, wfn_data.mo_down, input_data.neu, input_data.ned, wfn_data.atoms, wfn_data.shells))
    end = default_timer()
    print(f'total time {end-start}')
