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


@nb.jit(nopython=True, nogil=True, parallel=False)
def angular_part(r):
    """Angular part of gaussian WFN.
    :return:
    """
    x, y, z = r
    r2 = x*x + y*y + z*z
    return np.array([
        1,
        x,
        y,
        z,
        3 * z*z - r2,
        x*z,
        y*z,
        x*x - y*y,
        x*y,
        z * (5 * z*z - 3 * r2) / 2,
        3 * x * (5 * z*z - r2) / 2,
        3 * y * (5 * z*z - r2) / 2,
        15 * z * (x*x - y*y),
        30 * x * y*z,
        15 * x * (x*x - 3 * y*y),
        15 * y * (3 * x*x - y*y),
        (35 * z*z*z*z - 30 * z*z * r2 + 3 * r2 * r2) / 8,
        5 * x*z * (7 * z*z - 3 * r2) / 2,
        5 * y*z * (7 * z*z - 3 * r2) / 2,
        15 * (x*x - y*y) * (7 * z*z - r2) / 2,
        30 * x*y * (7 * z*z - r2) / 2,
        105 * x*z * (x*x - 3 * y*y),
        105 * y*z * (3 * x*x - y*y),
        105 * (x*x*x*x - 6 * x*x * y*y + y*y*y*y),
        420 * x*y * (x*x - y*y)
    ])


@nb.jit(nopython=True, nogil=True, parallel=False)
def gradient_angular_part(r):
    """Angular part of gaussian WFN.
    :return:
    """
    x, y, z = r
    return np.array([
        0,
        1,
        1,
        1,
        2 * (2 * z - x - y),
        x + z,
        y + z,
        2 * (x - y),
        x + y,
        (-1.5*x**2 - 3.0*x*z - 1.5*y**2 - 3.0*y*z + 3.0*z**2),
        (-4.5*x**2 - 3.0*x*y + 12.0*x*z - 1.5*y**2 + 6.0*z**2),
        (-1.5*x**2 - 3.0*x*y - 4.5*y**2 + 12.0*y*z + 6.0*z**2),
        (15.0*x**2 + 30.0*x*z - 15.0*y**2 - 30.0*y*z),
        30 * (x*y + x*z + y*z),
        45 * (x**2 - 2*x*y - y**2),
        45 * (x**2 + 2*x*y - y**2),
        (-7.5*x*z**2 + 1.5*x*(x**2 + y**2 + z**2) - 7.5*y*z**2 + 1.5*y*(x**2 + y**2 + z**2) + 10.0*z**3 - 6.0*z*(x**2 + y**2 + z**2)),
        (-15.0*x**2*z - 15.0*x*y*z + 37.5*x*z**2 - 7.5*x*(x**2 + y**2 + z**2) + 17.5*z**3 - 7.5*z*(x**2 + y**2 + z**2)),
        (-15.0*x*y*z - 15.0*y**2*z + 37.5*y*z**2 - 7.5*y*(x**2 + y**2 + z**2) + 17.5*z**3 - 7.5*z*(x**2 + y**2 + z**2)),
        (-30.0*x**3 + 90.0*x**2*z + 90.0*x*z**2 + 30.0*y**3 - 90.0*y**2*z - 90.0*y*z**2),
        (-30.0*x**2*y - 30.0*x*y**2 + 180.0*x*y*z + 105.0*x*z**2 - 15.0*x*(x**2 + y**2 + z**2) + 105.0*y*z**2 - 15.0*y*(x**2 + y**2 + z**2)),
        (105.0*x**3 + 315.0*x**2*z - 315.0*x*y**2 - 630.0*x*y*z - 315.0*y**2*z),
        (315.0*x**2*y + 315.0*x**2*z + 630.0*x*y*z - 105.0*y**3 - 315.0*y**2*z),
        (420.0*x**3 - 1260.0*x**2*y - 1260.0*x*y**2 + 420.0*y**3),
        (420.0*x**3 + 1260.0*x**2*y - 1260.0*x*y**2 - 420.0*y**3)
    ])


@nb.jit(nopython=True, nogil=True, parallel=False)
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
            angular_part_data = angular_part(rI)
            for nshell in range(atoms[natom].shells[0], atoms[natom].shells[1]):
                l = shells[nshell].moment
                radial_part = 0.0
                if shells[nshell].type == GAUSSIAN_TYPE:
                    for primitive in range(shells[nshell].primitives):
                        radial_part += shells[nshell].coefficients[primitive] * np.exp(-shells[nshell].exponents[primitive] * r2)  # 20s from 60s
                elif shells[nshell].type == SLATER_TYPE:
                    r = np.sqrt(r2)
                    for primitive in range(shells[nshell].primitives):
                        radial_part += r**shells[nshell].order * shells[nshell].coefficients[primitive] * np.exp(-shells[nshell].exponents[primitive] * r)
                for j in range(2 * l + 1):
                    orbital[i, ao+j] = radial_part * angular_part_data[l*l+j]
                ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True)
def gradient(re, mo, atoms, shells):
    """Gradient matrix."""
    orbital = np.zeros(mo.shape)
    for i in range(mo.shape[0]):
        ao = 0
        for natom in range(atoms.shape[0]):
            atom = atoms[natom]
            rI = re[i] - atom.position
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            angular_part_data = angular_part(rI)
            gradient_angular_part_data = gradient_angular_part(rI)
            grad_r = rI[0] + rI[1] + rI[2]
            for nshell in range(atom.shells[0], atom.shells[1]):
                radial_part_1 = 0.0
                radial_part_2 = 0.0
                if shells[nshell].type == GAUSSIAN_TYPE:
                    for primitive in range(shells[nshell].primitives):
                        alpha = shells[nshell].exponents[primitive]
                        exponent = shells[nshell].coefficients[primitive] * np.exp(-alpha * r2)
                        radial_part_1 -= 2 * alpha * grad_r * exponent   # 20s from 60s
                        radial_part_2 += exponent
                elif shells[nshell].type == SLATER_TYPE:
                    r = np.sqrt(r2)
                    for primitive in range(shells[nshell].primitives):
                        alpha = shells[nshell].exponents[primitive]
                        exponent = shells[nshell].coefficients[primitive] * np.exp(-alpha * r)
                        radial_part_1 -= (alpha * grad_r)/r * exponent   # 20s from 60s
                        radial_part_2 += exponent
                l = shells[nshell].moment
                for j in range(2 * l + 1):
                    orbital[i, ao+j] = radial_part_1 * angular_part_data[l*l+j] + radial_part_2 * gradient_angular_part_data[l*l+j]
                ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True)
def laplacian(re, mo, atoms, shells):
    """Laplacian matrix."""
    orbital = np.zeros(mo.shape)
    for i in range(mo.shape[0]):
        ao = 0
        for natom in range(atoms.shape[0]):
            atom = atoms[natom]
            rI = re[i] - atom.position
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            angular_part_data = angular_part(rI)
            for nshell in range(atom.shells[0], atom.shells[1]):
                l = shells[nshell].moment
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
                for j in range(2 * l + 1):
                    orbital[i, ao+j] = radial_part * angular_part_data[l*l+j]
                ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True, nogil=True, parallel=False)
def wfn_det(r_u, r_d, mo_u, mo_d, atoms, shells):
    """Slater determinant without norm factor 1/sqrt(N!).
    """
    u_orb = wfn(r_u, mo_u, atoms, shells)
    d_orb = wfn(r_d, mo_d, atoms, shells)
    return np.linalg.det(u_orb) * np.linalg.det(d_orb)


@nb.jit(nopython=True)
def gradient_log(r, mo, atoms, shells):
    """∇(phi)/phi.
    """
    orb = wfn(r, mo, atoms, shells)
    grad = gradient(r, mo, atoms, shells)
    cond = np.arange(r.shape[0]) * np.ones(orb.shape)

    res = 0
    for i in range(r.shape[0]):
        res += np.linalg.det(np.where(cond == i, grad, orb))

    return res / np.linalg.det(orb)


@nb.jit(nopython=True)
def laplacian_log(r, mo, atoms, shells):
    """∇²(phi)/phi.
    """
    orb = wfn(r, mo, atoms, shells)
    lap = laplacian(r, mo, atoms, shells)
    cond = np.arange(r.shape[0]) * np.ones(orb.shape)

    res = 0
    for i in range(r.shape[0]):
        res += np.linalg.det(np.where(cond == i, lap, orb))

    return res / np.linalg.det(orb)


@nb.jit(nopython=True)
def numerical_gradient_log(r, mo, atoms, shells):
    """Numerical gradient
    :param r: up/down electrons coordinates shape = (nelec, 3)
    """
    delta = 0.00001

    det = np.linalg.det(wfn(r, mo, atoms, shells))
    res = 0
    for i in range(r.shape[0]):
        res_1 = np.zeros((3,))
        for j in range(r.shape[1]):
            r[i, j] -= delta
            res_1[j] -= np.linalg.det(wfn(r, mo, atoms, shells))
            r[i, j] += 2 * delta
            res_1[j] += np.linalg.det(wfn(r, mo, atoms, shells))
            r[i, j] -= delta
        res += res_1[0] * res_1[0] + res_1[1] * res_1[1] + res_1[2] * res_1[2]

    return res / det / delta / 2


@nb.jit(nopython=True)
def numerical_laplacian_log(r, mo, atoms, shells):
    """Numerical laplacian
    :param r: up/down electrons coordinates shape = (nelec, 3)
    """
    delta = 0.00001

    det = np.linalg.det(wfn(r, mo, atoms, shells))
    res = 0
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r[i, j] -= delta
            res += np.linalg.det(wfn(r, mo, atoms, shells))
            r[i, j] += 2 * delta
            res += np.linalg.det(wfn(r, mo, atoms, shells))
            r[i, j] -= delta

    return (res / det - 2 * r.size) / delta / delta


@nb.jit(nopython=True)
def F(r_u, r_d, mo_u, mo_d, atoms, shells):
    """sum(|Fi|)"""
    return (numerical_gradient_log(r_u, mo_u, atoms, shells) + numerical_gradient_log(r_d, mo_d, atoms, shells)) / 2


@nb.jit(nopython=True)
def T(r_u, r_d, mo_u, mo_d, atoms, shells):
    """sum(Ti)"""
    return (
            numerical_gradient_log(r_u, mo_u, atoms, shells) - laplacian_log(r_u, mo_u, atoms, shells) +
            numerical_gradient_log(r_d, mo_d, atoms, shells) - laplacian_log(r_d, mo_d, atoms, shells)
    ) / 4


@nb.jit(nopython=True)
def local_kinetic(r_u, r_d, mo_u, mo_d, atoms, shells):
    """local kinetic energy on the point.
    -1/2 * ∇²(phi) / phi
    """
    # return -(laplacian_log(r_u, mo_u, atoms, shells) + laplacian_log(r_d, mo_d, atoms, shells)) / 2
    return F(r_u, r_d, mo_u, mo_d, atoms, shells)
    # return -(numerical_laplacian_log(r_u, mo_u, atoms, shells) + numerical_laplacian_log(r_d, mo_d, atoms, shells)) / 2


@nb.jit(nopython=True)
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


@nb.jit(nopython=True)
def local_energy(r_u, r_d, mo_u, mo_d, atoms, shells):
    return coulomb(r_u, r_d, atoms) + local_kinetic(r_u, r_d, mo_u, mo_d, atoms, shells)


@nb.jit(nopython=True, nogil=True, parallel=False)
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
