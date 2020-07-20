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

from overload import subtract_outer
from readers.wfn import Gwfn, Stowfn, GAUSSIAN_TYPE, SLATER_TYPE
from readers.input import Input
from readers.jastrow import Jastrow
from jastrow import jastrow


@nb.jit(nopython=True, nogil=True, parallel=False)
def angular_part(r):
    """Angular part of gaussian WFN.
    :return:
    """
    x, y, z = r
    r2 = x*x + y*y + z*z
    return [
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
    ]


@nb.jit(nopython=True, nogil=True, parallel=False)
def gradient_angular_part(r):
    """Angular part of gaussian WFN gradient.
    :return:
    """
    x, y, z = r
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
        [-3.0*x*z, -3.0*y*z, -1.5*x**2 - 1.5*y**2 + 3.0*z**2],
        [-4.5*x**2 - 1.5*y**2 + 6.0*z**2, -3.0*x*y, 12.0*x*z],
        [-3.0*x*y, -1.5*x**2 - 4.5*y**2 + 6.0*z**2, 12.0*y*z],
        [30.0*x*z, -30.0*y*z, 15.0*x**2 - 15.0*y**2],
        [30.0*y*z, 30.0*x*z, 30.0*x*y],
        [45.0*x**2 - 45.0*y**2, -90.0*x*y, 0],
        [90.0*x*y, 45.0*x**2 - 45.0*y**2, 0],
        [x*(1.5*x**2 + 1.5*y**2 - 6.0*z**2), y*(1.5*x**2 + 1.5*y**2 - 6.0*z**2), z*(-6.0*x**2 - 6.0*y**2 + 4.0*z**2)],
        [z*(-22.5*x**2 - 7.5*y**2 + 10.0*z**2), -15.0*x*y*z, x*(-7.5*x**2 - 7.5*y**2 + 30.0*z**2)],
        [-15.0*x*y*z, z*(-7.5*x**2 - 22.5*y**2 + 10.0*z**2), y*(-7.5*x**2 - 7.5*y**2 + 30.0*z**2)],
        [x*(-30.0*x**2 + 90.0*z**2), y*(30.0*y**2 - 90.0*z**2), 90.0*z*(x**2 - y**2)],
        [y*(-45.0*x**2 - 15.0*y**2 + 90.0*z**2), x*(-15.0*x**2 - 45.0*y**2 + 90.0*z**2), 180.0*x*y*z],
        [315.0*z*(x**2 - y**2), -630.0*x*y*z, x*(105.0*x**2 - 315.0*y**2)],
        [630.0*x*y*z, 315.0*z*(x**2 - y**2), y*(315.0*x**2 - 105.0*y**2)],
        [x*(420.0*x**2 - 1260.0*y**2), y*(-1260.0*x**2 + 420.0*y**2), 0],
        [y*(1260.0*x**2 - 420.0*y**2), x*(420.0*x**2 - 1260.0*y**2), 0]
    ])


@nb.jit(nopython=True, nogil=True, parallel=False)
def wfn(r_eI, mo, atoms, shells):
    """
    Slater matrix
    :param r_eI: electrons coordinates shape = (nelec, natom, 3)
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
    for i in range(mo.shape[0]):
        ao = 0
        for natom in range(atoms.shape[0]):
            rI = r_eI[i, natom]
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            angular_part_data = angular_part(rI)
            for nshell in range(atoms[natom].shells[0], atoms[natom].shells[1]):
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
def wfn_gradient(r_eI, mo, atoms, shells):
    """Gradient matrix."""
    orbital_x = np.zeros(mo.shape)
    orbital_y = np.zeros(mo.shape)
    orbital_z = np.zeros(mo.shape)
    for i in range(mo.shape[0]):
        ao = 0
        for natom in range(atoms.shape[0]):
            rI = r_eI[i, natom]
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            angular_part_data = angular_part(rI)
            gradient_angular_part_data = gradient_angular_part(rI)
            for nshell in range(atoms[natom].shells[0], atoms[natom].shells[1]):
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
    return np.dot(mo, orbital_x.T), np.dot(mo, orbital_y.T), np.dot(mo, orbital_z.T)


@nb.jit(nopython=True)
def wfn_laplacian(r_eI, mo, atoms, shells):
    """Laplacian matrix."""
    orbital = np.zeros(mo.shape)
    for i in range(mo.shape[0]):
        ao = 0
        for natom in range(atoms.shape[0]):
            rI = r_eI[i, natom]
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            angular_part_data = angular_part(rI)
            for nshell in range(atoms[natom].shells[0], atoms[natom].shells[1]):
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
def wfn_numerical_gradient(r_eI, mo, atoms, shells):
    """Numerical gradient
    :param r_eI: up/down electrons coordinates shape = (nelec, natom, 3)
    """
    delta = 0.00001

    res = np.zeros((r_eI.shape[0], r_eI.shape[0], 3))
    for j in range(3):
        r_eI[:, :, j] -= delta
        res[:, :, j] -= wfn(r_eI, mo, atoms, shells)
        r_eI[:, :, j] += 2 * delta
        res[:, :, j] += wfn(r_eI, mo, atoms, shells)
        r_eI[:, :, j] -= delta

    return res[:, :, 0] / delta / 2, res[:, :, 1] / delta / 2, res[:, :, 2] / delta / 2


@nb.jit(nopython=True)
def wfn_numerical_laplacian(r_eI, mo, atoms, shells):
    """Numerical laplacian
    :param r_eI: up/down electrons coordinates shape = (nelec, natom, 3)
    """
    delta = 0.00001

    res = -6 * wfn(r_eI, mo, atoms, shells)
    for j in range(3):
        r_eI[:, :, j] -= delta
        res += wfn(r_eI, mo, atoms, shells)
        r_eI[:, :, j] += 2 * delta
        res += wfn(r_eI, mo, atoms, shells)
        r_eI[:, :, j] -= delta

    return res / delta / delta


@nb.jit(nopython=True, nogil=True, parallel=False)
def wfn_det(r_uI, r_dI, mo_u, mo_d, atoms, shells):
    """Slater determinant without norm factor 1/sqrt(N!).
    """
    u_orb = wfn(r_uI, mo_u, atoms, shells)
    d_orb = wfn(r_dI, mo_d, atoms, shells)
    return np.linalg.det(u_orb) * np.linalg.det(d_orb)


@nb.jit(nopython=True)
def wfn_gradient_log(r_eI, mo, atoms, shells):
    """∇(phi)/phi.
    """
    orb = wfn(r_eI, mo, atoms, shells)
    grad_x, grad_y, grad_z = wfn_gradient(r_eI, mo, atoms, shells)
    cond = np.arange(r_eI.shape[0]) * np.ones(orb.shape)

    res = np.zeros((r_eI.shape[0], 3))
    for i in range(r_eI.shape[0]):
        res[i, 0] = np.linalg.det(np.where(cond == i, grad_x, orb))
        res[i, 1] = np.linalg.det(np.where(cond == i, grad_y, orb))
        res[i, 2] = np.linalg.det(np.where(cond == i, grad_z, orb))

    return res / np.linalg.det(orb)


@nb.jit(nopython=True)
def wfn_laplacian_log(r_eI, mo, atoms, shells):
    """∇²(phi)/phi.
    """
    orb = wfn(r_eI, mo, atoms, shells)
    lap = wfn_laplacian(r_eI, mo, atoms, shells)
    cond = np.arange(r_eI.shape[0]) * np.ones(orb.shape)

    res = 0
    for i in range(r_eI.shape[0]):
        res += np.linalg.det(np.where(cond == i, lap, orb))

    return res / np.linalg.det(orb)


@nb.jit(nopython=True)
def F(r_uI, r_dI, mo_u, mo_d, atoms, shells):
    """sum(|Fi|²)"""
    return (np.linalg.norm(wfn_gradient_log(r_uI, mo_u, atoms, shells))**2 + np.linalg.norm(wfn_gradient_log(r_dI, mo_d, atoms, shells)))**2 / 2


@nb.jit(nopython=True)
def T(r_uI, r_dI, mo_u, mo_d, atoms, shells):
    """sum(Ti)"""
    return (
            np.linalg.norm(wfn_gradient_log(r_uI, mo_u, atoms, shells))**2 - wfn_laplacian_log(r_uI, mo_u, atoms, shells) +
            np.linalg.norm(wfn_gradient_log(r_dI, mo_d, atoms, shells))**2 - wfn_laplacian_log(r_dI, mo_d, atoms, shells)
    ) / 4


@nb.jit(nopython=True)
def wfn_kinetic(r_uI, r_dI, mo_u, mo_d, atoms, shells):
    """local kinetic energy on the point.
    -1/2 * ∇²(phi) / phi
    """
    return 2 * T(r_uI, r_dI, mo_u, mo_d, atoms, shells) - F(r_uI, r_dI, mo_u, mo_d, atoms, shells)


@nb.jit(nopython=True, nogil=True, parallel=False)
def integral(low, high, neu, ned, steps, mo_u, mo_d, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, atomic_positions):
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
        r_uI = subtract_outer(X_u, atomic_positions)
        r_dI = subtract_outer(X_d, atomic_positions)
        result += jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, X_u, X_d, atoms) * wfn_det(r_uI, r_dI, mo_u, mo_d, atoms, shells) ** 2

    return result * dV / gamma(neu+1) / gamma(ned+1)


@nb.jit(nopython=True, nogil=True, parallel=True)
def p_integral(low, high, neu, ned, steps, mo_u, mo_d, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, atomic_positions):
    res = 0.0
    for i in nb.prange(4):
        res += integral(low, high, neu, ned, steps, mo_u, mo_d, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, atomic_positions)
    return res / 4


def main(mo_up, mo_down, neu, ned, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    steps = 10 * 1024 * 1024
    offset = 3.0

    low = np.min(atoms['position'], axis=0) - offset
    high = np.max(atoms['position'], axis=0) + offset

    atomic_positions = atoms['position']

    mo_u = mo_up[:neu]
    mo_d = mo_down[:ned]

    return integral(low, high, neu, ned, steps, mo_u, mo_d, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, atomic_positions)


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
    jastrow_data = Jastrow('test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/chi_term/correlation.out.5', wfn_data.atoms)
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
    res = main(
        wfn_data.mo_up, wfn_data.mo_down, input_data.neu, input_data.ned, wfn_data.atoms, wfn_data.shells, jastrow_data.trunc,
        jastrow_data.u_parameters, jastrow_data.u_cutoff,
        jastrow_data.chi_parameters, jastrow_data.chi_cutoff,
        jastrow_data.f_parameters, jastrow_data.f_cutoff
    )
    print(res)
    end = default_timer()
    print(f'total time {end-start}')
