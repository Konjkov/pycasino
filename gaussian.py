#!/usr/bin/env python3

from math import exp, sqrt, gamma

import numpy as np
import numba as nb

from readers.gwfn import Gwfn
from readers.input import Input
from utils import uniform


@nb.jit(nopython=True, cache=True)
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
        result[1] += radial * x * z
        result[2] += radial * y * z
        result[3] += radial * (x * x - y * y)
        result[4] += radial * x * y
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


@nb.jit(nopython=True, cache=True)
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


@nb.jit(nopython=True, cache=True)
def wfn_det(r, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """
    Slater determinant
    :param r: electrons coordinates shape = (nelec, 3)
    :param mo: MO-coefficients shape = (nbasis_functions, nelec)
    :param nshell:
    :param shell_types: l-number of the shell shape = (nshell, )
    :param shell_positions: centerd position of the shell shape = (nshell, 3)
    :param primitives: number of primitives on each shell shape = (nshell,)
    :param contraction_coefficients: contraction coefficients of a primitives shape = (nprimitives,)
    :param exponents: exponents of a primitives shape = (nprimitives,)
    :return: slater determinant shape = (nelec, nelec)
    """
    orbital = np.zeros(mo.shape)
    rI = np.zeros((3,))
    for i in range(mo.shape[0]):
        ao = 0
        p = 0
        for shell in range(nshell):
            for j in range(3):
                rI[j] = r[i, j] - shell_positions[shell][j]
            # angular momentum
            l = shell_types[shell]
            # radial part
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            radial_part = 0.0
            for primitive in range(p, p + primitives[shell]):
                radial_part += contraction_coefficients[primitive] * np.exp(-exponents[primitive] * r2)  # 20s from 60s
            p += primitives[shell]
            # angular part
            angular_part(rI, l, orbital[i, ao: ao+2*l+1], radial_part)  # 10s from 60s
            ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True, cache=True)
def gradient_det(r, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Orbital coefficients for every AO at electron position r."""
    orbital = np.zeros(mo.shape)
    rI = np.zeros((3,))
    for i in range(mo.shape[0]):
        ao = 0
        p = 0
        for shell in range(nshell):
            for j in range(3):
                rI[j] = r[i, j] - shell_positions[shell][j]
            # angular momentum
            l = shell_types[shell]
            # radial part
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            grad_r = rI[0] + rI[1] + rI[2]
            radial_part_1 = 0.0
            radial_part_2 = 0.0
            for primitive in range(p, p + primitives[shell]):
                alpha = exponents[primitive]
                exponent = np.exp(-alpha * r2)
                radial_part_1 += 2 * alpha * grad_r * contraction_coefficients[primitive] * exponent   # 20s from 60s
                radial_part_2 += exponent
            p += primitives[shell]
            # angular part
            angular_part(rI, l, orbital[i, ao: ao+2*l+1], radial_part_1)  # 10s from 60s
            gradient_angular_part(rI, l, orbital[i, ao: ao+2*l+1], radial_part_2)  # 10s from 60s
            ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True, cache=True)
def laplacian_det(r, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Orbital coefficients for every AO at electron position r."""
    orbital = np.zeros(mo.shape)
    rI = np.zeros((3,))
    for i in range(mo.shape[0]):
        ao = 0
        p = 0
        for shell in range(nshell):
            for j in range(3):
                rI[j] = r[i, j] - shell_positions[shell][j]
            # angular momentum
            l = shell_types[shell]
            # radial part
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            radial_part = 0.0
            for primitive in range(p, p + primitives[shell]):
                alpha = exponents[primitive]
                radial_part += 2 * alpha * (2 * alpha * r2 - 2 * l - 3) * contraction_coefficients[primitive] * np.exp(-alpha * r2)  # 20s from 60s
            p += primitives[shell]
            # angular part
            angular_part(rI, l, orbital[i, ao: ao+2*l+1], radial_part)  # 10s from 60s
            ao += 2*l+1
    return np.dot(mo, orbital.T)


@nb.jit(nopython=True, cache=True)
def wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """all electron wfn on the points without norm factor 1/sqrt(N!).

    Cauchy–Binet formula
    """
    u_orb = wfn_det(r_u, mo_u, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    if not mo_u.shape[0]:
        return np.linalg.det(u_orb)
    d_orb = wfn_det(r_d, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    return np.linalg.det(u_orb) * np.linalg.det(d_orb)


@nb.jit(nopython=True, cache=True)
def gradient(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """gradient
    """
    u_orb = wfn_det(r_u, mo_u, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    u_grad = gradient_det(r_u, mo_u, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    if not mo_u.shape[0]:
        return np.linalg.det(u_orb)
    d_orb = wfn_det(r_d, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    d_grad = gradient_det(r_d, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    return np.linalg.det(u_orb) * np.linalg.det(d_orb)


@nb.jit(nopython=True, cache=True)
def numerical_gradient(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Numerical gradient
    :param r_u: up electrons coordinates shape = (nelec, 3)
    :param r_d: down electrons coordinates shape = (nelec, 3)
    """
    delta = 0.001
    sum = 0
    for i in range(r_u.shape[0]):
        for j in range(r_u.shape[1]):
            r_u[i, j] -= delta
            sum -= wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
            r_u[i, j] += 2 * delta
            sum += wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
            r_u[i, j] -= delta
    for i in range(r_d.shape[0]):
        for j in range(r_d.shape[1]):
            r_d[i, j] -= delta
            sum -= wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
            r_d[i, j] += 2 * delta
            sum += wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
            r_d[i, j] -= delta
    return sum / delta


@nb.jit(nopython=True, cache=True)
def numerical_laplacian(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Numerical gradient
    :param r_u: up electrons coordinates shape = (nelec, 3)
    :param r_d: down electrons coordinates shape = (nelec, 3)
    """
    delta = 0.001
    sum = -2 * (r_u.shape[0] * r_u.shape[1] + r_d.shape[0] * r_d.shape[1]) * wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    for i in range(r_u.shape[0]):
        for j in range(r_u.shape[1]):
            r_u[i, j] -= delta
            sum += wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
            r_u[i, j] += 2 * delta
            sum += wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
            r_u[i, j] -= delta
    for i in range(r_d.shape[0]):
        for j in range(r_d.shape[1]):
            r_d[i, j] -= delta
            sum += wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
            r_d[i, j] += 2 * delta
            sum += wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
            r_d[i, j] -= delta
    return sum / delta


@nb.jit(nopython=True, cache=True)
def kinetic(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """single electron kinetic energy on the point.

    laplacian(phi) / 2

    param r: coordinat
    param mo: MO
    """
    return numerical_laplacian(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents) / 2


@nb.jit(nopython=True, cache=True)
def coulomb(r_u, r_d, atomic_positions, atom_charges):
    """Coulomb attraction between the electron and nucleus."""
    res = 0.0
    for atom in range(r_u.shape[0]):
        charge = atom_charges[atom]
        I = atomic_positions[atom]
        x = r_u[0] - I[0]
        y = r_u[1] - I[1]
        z = r_u[2] - I[2]
        r2 = x * x + y * y + z * z
        res += charge / sqrt(r2)

    for atom in range(r_d.shape[0]):
        charge = atom_charges[atom]
        I = atomic_positions[atom]
        x = r_d[0] - I[0]
        y = r_d[1] - I[1]
        z = r_d[2] - I[2]
        r2 = x * x + y * y + z * z
        res += charge / sqrt(r2)

    return -res


@nb.jit(nopython=True, cache=True)
def local_energy(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, atomic_positions, atom_charges):
    return coulomb(r_u, r_d, atomic_positions, atom_charges) + kinetic(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents) / wfn(r_u, r_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)


@nb.jit(nopython=True, cache=True)
def vmc(equlib, stat, mo, neu, ned, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, atomic_positions, atom_charges):

    dX_max = 0.4

    mo_u = mo[0][:neu]
    mo_d = mo[0][:ned]

    X_u = uniform(-dX_max, dX_max, (neu, 3))
    X_d = uniform(-dX_max, dX_max, (ned, 3))
    p = wfn(X_u, X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    for i in range(equlib):
        new_X_u = X_u + np.random.uniform(-dX_max, dX_max, size=3)
        new_X_d = X_d + np.random.uniform(-dX_max, dX_max, size=3)
        new_p = wfn(new_X_u, new_X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
        if new_p*new_p/p/p > np.random.random_sample(1)[0]:
            X_u, X_d, p = new_X_u, new_X_d, new_p

    j = 0
    sum = 0.0
    for dX in range(stat):
        new_X_u = X_u + np.random.uniform(-dX_max, dX_max, size=3)
        new_X_d = X_d + np.random.uniform(-dX_max, dX_max, size=3)
        new_p = wfn(X_u, X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
        if (new_p/p)**2 > np.random.random_sample(1)[0]:
            X_u, X_d, p = new_X_u, new_X_d, new_p
            j += 1
            sum += local_energy(X_u, X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, atomic_positions, atom_charges)
    return sum/j


@nb.jit(nopython=True, cache=True)
def main(mo, neu, ned, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    steps = 10 * 1000 * 1000
    offset = 3.0

    x_min = np.min(shell_positions[:, 0]) - offset
    y_min = np.min(shell_positions[:, 1]) - offset
    z_min = np.min(shell_positions[:, 2]) - offset
    x_max = np.max(shell_positions[:, 0]) + offset
    y_max = np.max(shell_positions[:, 1]) + offset
    z_max = np.max(shell_positions[:, 2]) + offset
    low = np.array([x_min, y_min, z_min])
    high = np.array([x_max, y_max, z_max])

    dV = (x_max - x_min)**(neu + ned) * (y_max - y_min)**(neu + ned) * (z_max - z_min)**(neu + ned) / steps

    mo_u = mo[0][:neu]
    mo_d = mo[0][:ned]

    integral = 0.0
    for i in range(steps):
        X_u = uniform(low, high, (neu, 3))
        X_d = uniform(low, high, (ned, 3))
        integral += wfn(X_u, X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents) ** 2

    return integral * dV / gamma(neu+1) / gamma(ned+1)


if __name__ == '__main__':
    """
    be HFcc-pVQZ

    steps = 5 * 1000 * 1000 * 1000
    offset = 4.5

    0.925763438273841

    real    464m11,715s
    user    464m11,194s
    sys     0m0,488s
    """

    # gwfn = Gwfn('test/h/HF/cc-pVQZ/gwfn.data')
    # input = Input('test/h/HF/cc-pVQZ/input')
    gwfn = Gwfn('test/be/HF/cc-pVQZ/gwfn.data')
    input = Input('test/be/HF/cc-pVQZ/input')
    # gwfn = Gwfn('test/be2/HF/cc-pVQZ/gwfn.data')
    # input = Input('test/be2/HF/cc-pVQZ/input')
    # gwfn = Gwfn('test/acetic/HF/cc-pVQZ/gwfn.data')
    # input = Input('test/acetic/HF/cc-pVQZ/input')
    # gwfn = Gwfn('test/acetaldehyde/HF/cc-pVQZ/gwfn.data')
    # input = Input('test/acetaldehyde/HF/cc-pVQZ/input')

    print(main(gwfn.mo, input.neu, input.ned, gwfn.nshell, gwfn.shell_types, gwfn.shell_positions, gwfn.primitives, gwfn.contraction_coefficients, gwfn.exponents))
