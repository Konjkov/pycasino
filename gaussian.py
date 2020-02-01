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
def wfn_det(r, ne, mo, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Orbital coefficients for every AO at electron position r."""
    orbital = np.zeros((nbasis_functions, ne))
    rI = np.zeros((3,))
    for i in range(ne):
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
            angular_part(rI, l, orbital[ao: ao+2*l+1, i], radial_part)  # 10s from 60s
            ao += 2*l+1
    return np.linalg.det(np.dot(mo, orbital))   # 9s from 60s


@nb.jit(nopython=True, cache=True)
def gradient_det(r, ne, mo, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Orbital coefficients for every AO at electron position r."""
    orbital = np.zeros((nbasis_functions, ne))
    rI = np.zeros((3,))
    for i in range(ne):
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
            angular_part(rI, l, orbital[ao: ao+2*l+1, i], radial_part_1)  # 10s from 60s
            gradient_angular_part(rI, l, orbital[ao: ao+2*l+1, i], radial_part_2)  # 10s from 60s
            ao += 2*l+1
    return np.linalg.det(np.dot(mo, orbital))   # 9s from 60s


@nb.jit(nopython=True, cache=True)
def laplacian_det(r, ne, mo, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Orbital coefficients for every AO at electron position r."""
    orbital = np.zeros((nbasis_functions, ne))
    rI = np.zeros((3,))
    for i in range(ne):
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
            angular_part(rI, l, orbital[ao: ao+2*l+1, i], radial_part)  # 10s from 60s
            ao += 2*l+1
    return np.linalg.det(np.dot(mo, orbital))   # 9s from 60s


@nb.jit(nopython=True, cache=True)
def wfn(r, mo, neu, ned, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """all electron wfn on the points without norm factor 1/sqrt(N!).

    param r: coordinates

    Cauchy–Binet formula

    """
    u_orb = wfn_det(r[:neu], neu, mo[0][:neu], nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    if not ned:
        return u_orb
    d_orb = wfn_det(r[neu:], ned, mo[0][:ned], nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    return u_orb * d_orb


def Fi(r, mo, neu, ned, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Kinetic energy Fi
    https://web.ornl.gov/~kentpr/thesis/pkthnode27.html
    """
    u_orb = wfn_det(r[:neu], neu, mo[0][:neu], nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    u_gradient = gradient_det(r[:neu], neu, mo[0][:neu], nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    if not ned:
        return u_orb / u_gradient
    d_orb = wfn_det(r[neu:], ned, mo[0][:ned], nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    d_gradient = gradient_det(r[neu:], ned, mo[0][:ned], nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    return u_orb / u_gradient + d_orb  / d_gradient


def Ti(r, mo, neu, ned, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Kinetic energy Ti
    https://web.ornl.gov/~kentpr/thesis/pkthnode27.html
    """
    u_orb = wfn_det(r[:neu], neu, mo[0][:neu], nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    u_laplacian = laplacian_det(r[:neu], neu, mo[0][:neu], nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    if not ned:
        return u_orb
    d_orb = wfn_det(r[neu:], ned, mo[0][:ned], nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    d_laplacian = laplacian_det(r[neu:], ned, mo[0][:ned], nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    return u_orb / u_laplacian * d_orb / d_laplacian


@nb.jit(nopython=True, cache=True)
def kinetic(r, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """single electron kinetic energy on the point.

    laplacian(phi) / 2

    param r: coordinat
    param mo: MO
    """

    res = 0.0
    ao = 0
    p = 0
    for shell in range(nshell):
        I = shell_positions[shell]
        x = r[0] - I[0]
        y = r[1] - I[1]
        z = r[2] - I[2]
        # angular momentum
        l = shell_types[shell]
        # radial part
        r2 = x * x + y * y + z * z
        prim_sum = 0.0
        for primitive in range(p, p + primitives[shell]):
            alpha = exponents[primitive]
            prim_sum += contraction_coefficients[primitive] * exp(-alpha * r2) * alpha * (2 * alpha * r2 - 2 * l - 3)
        p += primitives[shell]
        # angular part
        for m in range(2*l+1):
            angular = angular_part(x, y, z, l, m, r2)
            res += prim_sum * angular * mo[ao]
            ao += 1
    return -res


@nb.jit(nopython=True, cache=True)
def coulomb(r, natom, atomic_positions, atom_charges):
    """Coulomb attraction between the electron and nucleus."""
    res = 0.0
    for atom in range(natom):
        charge = atom_charges[atom]
        I = atomic_positions[atom]
        x = r[0] - I[0]
        y = r[1] - I[1]
        z = r[2] - I[2]
        r2 = x * x + y * y + z * z
        res += charge / sqrt(r2)
    return -res


@nb.jit(nopython=True, cache=True)
def local_energy(r, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, natom, atomic_positions, atom_charges):
    return coulomb(r, natom, atomic_positions, atom_charges) + kinetic(r, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents) / wfn(r, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)


@nb.jit(nopython=True, cache=True)
def vmc(equlib, stat, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, natom, atomic_positions, atom_charges):
    dX_max = 0.4
    X = np.random.uniform(-dX_max, dX_max, size=3)
    p = wfn(X, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    for i in range(equlib):
        new_X = X + np.random.uniform(-dX_max, dX_max, size=3)
        new_p = wfn(new_X, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
        if new_p*new_p/p/p > np.random.random_sample(1)[0]:
            X, p = new_X, new_p

    j = 0
    sum = 0.0
    for dX in range(stat):
        new_X = X + np.random.uniform(-dX_max, dX_max, size=3)
        new_p = wfn(new_X, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
        if (new_p/p)**2 > np.random.random_sample(1)[0]:
            X, p = new_X, new_p
            j += 1
            sum += local_energy(X, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, natom, atomic_positions, atom_charges)
    return sum/j


@nb.jit(nopython=True, cache=True)
def main(mo, neu, ned, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    steps = 5 * 1000 * 1000 * 1000
    offset = 4.5

    x_min = np.min(shell_positions[:, 0]) - offset
    y_min = np.min(shell_positions[:, 1]) - offset
    z_min = np.min(shell_positions[:, 2]) - offset
    x_max = np.max(shell_positions[:, 0]) + offset
    y_max = np.max(shell_positions[:, 1]) + offset
    z_max = np.max(shell_positions[:, 2]) + offset
    low = np.array([x_min, y_min, z_min])
    high = np.array([x_max, y_max, z_max])

    dV = (x_max - x_min)**(neu + ned) * (y_max - y_min)**(neu + ned) * (z_max - z_min)**(neu + ned) / steps

    integral = 0.0
    for i in range(steps):
        X = uniform(low, high, (neu + ned, 3))
        integral += wfn(X, mo, neu, ned, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents) ** 2

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

    print(main(gwfn.mo, input.neu, input.ned, gwfn.nbasis_functions, gwfn.nshell, gwfn.shell_types, gwfn.shell_positions, gwfn.primitives, gwfn.contraction_coefficients, gwfn.exponents))
