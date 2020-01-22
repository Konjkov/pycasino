#!/usr/bin/env python3

from math import exp, sqrt, factorial

import numpy as np
import numba as nb

from readers.gwfn import Gwfn
from readers.input import Input


@nb.jit(nopython=True, cache=True)
def angular_part(x, y, z, l, m, r2):
    """Angular part of gaussian WFN.
    :return:
    """
    if l == 0:
        return 1
    elif l == 1:
        if m == 0:
            return x
        elif m == 1:
            return y
        elif m == 2:
            return z
    elif l == 2:
        if m == 0:
            return 3 * z*z - r2
        elif m == 1:
            return x * z
        elif m == 2:
            return y * z
        elif m == 3:
            return x*x - y*y
        elif m == 4:
            return x*y
    elif l == 3:
        if m == 0:
            return z * (5 * z*z - 3 * r2) / 2
        if m == 1:
            return 3 * x * (5 * z*z - r2) / 2
        if m == 2:
            return 3 * y * (5 * z*z - r2) / 2
        if m == 3:
            return 15 * z * (x*x - y*y)
        if m == 4:
            return 30 * x*y*z
        if m == 5:
            return 15 * x * (x*x - 3 * y*y)
        if m == 6:
            return 15 * y * (3 * x*x - y*y)
    elif l == 4:
        if m == 0:
            return (35 * z*z*z*z - 30 * z*z * r2 + 3 * r2 * r2) / 8
        if m == 1:
            return 5 * x*z * (7 * z*z - 3 * r2) / 2
        if m == 2:
            return 5 * y*z * (7 * z*z - 3 * r2) / 2
        if m == 3:
            return 15 * (x*x - y*y) * (7 * z*z - r2) / 2
        if m == 4:
            return 30 * x*y * (7 * z*z - r2) / 2
        if m == 5:
            return 105 * x*z * (x*x - 3 * y*y)
        if m == 6:
            return 105 * y*z * (3 * x*x - y*y)
        if m == 7:
            return 105 * (x*x*x*x - 6 * x*x*y*y + y*y*y*y)
        if m == 8:
            return 420 * x*y * (x*x - y*y)
    return 0


@nb.jit(nopython=True, cache=True)
def orbitals(r, neu, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Orbital coefficients for every AO at electron position r."""
    res = np.zeros((nbasis_functions, neu))
    for i in range(neu):
        ao = 0
        p = 0
        rI = np.zeros((3,))
        for shell in range(nshell):
            for j in range(3):
                rI[j] = r[i, j] - shell_positions[shell][j]
            # angular momentum
            l = shell_types[shell]
            # radial part
            r2 = rI[0] * rI[0] + rI[1] * rI[1] + rI[2] * rI[2]
            prim_sum = 0.0
            for primitive in range(p, p + primitives[shell]):
                alpha = exponents[primitive]
                prim_sum += contraction_coefficients[primitive] * exp(-alpha * r2)
            p += primitives[shell]
            # angular part
            for m in range(2*l+1):
                angular = angular_part(rI[0], rI[1], rI[2], l, m, r2)
                res[ao, i] = prim_sum * angular
                ao += 1
    return res


# @nb.jit(nopython=True, cache=True)
def wfn(r, mo, neu, ned, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """all electron wfn on the points without norm factor 1/sqrt(N!).

    param r: coordinates

    Cauchy–Binet formula

    """
    orb = orbitals(r[:neu], neu, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    u_orb = np.einsum('ij,jk', mo[0][:input.neu], orb)
    res = np.linalg.det(u_orb)
    if not ned:
        return res
    orb = orbitals(r[neu:], ned, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    d_orb = np.einsum('ij,jk', mo[0][:input.ned], orb)
    return res * np.linalg.det(d_orb)


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


# @nb.jit(nopython=True, cache=True)
def main(mo, neu, ned, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    steps = 1000 * 1000
    offset = 3.5

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
        X = np.random.uniform(low, high, (neu + ned, 3))
        integral += wfn(X, mo, neu, ned, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents) ** 2

    return integral * dV / factorial(neu) / factorial(ned)


if __name__ == '__main__':
    """
    be HFcc-pVQZ

    steps = 1000 * 1000 * 500
    offset = 3.5

    0.7789653396996061

    real    364m31,210s
    user    364m31,430s
    sys     0m0,636s
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
