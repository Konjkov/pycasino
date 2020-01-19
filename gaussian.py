#!/usr/bin/env python3

from math import exp, sqrt
from random import uniform

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
def orbitals(r, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """Orbital coefficients for every AO at electron position r."""
    res = np.zeros((nbasis_functions,))
    ao = 0
    p = 0
    rI = np.zeros((3,))
    for shell in range(nshell):
        for i in range(3):
            rI[i] = r[i] - shell_positions[shell][i]
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
            res[ao] = prim_sum * angular
            ao += 1
    return res


@nb.jit(nopython=True, cache=True)
def wfn(r, mo, neu, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """single electron wfn on the point.

    param r: coordinat
    param mo: MO
    """
    orb = orbitals(r, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    # return np.einsum('ij,j', mo, orb)  # not supported
    res = np.zeros((neu,))
    for i in range(neu):
        for j in range(nbasis_functions):
            res[i] += mo[i, j] * orb[j]
    return res


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
def main(mo, neu, nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    steps = 10 * 1000 * 1000
    offset = 5.0

    x_min = np.min(shell_positions[:, 0]) - offset
    y_min = np.min(shell_positions[:, 1]) - offset
    z_min = np.min(shell_positions[:, 2]) - offset
    x_max = np.max(shell_positions[:, 0]) + offset
    y_max = np.max(shell_positions[:, 1]) + offset
    z_max = np.max(shell_positions[:, 2]) + offset
    # not supported
    # low = np.array([x_min, y_min, z_min])
    # high = np.array([x_max, y_max, z_max])

    dV = (x_max - x_min) * (y_max - y_min) * (z_max - z_min) / steps
    integral = np.zeros((neu,))
    for i in range(steps):
        # X = np.random.uniform(low, high)
        X = np.array([uniform(x_min, x_max), uniform(y_min, y_max), uniform(z_min, z_max)])
        integral += wfn(X, mo, neu,  nbasis_functions, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents) ** 2

    return integral * dV


if __name__ == '__main__':
    """
    0.999980184928134

    real    75m40,030s
    user    75m40,454s
    sys     0m0,656s

    """

    # gwfn = Gwfn('test/be/HF/cc-pVQZ/gwfn.data')
    # input = Input('test/be/HF/cc-pVQZ/input')
    gwfn = Gwfn('test/acetic/HF/cc-pVQZ/gwfn.data')
    input = Input('test/acetic/HF/cc-pVQZ/input')
    # gwfn = Gwfn('test/acetaldehyde/HF/cc-pVQZ/gwfn.data')
    # input = Input('test/acetaldehyde/HF/cc-pVQZ/input')

    mo = gwfn.mo[0]

    print(main(mo, input.neu, gwfn.nbasis_functions, gwfn.nshell, gwfn.shell_types, gwfn.shell_positions, gwfn.primitives, gwfn.contraction_coefficients, gwfn.exponents))
