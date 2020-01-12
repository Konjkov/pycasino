#!/usr/bin/env python3

from math import exp

import numpy as np
import numba as nb

from readers.gwfn import Gwfn


@nb.jit(nopython=True, cache=True)
def wfn_angular_part(x, y, z, l, m, r2):
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
            return x*x + y*y
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
def laplacian_angular_part(x, y, z, l, m, r2):
    """Angular part of laplacian WFN.
    :return:
    """
    if l == 0:
        return 2 * (2 * r2 - 3)
    elif l == 1:
        if m == 0:
            return 2 * x * (2 * r2 - 5)
        elif m == 1:
            return 2 * y * (2 * r2 - 5)
        elif m == 2:
            return 2 * z * (2 * r2 - 5)
    elif l == 2:
        if m == 0:
            return 4 * x**2 * z**2 + 4 * y**2 * z**2 + 8 * z**4 + 14 * x**2 + 14 * y**2 - 28 * z**2 - (4 * x**4 + 8 * x**2 * y**2 + 4 * y**4)
        if m == 1:
            return 2 * x * z * (2 * r2 - 7)
        if m == 2:
            return 2 * y * z * (2 * r2 - 7)
        if m == 3:
            return 2 * (2*x**4 + 2*x**2*z**2 - 2*y**4 - 2*y**2*z**2 - 7*x**2 + 7*y**2)
        if m == 4:
            return 2 * x * y * (2 * r2 - 7)


@nb.jit(nopython=True, cache=True)
def wfn(r, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """single electron wfn on the point.

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
        # radial part
        r2 = x * x + y * y + z * z
        prim_sum = 0.0
        for primitive in range(p, p + primitives[shell]):
            prim_sum += contraction_coefficients[primitive] * exp(-exponents[primitive] * r2)
        p += primitives[shell]
        # angular part
        l = shell_types[shell]
        for m in range(2*l+1):
            if True:
                angular = wfn_angular_part(x, y, z, l, m, r2)
            else:
                angular = laplacian_angular_part(x, y, z, l, m, r2) * exponents[primitive] * exponents[primitive]
            res += prim_sum * angular * mo[ao]
            ao += 1
    return res


def vmc(equlib, stat, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
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
            sum += wfn.local_energy(X)
    return sum/j


if __name__ == '__main__':

    # gwfn = Gwfn('test/be/HF/cc-pVQZ/gwfn.data')
    gwfn = Gwfn('test/acetic/HF/cc-pVQZ/gwfn.data')
    # gwfn = Gwfn('test/acetaldehyde/HF/cc-pVQZ/gwfn.data')

    mo = gwfn.mo[0, 0]

    steps = 140
    l = 10.0

    x_steps = y_steps = z_steps = steps
    x_min = y_min = z_min = -l
    x_max = y_max = z_max = l

    dV = 2 * l / (steps - 1) * 2 * l / (steps - 1) * 2 * l / (steps - 1)

    x = np.linspace(x_min, x_max, x_steps)
    y = np.linspace(y_min, y_max, y_steps)
    z = np.linspace(z_min, z_max, z_steps)

    grid = np.vstack(np.meshgrid(x, y, z)).reshape(3, -1).T

    integral = sum(wfn(r, mo, gwfn.nshell, gwfn.shell_types, gwfn.shell_positions, gwfn.primitives, gwfn.contraction_coefficients, gwfn.exponents) ** 2 for r in grid) * dV

    print(integral)

    # print(gwfn.vmc(500, 500000))
