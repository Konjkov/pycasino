#!/usr/bin/env python3

import pyblock
from math import sqrt
from random import random

import numpy as np
import numba as nb

from gaussian import wfn, local_energy
from readers.gwfn import Gwfn
from readers.input import Input
from utils import uniform


@nb.jit(nopython=True, cache=True)
def nuclear_repulsion(atomic_positions, atom_charges):

    res = 0.0
    for i in range(atomic_positions.shape[0]):
        for j in range(i+1, atomic_positions.shape[0]):
            x = atomic_positions[i][0] - atomic_positions[j][0]
            y = atomic_positions[i][1] - atomic_positions[j][1]
            z = atomic_positions[i][2] - atomic_positions[j][2]
            r2 = x * x + y * y + z * z
            res += atom_charges[i] * atom_charges[j] / sqrt(r2)
    return res


@nb.jit(nopython=True, cache=True)
def vmc(equlib, stat, mo, neu, ned, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, atomic_positions, atom_charges):

    dX_max = 0.2

    mo_u = mo[0][:neu]
    mo_d = mo[0][:ned]

    X_u = uniform(-np.array([dX_max, dX_max, dX_max]), np.array([dX_max, dX_max, dX_max]), (neu, 3))
    X_d = uniform(-np.array([dX_max, dX_max, dX_max]), np.array([dX_max, dX_max, dX_max]), (ned, 3))
    p = wfn(X_u, X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    for i in range(equlib):
        new_X_u = X_u + uniform(-np.array([dX_max, dX_max, dX_max]), np.array([dX_max, dX_max, dX_max]), (neu, 3))
        new_X_d = X_d + uniform(-np.array([dX_max, dX_max, dX_max]), np.array([dX_max, dX_max, dX_max]), (ned, 3))
        new_p = wfn(new_X_u, new_X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
        if (new_p/p)**2 > random():
            X_u, X_d, p = new_X_u, new_X_d, new_p

    j = 0
    E = np.zeros((stat,))
    while j < stat:
        new_X_u = X_u + uniform(-np.array([dX_max, dX_max, dX_max]), np.array([dX_max, dX_max, dX_max]), (neu, 3))
        new_X_d = X_d + uniform(-np.array([dX_max, dX_max, dX_max]), np.array([dX_max, dX_max, dX_max]), (ned, 3))
        new_p = wfn(new_X_u, new_X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
        if (new_p/p)**2 > random():
            X_u, X_d, p = new_X_u, new_X_d, new_p
            E[j] = local_energy(X_u, X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, atomic_positions, atom_charges)
            j += 1
    return E


if __name__ == '__main__':
    """
    be HF/cc-pVQZ

    stat = 1000 * 1000 * 1000

    """

    gwfn = Gwfn('test/be/HF/cc-pVQZ/gwfn.data')
    inp = Input('test/be/HF/cc-pVQZ/input')

    E = vmc(5000, 1000 * 1000 * 1000, gwfn.mo, inp.neu, inp.ned, gwfn.nshell, gwfn.shell_types, gwfn.shell_positions, gwfn.primitives, gwfn.contraction_coefficients, gwfn.exponents, gwfn.atomic_positions, gwfn.atom_charges)
    print(np.mean(E) + nuclear_repulsion( gwfn.atomic_positions, gwfn.atom_charges))

    reblock_data = pyblock.blocking.reblock(E)
    for reblock_iter in reblock_data:
        print(reblock_iter)

    opt = pyblock.blocking.find_optimal_block(E.size, reblock_data)
    print(opt)
    print(reblock_data[opt[0]])
