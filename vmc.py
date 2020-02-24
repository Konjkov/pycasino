#!/usr/bin/env python3

import pyblock
from random import random

import numpy as np
import numba as nb

from gaussian import wfn, local_energy, nuclear_repulsion
from readers.gwfn import Gwfn
from readers.input import Input


@nb.jit(nopython=True, cache=True)
def random_step(dX_max, ne):
    low = -np.array([dX_max, dX_max, dX_max])
    high = np.array([dX_max, dX_max, dX_max])
    return np.dstack((np.random.uniform(low[0], high[0], size=ne), np.random.uniform(low[1], high[1], size=ne), np.random.uniform(low[2], high[2], size=ne)))[0]


@nb.jit(nopython=True, cache=True)
def vmc(equlib, stat, mo, neu, ned, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, atomic_positions, atom_charges):

    dX_max = 0.2

    mo_u = mo[0][:neu]
    mo_d = mo[0][:ned]

    X_u = random_step(dX_max, neu)
    X_d = random_step(dX_max, ned)
    p = wfn(X_u, X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    for i in range(equlib):
        new_X_u = X_u + random_step(dX_max, neu)
        new_X_d = X_d + random_step(dX_max, neu)
        new_p = wfn(new_X_u, new_X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
        if (new_p/p)**2 > random():
            X_u, X_d, p = new_X_u, new_X_d, new_p

    j = 0
    E = np.zeros((stat,))
    while j < stat:
        new_X_u = X_u + random_step(dX_max, neu)
        new_X_d = X_d + random_step(dX_max, neu)
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

    E = vmc(5000, 10 * 1000 * 1000, gwfn.mo, inp.neu, inp.ned, gwfn.nshell, gwfn.shell_types, gwfn.shell_positions, gwfn.primitives, gwfn.contraction_coefficients, gwfn.exponents, gwfn.atomic_positions, gwfn.atom_charges)
    print(np.mean(E) + nuclear_repulsion( gwfn.atomic_positions, gwfn.atom_charges))

    reblock_data = pyblock.blocking.reblock(E)
    for reblock_iter in reblock_data:
        print(reblock_iter)

    opt = pyblock.blocking.find_optimal_block(E.size, reblock_data)
    print(opt)
    print(reblock_data[opt[0]])
