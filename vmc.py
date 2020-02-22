#!/usr/bin/env python3

from random import random

import numpy as np
import numba as nb

from gaussian import wfn, local_energy
from readers.gwfn import Gwfn
from readers.input import Input
from utils import uniform


@nb.jit(nopython=True, cache=True)
def vmc(equlib, stat, mo, neu, ned, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, atomic_positions, atom_charges):

    dX_max = 0.4

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
    E = 0.0
    while j < stat:
        new_X_u = X_u + uniform(-np.array([dX_max, dX_max, dX_max]), np.array([dX_max, dX_max, dX_max]), (neu, 3))
        new_X_d = X_d + uniform(-np.array([dX_max, dX_max, dX_max]), np.array([dX_max, dX_max, dX_max]), (ned, 3))
        new_p = wfn(new_X_u, new_X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
        if (new_p/p)**2 > random():
            X_u, X_d, p = new_X_u, new_X_d, new_p
            j += 1
            E += local_energy(X_u, X_d, mo_u, mo_d, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents, atomic_positions, atom_charges)
    return E / stat


if __name__ == '__main__':

    gwfn = Gwfn('test/be/HF/cc-pVQZ/gwfn.data')
    inp = Input('test/be/HF/cc-pVQZ/input')

    print(vmc(5000, 1000 * 1000, gwfn.mo, inp.neu, inp.ned, gwfn.nshell, gwfn.shell_types, gwfn.shell_positions, gwfn.primitives, gwfn.contraction_coefficients, gwfn.exponents, gwfn.atomic_positions, gwfn.atom_charges))
