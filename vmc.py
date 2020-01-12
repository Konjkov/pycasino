#!/usr/bin/env python3

import numpy as np
import numba as nb

from wfn import wfn, local_energy
from readers.gwfn import Gwfn


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


if __name__ == '__main__':

    gwfn = Gwfn('test/h/HF/cc-pVQZ/gwfn.data')
    mo = gwfn.mo[0, 0]

    print(vmc(5000, 10000000, mo, gwfn.nshell, gwfn.shell_types, gwfn.shell_positions, gwfn.primitives, gwfn.contraction_coefficients, gwfn.exponents, gwfn.natom, gwfn.atomic_positions, gwfn.atom_charges))
