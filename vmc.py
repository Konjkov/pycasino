#!/usr/bin/env python3

from math import sqrt
from random import random, randrange
from timeit import default_timer

import pyblock
import numpy as np
import numba as nb

from gaussian import wfn, local_energy, nuclear_repulsion
from readers.wfn import Gwfn, Stowfn
from readers.input import Input


@nb.jit(nopython=True, cache=True)
def initial_position(ne, atoms):
    """Initial positions of electrons"""
    natoms = atoms.shape[0]
    X = np.zeros((ne, 3))
    for i in range(ne):
        X[i] = atoms[randrange(natoms)].position
    return X + random_normal_step(1.0, ne)


@nb.jit(nopython=True, cache=True)
def optimal_vmc_step(neu, ned):
    """vmc step width """
    return 1 / (neu + ned)


@nb.jit(nopython=True, cache=True)
def random_laplace_step(dX, ne):
    """Random N-dim laplace distributed step"""
    return np.random.laplace(0.0, dX/2.2, ne*3).reshape((ne, 3))


@nb.jit(nopython=True, cache=True)
def random_triangular_step(dX, ne):
    """Random N-dim triangular distributed step"""
    return np.random.triangular(-1.5*dX, 0, 1.5*dX, ne*3).reshape((ne, 3))


@nb.jit(nopython=True, cache=True)
def random_square_step(dX, ne):
    """Random N-dim square distributed step"""
    return np.random.uniform(-dX, dX, ne*3).reshape((ne, 3))


@nb.jit(nopython=True, cache=True)
def random_normal_step(dX, ne):
    """Random normal distributed step"""
    return np.array([np.random.normal(0.0, dX/sqrt(3)) for i in range(ne*3)]).reshape((ne, 3))


random_step = random_normal_step


@nb.jit(nopython=True, cache=True)
def equilibration(steps, dX, X_u, X_d, p, neu, ned, mo_u, mo_d, atoms, shells):
    """VMC equilibration"""
    i = j = 0
    while i < steps:
        new_X_u = X_u + random_step(dX, neu)
        new_X_d = X_d + random_step(dX, ned)
        new_p = wfn(new_X_u, new_X_d, mo_u, mo_d, atoms, shells)
        j += 1
        if (new_p/p)**2 > random():
            X_u, X_d, p = new_X_u, new_X_d, new_p
            i += 1
    return j


@nb.jit(nopython=True, cache=True)
def accumulation(steps, dX, X_u, X_d, p, neu, ned, mo_u, mo_d, atoms, shells):
    """VMC simple accumulation"""
    j = 0
    E = np.zeros((steps,))
    while j < steps:
        new_X_u = X_u + random_step(dX, neu)
        new_X_d = X_d + random_step(dX, ned)
        new_p = wfn(new_X_u, new_X_d, mo_u, mo_d, atoms, shells)
        if (new_p/p)**2 > random():
            X_u, X_d, p = new_X_u, new_X_d, new_p
            E[j] = local_energy(X_u, X_d, mo_u, mo_d, atoms, shells)
            j += 1
    return E


@nb.jit(nopython=True, cache=True)
def averaging_accumulation(steps, dX, X_u, X_d, p, neu, ned, mo_u, mo_d, atoms, shells):
    """VMC accumulation with averaging local energies over proposed moves"""
    E = np.zeros((steps,))
    loc_E = local_energy(X_u, X_d, mo_u, mo_d, atoms, shells)
    for j in range(steps):
        new_X_u = X_u + random_step(dX, neu)
        new_X_d = X_d + random_step(dX, ned)
        new_p = wfn(new_X_u, new_X_d, mo_u, mo_d, atoms, shells)
        new_loc_E = local_energy(new_X_u, new_X_d, mo_u, mo_d, atoms, shells)
        E[j] = min((new_p/p)**2, 1) * new_loc_E + (1 - min((new_p/p)**2, 1)) * loc_E
        if (new_p/p)**2 > random():
            X_u, X_d, p, loc_E = new_X_u, new_X_d, new_p, new_loc_E
    return E


@nb.jit(nopython=True, cache=True)
def vmc(equlib, stat, mo_up, mo_down, neu, ned, atoms, shells):
    """configuration-by-configuration sampling (CBCS)"""

    dX = optimal_vmc_step(neu, ned)

    mo_u = mo_up[:neu]
    mo_d = mo_down[:ned]

    X_u = initial_position(neu, atoms)
    X_d = initial_position(ned, atoms)
    p = wfn(X_u, X_d, mo_u, mo_d, atoms, shells)

    equ = equilibration(equlib, dX, X_u, X_d, p, neu, ned, mo_u, mo_d, atoms, shells)
    print(equlib/equ)

    opt = equilibration(10000, dX, X_u, X_d, p, neu, ned, mo_u, mo_d, atoms, shells)
    print(10000/opt)

    return accumulation(stat, dX, X_u, X_d, p, neu, ned, mo_u, mo_d, atoms, shells)


if __name__ == '__main__':
    """
    be HF/cc-pVQZ

    stat = 1024 * 1024 * 1024

    """

    # gwfn_data = Gwfn('test/gwfn/h/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/h/HF/cc-pVQZ/input')
    # gwfn_data = Gwfn('test/gwfn/he/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/he/HF/cc-pVQZ/input')
    # gwfn_data = Gwfn('test/gwfn/be/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/be/HF/cc-pVQZ/input')
    gwfn_data = Gwfn('test/gwfn/b/HF/cc-pVQZ/gwfn.data')
    input_data = Input('test/gwfn/b/HF/cc-pVQZ/input')
    # gwfn_data = Gwfn('test/gwfn/be2/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/be2/HF/cc-pVQZ/input')
    # gwfn_data = Gwfn('test/gwfn/acetic/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/acetic/HF/cc-pVQZ/input')
    # gwfn_data = Gwfn('test/gwfn/acetaldehyde/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/acetaldehyde/HF/cc-pVQZ/input')
    # gwfn_data = Gwfn('test/gwfn/si2h6/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/si2h6/HF/cc-pVQZ/input')
    # gwfn_data = Gwfn('test/gwfn/s4-c2v/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/s4-c2v/HF/cc-pVQZ/input')

    start = default_timer()
    E = vmc(50000, 16 * 1024 * 1024, gwfn_data.mo_up, gwfn_data.mo_down, input_data.neu, input_data.ned, gwfn_data.atoms, gwfn_data.shells)
    end = default_timer()
    reblock_data = pyblock.blocking.reblock(E + nuclear_repulsion(gwfn_data.atoms))
    # for reblock_iter in reblock_data:
    #     print(reblock_iter)
    opt = pyblock.blocking.find_optimal_block(E.size, reblock_data)
    print(reblock_data[opt[0]])
    print(f'total time {end-start}')
