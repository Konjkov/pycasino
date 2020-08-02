#!/usr/bin/env python3

import os
from math import sqrt, pi
from random import random, randrange
from timeit import default_timer
from wfn import wfn, wfn_gradient_log, wfn_laplacian_log, wfn_numerical_gradient, wfn_numerical_laplacian
from jastrow import jastrow, jastrow_gradient, jastrow_laplacian, jastrow_numerical_gradient, jastrow_numerical_laplacian
from coulomb import coulomb, nuclear_repulsion

os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import pyblock
import numpy as np
import numba as nb

from overload import subtract_outer
from readers.wfn import Gwfn, Stowfn
from readers.input import Input
from readers.jastrow import Jastrow


@nb.jit(nopython=True)
def initial_position(ne, atoms):
    """Initial positions of electrons"""
    natoms = atoms.shape[0]
    X = np.zeros((ne, 3))
    for i in range(ne):
        X[i] = atoms[randrange(natoms)].position
    return X + random_normal_step(1.0, ne)


@nb.jit(nopython=True)
def optimal_vmc_step(neu, ned):
    """vmc step width """
    return 1 / (neu + ned)


@nb.jit(nopython=True)
def random_laplace_step(dX, ne):
    """Random N-dim laplace distributed step"""
    return np.random.laplace(0.0, dX/(3*pi/4), ne*3).reshape((ne, 3))


@nb.jit(nopython=True)
def random_triangular_step(dX, ne):
    """Random N-dim triangular distributed step"""
    return np.random.triangular(-1.5*dX, 0, 1.5*dX, ne*3).reshape((ne, 3))


@nb.jit(nopython=True)
def random_square_step(dX, ne):
    """Random N-dim square distributed step"""
    return np.random.uniform(-dX, dX, ne*3).reshape((ne, 3))


@nb.jit(nopython=True)
def random_normal_step(dX, ne):
    """Random normal distributed step"""
    return np.random.normal(0.0, dX/sqrt(3), ne*3).reshape((ne, 3))


@nb.jit(nopython=True)
def random_on_sphere_step(dX, ne):
    """Random on a sphere distributed step"""
    result = []
    for i in range(ne):
        x = np.random.normal(0.0, 1, 3)
        res = dX * x / np.linalg.norm(x)
        result.append(res[0])
        result.append(res[1])
        result.append(res[2])
    return np.array(result).reshape((ne, 3))


random_step = random_normal_step


@nb.jit(nopython=True)
def guiding_function(r_u, r_d, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """wave function in general form"""
    r_uI = subtract_outer(r_u, atomic_positions)
    r_dI = subtract_outer(r_d, atomic_positions)
    return (
        np.exp(jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_u, r_d, atoms)) *
        np.linalg.det(wfn(r_uI, mo_u, atoms, shells)) * np.linalg.det(wfn(r_dI, mo_d, atoms, shells))
    )


@nb.jit(nopython=True)
def local_energy(r_u, r_d, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    r_uI = subtract_outer(r_u, atomic_positions)
    r_dI = subtract_outer(r_d, atomic_positions)
    jg_u, jg_d = jastrow_gradient(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_u, r_d, atoms)
    j_l = jastrow_laplacian(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_u, r_d, atoms)
    wl_u = wfn_laplacian_log(r_uI, mo_u, atoms, shells)
    wl_d = wfn_laplacian_log(r_dI, mo_d, atoms, shells)
    wg_u = wfn_gradient_log(r_uI, mo_u, atoms, shells)
    wg_d = wfn_gradient_log(r_dI, mo_d, atoms, shells)
    F = (np.sum((wg_u + jg_u) * (wg_u + jg_u)) + np.sum((wg_d + jg_d) * (wg_d + jg_d))) / 2
    T = (np.sum(wg_u * wg_u) + np.sum(wg_d * wg_d) - wl_u - wl_d - j_l) / 4
    return coulomb(r_u, r_d, r_uI, r_dI, atoms) + 2 * T - F


@nb.jit(nopython=True)
def equilibration(steps, dX, r_u, r_d, neu, ned, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """VMC equilibration"""
    i = 0
    p = 0.0
    for j in range(steps):
        new_r_u = r_u + random_step(dX, neu)
        new_r_d = r_d + random_step(dX, ned)

        new_p = guiding_function(new_r_u, new_r_d, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
        j += 1
        if new_p**2 > random() * p**2:
            r_u, r_d, p = new_r_u, new_r_d, new_p
            i += 1
    return i


@nb.jit(nopython=True)
def simple_accumulation(steps, dX, r_u, r_d, neu, ned, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """VMC simple accumulation"""
    p = loc_E = 0.0
    E = np.zeros((steps,))
    for j in range(steps):
        new_r_u = r_u + random_step(dX, neu)
        new_r_d = r_d + random_step(dX, ned)

        new_p = guiding_function(new_r_u, new_r_d, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
        if new_p**2 > random() * p**2:
            r_u, r_d, p = new_r_u, new_r_d, new_p
            loc_E = local_energy(r_u, r_d, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
        E[j] = loc_E
    return E


@nb.jit(nopython=True)
def averaging_accumulation(steps, dX, r_u, r_d, p, neu, ned, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """VMC accumulation with averaging local energies over proposed moves"""
    E = np.zeros((steps,))
    loc_E = local_energy(r_u, r_d, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
    for j in range(steps):
        new_r_u = r_u + random_step(dX, neu)
        new_r_d = r_d + random_step(dX, ned)

        new_p = guiding_function(new_r_u, new_r_d, mo_u, mo_d, atoms, shells)
        new_loc_E = local_energy(new_r_u, new_r_d, mo_u, mo_d, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
        E[j] = min((new_p/p)**2, 1) * new_loc_E + (1 - min((new_p/p)**2, 1)) * loc_E
        if (new_p/p)**2 > random():
            r_u, r_d, p, loc_E = new_r_u, new_r_d, new_p, new_loc_E
    return E


accumulation = simple_accumulation


def vmc(equlib, stat, mo_up, mo_down, neu, ned, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """configuration-by-configuration sampling (CBCS)"""

    dX = optimal_vmc_step(neu, ned)

    mo_u = mo_up[:neu]
    mo_d = mo_down[:ned]

    atomic_positions = atoms['position']

    X_u = initial_position(neu, atoms)
    X_d = initial_position(ned, atoms)

    equ = equilibration(equlib, dX, X_u, X_d, neu, ned, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
    print(equ/equlib)

    opt = equilibration(10000, dX, X_u, X_d, neu, ned, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
    print(opt/10000)

    return accumulation(stat, dX, X_u, X_d, neu, ned, mo_u, mo_d, atoms, shells, atomic_positions, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)


if __name__ == '__main__':
    """
    be HF/cc-pVQZ

    stat = 1024 * 1024 * 1024

    """

    # wfn_data = Gwfn('test/gwfn/h/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/h/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/he/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/he/HF/cc-pVQZ/input')
    wfn_data = Gwfn('test/gwfn/be/HF/cc-pVQZ/gwfn.data')
    input_data = Input('test/gwfn/be/HF/cc-pVQZ/input')
    jastrow_data = Jastrow('test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/chi_term/correlation.out.5', wfn_data.atoms)
    # wfn_data = Gwfn('test/gwfn/b/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/b/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/n/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/n/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/al/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/al/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/h2/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/h2/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/be2/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/be2/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/acetic/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/acetic/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/acetaldehyde/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/acetaldehyde/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/si2h6/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/si2h6/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/alcl3/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/alcl3/HF/cc-pVQZ/input')
    # wfn_data = Gwfn('test/gwfn/s4-c2v/HF/cc-pVQZ/gwfn.data')
    # input_data = Input('test/gwfn/s4-c2v/HF/cc-pVQZ/input')

    # wfn_data = Stowfn('test/stowfn/he/HF/QZ4P/stowfn.data')
    # input_data = Input('test/stowfn/he/HF/QZ4P/input')
    # wfn_data = Stowfn('test/stowfn/be/HF/QZ4P/stowfn.data')
    # input_data = Input('test/stowfn/be/HF/QZ4P/input')

    start = default_timer()
    E = vmc(
        50000, 10 * 1024 * 1024, wfn_data.mo_up, wfn_data.mo_down, input_data.neu, input_data.ned, wfn_data.atoms, wfn_data.shells,
        jastrow_data.trunc, jastrow_data.u_parameters, jastrow_data.u_cutoff, jastrow_data.chi_parameters, jastrow_data.chi_cutoff,
        jastrow_data.f_parameters, jastrow_data.f_cutoff
    )
    end = default_timer()
    reblock_data = pyblock.blocking.reblock(E + nuclear_repulsion(wfn_data.atoms))
    # for reblock_iter in reblock_data:
    #     print(reblock_iter)
    opt = pyblock.blocking.find_optimal_block(E.size, reblock_data)
    print(reblock_data[opt[0]])
    print(f'total time {end-start}')
