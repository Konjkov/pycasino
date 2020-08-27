#!/usr/bin/env python3

import os
from timeit import default_timer
from wfn import Wfn
from jastrow import Jastrow
from coulomb import coulomb, nuclear_repulsion

os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import pyblock
import numpy as np
import numba as nb
import scipy as sp

from decorators import pool, thread
from readers.casino import Casino
from overload import subtract_outer


@nb.jit(nopython=True)
def initial_position(ne, atom_positions):
    """Initial positions of electrons"""
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        r_e[i] = atom_positions[np.random.randint(natoms)]
    return r_e + random_normal_step(1.0, ne)


def optimal_vmc_step(r_e, neu, ned, atom_positions, wfn, jastrow):
    """vmc step width """

    opt_steps = 10000

    def callback(tau, acc_ration):
        """dr = sqrt(3*dtvmc)"""
        print(f'dr * electrons = {tau[0] * (neu + ned):.5f}, acc_ration = {acc_ration[0] + 0.5:.5f}')

    def f(tau):
        return equilibration(opt_steps, tau, r_e, neu, ned, atom_positions, wfn, jastrow) - 0.5

    options = dict(jac_options=dict(alpha=1))
    res = sp.optimize.root(f, 1/(neu+ned), method='diagbroyden', tol=1/np.sqrt(opt_steps), callback=callback, options=options)
    return res.x


@nb.jit(nopython=True)
def random_laplace_step(dX, ne):
    """Random N-dim laplace distributed step"""
    return np.random.laplace(0.0, dX/(3*np.pi/4), ne*3).reshape((ne, 3))


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
    return np.random.normal(0.0, dX/np.sqrt(3), ne*3).reshape((ne, 3))


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
def guiding_function(e_vectors, n_vectors, neu, wfn, jastrow):
    """wave function in general form"""

    return np.exp(jastrow.value(e_vectors, n_vectors, neu)) * wfn.value(n_vectors, neu)


@nb.jit(nopython=True)
def local_energy(e_vectors, n_vectors, neu, ned, wfn, jastrow, atom_charges):

    j_g = jastrow.gradient(e_vectors, n_vectors, neu)
    j_l = jastrow.laplacian(e_vectors, n_vectors, neu)
    w = wfn.value(n_vectors, neu)
    w_g = wfn.gradient(n_vectors, neu, ned) / w
    w_l = wfn.laplacian(n_vectors, neu, ned) / w
    F = np.sum((w_g + j_g) * (w_g + j_g)) / 2
    T = (np.sum(w_g * w_g) - w_l - j_l) / 4
    return coulomb(e_vectors, n_vectors, atom_charges) + 2 * T - F


@nb.jit(nopython=True)
def equilibration(steps, tau, r_e, neu, ned, atom_positions, wfn, jastrow):
    """VMC equilibration"""
    i = 0
    p = 0.0
    for j in range(steps):
        new_r_e = r_e + random_step(tau, neu + ned)

        e_vectors = subtract_outer(new_r_e, new_r_e)
        n_vectors = subtract_outer(new_r_e, atom_positions)

        new_p = guiding_function(e_vectors, n_vectors, neu, wfn, jastrow)
        j += 1
        if new_p**2 > np.random.random() * p**2:
            r_e, p = new_r_e, new_p
            i += 1
    return i / steps


# @pool
@nb.jit(nopython=True, nogil=True, parallel=False)
def simple_accumulation(steps, tau, r_e, neu, ned, atom_positions, wfn, jastrow, atom_charges):
    """VMC simple accumulation"""
    p = loc_E = 0.0
    E = np.zeros((steps,))
    for j in range(steps):
        new_r_e = r_e + random_step(tau, neu + ned)

        e_vectors = subtract_outer(new_r_e, new_r_e)
        n_vectors = subtract_outer(new_r_e, atom_positions)

        new_p = guiding_function(e_vectors, n_vectors, neu, wfn, jastrow)
        if new_p**2 > np.random.random() * p**2:
            loc_E = local_energy(e_vectors, n_vectors, neu, ned, wfn, jastrow, atom_charges)
            r_e, p = new_r_e, new_p
        E[j] = loc_E
    return E


@nb.jit(nopython=True)
def averaging_accumulation(steps, tau, r_e, neu, ned, atom_positions, wfn, jastrow, atom_charges):
    """VMC accumulation with averaging local energies over proposed moves"""
    E = np.zeros((steps,))
    loc_E = local_energy(r_e, neu, ned, atom_positions, jastrow, atom_charges)
    for j in range(steps):
        new_r_e = r_e + random_step(tau, neu + ned)

        e_vectors = subtract_outer(new_r_e, new_r_e)
        n_vectors = subtract_outer(new_r_e, atom_positions)

        new_p = guiding_function(e_vectors, n_vectors, neu, wfn, jastrow)
        new_loc_E = local_energy(e_vectors, n_vectors, neu, ned, wfn, jastrow, atom_charges)

        E[j] = min((new_p/p)**2, 1) * new_loc_E + (1 - min((new_p/p)**2, 1)) * loc_E
        if (new_p/p)**2 > np.random.random():
            r_e, p, loc_E = new_r_e, new_p, new_loc_E
    return E


accumulation = simple_accumulation


def vmc(vmc_nstep, vmc_equil_nstep, neu, ned):
    """configuration-by-configuration sampling (CBCS)"""

    r_e = initial_position(neu + ned, casino.wfn.atom_positions)

    jastrow = Jastrow(
        casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_cutoff, casino.jastrow.chi_parameters,
        casino.jastrow.chi_cutoff, casino.jastrow.f_parameters, casino.jastrow.f_cutoff)

    wfn = Wfn(
        casino.wfn.nbasis_functions, casino.wfn.first_shells, casino.wfn.orbital_types, casino.wfn.shell_moments,
        casino.wfn.slater_orders, casino.wfn.primitives, casino.wfn.coefficients, casino.wfn.exponents,
        casino.mdet.mo_up, casino.mdet.mo_down, casino.mdet.coeff
    )

    acc_ratio = equilibration(vmc_equil_nstep, 1/(neu + ned), r_e, neu, ned, casino.wfn.atom_positions, wfn, jastrow)
    print(f'dr * electrons = 1.00000, acc_ration = {acc_ratio}')

    tau = optimal_vmc_step(r_e, neu, ned, casino.wfn.atom_positions, wfn, jastrow)

    return accumulation(vmc_nstep, tau, r_e, neu, ned, casino.wfn.atom_positions, wfn, jastrow, casino.wfn.atom_charges)


def main(casino):

    return vmc(casino.input.vmc_nstep, casino.input.vmc_equil_nstep, casino.input.neu, casino.input.ned)


if __name__ == '__main__':
    """
    be HF/cc-pVQZ

    stat = 1024 * 1024 * 1024

    """

    # path = 'test/gwfn/h/HF/cc-pVQZ/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/'
    path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_vmc/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/'
    # path = 'test/gwfn/be/HF-CASSCF(2.4)/def2-QZVP/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_vmc_cbc/'
    # path = 'test/gwfn/be/HF/def2-QZVP/VMC_OPT_BF/emin_BF/8_8_44__9_9_33'
    # path = 'test/gwfn/b/HF/cc-pVQZ/'
    # path = 'test/gwfn/n/HF/cc-pVQZ/'
    # path = 'test/gwfn/al/HF/cc-pVQZ/'
    # path = 'test/gwfn/h2/HF/cc-pVQZ/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/acetic/HF/cc-pVQZ/'
    # path = 'test/gwfn/acetaldehyde/HF/cc-pVQZ/'
    # path = 'test/gwfn/acetaldehyde/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/si2h6/HF/cc-pVQZ/'
    # path = 'test/gwfn/alcl3/HF/cc-pVQZ/'
    # path = 'test/gwfn/s4-c2v/HF/cc-pVQZ/'

    # path = 'test/stowfn/he/HF/QZ4P/'
    # path = 'test/stowfn/be/HF/QZ4P/'

    casino = Casino(path)

    start = default_timer()
    E = main(casino)
    end = default_timer()
    reblock_data = pyblock.blocking.reblock(E + nuclear_repulsion(casino.wfn.atom_positions, casino.wfn.atom_charges))
    # for reblock_iter in reblock_data:
    #     print(reblock_iter)
    opt = pyblock.blocking.find_optimal_block(E.size, reblock_data)
    opt_data = reblock_data[opt[0]]
    print(opt_data)
    # print(np.mean(opt_data.mean), '+/-', np.mean(opt_data.std_err) / np.sqrt(opt_data.std_err.size))
    print(f'total time {end-start}')
