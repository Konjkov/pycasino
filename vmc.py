#!/usr/bin/env python3

import os
from timeit import default_timer
from wfn import wfn, wfn_gradient, wfn_laplacian, wfn_numerical_gradient, wfn_numerical_laplacian
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
import scipy as sp

from decorators import multi_process
from readers.casino import Casino


@nb.jit(nopython=True)
def initial_position(ne, atoms):
    """Initial positions of electrons"""
    natoms = atoms.shape[0]
    X = np.zeros((ne, 3))
    for i in range(ne):
        X[i] = atoms[np.random.randint(natoms)]['position']
    return X + random_normal_step(1.0, ne)


def optimal_vmc_step(r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """vmc step width """

    opt_steps = 10000

    def callback(tau, acc_ration):
        print(f'tau * electrons = {tau[0] * (neu + ned):.5f}, acc_ration = {acc_ration[0] + 0.5:.5f}')

    def f(tau):
        return equilibration(opt_steps, tau, r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff) - 0.5

    return sp.optimize.root(f, 1/(neu+ned), method='diagbroyden', tol=1/np.sqrt(opt_steps), callback=callback, options=dict(jac_options=dict(alpha=1))).x


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
def guiding_function(r_e, nbasis_functions, neu, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """wave function in general form"""

    return (
        np.exp(jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)) *
        wfn(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, atoms, shells)
    )


@nb.jit(nopython=True)
def local_energy(r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    j_g = jastrow_gradient(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)
    j_l = jastrow_laplacian(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)
    w = wfn(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, atoms, shells)
    w_g = wfn_gradient(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, ned, atoms, shells) / w
    w_l = wfn_laplacian(r_e, nbasis_functions, mo_u, mo_d, coeff, neu, ned, atoms, shells) / w
    F = np.sum((w_g + j_g) * (w_g + j_g)) / 2
    T = (np.sum(w_g * w_g) - w_l - j_l) / 4
    return coulomb(r_e, atoms) + 2 * T - F


@nb.jit(nopython=True)
def equilibration(steps, tau, r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """VMC equilibration"""
    i = 0
    p = 0.0
    for j in range(steps):
        new_r_e = r_e + random_step(tau, neu + ned)

        new_p = guiding_function(new_r_e, nbasis_functions, neu, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
        j += 1
        if new_p**2 > np.random.random() * p**2:
            r_e, p = new_r_e, new_p
            i += 1
    return i / steps


@nb.jit(nopython=True, nogil=True, parallel=False)
def simple_accumulation(steps, tau, r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """VMC simple accumulation"""
    p = loc_E = 0.0
    E = np.zeros((steps,))
    for j in range(steps):
        new_r_e = r_e + random_step(tau, neu + ned)

        new_p = guiding_function(new_r_e, nbasis_functions, neu, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
        if new_p**2 > np.random.random() * p**2:
            r_e, p = new_r_e, new_p
            loc_E = local_energy(r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
        E[j] = loc_E
    return E


@nb.jit(nopython=True)
def averaging_accumulation(steps, tau, r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """VMC accumulation with averaging local energies over proposed moves"""
    E = np.zeros((steps,))
    loc_E = local_energy(r_e, nbasis_functions, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
    for j in range(steps):
        new_r_e = r_e + random_step(tau, neu + ned)

        new_p = guiding_function(new_r_e, nbasis_functions, neu, mo_u, mo_d, coeff, atoms, shells)
        new_loc_E = local_energy(new_r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
        E[j] = min((new_p/p)**2, 1) * new_loc_E + (1 - min((new_p/p)**2, 1)) * loc_E
        if (new_p/p)**2 > np.random.random():
            r_e, p, loc_E = new_r_e, new_p, new_loc_E
    return E


accumulation = simple_accumulation


# @multi_process
# @nb.jit(nopython=True, nogil=True, parallel=False)
def vmc(vmc_nstep, vmc_equil_nstep, neu, ned, nbasis_functions, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
    """configuration-by-configuration sampling (CBCS)"""

    r_e = initial_position(neu + ned, atoms)

    acc_ratio = equilibration(vmc_equil_nstep, 1/(neu + ned), r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
    print(f'tau * electrons = 1.00000, acc_ration = {acc_ratio}')

    tau = optimal_vmc_step(r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)

    acc_ratio = equilibration(vmc_equil_nstep, tau, r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)
    print(f'tau * electrons = {tau * (neu + ned):.5f}, acc_ration = {acc_ratio}')

    return accumulation(vmc_nstep, tau, r_e, nbasis_functions, neu, ned, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)


def main(casino):

    vmc_nstep, vmc_equil_nstep, neu, ned, nbasis_functions = casino.input.vmc_nstep, casino.input.vmc_equil_nstep, casino.input.neu, casino.input.ned, casino.wfn.nbasis_functions
    mo_u, mo_d, coeff = casino.mdet.mo_up, casino.mdet.mo_down, casino.mdet.coeff
    atoms, shells = casino.wfn.atoms, casino.wfn.shells
    trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff = casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_cutoff, casino.jastrow.chi_parameters, casino.jastrow.chi_cutoff,   casino.jastrow.f_parameters, casino.jastrow.f_cutoff

    return vmc(vmc_nstep, vmc_equil_nstep, neu, ned, nbasis_functions, mo_u, mo_d, coeff, atoms, shells, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff)


if __name__ == '__main__':
    """
    be HF/cc-pVQZ

    stat = 1024 * 1024 * 1024

    """

    # path = 'test/gwfn/h/HF/cc-pVQZ/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/'
    # path = 'test/gwfn/be/HF-CASSCF(2.4)/def2-QZVP/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/be/HF/def2-QZVP/VMC_OPT_BF/emin_BF/8_8_44__9_9_33'
    # path = 'test/gwfn/b/HF/cc-pVQZ/'
    # path = 'test/gwfn/n/HF/cc-pVQZ/'
    # path = 'test/gwfn/al/HF/cc-pVQZ/'
    # path = 'test/gwfn/h2/HF/cc-pVQZ/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/'
    path = 'test/gwfn/be2/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
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
    reblock_data = pyblock.blocking.reblock(E + nuclear_repulsion(casino.wfn.atoms))
    # for reblock_iter in reblock_data:
    #     print(reblock_iter)
    opt = pyblock.blocking.find_optimal_block(E.size, reblock_data)
    opt_data = reblock_data[opt[0]]
    print(opt_data)
    # print(np.mean(opt_data.mean), '+/-', np.mean(opt_data.std_err) / np.sqrt(opt_data.std_err.size))
    print(f'total time {end-start}')
