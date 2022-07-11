#!/usr/bin/env python3

import os
from timeit import default_timer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import numba as nb

from overload import subtract_outer
from logger import logging
from readers.casino import CasinoConfig
from cusp import CuspFactory, TestCuspFactory
from slater import Slater
from jastrow import Jastrow
from backflow import Backflow


logger = logging.getLogger('vmc')


@nb.jit(forceobj=True)
def initial_position(ne, atom_positions, atom_charges):
    """Initial positions of electrons."""
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
    return r_e


@nb.jit(nopython=True, nogil=True, cache=True)
def random_step(step, ne):
    """Random N-dim square distributed step"""
    return step * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))


# @thread
@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def slater_value(dx, neu, ned, steps, atom_positions, slater, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        n_vectors = subtract_outer(atom_positions, r_e)
        slater.value(n_vectors)


# @thread
@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def slater_gradient(dx, neu, ned, steps, atom_positions, slater, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        n_vectors = subtract_outer(atom_positions, r_e)
        slater.gradient(n_vectors)


# @thread
@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def slater_laplacian(dx, neu, ned, steps, atom_positions, slater, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        n_vectors = subtract_outer(atom_positions, r_e)
        slater.laplacian(n_vectors)


# @thread
@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def slater_hessian(dx, neu, ned, steps, atom_positions, slater, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        n_vectors = subtract_outer(atom_positions, r_e)
        slater.hessian(n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def jastrow_value(dx, neu, ned, steps, atom_positions, jastrow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        jastrow.value(e_vectors, n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def jastrow_gradient(dx, neu, ned, steps, atom_positions, jastrow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        jastrow.gradient(e_vectors, n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def jastrow_laplacian(dx, neu, ned, steps, atom_positions, jastrow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        jastrow.laplacian(e_vectors, n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def backflow_value(dx, neu, ned, steps, atom_positions, backflow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        backflow.value(e_vectors, n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def backflow_gradient(dx, neu, ned, steps, atom_positions, backflow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        backflow.gradient(e_vectors, n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def backflow_laplacian(dx, neu, ned, steps, atom_positions, backflow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        backflow.laplacian(e_vectors, n_vectors)


def slater_profiling(config):
    """For multithreaded
    https://numba.pydata.org/numba-doc/latest/user/threading-layer.html
    """
    dx = 3.0
    neu, ned = config.input.neu, config.input.ned

    # if config.input.cusp_correction:
    #     cusp = CuspFactory(
    #         neu, ned, config.mdet.mo_up, config.mdet.mo_down,
    #         config.wfn.nbasis_functions, config.wfn.first_shells, config.wfn.shell_moments, config.wfn.primitives,
    #         config.wfn.coefficients, config.wfn.exponents, config.wfn.atom_positions, config.wfn.atom_charges
    #     ).create()
    # else:
    cusp = None

    slater = Slater(
        neu, ned,
        config.wfn.nbasis_functions, config.wfn.first_shells, config.wfn.orbital_types, config.wfn.shell_moments,
        config.wfn.slater_orders, config.wfn.primitives, config.wfn.coefficients, config.wfn.exponents,
        config.mdet.mo_up, config.mdet.mo_down, config.mdet.coeff, cusp
    )

    r_initial = initial_position(neu + ned, config.wfn.atom_positions, config.wfn.atom_charges)

    start = default_timer()
    slater_value(dx, neu, ned, config.input.vmc_nstep, config.wfn.atom_positions, slater, r_initial)
    end = default_timer()
    logger.info(' value     %8.1f', end - start)
    # stats = rtsys.get_allocation_stats()
    # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    start = default_timer()
    slater_laplacian(dx, neu, ned, config.input.vmc_nstep, config.wfn.atom_positions, slater, r_initial)
    end = default_timer()
    logger.info(' laplacian %8.1f', end - start)
    # stats = rtsys.get_allocation_stats()
    # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    start = default_timer()
    slater_gradient(dx, neu, ned, config.input.vmc_nstep, config.wfn.atom_positions, slater, r_initial)
    end = default_timer()
    logger.info(' gradient  %8.1f', end - start)
    # stats = rtsys.get_allocation_stats()
    # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    start = default_timer()
    slater_hessian(dx, neu, ned, config.input.vmc_nstep, config.wfn.atom_positions, slater, r_initial)
    end = default_timer()
    logger.info(' hessian   %8.1f', end - start)
    # stats = rtsys.get_allocation_stats()
    # logger.info(f'{stats} total: {stats[0] - stats[1]}')


def jastrow_profiling(casino):
    dx = 3.0

    jastrow = Jastrow(
        casino.input.neu, casino.input.ned,
        casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_mask, casino.jastrow.u_cutoff, casino.jastrow.u_cusp_const,
        casino.jastrow.chi_parameters, casino.jastrow.chi_mask, casino.jastrow.chi_cutoff, casino.jastrow.chi_labels,
        casino.jastrow.f_parameters, casino.jastrow.f_mask, casino.jastrow.f_cutoff, casino.jastrow.f_labels,
        casino.jastrow.no_dup_u_term, casino.jastrow.no_dup_chi_term, casino.jastrow.chi_cusp
    )

    r_initial = initial_position(casino.input.neu + casino.input.ned, casino.wfn.atom_positions, casino.wfn.atom_charges)

    start = default_timer()
    jastrow_value(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, jastrow, r_initial)
    end = default_timer()
    logger.info(' value     %8.1f', end - start)

    start = default_timer()
    jastrow_laplacian(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, jastrow, r_initial)
    end = default_timer()
    logger.info(' laplacian %8.1f', end - start)

    start = default_timer()
    jastrow_gradient(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, jastrow, r_initial)
    end = default_timer()
    logger.info(' gradient  %8.1f', end - start)


def backflow_profiling(casino):
    dx = 3.0

    backflow = Backflow(
        casino.input.neu, casino.input.ned,
        casino.backflow.trunc, casino.backflow.eta_parameters, casino.backflow.eta_cutoff,
        casino.backflow.mu_parameters, casino.backflow.mu_cutoff, casino.backflow.mu_labels,
        casino.backflow.phi_parameters, casino.backflow.theta_parameters, casino.backflow.phi_cutoff,
        casino.backflow.phi_labels, casino.backflow.phi_irrotational, casino.backflow.ae_cutoff
    )

    r_initial = initial_position(casino.input.neu + casino.input.ned, casino.wfn.atom_positions, casino.wfn.atom_charges)

    start = default_timer()
    backflow_value(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, backflow, r_initial)
    end = default_timer()
    logger.info(' value     %8.1f', end - start)

    start = default_timer()
    backflow_gradient(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, backflow, r_initial)
    end = default_timer()
    logger.info(' gradient  %8.1f', end - start)

    start = default_timer()
    backflow_laplacian(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, backflow, r_initial)
    end = default_timer()
    logger.info(' laplacian %8.1f', end - start)


if __name__ == '__main__':
    """
    Slater:
        He:
         value         28.7
         laplacian     53.7
         gradient      74.1
         hessian      251.2
        Be:
         value         50.5
         laplacian    100.9
         gradient     136.9
         hessian      365.3
        Ne:
         value        116.1
         laplacian    228.9
         gradient     294.9
         hessian      777.2
        Ar:
         value        269.5
         laplacian    555.6
         gradient     657.0
         hessian     1670.7
        -- old --
        Kr:
         value        781.9
         laplacian   1589.5
         gradient    2526.0
         hessian     6741.8
        O3:
         value        655.3
         laplacian   1300.4
         gradient    2669.2
    Gaussian:
        He:
         value         29.1
         laplacian     55.0
         gradient     114.3
         hessian      386.1
        Be:
         value         55.9
         laplacian    110.2
         gradient     243.6
         hessian      769.3
        Ne:
         value        125.0
    """

    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr', 'O3'):
        # path = f'test/gwfn/{mol}/HF/cc-pVQZ/CBCS/Slater/'
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Slater/'
        logger.info('%s:', mol)
        slater_profiling(CasinoConfig(path))

    """
    He:
     value         25.6
     laplacian     30.4
     gradient      37.6
    Be:
     value         57.5
     laplacian     93.9
     gradient     112.4
    Ne:
     value        277.5
     laplacian    481.7
     gradient     536.5
    Ar:
     value        875.4
     laplacian   1612.5
     gradient    1771.5
    Kr:
     value       3174.8
    """

    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr'):
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Jastrow/'
        logger.info('%s:', mol)
        jastrow_profiling(CasinoConfig(path))

    """
    He:
     value         40.0
     gradient     121.6
     laplacian    138.8
    Be:
     value         99.5
     gradient     481.4
     laplacian    573.4
    Ne:
     value        415.0
     gradient    1897.3
     laplacian   2247.9
    Ar:
     value       1501.9
    """

    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr'):
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Backflow/'
        logger.info('%s:', mol)
        backflow_profiling(CasinoConfig(path))
