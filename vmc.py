#!/usr/bin/env python3

import os
from timeit import default_timer
from slater import Slater
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
from logger import logging
from random_steps import initial_position, random_step


logger = logging.getLogger('vmc')
numba_logger = logging.getLogger('numba')


def optimize_vmc_step(opt_steps, r_e, initial_tau, neu, ned, atom_positions, slater, jastrow):
    """Optimize vmc step size."""

    def callback(tau, acc_ration):
        """dr = sqrt(3*dtvmc)"""
        logger.info('dr * electrons = %.5f, acc_ration = %.5f', tau[0] * (neu + ned), acc_ration[0] + 0.5)

    def f(tau):
        weight, _, _ = random_walk(casino.input.vmc_equil_nstep, tau, r_e, neu, ned, atom_positions, slater, jastrow)
        return weight.size / casino.input.vmc_equil_nstep - 0.5

    options = dict(jac_options=dict(alpha=1))
    res = sp.optimize.root(f, initial_tau, method='diagbroyden', tol=1/np.sqrt(opt_steps), callback=callback, options=options)
    return np.abs(res.x)


@nb.jit(nopython=True)
def guiding_function(e_vectors, n_vectors, neu, slater, jastrow):
    """Wave function in general form"""

    return np.exp(jastrow.value(e_vectors, n_vectors, neu)) * slater.value(n_vectors, neu)


@nb.jit(nopython=True)
def make_step(p, tau, r_e, neu, ned, atom_positions, slater, jastrow):
    new_r_e = r_e + random_step(tau, neu + ned)
    e_vectors = subtract_outer(new_r_e, new_r_e)
    n_vectors = subtract_outer(new_r_e, atom_positions)
    new_p = guiding_function(e_vectors, n_vectors, neu, slater, jastrow)
    if cond := new_p ** 2 > np.random.random() * p ** 2:
        return new_r_e, new_p, cond
    else:
        return r_e, p, cond


@nb.jit(nopython=True)
def random_walk(steps, tau, r_e, neu, ned, atom_positions, slater, jastrow):
    """Metropolis-Hastings random walk.
    :param steps: steps to walk
    :param tau: step size
    :param r_e: position preceding starts position of electrons (last position of previous run)
    :param neu: number of up electrons
    :param ned: number of down electrons
    :param atom_positions: atomic positions
    :param slater: instance of Slater class
    :param jastrow: instance of Jastrow class
    :return:
    """

    weights = np.ones((steps, ), np.int64)
    position = np.zeros((steps, r_e.shape[0], r_e.shape[1]))
    function = np.ones((steps, ), np.float64)

    e_vectors = subtract_outer(r_e, r_e)
    n_vectors = subtract_outer(r_e, atom_positions)
    p = guiding_function(e_vectors, n_vectors, neu, slater, jastrow)

    i = 0
    # first step
    r_e, p, _ = make_step(p, tau, r_e, neu, ned, atom_positions, slater, jastrow)
    position[i] = r_e
    function[i] = p
    # other steps
    for _ in range(1, steps):
        r_e, p, cond = make_step(p, tau, r_e, neu, ned, atom_positions, slater, jastrow)
        if cond:
            i += 1
            position[i] = r_e
            function[i] = p
        else:
            weights[i] += 1

    return weights[:i+1], position[:i+1], function[:i+1]


@nb.jit(nopython=True, nogil=True, parallel=False)
def local_energy(position, neu, ned, atom_positions, atom_charges, slater, jastrow):
    """
    :param position:
    :param neu:
    :param ned:
    :param atom_positions:
    :param slater:
    :param jastrow:
    :param atom_charges:
    :return:
    """

    res = np.zeros((position.shape[0], ))
    for i in range(position.shape[0]):
        r_e = position[i]
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(r_e, atom_positions)

        s = slater.value(n_vectors, neu)
        s_l = slater.laplacian(n_vectors, neu, ned) / s
        res[i] = coulomb(e_vectors, n_vectors, atom_charges)
        if jastrow.enabled:
            j_g = jastrow.gradient(e_vectors, n_vectors, neu)
            j_l = jastrow.laplacian(e_vectors, n_vectors, neu)
            s_g = slater.gradient(n_vectors, neu, ned) / s
            F = np.sum((s_g + j_g) * (s_g + j_g)) / 2
            T = (np.sum(s_g * s_g) - s_l - j_l) / 4
            res[i] += 2 * T - F
        else:
            res[i] -= s_l / 2
    return res


@nb.jit(nopython=True, nogil=True, parallel=False)
def local_energy_gradient(position, neu, ned, atom_positions, atom_charges, slater, jastrow):
    """
    :param position:
    :param neu:
    :param ned:
    :param atom_positions:
    :param slater:
    :param jastrow:
    :param atom_charges:
    :return:
    """

    r_e = position[0]
    e_vectors = subtract_outer(r_e, r_e)
    n_vectors = subtract_outer(r_e, atom_positions)
    first_res = jastrow.parameters_numerical_first_deriv(e_vectors, n_vectors, neu)
    res = np.zeros((position.shape[0], ) + first_res.shape)
    res[0] = first_res

    for i in range(1, position.shape[0]):
        r_e = position[i]
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(r_e, atom_positions)
        res[i] = jastrow.parameters_numerical_first_deriv(e_vectors, n_vectors, neu)
    return res


@nb.jit(nopython=True, nogil=True, parallel=False)
def expand(weight, value):
    res = np.zeros((weight.sum(), ) + value.shape[1:])
    n = 0
    for i in range(value.shape[0]):
        for j in range(weight[i]):
            res[n] = value[i]
            n += 1
    return res


def main(casino):
    """Configuration-by-configuration sampling (CBCS)
    Should be pure python function.
    """

    jastrow = Jastrow(
        casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_cutoff, casino.jastrow.chi_parameters,
        casino.jastrow.chi_cutoff, casino.jastrow.chi_labels, casino.jastrow.f_parameters, casino.jastrow.f_cutoff,
        casino.jastrow.f_labels
    )

    slater = Slater(
        casino.wfn.nbasis_functions, casino.wfn.first_shells, casino.wfn.orbital_types, casino.wfn.shell_moments,
        casino.wfn.slater_orders, casino.wfn.primitives, casino.wfn.coefficients, casino.wfn.exponents,
        casino.mdet.mo_up, casino.mdet.mo_down, casino.mdet.coeff
    )

    neu, ned = casino.input.neu, casino.input.ned
    tau = 1 / (neu + ned)
    r_e = initial_position(neu + ned, casino.wfn.atom_positions, casino.wfn.atom_charges) + random_step(tau, neu + ned)

    weights, position, _ = random_walk(casino.input.vmc_equil_nstep, tau, r_e, neu, ned, casino.wfn.atom_positions, slater, jastrow)
    logger.info('dr * electrons = 1.00000, acc_ration = %.5f', weights.size / casino.input.vmc_equil_nstep)
    tau = optimize_vmc_step(10000, position[-1], tau, neu, ned, casino.wfn.atom_positions, slater, jastrow)

    repulsion = nuclear_repulsion(casino.wfn.atom_positions, casino.wfn.atom_charges)

    rounds = 10
    E = np.zeros((rounds, ))
    check_point_1 = default_timer()
    for i in range(rounds):
        weights, position, _ = random_walk(casino.input.vmc_nstep // rounds, tau, position[-1], neu, ned, casino.wfn.atom_positions, slater, jastrow)
        energy = local_energy(position, neu, ned, casino.wfn.atom_positions, casino.wfn.atom_charges, slater, jastrow)
        E[i] = np.average(energy, weights=weights)
        check_point_2 = default_timer()
        mean_energy = np.average(E[:i + 1])
        std_err = np.std(E[:i + 1], ddof=0) / np.sqrt(i)
        logger.info(f'{E[i] + repulsion}, {mean_energy + repulsion}, {std_err}, total time {check_point_2-check_point_1}')

    # energy_gradient = local_energy_gradient(position, neu, ned, casino.wfn.atom_positions, casino.wfn.atom_charges, slater, jastrow)
    # gradient = 2 * (
    #     np.average((energy_gradient * energy[:, np.newaxis]), axis=0, weights=weights) -
    #     np.average(energy, weights=weights) * np.average(energy_gradient, axis=0, weights=weights)
    # )
    # print(gradient)

    return expand(weights, energy)


if __name__ == '__main__':
    """
    be HF/cc-pVQZ

    stat = 1024 * 1024 * 1024

    """

    # path = 'test/gwfn/h/HF/cc-pVQZ/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_vmc/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/u_term/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/chi_term/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/be/HF-CASSCF(2.4)/def2-QZVP/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_vmc_cbc/'
    # path = 'test/gwfn/be/HF/def2-QZVP/VMC_OPT_BF/emin_BF/8_8_44__9_9_33'
    # path = 'test/gwfn/b/HF/cc-pVQZ/'
    # path = 'test/gwfn/n/HF/cc-pVQZ/'
    # path = 'test/gwfn/al/HF/cc-pVQZ/'
    # path = 'test/gwfn/cl/HF/cc-pVQZ/'
    # path = 'test/gwfn/h2/HF/cc-pVQZ/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/VMC_OPT/emin/legacy/u_term/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/VMC_OPT/emin/legacy/chi_term/'
    path = 'test/gwfn/be2/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/ch4/HF/cc-pVQZ/'
    # path = 'test/gwfn/acetic/HF/cc-pVQZ/'
    # path = 'test/gwfn/acetaldehyde/HF/cc-pVQZ/'
    # path = 'test/gwfn/acetaldehyde/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/si2h6/HF/cc-pVQZ/'
    # path = 'test/gwfn/alcl3/HF/cc-pVQZ/'
    # path = 'test/gwfn/s4-c2v/HF/cc-pVQZ/'

    # path = 'test/stowfn/he/HF/QZ4P/'
    # path = 'test/stowfn/be/HF/QZ4P/'

    casino = Casino(path)

    E = main(casino)
    reblock_data = pyblock.blocking.reblock(E + nuclear_repulsion(casino.wfn.atom_positions, casino.wfn.atom_charges))
    # for reblock_iter in reblock_data:
    #     print(reblock_iter)
    opt = pyblock.blocking.find_optimal_block(E.size, reblock_data)
    opt_data = reblock_data[opt[0]]
    logger.info(opt_data)
    # print(np.mean(opt_data.mean), '+/-', np.mean(opt_data.std_err) / np.sqrt(opt_data.std_err.size))

