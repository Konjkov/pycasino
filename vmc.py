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

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('r_e', nb.float64[:, :]),
    ('atom_positions', nb.float64[:, :]),
    ('atom_charges', nb.float64[:]),
    ('nuclear_repulsion', nb.float64),
    ('slater', Slater.class_type.instance_type),
    ('jastrow', Jastrow.class_type.instance_type),
]


@nb.experimental.jitclass(spec)
class Metropolis:

    def __init__(self, neu, ned, atom_positions, atom_charges, slater, jastrow):
        """Metropolis-Hastings random walk.
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param atom_positions: atomic positions
        :param atom_charges: atomic charges
        :param slater: instance of Slater class
        :param jastrow: instance of Jastrow class
        :return:
        """
        self.neu = neu
        self.ned = ned
        self.r_e = np.zeros((neu + ned, 3))
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.nuclear_repulsion = nuclear_repulsion(atom_positions, atom_charges)
        self.slater = slater
        self.jastrow = jastrow

    def guiding_function(self, e_vectors, n_vectors):
        """Wave function in general form"""
        return np.exp(self.jastrow.value(e_vectors, n_vectors, self.neu)) * self.slater.value(n_vectors, self.neu)

    def make_step(self, p, tau, r_e):
        new_r_e = r_e + random_step(tau, self.neu + self.ned)
        e_vectors = subtract_outer(new_r_e, new_r_e)
        n_vectors = subtract_outer(new_r_e, self.atom_positions)
        new_p = self.guiding_function(e_vectors, n_vectors)
        if cond := new_p ** 2 > np.random.random() * p ** 2:
            return new_r_e, new_p, cond
        else:
            return r_e, p, cond

    def random_walk(self, steps, tau):
        """Metropolis-Hastings random walk.
        :param steps: steps to walk
        :return:
        """

        r_e = self.r_e
        weight = np.ones((steps, ), np.int64)
        position = np.zeros((steps, r_e.shape[0], r_e.shape[1]))
        function = np.ones((steps, ), np.float64)

        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(r_e, self.atom_positions)
        p = self.guiding_function(e_vectors, n_vectors)

        i = 0
        # first step
        r_e, p, _ = self.make_step(p, tau, r_e)
        position[i] = r_e
        function[i] = p
        # other steps
        for _ in range(1, steps):
            r_e, p, cond = self.make_step(p, tau, r_e)
            if cond:
                i += 1
                position[i] = r_e
                function[i] = p
            else:
                weight[i] += 1

        self.r_e = r_e
        return weight[:i+1], position[:i+1], function[:i+1]

    def local_energy(self, position):
        """
        :param position: random walk positions
        :return:
        """

        res = np.zeros((position.shape[0], ))
        for i in range(position.shape[0]):
            r_e = position[i]
            e_vectors = subtract_outer(r_e, r_e)
            n_vectors = subtract_outer(r_e, self.atom_positions)

            s = self.slater.value(n_vectors, self.neu)
            s_l = self.slater.laplacian(n_vectors, self.neu, self.ned) / s
            res[i] = coulomb(e_vectors, n_vectors, self.atom_charges)
            if self.jastrow.enabled:
                j_g = self.jastrow.gradient(e_vectors, n_vectors, self.neu)
                j_l = self.jastrow.laplacian(e_vectors, n_vectors, self.neu)
                s_g = self.slater.gradient(n_vectors, self.neu, self.ned) / s
                F = np.sum((s_g + j_g) * (s_g + j_g)) / 2
                T = (np.sum(s_g * s_g) - s_l - j_l) / 4
                res[i] += 2 * T - F
            else:
                res[i] -= s_l / 2
        return res

    def jastrow_gradient(self, position):
        """
        :param position: random walk positions
        :return:
        """
        r_e = position[0]
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(r_e, self.atom_positions)
        first_res = self.jastrow.parameters_numerical_d1(e_vectors, n_vectors, self.neu)
        res = np.zeros((position.shape[0], ) + first_res.shape)
        res[0] = first_res

        for i in range(1, position.shape[0]):
            r_e = position[i]
            e_vectors = subtract_outer(r_e, r_e)
            n_vectors = subtract_outer(r_e, self.atom_positions)
            res[i] = self.jastrow.parameters_numerical_d1(e_vectors, n_vectors, self.neu)
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


def optimize_vmc_step(opt_steps, initial_tau, metropolis):
    """Optimize vmc step size."""

    def callback(tau, acc_ration):
        """dr = sqrt(3*dtvmc)"""
        logger.info('dr * electrons = %.5f, acc_ration = %.5f', tau[0] * (metropolis.neu + metropolis.ned), acc_ration[0] + 0.5)

    def f(tau):
        weight, _, _ = metropolis.random_walk(casino.input.vmc_equil_nstep, tau)
        return weight.size / casino.input.vmc_equil_nstep - 0.5

    options = dict(jac_options=dict(alpha=1))
    res = sp.optimize.root(f, initial_tau, method='diagbroyden', tol=1/np.sqrt(opt_steps), callback=callback, options=options)
    return np.abs(res.x)


def jastrow_u_cutoff_gradient(weight, energy, energy_gradient):
    """
    :param weight:
    :param energy:
    :param energy_gradient:
    :return:
    """
    return 2 * (
        np.average((energy_gradient * energy[:, np.newaxis]), axis=0, weights=weight) -
        np.average(energy, weights=weight) * np.average(energy_gradient, axis=0, weights=weight)
    )


def jastrow_parameters_gradient(weight, energy, energy_gradient):
    """
    :param weight:
    :param energy:
    :param energy_gradient:
    :return:
    """
    return 2 * (
        np.average((energy_gradient * energy[:, np.newaxis]), axis=0, weights=weight) -
        np.average(energy, weights=weight) * np.average(energy_gradient, axis=0, weights=weight)
    )


def optimize_vmc_parameters(tau, metropolis):
    """Optimize vmc parameters."""

    weight, position, _ = metropolis.random_walk(casino.input.vmc_opt_nstep, tau)

    def callback(x):
        logger.info('u_cutoff = %.5f', x)

    def f(x):
        metropolis.jastrow.set_parameters(x)
        energy = metropolis.local_energy(position)
        energy_gradient = metropolis.jastrow_gradient(position)
        mean_energy = np.average(energy, weights=weight)
        mean_energy_gradient = jastrow_parameters_gradient(weight, energy, energy_gradient)
        logger.info('energy = %.5f, energy_gradient = %.5f, cutoff = %.5f', mean_energy, mean_energy_gradient[0], x)
        return mean_energy, mean_energy_gradient[0]

    # Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr.
    res = sp.optimize.minimize(f, metropolis.jastrow.get_parameters(), method='BFGS', jac=True, callback=callback)
    return np.abs(res.x)


def main(casino):
    """Configuration-by-configuration sampling (CBCS)
    Should be pure python function.
    """

    jastrow = Jastrow(
        casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_cutoff, casino.jastrow.u_spin_dep,
        casino.jastrow.chi_parameters, casino.jastrow.chi_cutoff, casino.jastrow.chi_labels, casino.jastrow.chi_spin_dep,
        casino.jastrow.f_parameters, casino.jastrow.f_cutoff, casino.jastrow.f_labels, casino.jastrow.f_spin_dep
    )

    slater = Slater(
        casino.wfn.nbasis_functions, casino.wfn.first_shells, casino.wfn.orbital_types, casino.wfn.shell_moments,
        casino.wfn.slater_orders, casino.wfn.primitives, casino.wfn.coefficients, casino.wfn.exponents,
        casino.mdet.mo_up, casino.mdet.mo_down, casino.mdet.coeff
    )

    neu, ned = casino.input.neu, casino.input.ned
    metropolis = Metropolis(neu, ned, casino.wfn.atom_positions, casino.wfn.atom_charges, slater, jastrow)

    tau = 1 / (neu + ned)
    metropolis.r_e = initial_position(neu + ned, metropolis.atom_positions, metropolis.atom_charges)
    weight, _, _ = metropolis.random_walk(casino.input.vmc_equil_nstep, tau)
    logger.info('dr * electrons = 1.00000, acc_ration = %.5f', weight.size / casino.input.vmc_equil_nstep)
    tau = optimize_vmc_step(10000, tau, metropolis)

    optimize_vmc_parameters(tau, metropolis)

    # check_point_1 = default_timer()
    # weight, position, _ = metropolis.random_walk(casino.input.vmc_nstep // 10, tau)
    # energy = metropolis.local_energy(position)
    # energy_gradient = metropolis.jastrow_gradient(position)
    # gradient = jastrow_parameters_gradient(weight, energy, energy_gradient)
    # check_point_2 = default_timer()
    # logger.info(f'{gradient}, total time {check_point_2-check_point_1}')

    # rounds = 10
    # E = np.zeros((rounds, ))
    # G = np.zeros((rounds, ))
    # check_point_1 = default_timer()
    # for i in range(rounds):
    #     weights, position, _ = metropolis.random_walk(casino.input.vmc_nstep // rounds, tau)
    #     energy = metropolis.local_energy(position)
    #     E[i] = np.average(energy, weights=weights)
    #     mean_energy = np.average(E[:i + 1])
    #     std_err = np.std(E[:i + 1], ddof=0) / np.sqrt(i)
    #
    #     energy_gradient = metropolis.jastrow_gradient(position)
    #     gradient = 2 * (
    #         np.average((energy_gradient * energy[:, np.newaxis]), axis=0, weights=weights) -
    #         np.average(energy, weights=weights) * np.average(energy_gradient, axis=0, weights=weights)
    #     )
    #     G[i] = gradient[0]
    #     mean_gradient = np.average(G[:i + 1])
    #     std_err_gradient = np.std(G[:i + 1], ddof=0) / np.sqrt(i)
    #     check_point_2 = default_timer()
    #     logger.info(f'{E[i] + metropolis.nuclear_repulsion}, {mean_energy + metropolis.nuclear_repulsion}, '
    #                 f'{std_err}, {G[i]}, {mean_gradient}, {std_err_gradient}, total time {check_point_2-check_point_1}')

    # return expand(weights, energy)


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
    path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/u_term_test/'
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
    # path = 'test/gwfn/be2/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
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
    # reblock_data = pyblock.blocking.reblock(E + nuclear_repulsion(casino.wfn.atom_positions, casino.wfn.atom_charges))
    # for reblock_iter in reblock_data:
    #     print(reblock_iter)
    # opt = pyblock.blocking.find_optimal_block(E.size, reblock_data)
    # opt_data = reblock_data[opt[0]]
    # logger.info(opt_data)
    # print(np.mean(opt_data.mean), '+/-', np.mean(opt_data.std_err) / np.sqrt(opt_data.std_err.size))

