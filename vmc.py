#!/usr/bin/env python3

import os
from timeit import default_timer
from slater import Slater
from jastrow import Jastrow
from backflow import Backflow
from wfn import Wfn
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


logger = logging.getLogger('vmc')
numba_logger = logging.getLogger('numba')

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('r_e', nb.float64[:, :]),
    ('step', nb.float64),
    ('atom_positions', nb.float64[:, :]),
    ('atom_charges', nb.float64[:]),
    ('wfn', Wfn.class_type.instance_type),
]


@nb.jit(forceobj=True)
def initial_position(ne, atom_positions, atom_charges):
    """Initial positions of electrons."""
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
    return r_e


@nb.experimental.jitclass(spec)
class Metropolis:

    def __init__(self, neu, ned, atom_positions, atom_charges, wfn):
        """Metropolis random walk.
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param atom_positions: atomic positions
        :param atom_charges: atomic charges
        :param wfn: instance of Wfn class
        :return:
        """
        self.neu = neu
        self.ned = ned
        self.r_e = np.zeros((neu + ned, 3))
        self.step = 1 / (neu + ned)
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.wfn = wfn

    def random_step(self):
        """Random N-dim square distributed step"""
        ne = self.neu + self.ned
        return np.random.uniform(-self.step, self.step, ne * 3).reshape((ne, 3))

    def make_step(self, p, r_e):
        """Make random step in configuration-by-configuration sampling (CBCS)"""
        new_r_e = r_e + self.random_step()
        e_vectors = subtract_outer(new_r_e, new_r_e)
        n_vectors = -subtract_outer(self.atom_positions, new_r_e)
        new_p = self.wfn.value(e_vectors, n_vectors)
        cond = new_p ** 2 > np.random.random() * p ** 2
        if cond:
            return new_r_e, new_p, cond
        else:
            return r_e, p, cond

    def random_walk(self, steps):
        """Metropolis-Hastings random walk.
        :param steps: steps to walk
        :return:
        """

        r_e = self.r_e
        weight = np.ones((steps, ), np.int64)
        position = np.zeros((steps, r_e.shape[0], r_e.shape[1]))
        function = np.ones((steps, ), np.float64)

        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)
        p = self.wfn.value(e_vectors, n_vectors)

        i = 0
        # first step
        r_e, p, _ = self.make_step(p, r_e)
        position[i] = r_e
        function[i] = p
        # other steps
        for _ in range(1, steps):
            r_e, p, cond = self.make_step(p, r_e)
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
            res[i] = self.wfn.energy(position[i])
        return res

    def jastrow_gradient(self, position):
        """Jastrow gradient with respect to jastrow parameters.
        :param position: random walk positions
        :return:
        """
        r_e = position[0]
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)
        first_res = self.wfn.jastrow.parameters_numerical_d1(e_vectors, n_vectors, self.neu)
        res = np.zeros((position.shape[0], ) + first_res.shape)
        res[0] = first_res

        for i in range(1, position.shape[0]):
            r_e = position[i]
            e_vectors = subtract_outer(r_e, r_e)
            n_vectors = -subtract_outer(self.atom_positions, r_e)
            res[i] = self.wfn.jastrow.parameters_numerical_d1(e_vectors, n_vectors, self.neu)
        return res

    def jastrow_hessian(self, position):
        """Jastrow hessian with respect to jastrow parameters.
        :param position: random walk positions
        :return:
        """
        r_e = position[0]
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)
        first_res = self.wfn.jastrow.parameters_numerical_d2(e_vectors, n_vectors, self.neu)
        res = np.zeros((position.shape[0], ) + first_res.shape)
        res[0] = first_res

        for i in range(1, position.shape[0]):
            r_e = position[i]
            e_vectors = subtract_outer(r_e, r_e)
            n_vectors = -subtract_outer(self.atom_positions, r_e)
            res[i] = self.wfn.jastrow.parameters_numerical_d2(e_vectors, n_vectors, self.neu)
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


def jastrow_parameters_gradient(weight, energy, energy_gradient):
    """
    :param weight:
    :param energy:
    :param energy_gradient:
    :return:
    """
    return 2 * (
        np.average((energy_gradient * energy[:, np.newaxis]), axis=0, weights=weight) -
        np.average(energy_gradient, axis=0, weights=weight) * np.average(energy, weights=weight)
    )


def jastrow_parameters_hessian(weight, energy, energy_gradient, energy_hessian):
    """Lin, Zhang and Rappe (LZR) hessian from
    Optimization of quantum Monte Carlo wave functions by energy minimization.
    :param weight:
    :param energy:
    :param energy_gradient:
    :param energy_hessian:
    :return:
    """
    t1 = np.einsum('ij,ik->ijk', energy_gradient, energy_gradient)
    A = 2 * (
        np.average(energy_hessian * energy[:, np.newaxis, np.newaxis], axis=0, weights=weight) -
        np.average(energy_hessian, axis=0, weights=weight) * np.average(energy, axis=0, weights=weight) -
        np.average(t1 * energy[..., np.newaxis, np.newaxis], axis=0, weights=weight) +
        np.average(t1, axis=0, weights=weight) * np.average(energy, weights=weight)
    )
    t2 = energy_gradient - np.average(energy_gradient, axis=0, weights=weight)
    t3 = (energy - np.average(energy, axis=0, weights=weight))
    B = 4 * np.average(np.einsum('ij,ik->ijk', t2, t2) * t3[..., np.newaxis, np.newaxis], axis=0, weights=weight)
    C = 0.0
    return A + B + C


class VMC:

    def __init__(self, casino):
        self.slater = Slater(
            casino.input.neu, casino.input.ned,
            casino.wfn.nbasis_functions, casino.wfn.first_shells, casino.wfn.orbital_types, casino.wfn.shell_moments,
            casino.wfn.slater_orders, casino.wfn.primitives, casino.wfn.coefficients, casino.wfn.exponents,
            casino.mdet.mo_up, casino.mdet.mo_down, casino.mdet.coeff
        )
        self.jastrow = casino.jastrow and Jastrow(
            casino.input.neu, casino.input.ned,
            casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_mask, casino.jastrow.u_cutoff,
            casino.jastrow.chi_parameters, casino.jastrow.chi_mask, casino.jastrow.chi_cutoff, casino.jastrow.chi_labels,
            casino.jastrow.f_parameters, casino.jastrow.f_mask, casino.jastrow.f_cutoff, casino.jastrow.f_labels,
            casino.jastrow.no_dup_u_term, casino.jastrow.no_dup_chi_term, casino.jastrow.chi_cusp
        )
        self.backflow = casino.backflow and Backflow(
            casino.input.neu, casino.input.ned,
            casino.backflow.trunc, casino.backflow.eta_parameters, casino.backflow.eta_cutoff,
            casino.backflow.mu_parameters, casino.backflow.mu_cutoff, casino.backflow.mu_labels,
            casino.backflow.phi_parameters, casino.backflow.theta_parameters, casino.backflow.phi_cutoff,
            casino.backflow.phi_labels, casino.backflow.phi_irrotational
        )
        self.wfn = Wfn(casino.input.neu, casino.input.ned, casino.wfn.atom_positions, casino.wfn.atom_charges, self.slater, self.jastrow, self.backflow)
        self.neu, self.ned = casino.input.neu, casino.input.ned
        self.metropolis = Metropolis(self.neu, self.ned, casino.wfn.atom_positions, casino.wfn.atom_charges, self.wfn)
        self.metropolis.r_e = initial_position(self.neu + self.ned, self.metropolis.atom_positions, self.metropolis.atom_charges)

    def equilibrate(self, steps):
        weight, _, _ = self.metropolis.random_walk(steps)
        logger.info('dr * electrons = 1.00000, acc_ration = %.5f', weight.size / steps)

    def optimize_vmc_step(self, steps):
        """Optimize vmc step size."""

        def callback(tau, acc_ration):
            """dr = sqrt(3*dtvmc)"""
            logger.info('dr * electrons = %.5f, acc_ration = %.5f', tau[0] * (self.neu + self.ned), acc_ration[0] + 0.5)

        def f(tau):
            self.metropolis.step = tau[0]
            weight, _, _ = self.metropolis.random_walk(steps)
            return weight.size / steps - 0.5

        options = dict(jac_options=dict(alpha=1))
        res = sp.optimize.root(f, [self.metropolis.step], method='diagbroyden', tol=1 / np.sqrt(steps), callback=callback, options=options)
        self.metropolis.step = np.abs(res.x[0])

    def energy(self, steps):
        self.optimize_vmc_step(10000)

        rounds = 10
        E = np.zeros((rounds,))
        check_point_1 = default_timer()
        for i in range(rounds):
            weights, position, _ = self.metropolis.random_walk(steps // rounds)
            energy = self.metropolis.local_energy(position) + self.metropolis.wfn.nuclear_repulsion
            E[i] = np.average(energy, weights=weights)
            mean_energy = np.average(E[:i + 1])
            if i:
                std_err = np.std(E[:i + 1], ddof=0) / np.sqrt(i)
            else:
                std_err = 0

            check_point_2 = default_timer()
            logger.info(f'{E[i]}, {mean_energy}, {std_err}, total time {check_point_2 - check_point_1}')

        E = expand(weights, energy)
        reblock_data = pyblock.blocking.reblock(E)
        # for reblock_iter in reblock_data:
        #     print(reblock_iter)
        opt = pyblock.blocking.find_optimal_block(E.size, reblock_data)
        opt_data = reblock_data[opt[0]]
        logger.info(opt_data)
        logger.info(f'{np.mean(opt_data.mean)} +/- {np.mean(opt_data.std_err) / np.sqrt(opt_data.std_err.size)}')

    def normal_test(self, weight, energy):
        """Test whether energy distribution differs from a normal one."""
        from scipy import stats
        E = expand(weight, energy)
        logger.info('skew = %s, kurtosis = %s', stats.skewtest(E), stats.kurtosistest(E))

    def vmc_variance_minimization(self, steps):
        """Minimise vmc variance by jastrow parameters optimization."""
        bounds = self.metropolis.jastrow.get_bounds()
        weight, position, _ = self.metropolis.random_walk(steps)

        def f(x, *args, **kwargs):
            self.metropolis.jastrow.set_parameters(x)
            energy = self.metropolis.local_energy(position) + self.metropolis.wfn.nuclear_repulsion
            energy_average = np.average(energy, weights=weight)
            energy_variance = np.average((energy - energy_average) ** 2, weights=weight)
            std_err = np.sqrt(energy_variance / weight.sum())
            logger.info('energy = %.5f +- %.5f, variance = %.5f', energy_average, std_err, energy_variance)
            return expand(weight, energy) - energy_average

        def jac(x, *args, **kwargs):
            self.metropolis.jastrow.set_parameters(x)
            return expand(weight, self.metropolis.jastrow_gradient(position))

        parameters = self.metropolis.jastrow.get_parameters()
        res = sp.optimize.least_squares(
            f, parameters, jac=jac, bounds=bounds, method='trf', max_nfev=20,
            x_scale='jac', loss='linear', tr_solver='lsmr', tr_options=dict(show=False, regularize=False),
            verbose=2
        )
        return res

    def vmc_energy_minimization(self, steps):
        """Minimise vmc energy by jastrow parameters optimization.
        Gradient only for : CG, BFGS, L-BFGS-B, TNC, SLSQP
        Gradient and Hessian is required for: Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact, trust-constr
        Constraints definition only for: COBYLA, SLSQP and trust-constr
        """
        bounds = self.metropolis.jastrow.get_bounds()
        weight, position, _ = self.metropolis.random_walk(steps)

        def callback(x, *args):
            logger.info('inner iteration x = %s', x)
            self.metropolis.jastrow.set_parameters(x)
            weight, position, _ = self.metropolis.random_walk(steps)

        def f(x, *args):
            self.metropolis.jastrow.set_parameters(x)
            energy = self.metropolis.local_energy(position) + self.metropolis.wfn.nuclear_repulsion
            energy_gradient = self.metropolis.jastrow_gradient(position)
            mean_energy = np.average(energy, weights=weight)
            mean_energy_gradient = jastrow_parameters_gradient(weight, energy, energy_gradient)
            energy_average = np.average(energy, weights=weight)
            energy_variance = np.average((energy - energy_average) ** 2, weights=weight)
            std_err = np.sqrt(energy_variance / weight.sum())
            logger.info('energy = %.5f +- %.5f, variance = %.5f', energy_average, std_err, energy_variance)
            return mean_energy, mean_energy_gradient

        def hess(x, *args):
            self.metropolis.jastrow.set_parameters(x)
            energy = self.metropolis.local_energy(position) + self.metropolis.wfn.nuclear_repulsion
            energy_gradient = self.metropolis.jastrow_gradient(position)
            energy_hessian = self.metropolis.jastrow_hessian(position)
            mean_energy_hessian = jastrow_parameters_hessian(weight, energy, energy_gradient, energy_hessian)
            logger.info('hessian = %s', mean_energy_hessian)
            return mean_energy_hessian

        parameters = self.metropolis.jastrow.get_parameters()
        options = dict(disp=True, maxfun=50)
        res = sp.optimize.minimize(f, parameters, method='TNC', jac=True, bounds=list(zip(*bounds)), options=options, callback=callback)
        # options = dict(disp=True)
        # res = sp.optimize.minimize(f, parameters, method='trust-ncg', jac=True, hess=hess, options=options, callback=callback)
        return res

    def varmin(self, steps, opt_cycles):
        for _ in range(opt_cycles):
            self.optimize_vmc_step(10000)
            res = self.vmc_variance_minimization(steps)
            logger.info('x = %s', res.x)
            self.metropolis.jastrow.set_parameters(res.x)

    def emin(self, steps, opt_cycles):
        for _ in range(opt_cycles):
            self.optimize_vmc_step(10000)
            res = self.vmc_energy_minimization(steps)
            logger.info('x = %s', res.x)
            self.metropolis.jastrow.set_parameters(res.x)


def main(casino):
    """Configuration-by-configuration sampling (CBCS)
    Should be pure python function.
    """

    vmc = VMC(casino)
    vmc.equilibrate(casino.input.vmc_equil_nstep)
    vmc.energy(casino.input.vmc_nstep)
    # vmc.varmin(casino.input.vmc_opt_nstep, 5)
    # vmc.emin(casino.input.vmc_opt_nstep, 5)


if __name__ == '__main__':
    """
    be HF/cc-pVQZ

    stat = 1024 * 1024 * 1024

    """

    # path = 'test/gwfn/h/HF/cc-pVQZ/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/u_term/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_vmc/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/u_term_test/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/chi_term/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    path = 'test/gwfn/be/HF/cc-pVQZ/varmin_BF/8_8_33__0_9_00/'
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
    # path = 'test/stowfn/ne/HF/QZ4P/'
    # path = 'test/stowfn/ar/HF/QZ4P/'
    # path = 'test/stowfn/kr/HF/QZ4P/'
    # path = 'test/stowfn/o3/HF/QZ4P/'
    # path = 'test/stowfn/be/HF/QZ4P/varmin_BF/0_0_00__9_0_00/'
    # path = 'test/stowfn/be/HF/QZ4P/varmin_BF/0_0_00__9_9_00/'
    # path = 'test/stowfn/be/HF/QZ4P/varmin_BF/8_8_33__0_9_00/'
    # path = 'test/stowfn/be/HF/QZ4P/varmin_BF/8_8_33__9_9_00/'
    # path = 'test/stowfn/ne/HF/QZ4P/VMC_OPT/emin/legacy/f_term/'
    casino = Casino(path)
    main(casino)
