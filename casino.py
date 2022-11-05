#!/usr/bin/env python3

import os
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer
from cusp import CuspFactory, TestCuspFactory
from slater import Slater
from jastrow import Jastrow
from backflow import Backflow
from markovchain import MarkovChain, vmc_observable
from wfn import Wfn

os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import numpy as np
import numba as nb
from scipy.optimize import least_squares, minimize, root, Bounds
import matplotlib.pyplot as plt

from readers.casino import CasinoConfig
from sem import correlated_sem
from psutil import cpu_count
from logger import logging

np.random.seed(31415926)

logger = logging.getLogger('vmc')


def jastrow_parameters_gradient(energy, energy_gradient):
    """
    :param energy:
    :param energy_gradient:
    :return:
    """
    return 2 * (
        np.average((energy_gradient * energy[:, np.newaxis]), axis=0) -
        np.average(energy_gradient, axis=0) * np.average(energy)
    )


def jastrow_parameters_hessian(energy, energy_gradient, energy_hessian):
    """Lin, Zhang and Rappe (LZR) hessian from
    Optimization of quantum Monte Carlo wave functions by energy minimization.
    :param energy:
    :param energy_gradient:
    :param energy_hessian:
    :return:
    """
    t1 = np.einsum('ij,ik->ijk', energy_gradient, energy_gradient)
    A = 2 * (
        np.average(energy_hessian * energy[:, np.newaxis, np.newaxis], axis=0) -
        np.average(energy_hessian, axis=0) * np.average(energy, axis=0) -
        np.average(t1 * energy[..., np.newaxis, np.newaxis], axis=0) +
        np.average(t1, axis=0) * np.average(energy)
    )
    t2 = energy_gradient - np.average(energy_gradient, axis=0)
    t3 = (energy - np.average(energy, axis=0))
    B = 4 * np.average(np.einsum('ij,ik->ijk', t2, t2) * t3[..., np.newaxis, np.newaxis], axis=0)
    C = 0.0
    return A + B + C


class Casino:

    def __init__(self, config_path: str):
        """Casino workflow.
        :param config_path: path to config file
        """
        self.config_path = config_path
        self.config = CasinoConfig(self.config_path)
        self.config.read(self.config_path)
        self.num_proc = cpu_count(logical=False)
        self.neu, self.ned = self.config.input.neu, self.config.input.ned
        self.r_e = self.initial_position(self.neu + self.ned, self.config.wfn.atom_positions, self.config.wfn.atom_charges)
        self.r_e += np.random.uniform(-1, 1, (self.neu + self.ned) * 3).reshape((self.neu + self.ned, 3))

        if self.config.input.cusp_correction:
            cusp = CuspFactory(
                self.config.input.neu, self.config.input.ned, self.config.wfn.mo_up, self.config.wfn.mo_down,
                self.config.mdet.permutation_up, self.config.mdet.permutation_down,
                self.config.wfn.first_shells, self.config.wfn.shell_moments, self.config.wfn.primitives,
                self.config.wfn.coefficients, self.config.wfn.exponents,
                self.config.wfn.atom_positions, self.config.wfn.atom_charges
            ).create()
            # cusp = TestCuspFactory(
            #     self.config.input.neu, self.config.input.ned, self.config.wfn.mo_up, self.config.wfn.mo_down,
            #     self.config.mdet.permutation_up, self.config.mdet.permutation_down,
            #     self.config.wfn.first_shells, self.config.wfn.shell_moments, self.config.wfn.primitives,
            #     self.config.wfn.coefficients, self.config.wfn.exponents
            # ).create()
        else:
            cusp = None

        slater = Slater(
            self.config.input.neu, self.config.input.ned,
            self.config.wfn.nbasis_functions, self.config.wfn.first_shells, self.config.wfn.orbital_types, self.config.wfn.shell_moments,
            self.config.wfn.slater_orders, self.config.wfn.primitives, self.config.wfn.coefficients, self.config.wfn.exponents,
            self.config.wfn.mo_up, self.config.wfn.mo_down, self.config.mdet.permutation_up, self.config.mdet.permutation_down, self.config.mdet.coeff, cusp
        )
        jastrow = (self.config.jastrow or None) and Jastrow(
            self.config.input.neu, self.config.input.ned,
            self.config.jastrow.trunc, self.config.jastrow.u_parameters, self.config.jastrow.u_parameters_optimizable, self.config.jastrow.u_mask,
            self.config.jastrow.u_cutoff, self.config.jastrow.u_cusp_const,
            self.config.jastrow.chi_parameters, self.config.jastrow.chi_parameters_optimizable, self.config.jastrow.chi_mask, self.config.jastrow.chi_cutoff,
            self.config.jastrow.chi_labels, self.config.jastrow.chi_cusp,
            self.config.jastrow.f_parameters, self.config.jastrow.f_parameters_optimizable, self.config.jastrow.f_mask, self.config.jastrow.f_cutoff,
            self.config.jastrow.f_labels,
            self.config.jastrow.no_dup_u_term, self.config.jastrow.no_dup_chi_term
        )
        backflow = (self.config.backflow or None) and Backflow(
            self.config.input.neu, self.config.input.ned,
            self.config.backflow.trunc, self.config.backflow.eta_parameters, self.config.backflow.eta_parameters_optimizable, self.config.backflow.eta_mask,
            self.config.backflow.eta_cutoff,
            self.config.backflow.mu_parameters, self.config.backflow.mu_parameters_optimizable, self.config.backflow.mu_mask, self.config.backflow.mu_cutoff,
            self.config.backflow.mu_cusp, self.config.backflow.mu_labels,
            self.config.backflow.phi_parameters, self.config.backflow.phi_parameters_optimizable, self.config.backflow.phi_mask,
            self.config.backflow.theta_parameters, self.config.backflow.theta_parameters_optimizable, self.config.backflow.theta_mask,
            self.config.backflow.phi_cutoff, self.config.backflow.phi_cusp, self.config.backflow.phi_labels, self.config.backflow.phi_irrotational,
            self.config.backflow.ae_cutoff
        )
        wfn = Wfn(
            self.config.input.neu, self.config.input.ned, self.config.wfn.atom_positions, self.config.wfn.atom_charges, slater, jastrow, backflow
        )

        if self.config.input.vmc_method == 1:
            # EBES
            step = 1 / np.log(self.neu + self.ned)
        elif self.config.input.vmc_method == 3:
            # CBCS
            step = 1 / (self.neu + self.ned)
        else:
            # wrong method
            step = 0
        self.markovchain = MarkovChain(step, wfn)

    def __reduce__(self):
        """to fix TypeError: cannot pickle '_io.TextIOWrapper' object"""
        step = self.markovchain.step
        parameters = self.markovchain.wfn.get_parameters(self.config.jastrow is not None, self.config.backflow is not None)
        return self.__class__, (self.config_path, ), {'r_e': self.r_e, 'step': step, 'parameters': parameters}

    def __setstate__(self, state):
        """set state"""
        self.r_e = state['r_e']
        self.markovchain.step = state['step']
        self.markovchain.wfn.set_parameters(state['parameters'], self.config.jastrow is not None, self.config.backflow is not None)

    def parallel_execution(self, function, *args):
        """Parallel execution of methods
        https://github.com/numba/numba/issues/1846

        Sharing big NumPy arrays across python processes
        https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2

        :param function: callable objects
        :param args: arguments
        :return:
        """
        if self.num_proc == 1:
            return [function(*args)]
        else:
            # FIXME: if not cached numba compiled it everytime
            with ProcessPoolExecutor(max_workers=self.num_proc) as executor:
                futures = [executor.submit(function, *args) for _ in range(self.num_proc)]
                # to get task results in order they were submitted
                return [res.result() for res in futures]

    def parallel_execution_map(self, function, *args):
        """Parallel execution of methods
        https://github.com/numba/numba/issues/1846

        Sharing big NumPy arrays across python processes
        https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2

        :param function: callable objects
        :param args: arguments iterables
        :return:
        """
        # FIXME: if not cached numba compiled it everytime
        with ProcessPoolExecutor(max_workers=self.num_proc) as executor:
            return list(executor.map(function, *args))

    def initial_position(self, ne, atom_positions, atom_charges):
        """Initial positions of electrons."""
        natoms = atom_positions.shape[0]
        r_e = np.zeros((ne, 3))
        for i in range(ne):
            r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
        return r_e

    def run(self):
        """Run Casino workflow.
        """
        self.equilibrate(self.config.input.vmc_equil_nstep)
        if self.config.input.runtype == 'vmc':
            start = default_timer()
            logger.info(
                ' ====================================\n'
                ' PERFORMING A SINGLE VMC CALCULATION.\n'
                ' ====================================\n\n'
            )
            # FIXME: in EBEC nstep = vmc_nstep * (neu + ned)
            self.optimize_vmc_step(10000)
            self.vmc_energy_accumulation()
            stop = default_timer()
            logger.info(
                f' =========================================================================\n\n'
                f' Total PyCasino real time : : :    {stop - start:.4f}'
            )
        elif self.config.input.runtype == 'vmc_opt':
            if self.config.input.opt_method == 'varmin':
                start = default_timer()
                self.config.write('.', 0)
                self.optimize_vmc_step(10000)
                self.vmc_energy_accumulation()
                for i in range(self.config.input.opt_cycles):
                    res = self.vmc_variance_minimization(
                        self.config.input.vmc_nconfig_write,
                        self.config.input.vmc_decorr_period,
                        self.config.input.opt_jastrow,
                        self.config.input.opt_backflow
                    )
                    self.markovchain.wfn.set_parameters(res.x, self.config.input.opt_jastrow, self.config.input.opt_backflow)
                    logger.info(res.x / self.markovchain.wfn.get_parameters_scale(self.config.input.opt_jastrow, self.config.input.opt_backflow))
                    self.config.jastrow.u_cutoff = self.markovchain.wfn.jastrow.u_cutoff
                    self.config.write('.', i + 1)
                    self.optimize_vmc_step(10000)
                    self.vmc_energy_accumulation()
                stop = default_timer()
                logger.info(
                    f' =========================================================================\n\n'
                    f' Total PyCasino real time : : :    {stop - start:.4f}'
                )
            elif self.config.input.opt_method == 'emin':
                start = default_timer()
                self.config.write('.', 0)
                self.optimize_vmc_step(10000)
                self.vmc_energy_accumulation()
                for i in range(self.config.input.opt_cycles):
                    res = self.vmc_energy_minimization(
                        self.config.input.vmc_nconfig_write,
                        self.config.input.vmc_decorr_period,
                        self.config.input.opt_jastrow,
                        self.config.input.opt_backflow
                    )
                    self.markovchain.wfn.set_parameters(res.x, self.config.input.opt_jastrow, self.config.input.opt_backflow)
                    logger.info(res.x / self.markovchain.wfn.get_parameters_scale(self.config.input.opt_jastrow, self.config.input.opt_backflow))
                    self.config.jastrow.u_cutoff = self.markovchain.wfn.jastrow.u_cutoff
                    self.config.write('.', i + 1)
                    self.optimize_vmc_step(10000)
                    self.vmc_energy_accumulation()
                stop = default_timer()
                logger.info(
                    f' =========================================================================\n\n'
                    f' Total PyCasino real time : : :    {stop - start:.4f}'
                )
        elif self.config.input.runtype == 'vmc_dmc':
            self.optimize_vmc_step(10000)
            # FIXME: decorr_period for dmc?
            block_start = default_timer()
            condition, position = self.markovchain.vmc_random_walk(self.r_e, self.config.input.vmc_nstep, 1)
            energy = vmc_observable(condition, position, self.markovchain.wfn.energy) + self.markovchain.wfn.nuclear_repulsion
            block_stop = default_timer()
            logger.info(
                f' =========================================================================\n'
                f' In block : {1}\n'
                f'  Number of VMC steps           = {self.config.input.vmc_nstep}\n\n'
                f'  Block average energies (au)\n\n'
                f'  Total energy                       (au) =       {energy.mean():18.12f}\n'
                f'  Standard error                        +/-       {energy.std():18.12f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
            r_e_list = [position[-i] for i in range(self.config.input.vmc_nconfig_write)]
            # FIXME: local variables?
            self.markovchain.step = self.config.input.dtdmc
            r_e_list = self.dmc_energy_equilibration(r_e_list)
            r_e_list = self.dmc_energy_accumulation(r_e_list)

    def equilibrate(self, steps):
        """
        :param steps: burn-in period
        :return:
        """
        condition, position = self.markovchain.vmc_random_walk(self.r_e, steps, 1)
        self.r_e = position[-1]
        logger.info(
            f'Running VMC equilibration ({steps} moves).'
        )
        logger.debug('dr * electrons = 1.00000, acc_ration = %.5f', condition.mean())

    def optimize_vmc_step(self, steps, acceptance_rate=0.5):
        """Optimize vmc step size."""

        def callback(tau, acc_ration):
            """dr = sqrt(3*dtvmc)"""
            logger.debug('dr * electrons = %.5f, acc_ration = %.5f', tau[0] * (self.neu + self.ned), acc_ration[0] + acceptance_rate)

        def f(tau):
            self.markovchain.step = tau[0]
            logger.debug('dr * electrons = %.5f', tau[0] * (self.neu + self.ned))
            if tau[0] > 0:
                condition, position = self.markovchain.vmc_random_walk(self.r_e, steps, 1)
                self.r_e = position[-1]
                acc_ration = condition.mean()
            else:
                acc_ration = 1
            return acc_ration - acceptance_rate

        options = dict(jac_options=dict(alpha=1))
        res = root(f, [self.markovchain.step], method='diagbroyden', tol=1/np.sqrt(steps), callback=callback, options=options)
        self.markovchain.step = np.abs(res.x[0])

    def get_decorr_period(self):
        """Decorr period"""
        if self.config.input.vmc_decorr_period == 0:
            return 3
        else:
            return self.config.input.vmc_decorr_period

    def vmc_energy(self, steps, decorr_period):
        """wrapper for VMC energy
            https://github.com/numba/numba/issues/4830
            https://github.com/numba/numba/issues/6522  - use structrefs instead of jitclasses
            https://numba.readthedocs.io/en/stable/extending/high-level.html#implementing-mutable-structures
        """
        condition, position = self.markovchain.vmc_random_walk(self.r_e, steps, decorr_period)
        self.r_e = position[-1]
        return vmc_observable(condition, position, self.markovchain.wfn.energy) + self.markovchain.wfn.nuclear_repulsion

    def vmc_energy_accumulation(self):
        """VMC energy accumulation"""
        steps = self.config.input.vmc_nstep
        nblock = self.config.input.vmc_nblock

        decorr_period = self.get_decorr_period()
        energy_block_mean = np.zeros(shape=(nblock, self.num_proc))
        energy_block_sem = np.zeros(shape=(nblock, self.num_proc))
        energy_block_var = np.zeros(shape=(nblock, self.num_proc))
        logger.info(
            f'Starting VMC.'
        )
        for i in range(nblock):
            block_start = default_timer()
            for j, energy in enumerate(self.parallel_execution(self.vmc_energy, steps // nblock // self.num_proc, decorr_period)):
                energy_block_mean[i, j] = energy.mean()
                energy_block_sem[i, j] = correlated_sem(energy)
                energy_block_var[i, j] = energy.var()
            block_stop = default_timer()
            logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n'
                f'  Number of VMC steps           = {steps // nblock}\n\n'
                f'  Block average energies (au)\n\n'
                f'  Total energy                       (au) =       {energy_block_mean[i].mean():18.12f}\n'
                f'  Standard error                        +/-       {energy_block_sem[i].mean() / np.sqrt(self.num_proc):18.12f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
        logger.info(
            f' =========================================================================\n'
            f' FINAL RESULT:\n\n'
            f'  VMC energy (au)    Standard error      Correction for serial correlation\n'
            f' {energy_block_mean.mean():.12f} +/- {energy_block_sem.mean() / np.sqrt(nblock * self.num_proc):.12f}      On-the-fly reblocking method\n\n'
            f' Sample variance of E_L (au^2/sim.cell) : {energy_block_var.mean():.12f}\n\n'
        )

    def dmc_energy_equilibration(self, r_e_list):
        """DMC energy equilibration"""
        logger.info(
            ' ===========================================\n'
            ' PERFORMING A DMC EQUILIBRATION CALCULATION.\n'
            ' ===========================================\n\n'
        )

        steps = self.config.input.dmc_equil_nstep
        nblock = self.config.input.dmc_equil_nblock
        energy_block_mean = np.zeros(shape=(nblock,))
        energy_block_sem = np.zeros(shape=(nblock,))

        for i in range(nblock):
            block_start = default_timer()
            energy, r_e_list = self.markovchain.dmc_random_walk(r_e_list, steps // nblock, self.config.input.dmc_target_weight)
            energy_block_mean[i] = energy.mean()
            energy_block_sem[i] = correlated_sem(energy)
            block_stop = default_timer()
            logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n'
                f'  Number of DMC steps           = {steps // nblock}\n\n'
                f'  Block average energies (au)\n\n'
                f'  Total energy                       (au) =       {energy_block_mean[i]:18.12f}\n'
                f'  Standard error                        +/-       {energy_block_sem[i]:18.12f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
        return r_e_list

    def dmc_energy_accumulation(self, r_e_list):
        """DMC energy accumulation"""
        logger.info(
            ' =====================================================\n'
            ' PERFORMING A DMC STATISTICS-ACCUMULATION CALCULATION.\n'
            ' =====================================================\n\n'
        )

        steps = self.config.input.dmc_stats_nstep
        nblock = self.config.input.dmc_stats_nblock
        energy_block_mean = np.zeros(shape=(nblock,))
        energy_block_sem = np.zeros(shape=(nblock,))

        for i in range(nblock):
            block_start = default_timer()
            energy, r_e_list = self.markovchain.dmc_random_walk(r_e_list, steps // nblock, self.config.input.dmc_target_weight)
            energy_block_mean[i] = energy.mean()
            energy_block_sem[i] = correlated_sem(energy)
            block_stop = default_timer()
            logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n'
                f'  Number of DMC steps           = {steps // nblock}\n\n'
                f'  Block average energies (au)\n\n'
                f'  Total energy                       (au) =       {energy_block_mean[i]:18.12f}\n'
                f'  Standard error                        +/-       {energy_block_sem[i]:18.12f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
        return r_e_list

    def normal_test(self, energy):
        """Test whether energy distribution differs from a normal one."""
        from scipy import stats
        logger.info(f'skew = {stats.skewtest(energy)}, kurtosis = {stats.kurtosistest(energy)}')
        plt.hist(
            energy,
            bins='auto',
            range=(energy.mean() - 5 * energy.std(), energy.mean() + 5 * energy.std()),
            density=True
        )
        plt.savefig('hist.png')
        plt.clf()

    def vmc_variance_minimization(self, steps, decorr_period, opt_jastrow, opt_backflow):
        """Minimise vmc variance by jastrow parameters optimization.
        https://github.com/scipy/scipy/issues/10634
        """
        condition, position = self.markovchain.vmc_random_walk(self.r_e, steps, decorr_period)

        def fun(x, *args, **kwargs):
            self.markovchain.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.markovchain.wfn.energy)
            return energy - energy.mean()

        return least_squares(
            fun, x0=self.markovchain.wfn.get_parameters(opt_jastrow, opt_backflow), jac='2-point', method='trf',
            max_nfev=7, x_scale=self.markovchain.wfn.get_parameters_scale(opt_jastrow, opt_backflow), loss='linear',
            f_scale=1, tr_solver='lsmr', tr_options=dict(regularize=False), verbose=2
        )

    def vmc_energy_minimization(self, steps, decorr_period, opt_jastrow=True, opt_backflow=True):
        """Minimise vmc energy by jastrow parameters optimization.
        Gradient only for : CG, BFGS, L-BFGS-B, TNC, SLSQP
        Gradient and Hessian is required for: Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact, trust-constr
        Constraints definition only for: COBYLA, SLSQP and trust-constr.
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods.

        SciPy, оптимизация с условиями - https://habr.com/ru/company/ods/blog/448054/
        """
        bounds = Bounds(*self.markovchain.wfn.jastrow.get_bounds(), keep_feasible=True)
        condition, position = self.markovchain.vmc_random_walk(self.r_e, steps, decorr_period)

        def fun(x, *args):
            self.markovchain.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.markovchain.wfn.energy)
            energy_gradient = vmc_observable(condition, position, self.markovchain.wfn.jastrow_parameters_numerical_d1)
            mean_energy_gradient = jastrow_parameters_gradient(energy, energy_gradient)
            return energy.mean(), mean_energy_gradient

        def hess(x, *args):
            self.markovchain.wfn.jastrow.set_parameters(x, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.markovchain.wfn.energy)
            energy_gradient = vmc_observable(condition, position, self.markovchain.wfn.jastrow_parameters_numerical_d1)
            energy_hessian = vmc_observable(condition, position, self.markovchain.wfn.jastrow_parameters_numerical_d2)
            mean_energy_hessian = jastrow_parameters_hessian(energy, energy_gradient, energy_hessian)
            logger.info('hessian = %s', mean_energy_hessian)
            return mean_energy_hessian

        parameters = self.markovchain.wfn.get_parameters(opt_jastrow, opt_backflow)
        res = minimize(fun, parameters, method='TNC', jac=True, bounds=bounds, options=dict(disp=True, maxfun=10))
        # res = minimize(f, parameters, method='trust-ncg', jac=True, hess=hess, options=dict(disp=True)
        return res


if __name__ == '__main__':
    """Tests
    """
    # path = 'test/gwfn/He/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Be/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Be/MP2-CASSCF(2.4)/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/N/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Ne/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Ar/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Kr/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/O3/HF/cc-pVQZ/CBCS/Slater/'

    # path = 'test/gwfn/He/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/Be/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/Be/MP2-CASSCF(2.4)/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/N/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/Ne/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/Ar/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/Kr/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/O3/HF/cc-pVQZ/CBCS/Jastrow/'

    # path = 'test/gwfn/He/HF/cc-pVQZ/CBCS/Backflow/'
    # path = 'test/gwfn/Be/HF/cc-pVQZ/CBCS/Backflow/'
    # path = 'test/gwfn/Be/MP2-CASSCF(2.4)/cc-pVQZ/CBCS/Backflow/'
    # path = 'test/gwfn/N/HF/cc-pVQZ/CBCS/Backflow/'
    # path = 'test/gwfn/Ne/HF/cc-pVQZ/CBCS/Backflow/'
    # path = 'test/gwfn/Ar/HF/cc-pVQZ/CBCS/Backflow/'
    # path = 'test/gwfn/Kr/HF/cc-pVQZ/CBCS/Backflow/'
    # path = 'test/gwfn/O3/HF/cc-pVQZ/CBCS/Backflow/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/N/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Slater/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Jastrow_varmin/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Jastrow_varmin/'
    # path = 'test/stowfn/N/HF/QZ4P/CBCS/Jastrow_varmin/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Jastrow_varmin/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Jastrow_varmin/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Jastrow_varmin/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Jastrow_varmin/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Backflow_varmin/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Backflow_varmin/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/N/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Jastrow/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/N/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Backflow/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Jastrow_dmc/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Jastrow_dmc/'
    # path = 'test/stowfn/N/HF/QZ4P/CBCS/Jastrow_dmc/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Jastrow_dmc/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Jastrow_dmc/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Jastrow_dmc/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Jastrow_dmc/'

    # Casino(path).run()
