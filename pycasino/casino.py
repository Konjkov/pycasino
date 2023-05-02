#!/usr/bin/env python3
import os
import sys
import argparse
from timeit import default_timer
from numpy_config import np
from mpi4py import MPI
import scipy as sp
from scipy.optimize import least_squares, minimize, curve_fit, line_search
import matplotlib.pyplot as plt

from cusp import CuspFactory
from slater import Slater
from jastrow import Jastrow
from backflow import Backflow
from markovchain import VMCMarkovChain, DMCMarkovChain, vmc_observable
from wfn import Wfn
from readers.casino import CasinoConfig
from sem import correlated_sem
from logger import logging, StreamToLogger


# @nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def energy_parameters_gradient(energy, wfn_gradient):
    """Gradient estimator of local energy from
    Optimization of quantum Monte Carlo wave functions by energy minimization.
    Julien Toulouse, C. J. Umrigar
    :param energy:
    :param wfn_gradient:
    :return:
    """
    return 2 * (
        # numba doesn't support kwarg for mean
        np.mean(wfn_gradient * np.expand_dims(energy, 1), axis=0) -
        np.mean(wfn_gradient, axis=0) * np.mean(energy)
    )


# @nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def energy_parameters_hessian(wfn_gradient, wfn_hessian, energy, energy_gradient):
    """Hessian estimators of local energy from
    Optimization of quantum Monte Carlo wave functions by energy minimization.
    Julien Toulouse, C. J. Umrigar
    :param wfn_gradient:
    :param wfn_hessian: wfn_hessian - np.outer(wfn_gradient, wfn_gradient)
    :param energy:
    :param energy_gradient:
    :return:
    """
    mean_energy = np.mean(energy)
    mean_wfn_gradient = np.mean(wfn_gradient, axis=0)
    A = 2 * (
        np.mean(wfn_hessian * np.expand_dims(energy, (1, 2)), axis=0) -
        np.mean(wfn_hessian, axis=0) * mean_energy -
        np.mean(np.expand_dims(wfn_gradient, 1) * np.expand_dims(wfn_gradient, 2) * np.expand_dims(energy, (1, 2)), axis=0) +
        np.mean(np.expand_dims(wfn_gradient, 1) * np.expand_dims(wfn_gradient, 2), axis=0) * mean_energy
    )
    t2 = wfn_gradient - mean_wfn_gradient
    B = 4 * np.mean(
        np.expand_dims(t2, 1) *
        np.expand_dims(t2, 2) *
        np.expand_dims(energy - mean_energy, (1, 2)),
        axis=0
    )
    # Umrigar and Filippi
    mean_energy_gradient = np.mean(energy_gradient, axis=0)
    D = (
        np.mean(np.expand_dims(wfn_gradient, 1) * np.expand_dims(energy_gradient, 2), axis=0) +
        np.mean(np.expand_dims(wfn_gradient, 2) * np.expand_dims(energy_gradient, 1), axis=0) -
        np.outer(mean_wfn_gradient, mean_energy_gradient) -
        np.outer(mean_energy_gradient, mean_wfn_gradient)
    )
    return A + B + D


# @nb.njit(nogil=True, parallel=False, cache=True)
def overlap_matrix(wfn_gradient):
    """Symmetric overlap matrix S"""
    size = wfn_gradient.shape[1] + 1
    S = np.zeros(shape=(size, size),)
    S[0, 0] = 1
    # numba doesn't support kwarg for mean
    mean_wfn_gradient = np.mean(wfn_gradient, axis=0)
    S[1:, 1:] = (
        np.mean(np.expand_dims(wfn_gradient, 1) * np.expand_dims(wfn_gradient, 2), axis=0) -
        np.outer(mean_wfn_gradient, mean_wfn_gradient)
    )
    return S


# @nb.njit(nogil=True, parallel=False, cache=True)
def hamiltonian_matrix(wfn_gradient, energy, energy_gradient):
    """Hamiltonian matrix H"""
    size = wfn_gradient.shape[1] + 1
    H = np.zeros(shape=(size, size))
    mean_energy = np.mean(energy)
    H[0, 0] = mean_energy
    H[1:, 0] = (
        # numba doesn't support kwarg for mean
        np.mean(wfn_gradient * np.expand_dims(energy, 1), axis=0) -
        np.mean(wfn_gradient, axis=0) * mean_energy
    )
    H[0, 1:] = H[1:, 0] + np.mean(energy_gradient, axis=0)
    mean_wfn_gradient = np.mean(wfn_gradient, axis=0)
    mean_energy_gradient = np.mean(energy_gradient, axis=0)
    mean_wfn_gradient_energy = np.mean(wfn_gradient * np.expand_dims(energy, 1), axis=0)
    H[1:, 1:] = (
        np.mean(np.expand_dims(wfn_gradient, 1) * np.expand_dims(wfn_gradient, 2) * np.expand_dims(energy, (1, 2)), axis=0) -
        np.outer(mean_wfn_gradient, mean_wfn_gradient_energy) -
        np.outer(mean_wfn_gradient_energy, mean_wfn_gradient) +
        np.outer(mean_wfn_gradient, mean_wfn_gradient) * mean_energy +
        np.mean(np.expand_dims(wfn_gradient, 2) * np.expand_dims(energy_gradient, 1), axis=0) -
        np.outer(mean_wfn_gradient, mean_energy_gradient)
    )
    return H


class Casino:

    def __init__(self, config_path: str):
        """Casino workflow.
        :param config_path: path to config file
        """
        self.mpi_comm = MPI.COMM_WORLD
        self.config = CasinoConfig(config_path)
        self.config.read()
        self.neu, self.ned = self.config.input.neu, self.config.input.ned
        self.logger = logging.getLogger('vmc')
        self.root = self.mpi_comm.rank == 0
        if self.root:
            # to redirect scipy.optimize stdout to log-file
            sys.stdout = StreamToLogger(self.logger, logging.INFO)
            # sys.stderr = StreamToLogger(self.logger, logging.ERROR)
        else:
            self.logger.level = logging.ERROR

        if self.config.input.cusp_correction:
            cusp = CuspFactory(
                self.config.input.neu, self.config.input.ned, self.config.wfn.mo_up, self.config.wfn.mo_down,
                self.config.mdet.permutation_up, self.config.mdet.permutation_down,
                self.config.wfn.first_shells, self.config.wfn.shell_moments, self.config.wfn.primitives,
                self.config.wfn.coefficients, self.config.wfn.exponents,
                self.config.wfn.atom_positions, self.config.wfn.atom_charges
            ).create()
        else:
            cusp = None

        slater = Slater(
            self.config.input.neu, self.config.input.ned,
            self.config.wfn.nbasis_functions, self.config.wfn.first_shells, self.config.wfn.orbital_types, self.config.wfn.shell_moments,
            self.config.wfn.slater_orders, self.config.wfn.primitives, self.config.wfn.coefficients, self.config.wfn.exponents,
            self.config.wfn.mo_up, self.config.wfn.mo_down, self.config.mdet.permutation_up, self.config.mdet.permutation_down, self.config.mdet.coeff, cusp
        )

        if self.config.jastrow:
            jastrow = Jastrow(
                self.config.input.neu, self.config.input.ned,
                self.config.jastrow.trunc, self.config.jastrow.u_parameters, self.config.jastrow.u_parameters_optimizable,
                self.config.jastrow.u_cutoff, self.config.jastrow.u_cusp_const,
                self.config.jastrow.chi_parameters, self.config.jastrow.chi_parameters_optimizable, self.config.jastrow.chi_cutoff,
                self.config.jastrow.chi_labels, self.config.jastrow.chi_cusp,
                self.config.jastrow.f_parameters, self.config.jastrow.f_parameters_optimizable, self.config.jastrow.f_cutoff, self.config.jastrow.f_labels,
                self.config.jastrow.no_dup_u_term, self.config.jastrow.no_dup_chi_term
            )
        else:
            jastrow = None

        if self.config.backflow:
            backflow = Backflow(
                self.config.input.neu, self.config.input.ned,
                self.config.backflow.trunc, self.config.backflow.eta_parameters, self.config.backflow.eta_parameters_optimizable,
                self.config.backflow.eta_cutoff,
                self.config.backflow.mu_parameters, self.config.backflow.mu_parameters_optimizable, self.config.backflow.mu_cutoff,
                self.config.backflow.mu_cusp, self.config.backflow.mu_labels,
                self.config.backflow.phi_parameters, self.config.backflow.phi_parameters_optimizable,
                self.config.backflow.theta_parameters, self.config.backflow.theta_parameters_optimizable,
                self.config.backflow.phi_cutoff, self.config.backflow.phi_cusp, self.config.backflow.phi_labels, self.config.backflow.phi_irrotational,
                self.config.backflow.ae_cutoff
            )
        else:
            backflow = None

        self.wfn = Wfn(
            self.config.input.neu, self.config.input.ned, self.config.wfn.atom_positions, self.config.wfn.atom_charges, slater, jastrow, backflow
        )

        self.vmc_markovchain = VMCMarkovChain(
            self.initial_position(self.config.wfn.atom_positions, self.config.wfn.atom_charges),
            self.approximate_step_size,
            self.wfn
        )

    def initial_position(self, atom_positions, atom_charges):
        """Initial positions of electrons."""
        ne = self.neu + self.ned
        natoms = atom_positions.shape[0]
        r_e = np.zeros((ne, 3))
        for i in range(ne):
            # electrons randomly centered on atoms
            r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
        return r_e + np.random.uniform(-1, 1, ne * 3).reshape(ne, 3)

    @property
    def approximate_step_size(self):
        """Approximation to VMC step size."""
        if self.config.input.vmc_method == 1:
            # EBES
            return 1 / np.log(self.neu + self.ned)
        elif self.config.input.vmc_method == 3:
            # CBCS
            return 1 / (self.neu + self.ned)
        else:
            # wrong method
            return 0

    def vmc_step_graph(self):
        """Acceptance probability vs step size to plot a graph."""
        n = 5
        step_size = self.approximate_step_size
        for x in range(4 * n):
            self.vmc_markovchain.step_size = step_size * (x + 1) / n
            condition, _ = self.vmc_markovchain.random_walk(1000000)
            acc_ration = condition.mean()
            self.logger.info(
                'step_size * electrons = %.5f, acc_ration = %.5f', self.vmc_markovchain.step_size * (self.neu + self.ned), acc_ration
            )

    def optimize_vmc_step(self, steps):
        """Optimize vmc step size."""
        xdata = np.linspace(0, 2, 11)
        ydata = np.ones_like(xdata)
        approximate_step_size = self.approximate_step_size
        for i in range(1, xdata.size):
            self.vmc_markovchain.step_size = approximate_step_size * xdata[i]
            condition, position = self.vmc_markovchain.random_walk(steps)
            acc_rate = condition.mean()
            ydata[i] = self.mpi_comm.allreduce(acc_rate) / self.mpi_comm.size

        def f(ts, a, ts0):
            """Dependence of the acceptance probability on the step size in CBCS case looks like:
            p(ts) = (exp(a/ts0) - 1)/(exp(a/ts0) + exp(ts/ts0) - 2)
            :param ts: step_size
            :param a: step_size for 50% acceptance probability
            :param ts0: scale factor
            :return: acceptance probability
            """
            return (np.exp(a/ts0) - 1) / (np.exp(a/ts0) + np.exp(ts/ts0) - 2)

        step_size = None
        if self.root:
            popt, pcov = curve_fit(f, xdata, ydata)
            step_size = approximate_step_size * popt[0]

        step_size = self.mpi_comm.bcast(step_size)
        self.logger.info(
            f'Performing time-step optimization.\n'
            f'Optimized step size: {step_size:.5f}\n'
        )
        self.vmc_markovchain.step_size = step_size

    @property
    def decorr_period(self):
        """Decorr period"""
        if self.config.input.vmc_decorr_period == 0:
            return 3
        else:
            return self.config.input.vmc_decorr_period

    def run(self):
        """Run Casino workflow.
        """
        start = default_timer()
        if self.config.input.runtype == 'vmc':
            self.logger.info(
                ' ====================================\n'
                ' PERFORMING A SINGLE VMC CALCULATION.\n'
                ' ====================================\n\n'
            )
            self.vmc_energy_accumulation()
        elif self.config.input.runtype == 'vmc_opt':
            if self.root:
                self.config.write('.', 0)
            self.vmc_energy_accumulation()
            for i in range(self.config.input.opt_cycles):
                self.logger.info(
                    f' ==========================================\n'
                    f' PERFORMING OPTIMIZATION CALCULATION No. {i+1}.\n'
                    f' ==========================================\n\n'
                )
                if self.config.input.opt_method == 'varmin':
                    self.vmc_unreweighted_variance_minimization(
                        self.config.input.vmc_nconfig_write,
                        self.config.input.opt_jastrow,
                        self.config.input.opt_backflow
                    )
                elif self.config.input.opt_method == 'emin':
                    self.vmc_energy_minimization(
                        self.config.input.vmc_nconfig_write,
                        self.config.input.opt_jastrow,
                        self.config.input.opt_backflow
                    )
                self.config.jastrow.u_cutoff[0]['value'] = self.wfn.jastrow.u_cutoff
                if self.root:
                    self.config.write('.', i + 1)
                self.vmc_energy_accumulation()
        elif self.config.input.runtype == 'vmc_dmc':
            self.logger.info(
                 ' ======================================================\n'
                 ' PERFORMING A VMC CONFIGURATION-GENERATION CALCULATION.\n'
                 ' ======================================================\n\n'
            )
            _, position = self.vmc_energy_accumulation()
            r_e_list = position[-self.config.input.vmc_nconfig_write // self.mpi_comm.size:]
            self.dmc_markovchain = DMCMarkovChain(
                r_e_list, self.config.input.alimit, self.config.input.nucleus_gf_mods,
                self.config.input.dtdmc, self.config.input.dmc_target_weight, self.wfn
            )
            self.dmc_energy_equilibration()
            self.dmc_energy_accumulation()

        stop = default_timer()
        self.logger.info(
            f' =========================================================================\n\n'
            f' Total PyCasino real time : : :    {stop - start:.4f}'
        )

    def equilibrate(self, steps):
        """Burn-in.
        :param steps: burn-in period
        :return:
        """
        condition, _ = self.vmc_markovchain.random_walk(steps)
        self.logger.info(
            f'Running VMC equilibration ({steps} moves).'
        )

    def vmc_energy_accumulation(self):
        """VMC energy accumulation"""
        self.logger.info(
            f' BEGIN VMC CALCULATION\n'
            f' =====================\n'
        )
        self.equilibrate(self.config.input.vmc_equil_nstep)
        self.optimize_vmc_step(1000)

        steps = self.config.input.vmc_nstep
        nblock = self.config.input.vmc_nblock

        energy_block_mean = np.zeros(shape=(nblock,))
        energy_block_sem = np.zeros(shape=(nblock,))
        energy_block_var = np.zeros(shape=(nblock,))
        self.logger.info(
            f'Starting VMC.\n'
        )
        for i in range(nblock):
            block_start = default_timer()
            block_energy = np.zeros(shape=(steps // nblock,))
            condition, position = self.vmc_markovchain.random_walk(steps // nblock // self.mpi_comm.size, self.decorr_period)
            energy = vmc_observable(condition, position, self.wfn.energy)
            self.mpi_comm.Gather(energy, block_energy, root=0)
            if self.mpi_comm.rank == 0:
                energy_block_mean[i] = block_energy.mean()
                energy_block_var[i] = block_energy.var()
                energy_block_sem[i] = np.mean([
                    correlated_sem(block_energy.reshape(self.mpi_comm.size, steps // nblock // self.mpi_comm.size)[j]) for j in range(self.mpi_comm.size)
                ]) / np.sqrt(self.mpi_comm.size)
            block_stop = default_timer()
            self.logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n'
                f'  Number of VMC steps           = {steps // nblock}\n\n'
                f'  Block average energies (au)\n\n'
                f'  Total energy                       (au) =       {energy_block_mean[i]:18.12f}\n'
                f'  Standard error                        +/-       {energy_block_sem[i]:18.12f}\n\n'
                f'  Constant energy contributions      (au) =       {self.wfn.nuclear_repulsion:18.12f}\n\n'
                f'  Variance of local energy           (au) =       {energy_block_var[i]:18.12f}\n'
                f'  Standard error                        +/-       {0:18.12f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
        self.logger.info(
            f' =========================================================================\n'
            f' FINAL RESULT:\n\n'
            f'  VMC energy (au)    Standard error      Correction for serial correlation\n\n'
            f' {energy_block_mean.mean():.12f} +/- {energy_block_sem.mean() / np.sqrt(nblock):.12f}      On-the-fly reblocking method\n\n'
            f' Sample variance of E_L (au^2/sim.cell) : {energy_block_var.mean():.12f}\n\n'
        )
        return condition, position

    def dmc_energy_equilibration(self):
        """DMC energy equilibration"""
        self.logger.info(
            f' *     *     *     *     *     *     *     *     *     *     *     *\n\n'
            f' ===========================================\n'
            f' PERFORMING A DMC EQUILIBRATION CALCULATION.\n'
            f' ===========================================\n\n'
            f' BEGIN DMC CALCULATION\n'
            f' =====================\n\n'
            f' Random number generator reset to state in config.in.\n\n'
            f' EBEST = {self.dmc_markovchain.best_estimate_energy} (au/prim cell inc. N-N)\n'
            f' EREF  = {self.dmc_markovchain.energy_t}\n\n'
        )

        steps = self.config.input.dmc_equil_nstep
        nblock = self.config.input.dmc_equil_nblock

        for i in range(nblock):
            block_start = default_timer()
            energy = self.dmc_markovchain.random_walk(steps // nblock)
            block_stop = default_timer()
            self.logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n\n'
                f' Number of moves in block                 : {steps // nblock}\n'
                f' Load-balancing efficiency (%)            : {100 * np.mean(self.dmc_markovchain.efficiency_list):.3f}\n'
                f' Number of config transfers               : {self.dmc_markovchain.ntransfers_tot}\n'
                f' New best estimate of DMC energy (au)     : {energy.mean():.8f}\n'
                f' New best estimate of effective time step : {self.dmc_markovchain.step_eff:.8f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )

    def dmc_energy_accumulation(self):
        """DMC energy accumulation"""
        self.logger.info(
            f' *     *     *     *     *     *     *     *     *     *     *     *\n\n'
            f' =====================================================\n'
            f' PERFORMING A DMC STATISTICS-ACCUMULATION CALCULATION.\n'
            f' =====================================================\n\n'
            f' BEGIN DMC CALCULATION\n'
            f' =====================\n\n'
            f' Random number generator reset to state in config.in.\n\n'
            f' EBEST = {self.dmc_markovchain.best_estimate_energy} (au/prim cell inc. N-N)\n'
            f' EREF  = {self.dmc_markovchain.energy_t}\n\n'
            f' Number of previous DMC stats accumulation moves : 0\n'
        )

        steps = self.config.input.dmc_stats_nstep
        nblock = self.config.input.dmc_stats_nblock
        block_steps = steps // nblock
        energy = np.zeros(shape=(steps,))

        for i in range(nblock):
            block_start = default_timer()
            energy[block_steps * i:block_steps * (i + 1)] = self.dmc_markovchain.random_walk(block_steps)
            energy_mean = energy[:block_steps * (i + 1)].mean()
            block_stop = default_timer()
            self.logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n\n'
                f' Number of moves in block                 : {block_steps}\n'
                f' Load-balancing efficiency (%)            : {100 * np.mean(self.dmc_markovchain.efficiency_list):.3f}\n'
                f' Number of config transfers               : {self.dmc_markovchain.ntransfers_tot}\n'
                f' New best estimate of DMC energy (au)     : {energy_mean:.8f}\n'
                f' New best estimate of effective time step : {self.dmc_markovchain.step_eff:.8f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
        self.logger.info(
            f'Mixed estimators of the energies at the end of the run\n'
            f'------------------------------------------------------\n\n'
            f'Total energy                 =       {energy.mean():.12f} +/- {correlated_sem(energy):.12f}\n'
        )

    def normal_test(self, energy):
        """Test whether energy distribution differs from a normal one."""
        from scipy import stats
        self.logger.info(f'skew = {stats.skewtest(energy)}, kurtosis = {stats.kurtosistest(energy)}')
        plt.hist(
            energy,
            bins='auto',
            range=(energy.mean() - 5 * energy.std(), energy.mean() + 5 * energy.std()),
            density=True
        )
        plt.savefig('hist.png')
        plt.clf()

    def vmc_unreweighted_variance_minimization(self, steps, opt_jastrow, opt_backflow, verbose=2):
        """Minimize vmc unreweighted variance.
        https://github.com/scipy/scipy/issues/10634
        :param steps: number of configs
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param verbose:
            0 : work silently.
            1 : display a termination report.
            2 : display progress during iterations.
        """
        steps = steps // self.mpi_comm.size * self.mpi_comm.size
        # rescale for "Cost column" in output of scipy.optimize.least_squares to be a variance of E local
        scale = np.sqrt(2) / np.sqrt(steps - 1)
        condition, position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, self.decorr_period)
        steps_eff = self.mpi_comm.allreduce(condition.sum())

        # for pos in position:
        #     self.logger.info(self.wfn.value_parameters_d1(pos) / self.wfn.value_parameters_numerical_d1(pos))
        #     self.logger.info(self.wfn.energy_parameters_d1(pos) / self.wfn.energy_parameters_numerical_d1(pos))

        def fun(x, *args, **kwargs):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy = np.empty(shape=(steps,))
            energy_part = vmc_observable(condition, position, self.wfn.energy)
            self.mpi_comm.Allgather(energy_part, energy)
            return scale * (energy - energy.mean())

        def jac(x, *args, **kwargs):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy_gradient = np.empty(shape=(steps, x.size))
            energy_gradient_part = vmc_observable(condition, position, self.wfn.energy_parameters_d1)
            self.mpi_comm.Allgather(energy_gradient_part, energy_gradient)
            return scale * (energy_gradient - energy_gradient.mean(axis=0))

        self.logger.info(
            ' Optimization start\n'
            ' =================='
        )

        x = self.wfn.get_parameters(opt_jastrow, opt_backflow)
        res = least_squares(
            fun, x0=x, jac=jac, method='trf', ftol=1/np.sqrt(steps_eff-1),
            tr_solver='exact', verbose=self.root and verbose
        )
        parameters = res.x
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)
        self.logger.info('Jacobian matrix at the solution:')
        self.logger.info(res.jac.mean(axis=0))

    def vmc_reweighted_variance_minimization(self, steps, opt_jastrow, opt_backflow, verbose=2):
        """Minimize vmc reweighted variance.
        https://github.com/scipy/scipy/issues/10634
        :param steps: number of configs
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param verbose:
            0 : work silently.
            1 : display a termination report.
            2 : display progress during iterations.
        """
        steps = steps // self.mpi_comm.size * self.mpi_comm.size
        condition, position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, self.decorr_period)
        steps_eff = self.mpi_comm.allreduce(condition.sum())
        wfn_0 = np.empty(shape=(steps,))
        wfn_0_part = vmc_observable(condition, position, self.wfn.value)
        self.mpi_comm.Allgather(wfn_0_part, wfn_0)

        def fun(x, *args, **kwargs):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            wfn = np.empty(shape=(steps,))
            energy = np.empty(shape=(steps,))
            wfn_part = vmc_observable(condition, position, self.wfn.value)
            energy_part = vmc_observable(condition, position, self.wfn.energy)
            self.mpi_comm.Allgather(wfn_part, wfn)
            self.mpi_comm.Allgather(energy_part, energy)
            weights = (wfn / wfn_0)**2
            ddof = (weights**2).sum() / weights.sum()  # Delta Degrees of Freedom
            mean_energy = np.average(energy, weights=weights)
            # rescale for "Cost column" in output of scipy.optimize.least_squares to be variance of E local
            return np.sqrt(2) * (energy - mean_energy) * np.sqrt(weights / (weights.sum() - ddof))

        def jac(x, *args, **kwargs):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            wfn = np.empty(shape=(steps,))
            energy_gradient = np.empty(shape=(steps, x.size))
            wfn_part = vmc_observable(condition, position, self.wfn.value)
            energy_gradient_part = vmc_observable(condition, position, self.wfn.energy_parameters_d1)
            self.mpi_comm.Allgather(wfn_part, wfn)
            self.mpi_comm.Allgather(energy_gradient_part, energy_gradient)
            weights = (wfn / wfn_0)**2
            ddof = (weights**2).sum() / weights.sum()  # Delta Degrees of Freedom
            mean_energy_gradient = np.average(energy_gradient, axis=0, weights=weights)
            # rescale for "Cost column" in output of scipy.optimize.least_squares to be a variance of E local
            return np.sqrt(2) * (energy_gradient - mean_energy_gradient) * np.sqrt(np.expand_dims(weights, 1) / (weights.sum() - ddof))

        self.logger.info(
            ' Optimization start\n'
            ' =================='
        )

        x = self.wfn.get_parameters(opt_jastrow, opt_backflow)
        res = least_squares(
            fun, x0=x, jac=jac, method='trf', ftol=1/np.sqrt(steps_eff-1),
            tr_solver='exact', verbose=self.root and verbose
        )
        parameters = res.x
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)
        self.logger.info('Jacobian matrix at the solution:')
        self.logger.info(res.jac.mean(axis=0))

    def vmc_energy_minimization_newton(self, steps, opt_jastrow=True, opt_backflow=True):
        """Minimize vmc energy by Newton or gradient descent methods.
        For SJB wfn = exp(J(r)) * S(Bf(r))
            second derivatives by Jastrow parameters is:
        d²exp(J(p)) * S(Bf(r))/dp² = d(dJ(p)/dp * wfn)/dp = (d²J(p)/dp² + dJ(p)/dp * dJ(p)/dp) * wfn
            second derivatives by backflow parameters is:
        d²exp(J(r)) * S(Bf(p))/dp² = d(J(r) * dS(r)/dr * dBf(p)/dp)/dp =
        J(r) * (d²S(r)/dr² * dBf(p)/dp * dBf(p)/dp + dS(r)/dr * d²Bf(p)/dp²) =
        (1/S * d²S(r)/dr² * dBf(p)/dp * dBf(p)/dp + 1/S * dS(r)/dr * d²Bf(p)/dp²) * wfn
        :param steps: number of configs
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param exact: exact or dogleg trust region optimization
        """
        steps = steps // self.mpi_comm.size * self.mpi_comm.size
        scale = self.wfn.get_parameters_scale(opt_jastrow, opt_backflow)
        self.wfn.set_parameters(self.wfn.get_parameters(opt_jastrow, opt_backflow), opt_jastrow, opt_backflow)
        condition, position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, self.decorr_period)

        def fun(x, *args):
            """For Nelder-Mead, Powell, COBYLA and those listed in jac and hess methods."""
            self.wfn.set_parameters(x * scale, opt_jastrow, opt_backflow)
            energy_mean = vmc_observable(condition, position, self.wfn.energy).mean()
            return self.mpi_comm.allreduce(energy_mean) / self.mpi_comm.size

        def jac(x, *args):
            """Only for CG, BFGS, L-BFGS-B, TNC, SLSQP and those listed in hess method."""
            self.wfn.set_parameters(x * scale, opt_jastrow, opt_backflow)
            energy_part = vmc_observable(condition, position, self.wfn.energy)
            energy = np.empty(shape=(steps,)) if self.root else None
            self.mpi_comm.Gather(energy_part, energy)
            wfn_gradient_part = vmc_observable(condition, position, self.wfn.value_parameters_d1)
            wfn_gradient = np.empty(shape=(steps, x.size)) if self.root else None
            self.mpi_comm.Gather(wfn_gradient_part, wfn_gradient)
            energy_gradient = energy_parameters_gradient(energy, wfn_gradient) if self.root else np.empty(shape=(x.size, ))
            self.mpi_comm.Bcast(energy_gradient)
            return energy_gradient * scale

        def hess(x, *args):
            """Only for Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr."""
            self.wfn.set_parameters(x * scale, opt_jastrow, opt_backflow)
            energy_part = vmc_observable(condition, position, self.wfn.energy)
            energy = np.empty(shape=(steps,)) if self.root else None
            self.mpi_comm.Gather(energy_part, energy)
            wfn_gradient_part = vmc_observable(condition, position, self.wfn.value_parameters_d1)
            wfn_gradient = np.empty(shape=(steps, x.size)) if self.root else None
            self.mpi_comm.Gather(wfn_gradient_part, wfn_gradient)
            energy_gradient_part = vmc_observable(condition, position, self.wfn.energy_parameters_d1)
            energy_gradient = np.empty(shape=(steps, x.size)) if self.root else None
            self.mpi_comm.Gather(energy_gradient_part, energy_gradient)
            # wfn_hessian_part = vmc_observable(condition, position, self.wfn.value_parameters_d2)
            # wfn_hessian = np.empty(shape=(steps, x.size, x.size)) if root else None
            # self.mpi_comm.Gather(wfn_hessian_part, wfn_hessian)
            wfn_hessian = np.expand_dims(wfn_gradient, 1) * np.expand_dims(wfn_gradient, 2) if self.root else None
            energy_hessian = energy_parameters_hessian(wfn_gradient, wfn_hessian, energy, energy_gradient) if self.root else np.empty(shape=(x.size, x.size))
            self.mpi_comm.Bcast(energy_hessian)
            return energy_hessian * np.outer(scale, scale)

        self.logger.info(
            ' Optimization start\n'
            ' =================='
        )

        x = self.wfn.get_parameters(opt_jastrow, opt_backflow) / scale
        options = dict(disp=self.root, initial_trust_radius=1, max_trust_radius=10)
        res = minimize(fun, x0=x, method='trust-exact', jac=jac, hess=hess, options=options)
        self.logger.info('Scaled Jacobian matrix at the solution:')
        self.logger.info(res.jac / scale)
        parameters = res.x * scale
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)

    def vmc_energy_minimization_linear_method(self, steps, opt_jastrow=True, opt_backflow=True):
        """Minimize vmc energy by linear method.
        Another way to energy-optimize linear parameters of wfn is to diagonalize the Hamiltonian
        in the variational space that they define, leading to a generalized eigenvalue equation.
        Energy calculated with wave function depended on parameters p is:
                                           E(p) = <ψ(p)|Ĥ|ψ(p)>/<ψ(p)|ψ(p)>
        which is Rayleigh quotient. To determine the stationary points of E(p) or solving ∇E(p) = 0 we have to solve
        following generalized eigenvalue problem, with ψ(p) expand to first-order in the parameters p:
                                           H · Δp = E(p) * S · Δp
        where elements of the matrices S and H approach the standard quantum mechanical overlap integrals and Hamiltonian matrix elements in
        the limit of an infinite Monte Carlo sample or exact ψ(p), hence their names. Thus, the extremum points of ψ(p*) (extremum values E(p*))
        of the Rayleigh quotient are obtained as the eigenvectors e (eigenvalues λ(e)) of the corresponding generalized eigenproblem.
        If the second-order expansion of ψ(p) is not too small, this does not ensure the convergence in one step and may require uniformly rescaling
        of ∆p to stabilise iterative process.
        :param steps: number of configs
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        """
        sparse = True
        steps = steps // self.mpi_comm.size * self.mpi_comm.size
        self.wfn.set_parameters(self.wfn.get_parameters(opt_jastrow, opt_backflow), opt_jastrow, opt_backflow)
        condition, position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, self.decorr_period)

        # for pos in position:
        #     self.logger.info(self.wfn.value_parameters_d1(pos) / self.wfn.value_parameters_numerical_d1(pos))
        #     self.logger.info(self.wfn.energy_parameters_d1(pos) / self.wfn.energy_parameters_numerical_d1(pos))

        self.logger.info(
            ' Optimization start\n'
            ' =================='
        )

        parameters = self.wfn.get_parameters(opt_jastrow, opt_backflow)
        energy_part = vmc_observable(condition, position, self.wfn.energy)
        energy = np.empty(shape=(steps,)) if self.root else None
        self.mpi_comm.Gather(energy_part, energy)
        wfn_gradient_part = vmc_observable(condition, position, self.wfn.value_parameters_d1)
        wfn_gradient = np.empty(shape=(steps, parameters.size)) if self.root else None
        self.mpi_comm.Gather(wfn_gradient_part, wfn_gradient)
        energy_gradient_part = vmc_observable(condition, position, self.wfn.energy_parameters_d1)
        energy_gradient = np.empty(shape=(steps, parameters.size)) if self.root else None
        self.mpi_comm.Gather(energy_gradient_part, energy_gradient)
        dp = np.empty_like(parameters)
        if self.root:
            S = overlap_matrix(wfn_gradient)
            self.logger.info(f'S is positive definite: {np.all(np.linalg.eigvals(S) > 0)}')
            H = hamiltonian_matrix(wfn_gradient, energy, energy_gradient)
            if sparse:
                # get normalized right eigenvector corresponding to the eigenvalue
                eigvals, eigvectors = sp.sparse.linalg.eigs(A=H, k=1, M=S, v0=S[0], which='SR')
            else:
                eigvals, eigvectors = sp.linalg.eig(H, S)
            # since imaginary parts only arise from statistical noise, discard them
            eigvals, eigvectors = np.real(eigvals), np.real(eigvectors)
            idx = eigvals.argmin()
            eigval, eigvector = eigvals[idx], eigvectors[:, idx]
            dp = eigvector[0] * eigvector[1:]
            self.logger.info(f'E lin {eigval}')
            # eigvector[0] ** 2 + eigvector[1:] @ S[1:, 1:] @ eigvector[1:] = eigvector @ S @ eigvector = 1
            self.logger.info(f'eigvector[0] {eigvector[0]}')
            if parameters.all():
                self.logger.info(f'delta p / p\n{dp/parameters}')
            else:
                self.logger.info(f'delta p\n{dp}')

        self.mpi_comm.Bcast(dp)
        for i in range(11):
            self.wfn.set_parameters(parameters + i * dp / 10, opt_jastrow, opt_backflow)
            _condition, _position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, self.decorr_period)
            energy_part_mean = vmc_observable(_condition, _position, self.wfn.energy).mean()
            energy_mean = self.mpi_comm.allreduce(energy_part_mean) / self.mpi_comm.size
            self.logger.info(f'* {i/10} {energy_mean}')

        parameters += dp
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)

    def vmc_energy_minimization_stochastic_reconfiguration(self, steps, opt_jastrow=True, opt_backflow=True):
        """Minimize vmc energy by stochastic reconfiguration.
        :param steps: number of configs
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        """
        steps = steps // self.mpi_comm.size * self.mpi_comm.size
        self.wfn.set_parameters(self.wfn.get_parameters(opt_jastrow, opt_backflow), opt_jastrow, opt_backflow)
        condition, position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, self.decorr_period)

        def fun(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy_mean = vmc_observable(condition, position, self.wfn.energy).mean()
            return self.mpi_comm.allreduce(energy_mean) / self.mpi_comm.size

        def jac(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy_part = vmc_observable(condition, position, self.wfn.energy)
            energy = np.empty(shape=(steps,)) if self.root else None
            self.mpi_comm.Gather(energy_part, energy)
            wfn_gradient_part = vmc_observable(condition, position, self.wfn.value_parameters_d1)
            wfn_gradient = np.empty(shape=(steps, x.size)) if self.root else None
            self.mpi_comm.Gather(wfn_gradient_part, wfn_gradient)
            energy_gradient_part = vmc_observable(condition, position, self.wfn.energy_parameters_d1)
            energy_gradient = np.empty(shape=(steps, x.size)) if self.root else None
            self.mpi_comm.Gather(energy_gradient_part, energy_gradient)
            H = hamiltonian_matrix(wfn_gradient, energy, energy_gradient) if self.root else np.empty(shape=(x.size + 1, x.size + 1))
            self.mpi_comm.Bcast(H)
            return H[1:, 0]

        def hess(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            wfn_gradient_part = vmc_observable(condition, position, self.wfn.value_parameters_d1)
            wfn_gradient = np.empty(shape=(steps, x.size)) if self.root else None
            self.mpi_comm.Gather(wfn_gradient_part, wfn_gradient)
            S = overlap_matrix(wfn_gradient) if self.root else np.empty(shape=(x.size + 1, x.size + 1))
            self.mpi_comm.Bcast(S)
            return np.linalg.inv(S[1:, 1:])

        def epsilon(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            wfn_gradient_part = vmc_observable(condition, position, self.wfn.value_parameters_d1)
            wfn_gradient = np.empty(shape=(steps, x.size)) if self.root else None
            self.mpi_comm.Gather(wfn_gradient_part, wfn_gradient)
            S = overlap_matrix(wfn_gradient) if self.root else np.empty(shape=(x.size + 1, x.size + 1))
            self.mpi_comm.Bcast(S)
            energy_part = vmc_observable(condition, position, self.wfn.energy)
            energy = np.empty(shape=(steps,)) if self.root else None
            self.mpi_comm.Gather(energy_part, energy)
            energy_gradient_part = vmc_observable(condition, position, self.wfn.energy_parameters_d1)
            energy_gradient = np.empty(shape=(steps, x.size)) if self.root else None
            self.mpi_comm.Gather(energy_gradient_part, energy_gradient)
            H = hamiltonian_matrix(wfn_gradient, energy, energy_gradient) if self.root else np.empty(shape=(x.size + 1, x.size + 1))
            self.mpi_comm.Bcast(H)
            return np.diag(H[1:, 1:]) / np.diag(S[1:, 1:]) - H[0, 0]

        self.logger.info(
            ' Optimization start\n'
            ' =================='
        )

        parameters = self.wfn.get_parameters(opt_jastrow, opt_backflow)
        # S = np.linalg.inv(hess(parameters))
        # H = jac(parameters)
        # eps = epsilon(parameters)
        # if self.root:
        #     parameters -= S @ H / eps
        # self.mpi_comm.Bcast(parameters)
        # self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)

        options = dict(disp=self.root, eps=1/epsilon(parameters))
        res = minimize(fun, x0=parameters, method='Newton-CG', jac=jac, hess=hess, options=options)
        self.logger.info('Scaled Jacobian matrix at the solution:')
        self.logger.info(res.jac)
        parameters = res.x
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)

    vmc_energy_minimization = vmc_energy_minimization_linear_method


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="This script run CASINO workflow.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('config_path', type=str, help="path to CASINO config dir")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.config_path, 'input')):
        Casino(args.config_path).run()
    else:
        print(f'File {args.config_path}input not found...')
        sys.exit(1)
