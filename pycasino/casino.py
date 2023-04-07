#!/usr/bin/env python3
import os
import sys
import argparse
from timeit import default_timer
from numpy_config import np
from mpi4py import MPI
import scipy as sp
import mpmath as mp  # inherited from sympy, not need to install
from scipy.optimize import least_squares, minimize, curve_fit
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
        if self.mpi_comm.rank == 0:
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
        if self.mpi_comm.rank == 0:
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
        self.equilibrate(self.config.input.vmc_equil_nstep)
        if self.config.input.runtype == 'vmc':
            start = default_timer()
            self.logger.info(
                ' ====================================\n'
                ' PERFORMING A SINGLE VMC CALCULATION.\n'
                ' ====================================\n\n'
            )
            self.optimize_vmc_step(1000)
            self.vmc_energy_accumulation()
            stop = default_timer()
            self.logger.info(
                f' =========================================================================\n\n'
                f' Total PyCasino real time : : :    {stop - start:.4f}'
            )
        elif self.config.input.runtype == 'vmc_opt':
            if self.config.input.opt_method == 'varmin':
                start = default_timer()
                self.config.write('.', 0)
                self.optimize_vmc_step(1000)
                self.vmc_energy_accumulation()
                for i in range(self.config.input.opt_cycles):
                    self.logger.info(
                        f' ==========================================\n'
                        f' PERFORMING OPTIMIZATION CALCULATION No. {i+1}.\n'
                        f' ==========================================\n\n'
                    )
                    self.vmc_unreweighted_variance_minimization(
                        self.config.input.vmc_nconfig_write,
                        self.config.input.opt_jastrow,
                        self.config.input.opt_backflow
                    )
                    self.config.jastrow.u_cutoff[0]['value'] = self.wfn.jastrow.u_cutoff
                    if self.mpi_comm.rank == 0:
                        self.config.write('.', i + 1)
                    self.optimize_vmc_step(1000)
                    self.vmc_energy_accumulation()
                stop = default_timer()
                self.logger.info(
                    f' =========================================================================\n\n'
                    f' Total PyCasino real time : : :    {stop - start:.4f}'
                )
            elif self.config.input.opt_method == 'emin':
                start = default_timer()
                self.config.write('.', 0)
                self.optimize_vmc_step(1000)
                self.vmc_energy_accumulation()
                for i in range(self.config.input.opt_cycles):
                    self.logger.info(
                        f' ==========================================\n'
                        f' PERFORMING OPTIMIZATION CALCULATION No. {i+1}.\n'
                        f' ==========================================\n\n'
                    )
                    self.vmc_energy_minimization(
                        self.config.input.vmc_nconfig_write,
                        self.config.input.opt_jastrow,
                        self.config.input.opt_backflow
                    )
                    self.config.jastrow.u_cutoff[0]['value'] = self.wfn.jastrow.u_cutoff
                    if self.mpi_comm.rank == 0:
                        self.config.write('.', i + 1)
                    self.optimize_vmc_step(1000)
                    self.vmc_energy_accumulation()
                stop = default_timer()
                self.logger.info(
                    f' =========================================================================\n\n'
                    f' Total PyCasino real time : : :    {stop - start:.4f}'
                )
        elif self.config.input.runtype == 'vmc_dmc':
            start = default_timer()
            self.optimize_vmc_step(1000)
            block_start = default_timer()
            condition, position = self.vmc_markovchain.random_walk(self.config.input.vmc_nstep, self.decorr_period)
            energy = vmc_observable(condition, position, self.wfn.energy) + self.wfn.nuclear_repulsion
            block_stop = default_timer()
            self.logger.info(
                f' =========================================================================\n'
                f' In block : {1}\n'
                f'  Number of VMC steps           = {self.config.input.vmc_nstep}\n\n'
                f'  Block average energies (au)\n\n'
                f'  Total energy                       (au) =       {energy.mean():18.12f}\n'
                f'  Standard error                        +/-       {energy.std():18.12f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
            r_e_list = position[-self.config.input.vmc_nconfig_write // self.mpi_comm.size:]
            self.dmc_markovchain = DMCMarkovChain(
                r_e_list, self.config.input.alimit, self.config.input.nucleus_gf_mods,
                self.config.input.dtdmc, self.config.input.dmc_target_weight, self.wfn
            )
            self.logger.info(
                f' *     *     *     *     *     *     *     *     *     *     *     *\n'
            )
            self.dmc_energy_equilibration()
            self.logger.info(
                f' *     *     *     *     *     *     *     *     *     *     *     *\n'
            )
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
        steps = self.config.input.vmc_nstep
        nblock = self.config.input.vmc_nblock

        energy_block_mean = np.zeros(shape=(nblock,))
        energy_block_sem = np.zeros(shape=(nblock,))
        energy_block_var = np.zeros(shape=(nblock,))
        self.logger.info(
            f'Starting VMC.'
        )
        for i in range(nblock):
            block_start = default_timer()
            block_energy = np.zeros(shape=(steps // nblock,))
            condition, position = self.vmc_markovchain.random_walk(steps // nblock // self.mpi_comm.size, self.decorr_period)
            energy = vmc_observable(condition, position, self.wfn.energy) + self.wfn.nuclear_repulsion
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
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
        self.logger.info(
            f' =========================================================================\n'
            f' FINAL RESULT:\n\n'
            f'  VMC energy (au)    Standard error      Correction for serial correlation\n'
            f' {energy_block_mean.mean():.12f} +/- {energy_block_sem.mean() / np.sqrt(nblock):.12f}      On-the-fly reblocking method\n\n'
            f' Sample variance of E_L (au^2/sim.cell) : {energy_block_var.mean():.12f}\n\n'
        )

    def dmc_energy_equilibration(self):
        """DMC energy equilibration"""
        self.logger.info(
            f' ===========================================\n'
            f' PERFORMING A DMC EQUILIBRATION CALCULATION.\n'
            f' ===========================================\n\n'
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
            f' =====================================================\n'
            f' PERFORMING A DMC STATISTICS-ACCUMULATION CALCULATION.\n'
            f' =====================================================\n\n'
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
        :param steps:
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

        res = least_squares(
            fun, x0=self.wfn.get_parameters(opt_jastrow, opt_backflow), jac=jac, method='trf',
            ftol=1/np.sqrt(steps_eff-1), tr_solver='exact', verbose=0 if self.mpi_comm.rank else verbose
        )
        parameters = res.x
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)
        self.logger.info('Jacobian matrix at the solution:')
        self.logger.info(res.jac.mean(axis=0))

    def vmc_reweighted_variance_minimization(self, steps, opt_jastrow, opt_backflow, verbose=2):
        """Minimize vmc reweighted variance.
        https://github.com/scipy/scipy/issues/10634
        :param steps:
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

        res = least_squares(
            fun, x0=self.wfn.get_parameters(opt_jastrow, opt_backflow), jac=jac, method='trf',
            ftol=1/np.sqrt(steps_eff-1), tr_solver='exact', verbose=0 if self.mpi_comm.rank else verbose
        )
        parameters = res.x
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)
        self.logger.info('Jacobian matrix at the solution:')
        self.logger.info(res.jac.mean(axis=0))

    def vmc_energy_minimization_newton_cg(self, steps, decorr_period, opt_jastrow=True, opt_backflow=True):
        """Minimize vmc energy by Newton method.
        :param steps:
        :param decorr_period:
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        """
        steps = steps // self.mpi_comm.size * self.mpi_comm.size
        self.wfn.jastrow.fix_u_parameters()
        self.wfn.jastrow.fix_chi_parameters()
        self.wfn.jastrow.fix_f_parameters()
        condition, position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, decorr_period)

        def fun(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.wfn.energy)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, energy)
            mean_energy = energy.mean() / self.mpi_comm.size
            self.logger.info(f'energy {mean_energy}')
            return mean_energy

        def jac(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy_part = vmc_observable(condition, position, self.wfn.energy)
            energy = np.empty(shape=(steps,)) if self.mpi_comm.rank == 0 else None
            self.mpi_comm.Gather(energy_part, energy)
            wfn_gradient_part = vmc_observable(condition, position, self.wfn.value_parameters_d1)
            wfn_gradient = np.empty(shape=(steps, wfn_gradient_part.shape[1])) if self.mpi_comm.rank == 0 else None
            self.mpi_comm.Gather(wfn_gradient_part, wfn_gradient)
            mean_energy_gradient = energy_parameters_gradient(energy, wfn_gradient)
            self.logger.info(f'projected gradient values min {mean_energy_gradient.min()} max {mean_energy_gradient.max()}')
            return mean_energy_gradient

        def hess(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.wfn.energy)
            wfn_gradient = vmc_observable(condition, position, self.wfn.value_parameters_d1)
            wfn_hessian = vmc_observable(condition, position, self.wfn.value_parameters_d2) + np.expand_dims(wfn_gradient, 1) * np.expand_dims(wfn_gradient, 2)
            energy_gradient = vmc_observable(condition, position, self.wfn.energy_parameters_d1)
            mean_projected_energy_hessian = energy_parameters_hessian(wfn_gradient, wfn_hessian, energy, energy_gradient)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, mean_projected_energy_hessian)
            mean_energy_hessian = mean_projected_energy_hessian / self.mpi_comm.size
            eigvals = np.linalg.eigvalsh(mean_energy_hessian)
            self.logger.info(f'projected hessian eigenvalues min {eigvals.min()} max {eigvals.max()}')
            self.logger.info(f'projected hessian rank {np.linalg.matrix_rank(mean_energy_hessian)} {mean_energy_hessian.shape}')
            # if projected_eigvals.min() < 0:
            #     mean_energy_hessian -= projected_eigvals.min() * np.eye(x.size)
            return mean_energy_hessian

        self.logger.info(
            ' Optimization start\n'
            ' =================='
        )

        disp = self.mpi_comm.rank == 0
        x0 = self.wfn.get_parameters(opt_jastrow, opt_backflow)
        options = dict(disp=disp)
        res = minimize(fun, x0=x0, method='Newton-CG', jac=jac, hess=hess, options=options)
        self.logger.info('scaled x at the solution:')
        self.logger.info(res.x)
        parameters = res.x
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)

    def vmc_energy_minimization_linear_method(self, steps, opt_jastrow=True, opt_backflow=True, precision=None):
        """Minimize vmc energy by linear method.
        The most straightforward way to energy-optimize linear parameters in wave functions is to diagonalize the Hamiltonian
        in the variational space that they define, leading to a generalized eigenvalue equation.
        Energy calculated with wave function depended on parameters p is:
                                           E(p) = <ψ(p)|Ĥ|ψ(p)>/<ψ(p)|ψ(p)>
        which is Rayleigh quotient. To determine the stationary points of E(p) or solving ∇E(p) = 0 we have to solve
        following generalized eigenvalue problem, with ψ(p) expand to first-order in the parameters p:
                                           H · Δp = E(p) * S · Δp
        where elements of the matrices S and H approach the standard quantum mechanical overlap integrals and Hamiltonian matrix elements in
        the limit of an infinite Monte Carlo sample or exact ψ(p), hence their names. Thus, the extremum points of ψ(p*) (extremum values E(p*))
        of the Rayleigh quotient are obtained as the eigenvectors e (eigenvalues λ(e)) of the corresponding generalized eigenproblem.
        If the second-order expansion of ψ(p) is not small, this does not ensure the convergence in one step and may require uniformly rescaling
        of ∆p to stabilise iterative process.
        :param steps:
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param precision: decimal precision in float-point arithmetic with mpmath.
            np.finfo(np.double).precision -> 15
            np.finfo(np.longdouble).precision -> 18
        check also np.show_config() and sp.show_config()
        """
        sparse = True
        steps = steps // self.mpi_comm.size * self.mpi_comm.size
        self.wfn.jastrow.fix_u_parameters()
        self.wfn.jastrow.fix_chi_parameters()
        self.wfn.jastrow.fix_f_parameters()
        condition, position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, self.decorr_period)

        # for pos in position:
        #     self.logger.info(self.wfn.value_parameters_d1(pos) / self.wfn.value_parameters_numerical_d1(pos))
        #     self.logger.info(self.wfn.energy_parameters_d1(pos) / self.wfn.energy_parameters_numerical_d1(pos))

        self.logger.info(
            ' Optimization start\n'
            ' =================='
        )

        energy_part = vmc_observable(condition, position, self.wfn.energy)
        energy = np.empty(shape=(steps,)) if self.mpi_comm.rank == 0 else None
        self.mpi_comm.Gather(energy_part, energy)
        wfn_gradient_part = vmc_observable(condition, position, self.wfn.value_parameters_d1)
        wfn_gradient = np.empty(shape=(steps, wfn_gradient_part.shape[1])) if self.mpi_comm.rank == 0 else None
        self.mpi_comm.Gather(wfn_gradient_part, wfn_gradient)
        energy_gradient_part = vmc_observable(condition, position, self.wfn.energy_parameters_d1)
        energy_gradient = np.empty(shape=(steps, wfn_gradient_part.shape[1])) if self.mpi_comm.rank == 0 else None
        self.mpi_comm.Gather(energy_gradient_part, energy_gradient)
        if self.mpi_comm.rank == 0:
            S = overlap_matrix(wfn_gradient)
            self.logger.info(f'S is positive definite: {np.all(np.linalg.eigvals(S) > 0)}')
            H = hamiltonian_matrix(wfn_gradient, energy, energy_gradient)
            if precision is not None:
                with mp.workdps(precision):
                    # https://github.com/mpmath/mpmath/blob/master/mpmath/matrices/eigen.py
                    # get right eigenvector corresponding to the eigenvalue sorted by increasing real part
                    E, ER = mp.eig(mp.matrix(S)**-1 * mp.matrix(H), overwrite_a=True)
                    E, ER = mp.eig_sort(E, ER=ER)
                    # since imaginary parts only arise from statistical noise, discard them
                    eigval, eigvector = float(mp.re(E[0])), np.array(list(map(mp.re, ER[:, 0])), dtype=np.float64)
            else:
                if sparse:
                    # get normalized right eigenvector corresponding to the eigenvalue
                    eigvals, eigvectors = sp.sparse.linalg.eigs(A=H, k=1, M=S, v0=S[0], which='SR')
                else:
                    eigvals, eigvectors = sp.linalg.eig(H, S)
                # since imaginary parts only arise from statistical noise, discard them
                eigvals, eigvectors = np.real(eigvals), np.real(eigvectors)
                idx = eigvals.argmin()
                eigval, eigvector = eigvals[idx], eigvectors[:, idx]
            dp = eigvector[1:] / eigvector[0]
            dp_S_dp = np.sum(S[1:, 1:] * np.outer(dp, dp))
            norm = 1 / (1 + dp_S_dp)
            self.logger.info(f'E lin {eigval}')
            self.logger.info(f'norm {norm}')
            parameters = self.wfn.get_parameters(opt_jastrow, opt_backflow)
            if parameters.all():
                self.logger.info(f'delta p / p\n{norm * dp/parameters}')
            else:
                self.logger.info(f'delta p\n{norm * dp}')
            parameters += norm * dp
        else:
            parameters = self.wfn.get_parameters(opt_jastrow, opt_backflow)
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)

    def vmc_energy_minimization_stochastic_reconfiguration(self, steps, decorr_period, opt_jastrow=True, opt_backflow=True):
        """Minimize vmc energy by stochastic reconfiguration.
        :param steps:
        :param decorr_period:
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        """
        steps = steps // self.mpi_comm.size * self.mpi_comm.size
        self.wfn.jastrow.fix_u_parameters()
        a, b = self.wfn.jastrow.get_parameters_constraints()
        p = np.eye(a.shape[1]) - a.T @ np.linalg.inv(a @ a.T) @ a
        mask_idx = np.argwhere(self.wfn.jastrow.get_parameters_mask()).ravel()
        condition, position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, decorr_period)

        def fun(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.wfn.energy)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, energy)
            mean_energy = energy.mean() / self.mpi_comm.size
            self.logger.info(f'energy {mean_energy}')
            return mean_energy

        def jac(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.wfn.energy)
            wfn_gradient = vmc_observable(condition, position, self.wfn.value_parameters_d1)
            projected_wfn_gradient = (wfn_gradient @ p)[:, mask_idx]
            mean_projected_energy_gradient = energy_parameters_gradient(energy, projected_wfn_gradient)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, mean_projected_energy_gradient)
            mean_energy_gradient = mean_projected_energy_gradient / self.mpi_comm.size
            self.logger.info(f'projected gradient values min {mean_energy_gradient.min()} max {mean_energy_gradient.max()}')
            return mean_energy_gradient

        def hess(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            wfn_gradient = vmc_observable(condition, position, self.wfn.value_parameters_d1)
            projected_wfn_gradient = (wfn_gradient @ p)[:, mask_idx]
            overlap = overlap_matrix(projected_wfn_gradient)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, overlap)
            overlap = overlap / self.mpi_comm.size
            eigvals = np.linalg.eigvalsh(overlap)
            self.logger.info(f'projected hessian eigenvalues min {eigvals.min()} max {eigvals.max()}')
            return overlap

        def epsilon(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.wfn.energy)
            wfn_gradient = vmc_observable(condition, position, self.wfn.value_parameters_d1)
            projected_wfn_gradient = (wfn_gradient @ p)[:, mask_idx]
            energy_gradient = vmc_observable(condition, position, self.wfn.energy_parameters_d1)
            projected_energy_gradient = (energy_gradient @ p)[:, mask_idx]
            overlap = overlap_matrix(projected_wfn_gradient)
            hamiltonian = hamiltonian_matrix(projected_wfn_gradient, energy, projected_energy_gradient)
            epsilon = np.diag(hamiltonian) / np.diag(overlap) - energy.mean()
            self.mpi_comm.Allreduce(MPI.IN_PLACE, epsilon)
            epsilon = epsilon / self.mpi_comm.size
            self.logger.info(f'epsilon {epsilon}')
            return epsilon

        self.logger.info(
            ' Optimization start\n'
            ' =================='
        )

        disp = self.mpi_comm.rank == 0
        x0 = self.wfn.get_parameters(opt_jastrow, opt_backflow)
        options = dict(disp=disp)
        res = minimize(fun, x0=x0, method='Newton-CG', jac=jac, hess=hess, options=options)
        self.logger.info('scaled x at the solution:')
        self.logger.info(res.x)
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
