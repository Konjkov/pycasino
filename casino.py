#!/usr/bin/env python3
import argparse
from timeit import default_timer
from numpy_config import np
from mpi4py import MPI
from scipy.optimize import least_squares, minimize, root, Bounds, curve_fit
import matplotlib.pyplot as plt

from cusp import CuspFactory
from slater import Slater
from jastrow import Jastrow
from backflow import Backflow
from markovchain import VMCMarkovChain, DMCMarkovChain, vmc_observable
from wfn import Wfn
from readers.casino import CasinoConfig
from sem import correlated_sem
from logger import logging


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
        self.mpi_comm = MPI.COMM_WORLD
        self.config = CasinoConfig(config_path)
        self.config.read()
        self.neu, self.ned = self.config.input.neu, self.config.input.ned
        self.logger = logging.getLogger('vmc')
        if self.mpi_comm.rank > 0:
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
                self.config.jastrow.trunc, self.config.jastrow.u_parameters, self.config.jastrow.u_parameters_optimizable, self.config.jastrow.u_mask,
                self.config.jastrow.u_cutoff, self.config.jastrow.u_cusp_const,
                self.config.jastrow.chi_parameters, self.config.jastrow.chi_parameters_optimizable, self.config.jastrow.chi_mask, self.config.jastrow.chi_cutoff,
                self.config.jastrow.chi_labels, self.config.jastrow.chi_cusp,
                self.config.jastrow.f_parameters, self.config.jastrow.f_parameters_optimizable, self.config.jastrow.f_mask, self.config.jastrow.f_cutoff,
                self.config.jastrow.f_labels,
                self.config.jastrow.no_dup_u_term, self.config.jastrow.no_dup_chi_term
            )
        else:
            jastrow = None

        if self.config.backflow:
            backflow = Backflow(
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
        else:
            backflow = None

        self.wfn = Wfn(
            self.config.input.neu, self.config.input.ned, self.config.wfn.atom_positions, self.config.wfn.atom_charges, slater, jastrow, backflow
        )

        self.vmc_markovchain = VMCMarkovChain(
            self.initial_position(self.config.wfn.atom_positions, self.config.wfn.atom_charges),
            self.initial_step_size(),
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

    def initial_step_size(self):
        if self.config.input.vmc_method == 1:
            # EBES
            return 1 / np.log(self.neu + self.ned)
        elif self.config.input.vmc_method == 3:
            # CBCS
            return 1 / (self.neu + self.ned)
        else:
            # wrong method
            return 0

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
                    self.vmc_variance_minimization(
                        self.config.input.vmc_nconfig_write,
                        self.config.input.vmc_decorr_period,
                        self.config.input.opt_jastrow,
                        self.config.input.opt_backflow
                    )
                    self.config.jastrow.u_cutoff = self.wfn.jastrow.u_cutoff
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
                    self.vmc_energy_minimization(
                        self.config.input.vmc_nconfig_write,
                        self.config.input.vmc_decorr_period,
                        self.config.input.opt_jastrow,
                        self.config.input.opt_backflow
                    )
                    self.config.jastrow.u_cutoff = self.wfn.jastrow.u_cutoff
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
            self.optimize_vmc_step(1000)
            # FIXME: decorr_period for dmc?
            block_start = default_timer()
            condition, position = self.vmc_markovchain.random_walk(self.config.input.vmc_nstep, 1)
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
            r_e_list = position[-self.config.input.vmc_nconfig_write:]
            self.dmc_markovchain = DMCMarkovChain(r_e_list, self.config.input.dtdmc, self.config.input.dmc_target_weight, self.wfn)
            self.dmc_energy_equilibration()
            self.dmc_energy_accumulation()

    def equilibrate(self, steps):
        """
        :param steps: burn-in period
        :return:
        """
        condition, _ = self.vmc_markovchain.random_walk(steps, 1)
        self.logger.info(
            f'Running VMC equilibration ({steps} moves).'
        )
        self.logger.debug('dr * electrons = 1.00000, acc_ration = %.5f', condition.mean())

    def vmc_step_graph(self, steps):
        """Acceptance probability vs step size data to plot graph."""
        n = 5
        step_size = self.initial_step_size()
        for x in range(4 * n):
            self.vmc_markovchain.step_size = step_size * (x + 1) / n
            condition, _ = self.vmc_markovchain.random_walk(1000000, 1)
            acc_ration = condition.mean()
            self.logger.info('dr * electrons = %.5f, acc_ration = %.5f', self.vmc_markovchain.step_size * (self.neu + self.ned), acc_ration)

    def optimize_vmc_step(self, steps):
        """Optimize vmc step size."""
        xdata = np.linspace(0, 2, 11)
        ydata = np.ones_like(xdata)
        initial_step_size = self.initial_step_size()
        for i in range(1, xdata.size):
            self.vmc_markovchain.step_size = initial_step_size * xdata[i]
            condition, position = self.vmc_markovchain.random_walk(steps, 1)
            acc_rate = condition.mean()
            if acc_rate == 0:
                print(self.vmc_markovchain.wfn.value(position[0]))
            ydata[i] = self.mpi_comm.allreduce(acc_rate) / self.mpi_comm.size

        def f(ts, a, ts0):
            """Dependence of the acceptance probability on the step size in the case of CBCS looks like:
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
            step_size = initial_step_size * popt[0]

        self.vmc_markovchain.step_size = self.mpi_comm.bcast(step_size)

    def get_decorr_period(self):
        """Decorr period"""
        if self.config.input.vmc_decorr_period == 0:
            return 3
        else:
            return self.config.input.vmc_decorr_period

    def vmc_energy_accumulation(self):
        """VMC energy accumulation"""
        steps = self.config.input.vmc_nstep
        nblock = self.config.input.vmc_nblock

        decorr_period = self.get_decorr_period()
        energy_block_mean = np.zeros(shape=(nblock,))
        energy_block_sem = np.zeros(shape=(nblock,))
        energy_block_var = np.zeros(shape=(nblock,))
        self.logger.info(
            f'Starting VMC.'
        )
        for i in range(nblock):
            block_start = default_timer()
            block_energy = np.zeros(shape=(steps // nblock,))
            condition, position = self.vmc_markovchain.random_walk(steps // nblock // self.mpi_comm.size, decorr_period)
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
            energy = self.dmc_markovchain.random_walk(steps // nblock)
            energy_block_mean[i] = energy.mean()
            energy_block_sem[i] = correlated_sem(energy)
            block_stop = default_timer()
            self.logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n'
                f'  Number of DMC steps           = {steps // nblock}\n\n'
                f'  Block average energies (au)\n\n'
                f'  Total energy                       (au) =       {energy_block_mean[i]:18.12f}\n'
                f'  Standard error                        +/-       {energy_block_sem[i]:18.12f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )

    def dmc_energy_accumulation(self):
        """DMC energy accumulation"""
        self.logger.info(
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
            energy = self.dmc_markovchain.random_walk(steps // nblock)
            energy_block_mean[i] = energy.mean()
            energy_block_sem[i] = correlated_sem(energy)
            block_stop = default_timer()
            self.logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n'
                f'  Number of DMC steps           = {steps // nblock}\n\n'
                f'  Block average energies (au)\n\n'
                f'  Total energy                       (au) =       {energy_block_mean[i]:18.12f}\n'
                f'  Standard error                        +/-       {energy_block_sem[i]:18.12f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
        self.logger.info(
            f'Mixed estimators of the energies at the end of the run\n'
            f'------------------------------------------------------\n'
            f'Total energy                 =       {energy_block_mean.mean():.12f} +/-  {energy_block_sem.mean() / np.sqrt(nblock):.12f}\n'
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

    def vmc_variance_minimization(self, steps, decorr_period, opt_jastrow, opt_backflow, verbose=2):
        """Minimise vmc variance by jastrow parameters optimization.
        https://github.com/scipy/scipy/issues/10634
        :param steps:
        :param decorr_period:
        :param opt_jastrow:
        :param opt_backflow:
        :param verbose:
            0 : work silently.
            1 : display a termination report.
            2 : display progress during iterations.
        """
        scale = self.wfn.get_parameters_scale(opt_jastrow, opt_backflow)
        condition, position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, decorr_period)

        def fun(x, *args, **kwargs):
            self.wfn.set_parameters(x * scale, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.wfn.energy) / np.sqrt(steps // self.mpi_comm.size - 1)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, energy)
            return energy - energy.mean()

        res = least_squares(
            fun, x0=self.wfn.get_parameters(opt_jastrow, opt_backflow) / scale, jac='2-point',
            method='trf', ftol=1/np.sqrt(steps-1), x_scale='jac', loss='linear',
            tr_solver='exact', verbose=0 if self.mpi_comm.rank else verbose
        )
        parameters = res.x * scale
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)
        self.logger.info(f'{res.message}\n')
        self.logger.info('gradient norm:')
        self.logger.info(np.linalg.norm(res.jac, axis=0))

    def vmc_energy_minimization(self, steps, decorr_period, opt_jastrow=True, opt_backflow=True):
        """Minimise vmc energy by jastrow parameters optimization.
        Gradient only for : CG, BFGS, L-BFGS-B, TNC, SLSQP
        Gradient and Hessian is required for: Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact, trust-constr
        Constraints definition only for: COBYLA, SLSQP and trust-constr.
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods.

        SciPy, оптимизация с условиями - https://habr.com/ru/company/ods/blog/448054/
        """
        bounds = Bounds(*self.wfn.jastrow.get_bounds(), keep_feasible=True)
        condition, position = self.vmc_markovchain.random_walk(steps // self.mpi_comm.size, decorr_period)

        def fun(x, *args):
            self.wfn.set_parameters(x, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.wfn.energy)
            energy_gradient = vmc_observable(condition, position, self.wfn.jastrow_parameters_numerical_d1)
            mean_energy_gradient = jastrow_parameters_gradient(energy, energy_gradient)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, mean_energy_gradient)
            return self.mpi_comm.allreduce(energy.mean()), mean_energy_gradient

        def hess(x, *args):
            self.wfn.jastrow.set_parameters(x, opt_jastrow, opt_backflow)
            energy = vmc_observable(condition, position, self.wfn.energy)
            energy_gradient = vmc_observable(condition, position, self.wfn.jastrow_parameters_numerical_d1)
            energy_hessian = vmc_observable(condition, position, self.wfn.jastrow_parameters_numerical_d2)
            mean_energy_hessian = jastrow_parameters_hessian(energy, energy_gradient, energy_hessian)
            self.mpi_comm.Allreduce(MPI.IN_PLACE, mean_energy_hessian)
            self.logger.info('hessian = %s', mean_energy_hessian)
            return mean_energy_hessian

        res = minimize(
            fun, x0=self.wfn.get_parameters(opt_jastrow, opt_backflow), method='TNC',
            jac=True, bounds=bounds, options=dict(disp=True, maxfun=10)
        )
        # res = minimize(
        #     fun, x0=self.wfn.get_parameters(opt_jastrow, opt_backflow), method='trust-ncg',
        #     jac=True, hess=hess, options=dict(disp=True)
        # )
        parameters = res.x
        self.mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters, opt_jastrow, opt_backflow)
        # self.logger.info(parameters / self.wfn.get_parameters_scale(opt_jastrow, opt_backflow))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="This script run CASINO workflow.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('config_path', type=str, help="path to CASINO config dir")
    args = parser.parse_args()

    import os
    import sys
    if os.path.exists(os.path.join(args.config_path, 'input')):
        Casino(args.config_path).run()
    else:
        print(f'File {args.config_path}/input not found...')
        sys.exit(1)
