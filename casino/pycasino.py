import os
import sys
import logging
import warnings
import argparse
from timeit import default_timer
from mpi4py import MPI
import numba as nb
import numpy as np
import scipy as sp
from scipy.optimize import least_squares, minimize, curve_fit, OptimizeWarning
import matplotlib.pyplot as plt

from casino.wfn import Wfn
from casino.cusp import CuspFactory
from casino.slater import Slater
from casino.jastrow import Jastrow
# from casino.gjastrow import Gjastrow
from casino.backflow import Backflow
from casino.ppotential import PPotential
from casino.vmc import VMCMarkovChain, vmc_observable
from casino.dmc import DMCMarkovChain
from casino.readers import CasinoConfig
from casino.sem import correlated_sem

logger = logging.getLogger(__name__)

mpi_comm = MPI.COMM_WORLD

double_size = MPI.DOUBLE.Get_size()


@nb.njit(nogil=True, parallel=False, cache=True)
def expand(np_array):
    """Set NaN to previous value"""
    for i in range(1, np_array.shape[0]):
        if np.isnan(np_array[i]).all():
            np_array[i] = np_array[i-1]


# @nb.njit(nogil=True, parallel=False, cache=True)
def overlap_matrix(wfn_gradient):
    """Overlap matrix S.
    As any covariance matrix it is symmetric and positive semi-definite.
    Cov(x1 + x2, y1 + y2) = Cov(x1, y1) + Cov(x1, y2) + Cov(x2, y1) + Cov(x2, y2)
    """
    size_0 = wfn_gradient.shape[0]
    size_1 = wfn_gradient.shape[1] + 1
    extended_wfn_gradient = np.ones(shape=(size_0, size_1))
    extended_wfn_gradient[:, 1:] = wfn_gradient - np.mean(wfn_gradient, axis=0)
    return extended_wfn_gradient.T @ extended_wfn_gradient / size_0


# @nb.njit(nogil=True, parallel=False, cache=True)
def hamiltonian_matrix(wfn_gradient, energy, energy_gradient):
    """Hamiltonian matrix H.
    <(X-<X>)(Y-<Y>)> = <XY> - <X><Y> = X.T @ Y / size
    <(X-<X>)Z(Y-<Y>)> = <XYZ> - <XZ><Y> - <X><YZ> + <X><Y><Z> = X.T @ diag(Z) @ Y / size
    """
    size_0 = wfn_gradient.shape[0]
    size_1 = wfn_gradient.shape[1] + 1
    extended_wfn_gradient = np.ones(shape=(size_0, size_1))
    extended_wfn_gradient[:, 1:] = wfn_gradient - np.mean(wfn_gradient, axis=0)
    extended_energy_gradient = np.zeros(shape=(size_0, size_1))
    extended_energy_gradient[:, 1:] = energy_gradient
    return extended_wfn_gradient.T @ (np.expand_dims(energy, 1) * extended_wfn_gradient + extended_energy_gradient) / size_0


def S_inv_H_matrix(wfn_gradient, energy, energy_gradient):
    """S^-1 @ H"""
    size_0 = wfn_gradient.shape[0]
    size_1 = wfn_gradient.shape[1] + 1
    extended_wfn_gradient = np.ones(shape=(size_0, size_1))
    extended_wfn_gradient[:, 1:] = wfn_gradient - np.mean(wfn_gradient, axis=0)
    extended_energy_gradient = np.zeros(shape=(size_0, size_1))
    extended_energy_gradient[:, 1:] = energy_gradient
    return sp.linalg.pinv(extended_wfn_gradient) @ (np.expand_dims(energy, 1) * extended_wfn_gradient + extended_energy_gradient)


class Casino:

    def __init__(self, config_path: str):
        """Casino workflow.
        :param config_path: path to config file
        """
        self.root = mpi_comm.rank == 0
        self.config = CasinoConfig(config_path)
        self.config.read()
        self.neu, self.ned = self.config.input.neu, self.config.input.ned

        if self.config.input.cusp_correction and not self.config.wfn.is_pseudoatom.all():
            cusp_factory = CuspFactory(
                self.config.input.neu, self.config.input.ned, self.config.input.cusp_threshold, self.config.wfn.mo_up, self.config.wfn.mo_down,
                self.config.mdet.permutation_up, self.config.mdet.permutation_down,
                self.config.wfn.first_shells, self.config.wfn.shell_moments, self.config.wfn.primitives,
                self.config.wfn.coefficients, self.config.wfn.exponents,
                self.config.wfn.atom_positions, self.config.wfn.atom_charges, self.config.wfn.unrestricted, self.config.wfn.is_pseudoatom
            )
            cusp = cusp_factory.create()
            if self.config.input.cusp_info:
                cusp_factory.cusp_info()
        else:
            cusp = None

        if self.config.wfn.is_pseudoatom.any():
            for atom, vmc_nonlocal_grid in enumerate(self.config.wfn.vmc_nonlocal_grid):
                if self.config.wfn.is_pseudoatom[atom]:
                    l_exact = [0, 2, 3, 5, 5, 7, 11]
                    n_points = [1, 4, 6, 12, 18, 26, 50]
                    vmc_nonlocal_grid = vmc_nonlocal_grid or self.config.input.non_local_grid
                    logger.info(
                         f' Non-local integration grids\n'
                         f' ===========================\n'
                         f' Ion type            :  {atom+1}\n'
                         f' Non-local grid no.  :  {vmc_nonlocal_grid}\n'
                         f' Lexact              :  {l_exact[vmc_nonlocal_grid-1]}\n'
                         f' Number of points    :  {n_points[vmc_nonlocal_grid-1]}\n'
                    )
            ppotential = PPotential(
                self.config.input.neu, self.config.input.ned, self.config.input.lcutofftol, self.config.input.nlcutofftol, self.config.wfn.atom_charges,
                self.config.wfn.vmc_nonlocal_grid, self.config.wfn.dmc_nonlocal_grid, self.config.wfn.local_angular_momentum, self.config.wfn.ppotential,
                self.config.wfn.is_pseudoatom
            )
        else:
            ppotential = None

        slater = Slater(
            self.config.input.neu, self.config.input.ned,
            self.config.wfn.nbasis_functions, self.config.wfn.first_shells, self.config.wfn.orbital_types, self.config.wfn.shell_moments,
            self.config.wfn.slater_orders, self.config.wfn.primitives, self.config.wfn.coefficients, self.config.wfn.exponents,
            self.config.wfn.mo_up, self.config.wfn.mo_down, self.config.mdet.permutation_up, self.config.mdet.permutation_down, self.config.mdet.coeff, cusp
        )

        jastrow = None
        if self.config.jastrow:
            if self.config.input.use_jastrow:
                jastrow = Jastrow(
                    self.config.input.neu, self.config.input.ned,
                    self.config.jastrow.trunc, self.config.jastrow.u_parameters, self.config.jastrow.u_parameters_optimizable,
                    self.config.jastrow.u_cutoff,
                    self.config.jastrow.chi_parameters, self.config.jastrow.chi_parameters_optimizable, self.config.jastrow.chi_cutoff,
                    self.config.jastrow.chi_labels, self.config.jastrow.chi_cusp,
                    self.config.jastrow.f_parameters, self.config.jastrow.f_parameters_optimizable, self.config.jastrow.f_cutoff, self.config.jastrow.f_labels,
                    self.config.jastrow.no_dup_u_term, self.config.jastrow.no_dup_chi_term
                )
            elif self.config.input.use_gjastrow:
                self.config.jastrow.write('.', 0)
                # gjastrow = Gjastrow(
                #     self.config.input.neu, self.config.input.ned, self.config.jastrow.rank, self.config.jastrow.cusp,
                #     self.config.jastrow.ee_basis_type, self.config.jastrow.en_basis_type,
                #     self.config.jastrow.ee_cutoff_type, self.config.jastrow.en_cutoff_type,
                #     self.config.jastrow.ee_constants, self.config.jastrow.en_constants,
                #     self.config.jastrow.ee_basis_parameters, self.config.jastrow.en_basis_parameters,
                #     self.config.jastrow.ee_cutoff_parameters, self.config.jastrow.en_cutoff_parameters,
                #     self.config.jastrow.linear_parameters, self.config.jastrow.linear_parameters_shape,
                # )

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
                self.config.backflow.ae_cutoff, self.config.backflow.ae_cutoff_optimizable
            )
        else:
            backflow = None

        self.wfn = Wfn(
            self.config.input.neu, self.config.input.ned, self.config.wfn.atom_positions, self.config.wfn.atom_charges, slater, jastrow, backflow, ppotential
        )

        self.vmc_markovchain = VMCMarkovChain(
            self.initial_position(self.config.wfn.atom_positions, self.config.wfn.atom_charges),
            self.approximate_step_size, self.wfn, self.config.input.vmc_method
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
            return 1 / np.log(np.max(self.config.wfn.atom_charges))
        elif self.config.input.vmc_method == 2:
            # determinant-by-determinant sampling
            return 1 / (self.neu + self.ned)
        elif self.config.input.vmc_method == 3:
            # CBCS
            return 1 / (self.neu + self.ned)
        else:
            # wrong method
            return 0

    def vmc_step_graph(self):
        """Acceptance probability vs step size to plot a graph."""
        n = 5
        approximate_step_size = self.approximate_step_size
        for x in range(4 * n):
            self.vmc_markovchain.step_size = approximate_step_size * (x + 1) / n
            position = self.vmc_markovchain.random_walk(1000000, 1)
            acc_ration = (np.isfinite(position[:, 0, 0])).mean()
            acc_ration /= (self.neu + self.ned)
            logger.info(
                'step_size / approximate_step_size  = %.5f, acc_ratio = %.5f', self.vmc_markovchain.step_size / approximate_step_size, acc_ration
            )

    def optimize_vmc_step(self, steps):
        """Optimize vmc step size."""
        xdata = np.linspace(0, 2, 11)
        ydata = np.ones_like(xdata)
        step_size = self.approximate_step_size
        for i in range(1, xdata.size):
            self.vmc_markovchain.step_size = step_size * xdata[i]
            position = self.vmc_markovchain.random_walk(steps, 1)
            acc_ration = (np.isfinite(position[:, 0, 0])).mean()
            if self.config.input.vmc_method == 1:
                acc_ration /= (self.neu + self.ned)
            ydata[i] = mpi_comm.allreduce(acc_ration) / mpi_comm.size

        def f(ts, a, ts0):
            """Dependence of the acceptance probability on the step size in CBCS case looks like:
            p(ts) = (exp(a/ts0) - 1)/(exp(a/ts0) + exp(ts/ts0) - 2)
            :param ts: step_size
            :param a: step_size for 50% acceptance probability
            :param ts0: scale factor
            :return: acceptance probability
            """
            return (np.exp(a/ts0) - 1) / (np.exp(a/ts0) + np.exp(ts/ts0) - 2)

        logger.info(
            ' Performing time-step optimization.'
        )
        if self.root:
            warnings.simplefilter("error", OptimizeWarning)
            try:
                popt, pcov = curve_fit(f, xdata, ydata)
                step_size *= popt[0]
            except OptimizeWarning:
                logger.info(
                    f' time-step optimization failed for.\n'
                    f' ydata: {ydata}\n'
                    f' set step size to approximate'
                )
        self.vmc_markovchain.step_size = mpi_comm.bcast(step_size)

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
        if self.config.input.testrun:
            logger.info(
                ' TEST RUN only.\n'
                ' Quitting.\n'
            )
        elif self.config.input.runtype == 'vmc':
            logger.info(
                ' ====================================\n'
                ' PERFORMING A SINGLE VMC CALCULATION.\n'
                ' ====================================\n\n'
            )
            self.vmc_energy_accumulation()
        elif self.config.input.runtype == 'vmc_opt':
            if self.root:
                self.config.write('.', 0)
            opt_method = self.config.input.opt_method
            opt_cycles = self.config.input.opt_cycles
            if self.config.input.opt_plan:
                opt_cycles = len(self.config.input.opt_plan)
            for i in range(opt_cycles):
                if self.config.input.opt_plan:
                    opt_method = self.config.input.opt_plan[i].get('method', self.config.input.opt_method)
                    vm_reweight = self.config.input.opt_plan[i].get('reweight', self.config.input.vm_reweight)
                    self.wfn.opt_jastrow = self.config.input.opt_plan[i].get('jastrow', self.config.input.opt_jastrow)
                    self.wfn.opt_backflow = self.config.input.opt_plan[i].get('backflow', self.config.input.opt_backflow)
                    self.wfn.opt_orbitals = self.config.input.opt_plan[i].get('orbitals', self.config.input.opt_orbitals)
                    self.wfn.opt_det_coeff = self.config.input.opt_plan[i].get('det_coeff', self.config.input.opt_det_coeff)
                    if self.wfn.jastrow:
                        self.wfn.jastrow.cutoffs_optimizable = not self.config.input.opt_plan[i].get('fix_cutoffs', False)
                    if self.wfn.backflow:
                        self.wfn.backflow.cutoffs_optimizable = not self.config.input.opt_plan[i].get('fix_cutoffs', False)
                position = self.vmc_energy_accumulation()
                logger.info(
                    f' ==========================================\n'
                    f' PERFORMING OPTIMIZATION CALCULATION No. {i+1}.\n'
                    f' ==========================================\n\n'
                )
                if opt_method == 'varmin':
                    if vm_reweight:
                        self.vmc_reweighted_variance_minimization(self.config.input.vmc_nconfig_write)
                    else:
                        self.vmc_unreweighted_variance_minimization(self.config.input.vmc_nconfig_write)
                elif opt_method == 'madmin':
                    # https://optimization.cbe.cornell.edu/index.php?title=Optimization_with_absolute_values
                    # use scipy.optimize.linprog
                    raise NotImplementedError
                elif opt_method == 'emin':
                    if self.config.input.emin_method == 'newton':
                        self.vmc_energy_minimization_newton(self.config.input.vmc_nconfig_write)
                    elif self.config.input.emin_method == 'linear':
                        self.vmc_energy_minimization_linear_method(self.config.input.vmc_nconfig_write)
                    elif self.config.input.emin_method == 'reconf':
                        self.vmc_energy_minimization_stochastic_reconfiguration(self.config.input.vmc_nconfig_write)
                self.config.jastrow.u_cutoff[0]['value'] = self.wfn.jastrow.u_cutoff
                if self.root:
                    self.config.write('.', i + 1)
            if self.config.input.postfit_vmc:
                self.vmc_energy_accumulation()
        elif self.config.input.runtype == 'vmc_dmc':
            logger.info(
                 ' ======================================================\n'
                 ' PERFORMING A VMC CONFIGURATION-GENERATION CALCULATION.\n'
                 ' ======================================================\n\n'
            )
            position = self.vmc_energy_accumulation()
            r_e_list = position[-self.config.input.vmc_nconfig_write // mpi_comm.size:]
            expand(r_e_list)
            self.dmc_markovchain = DMCMarkovChain(
                r_e_list, self.config.input.alimit, self.config.input.nucleus_gf_mods, self.config.input.use_tmove,
                self.config.input.dtdmc, self.config.input.dmc_target_weight, self.wfn, self.config.input.dmc_method,
            )
            self.dmc_energy_equilibration()
            self.dmc_energy_accumulation()

        stop = default_timer()
        logger.info(
            f' =========================================================================\n\n'
            f' Total PyCasino real time : : :    {stop - start:.4f}'
        )

    def equilibrate(self, steps):
        """Burn-in.
        :param steps: burn-in period
        :return:
        """
        self.vmc_markovchain.random_walk(steps, self.decorr_period)
        logger.info(
            f' Running VMC equilibration ({steps} moves).'
        )

    def vmc_energy_accumulation(self):
        """VMC energy accumulation"""
        logger.info(
            ' BEGIN VMC CALCULATION\n'
            ' =====================\n'
        )
        self.equilibrate(self.config.input.vmc_equil_nstep)

        if self.config.input.opt_dtvmc == 0:
            self.vmc_markovchain.step_size = np.sqrt(3 * self.config.input.dtvmc)
        elif self.config.input.opt_dtvmc == 1:
            # to achieve an acceptance ratio of (roughly) 50% (EBES default).
            self.optimize_vmc_step(1000)
        elif self.config.input.opt_dtvmc == 2:
            # to maximize the diffusion constant with respect to dtvmc (CBCS default).
            raise NotImplementedError

        logger.info(
            f' Optimized step size: {self.vmc_markovchain.step_size:.5f}\n'
            f' DTVMC: {(self.vmc_markovchain.step_size**2)/3:.5f}\n'
        )

        nblock = self.config.input.vmc_nblock
        steps = self.config.input.vmc_nstep // nblock // mpi_comm.size * nblock * mpi_comm.size
        nblock_steps = steps // nblock // mpi_comm.size

        logger.info(
            ' Starting VMC.\n'
        )
        energy_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create energy numpy array whose data points to the shared buffer
        buffer, _ = energy_buffer.Shared_query(rank=0)
        energy = np.ndarray(buffer=buffer, shape=(mpi_comm.size, nblock, nblock_steps))

        for i in range(nblock):
            block_start = default_timer()
            position = self.vmc_markovchain.random_walk(nblock_steps, self.decorr_period)
            energy[mpi_comm.rank, i] = vmc_observable(position, self.wfn.energy)
            # wait until all processes have written to the array
            mpi_comm.Barrier()
            if self.root:
                energy_block_mean = np.mean(energy[:, i, :])
                energy_block_var = np.var(energy[:, i, :])
                energy_block_sem = np.mean(correlated_sem(energy[:, i, :])) / np.sqrt(mpi_comm.size)
                block_stop = default_timer()
                logger.info(
                    f' =========================================================================\n'
                    f' In block : {i + 1}\n'
                    f'  Number of VMC steps           = {steps // nblock}\n\n'
                    f'  Block average energies (au)\n\n'
                    f'  Total energy                       (au) =       {energy_block_mean:18.12f}\n'
                    f'  Standard error                        +/-       {energy_block_sem:18.12f}\n\n'
                    f'  Constant energy contributions      (au) =       {self.wfn.nuclear_repulsion:18.12f}\n\n'
                    f'  Variance of local energy           (au) =       {energy_block_var:18.12f}\n'
                    f'  Standard error                        +/-       {0:18.12f}\n\n'
                    f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
                )
        if self.root:
            energy_sem = np.mean(correlated_sem(energy.reshape(mpi_comm.size, nblock * nblock_steps))) / np.sqrt(mpi_comm.size)
            logger.info(
                f' =========================================================================\n'
                f' FINAL RESULT:\n\n'
                f'  VMC energy (au)    Standard error      Correction for serial correlation\n\n'
                f' {energy.mean():.12f} +/- {energy_sem:.12f}      On-the-fly reblocking method\n\n'
                f' Sample variance of E_L (au^2/sim.cell) : {energy.var():.12f}\n\n'
            )
        energy_buffer.Free()
        return position

    def dmc_energy_equilibration(self):
        """DMC energy equilibration"""
        logger.info(
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
            logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n\n'
                f' Number of moves in block                 : {steps // nblock}\n'
                f' Load-balancing efficiency (%)            : {100 * np.mean(self.dmc_markovchain.efficiency_list):.3f}\n'
                f' Acceptance ratio (%)                     : {100 * self.dmc_markovchain.step_eff / self.dmc_markovchain.step_size:.3f}\n'
                f' Number of config transfers               : {self.dmc_markovchain.ntransfers_tot}\n'
                f' New best estimate of DMC energy (au)     : {energy.mean():.8f}\n'
                f' New best estimate of effective time step : {self.dmc_markovchain.step_eff:.8f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )

    def dmc_energy_accumulation(self):
        """DMC energy accumulation"""
        logger.info(
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
            logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n\n'
                f' Number of moves in block                 : {block_steps}\n'
                f' Load-balancing efficiency (%)            : {100 * np.mean(self.dmc_markovchain.efficiency_list):.3f}\n'
                f' Acceptance ratio (%)                     : {100 * self.dmc_markovchain.step_eff / self.dmc_markovchain.step_size:.3f}\n'
                f' Number of config transfers               : {self.dmc_markovchain.ntransfers_tot}\n'
                f' New best estimate of DMC energy (au)     : {energy_mean:.8f}\n'
                f' New best estimate of effective time step : {self.dmc_markovchain.step_eff:.8f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
        logger.info(
            f'Mixed estimators of the energies at the end of the run\n'
            f'------------------------------------------------------\n\n'
            f'Total energy                 =       {energy.mean():.12f} +/- {correlated_sem(energy):.12f}\n'
        )

    def distribution(self, energy):
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

    def check(self, steps):
        """Check"""
        self.wfn.set_parameters_projector()
        self.wfn.opt_jastrow, self.wfn.opt_backflow, self.wfn.opt_det_coeff = False, False, False
        position = self.vmc_markovchain.random_walk(steps // mpi_comm.size, self.decorr_period)
        for pos in position:
            if not np.isnan(pos.sum()):
                e_vectors, n_vectors = self.wfn._relative_coordinates(pos)
                logger.info(self.wfn.slater.cusp.gradient(n_vectors)[0] / self.wfn.slater.cusp.numerical_gradient(n_vectors)[0])
                logger.info(self.wfn.slater.cusp.laplacian(n_vectors)[0] / self.wfn.slater.cusp.numerical_laplacian(n_vectors)[0])
                logger.info(self.wfn.slater.cusp.hessian(n_vectors)[0] / self.wfn.slater.cusp.numerical_hessian(n_vectors)[0])
                logger.info(self.wfn.slater.cusp.tressian(n_vectors)[0] / self.wfn.slater.cusp.numerical_tressian(n_vectors)[0])
                logger.info(self.wfn.slater.tressian(n_vectors)[0] / self.wfn.slater.tressian_v2(n_vectors)[0])
                logger.info(self.wfn.slater.tressian(n_vectors)[0] / self.wfn.slater.numerical_tressian(n_vectors))
                if self.wfn.jastrow is not None:
                    logger.info(self.wfn.jastrow.value_parameters_d1(e_vectors, n_vectors) / self.wfn.jastrow.value_parameters_numerical_d1(e_vectors, n_vectors, False))
                    logger.info(self.wfn.jastrow.gradient_parameters_d1(e_vectors, n_vectors) / self.wfn.jastrow.gradient_parameters_numerical_d1(e_vectors, n_vectors, False))
                    logger.info(self.wfn.jastrow.laplacian_parameters_d1(e_vectors, n_vectors) / self.wfn.jastrow.laplacian_parameters_numerical_d1(e_vectors, n_vectors, False))
                if self.wfn.backflow is not None:
                    logger.info(self.wfn.backflow.value_parameters_d1(e_vectors, n_vectors) / self.wfn.backflow.value_parameters_numerical_d1(e_vectors, n_vectors, False))
                    logger.info(self.wfn.backflow.parameters_projector.T @ self.wfn.backflow.gradient_parameters_d1(e_vectors, n_vectors)[0][:, :, 1] / self.wfn.backflow.gradient_parameters_numerical_d1(e_vectors, n_vectors, False)[:, :, 1])
                    logger.info(self.wfn.backflow.parameters_projector.T @ self.wfn.backflow.laplacian_parameters_d1(e_vectors, n_vectors)[0] / self.wfn.backflow.laplacian_parameters_numerical_d1(e_vectors, n_vectors, False))
                logger.info(self.wfn.value_parameters_d1(pos) / self.wfn.value_parameters_numerical_d1(pos))
                logger.info(self.wfn.energy_parameters_d1(pos) / self.wfn.energy_parameters_numerical_d1(pos))

    def vmc_unreweighted_variance_minimization(self, steps, verbose=2):
        """Minimize vmc unreweighted variance.
        https://github.com/scipy/scipy/issues/10634
        :param steps: number of configs
        :param verbose:
            0 : work silently.
            1 : display a termination report.
            2 : display progress during iterations.
        """
        steps = steps // mpi_comm.size * mpi_comm.size
        start, stop = mpi_comm.rank * steps // mpi_comm.size, (mpi_comm.rank + 1) * steps // mpi_comm.size
        # rescale for "Cost column" in output of scipy.optimize.least_squares to be a variance of E local
        scale = np.sqrt(2) / np.sqrt(steps - 1)
        x0 = self.wfn.get_parameters()
        # FIXME: reuse from vmc_energy_accumulation run
        position = self.vmc_markovchain.random_walk(steps // mpi_comm.size, self.decorr_period)
        logger.info(
            ' Optimization start\n'
            ' =================='
        )
        energy_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create energy numpy array whose data points to the shared buffer
        buffer, _ = energy_buffer.Shared_query(rank=0)
        energy = np.ndarray(buffer=buffer, shape=(steps, ))
        energy_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create energy_gradient numpy array whose data points to the shared buffer
        buffer, _ = energy_gradient_buffer.Shared_query(rank=0)
        energy_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))

        def fun(x, *args, **kwargs):
            self.wfn.set_parameters(x)
            energy[start:stop] = vmc_observable(position, self.wfn.energy)
            mpi_comm.Barrier()
            return scale * (energy - energy.mean())

        def jac(x, *args, **kwargs):
            self.wfn.set_parameters(x)
            self.wfn.set_parameters_projector()
            if self.config.input.opt_fixnl:
                energy_gradient[start:stop] = vmc_observable(position, self.wfn.kinetic_energy_parameters_d1)
            else:
                energy_gradient[start:stop] = vmc_observable(position, self.wfn.energy_parameters_d1)
            mpi_comm.Barrier()
            return scale * (energy_gradient - energy_gradient.mean(axis=0))

        res = least_squares(
            fun, x0=x0, jac=jac, method='trf', ftol=2/np.sqrt(steps-1), x_scale='jac',
            tr_solver='exact', max_nfev=self.config.input.opt_maxeval, verbose=self.root and verbose
        )
        parameters = res.x
        energy_buffer.Free()
        energy_gradient_buffer.Free()
        mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters)
        logger.info(
            f'Norm of Jacobian at the solution: {np.linalg.norm(res.jac.mean(axis=0)):.5e}\n'
        )

    def vmc_reweighted_variance_minimization(self, steps, verbose=2):
        """Minimize vmc reweighted variance.
        https://github.com/scipy/scipy/issues/10634
        :param steps: number of configs
        :param verbose:
            0 : work silently.
            1 : display a termination report.
            2 : display progress during iterations.
        """
        steps = steps // mpi_comm.size * mpi_comm.size
        start, stop = mpi_comm.rank * steps // mpi_comm.size, (mpi_comm.rank + 1) * steps // mpi_comm.size
        x0 = self.wfn.get_parameters()
        # FIXME: reuse from vmc_energy_accumulation run
        position = self.vmc_markovchain.random_walk(steps // mpi_comm.size, self.decorr_period)
        logger.info(
            ' Optimization start\n'
            ' =================='
        )
        wfn_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create wfn numpy array whose data points to the shared buffer
        buffer, _ = wfn_buffer.Shared_query(rank=0)
        wfn = np.ndarray(buffer=buffer, shape=(steps, ))
        wfn_0_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create wfn_0 numpy array whose data points to the shared buffer
        buffer, _ = wfn_0_buffer.Shared_query(rank=0)
        wfn_0 = np.ndarray(buffer=buffer, shape=(steps, ))
        energy_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create energy numpy array whose data points to the shared buffer
        buffer, _ = energy_buffer.Shared_query(rank=0)
        energy = np.ndarray(buffer=buffer, shape=(steps, ))
        wfn_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create wfn_gradient numpy array whose data points to the shared buffer
        buffer, _ = wfn_gradient_buffer.Shared_query(rank=0)
        wfn_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))
        energy_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create energy_gradient numpy array whose data points to the shared buffer
        buffer, _ = energy_gradient_buffer.Shared_query(rank=0)
        energy_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))
        wfn_0[start:stop] = vmc_observable(position, self.wfn.value)
        mpi_comm.Barrier()

        def fun(x, *args, **kwargs):
            self.wfn.set_parameters(x)
            wfn[start:stop] = vmc_observable(position, self.wfn.value)
            energy[start:stop] = vmc_observable(position, self.wfn.energy)
            mpi_comm.Barrier()
            weights = (wfn / wfn_0)**2
            mean_energy = np.average(energy, weights=weights)
            ddof = np.average(weights, weights=weights)  # Delta Degrees of Freedom
            # rescale for "Cost column" in output of scipy.optimize.least_squares to be variance of E local
            return np.sqrt(2) * (energy - mean_energy) * np.sqrt(weights / (weights.sum() - ddof))

        def jac(x, *args, **kwargs):
            """
            diff(weights, p) = 2 * wfn_gradient * weights
            diff(np.average(x, weights=weights), p) =
                       np.average(diff(x, p), weights=weights) +
                       2 * np.average(x * wfn_gradient, weights=weights) -
                       2 * np.average(wfn_gradient, weights=weights) * np.average(x, weights=weights)
            diff(ddof, p) = diff(np.average(weights, weights=weights), p) =
                       2 * np.average(wfn_gradient * weights, weights=weights) +
                       2 * np.average(weights * wfn_gradient, weights=weights) -
                       2 * np.average(wfn_gradient, weights=weights) * ddof
            """
            self.wfn.set_parameters(x)
            self.wfn.set_parameters_projector()
            # jac(x) call allways follows fun(x) call
            # wfn[start:stop] = vmc_observable(position, self.wfn.value)
            # energy[start:stop] = vmc_observable(position, self.wfn.energy)
            wfn_gradient[start:stop] = vmc_observable(position, self.wfn.value_parameters_d1)
            if self.config.input.opt_fixnl:
                energy_gradient[start:stop] = vmc_observable(position, self.wfn.kinetic_energy_parameters_d1)
            else:
                energy_gradient[start:stop] = vmc_observable(position, self.wfn.energy_parameters_d1)
            mpi_comm.Barrier()
            weights = (wfn / wfn_0)**2
            mean_energy = np.average(energy, weights=weights)
            mean_wfn_gradient = np.average(wfn_gradient, axis=0, weights=weights)
            mean_energy_gradient = np.average(energy_gradient, axis=0, weights=weights)
            ddof = np.average(weights, weights=weights)  # Delta Degrees of Freedom
            half_ddof_gradient = 2 * np.average(wfn_gradient * np.expand_dims(weights, 1), axis=0, weights=weights) - ddof * mean_wfn_gradient
            # rescale for "Cost column" in output of scipy.optimize.least_squares to be a variance of E local
            return np.sqrt(2) * (
                energy_gradient - mean_energy_gradient +
                2 * (np.average(wfn_gradient * np.expand_dims(energy, 1), axis=0, weights=weights) - mean_energy * mean_wfn_gradient) +
                np.expand_dims((energy - mean_energy), 1) * (mean_wfn_gradient - (mean_wfn_gradient * weights.sum() - half_ddof_gradient) / (weights.sum() - ddof))
            ) * np.sqrt(np.expand_dims(weights, 1) / (weights.sum() - ddof))

        res = least_squares(
            fun, x0=x0, jac=jac, method='trf', ftol=2/np.sqrt(steps-1),
            tr_solver='exact', max_nfev=self.config.input.opt_maxeval, verbose=self.root and verbose
        )
        parameters = res.x
        wfn_buffer.Free()
        wfn_0_buffer.Free()
        energy_buffer.Free()
        wfn_gradient_buffer.Free()
        energy_gradient_buffer.Free()
        mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters)
        logger.info(
            f'Norm of Jacobian at the solution: {np.linalg.norm(res.jac.mean(axis=0)):.5e}\n'
        )

    def energy_parameters_gradient(self, data):
        """Gradient estimator of local energy from
        Optimization of quantum Monte Carlo wave functions by energy minimization.
        Julien Toulouse, C. J. Umrigar
        :param data: data structure
        :return:
        """
        parameters_size = data['wfn_gradient_mean'].size
        energy = data['energy'] - data['energy_mean']
        wfn_gradient = data['wfn_gradient'] - data['wfn_gradient_mean']
        jacobian = 2 * wfn_gradient.T @ energy / parameters_size
        mpi_comm.Allreduce(MPI.IN_PLACE, jacobian)
        return jacobian / mpi_comm.size

    def energy_parameters_hessian(self, data):
        """Hessian estimators of local energy from
        Optimization of quantum Monte Carlo wave functions by energy minimization.
        Julien Toulouse, C. J. Umrigar
        :param data: data structure
        :return:
        """
        parameters_size = data['wfn_gradient_mean'].size
        energy = data['energy'] - data['energy_mean']
        wfn_gradient = data['wfn_gradient'] - data['wfn_gradient_mean']
        A = 2 * data['wfn_hessian'].T @ energy
        B = 4 * wfn_gradient.T @ (wfn_gradient * np.expand_dims(energy, 1))
        # Umrigar and Filippi
        half_D = wfn_gradient.T @ data['energy_gradient']
        hessian = (A + B + half_D + half_D.T) / parameters_size
        mpi_comm.Allreduce(MPI.IN_PLACE, hessian)
        return hessian / mpi_comm.size

    def vmc_energy_minimization_newton(self, steps, method='Newton-CG'):
        """Minimize vmc energy by Newton or gradient descent methods.
        For SJB wfn = exp(J(r)) * S(Bf(r))
            second derivatives by Jastrow parameters is:
        d²exp(J(p)) * S(Bf(r))/dp² = d(dJ(p)/dp * wfn)/dp = (d²J(p)/dp² + dJ(p)/dp * dJ(p)/dp) * wfn
            second derivatives by backflow parameters is:
        exp(J(r)) * d²S(Bf(p))/dp² = exp(J(r)) * d(dS(r)/dr * dBf(p)/dp)/dp =
        exp(J(r)) * (d²S(r)/dr² * dBf(p)/dp * dBf(p)/dp + dS(r)/dr * d²Bf(p)/dp²) =
        exp(J(r)) * (d²S(r)/dr² * dBf(p)/dp * dBf(p)/dp + dS(r)/dr * d²Bf(p)/dp²)
        :param steps: number of configs
        :param method: type of solver
        """
        data = dict()
        steps = steps // mpi_comm.size * mpi_comm.size
        x0 = self.wfn.get_parameters()
        self.wfn.set_parameters(x0)
        # FIXME: reuse from vmc_energy_accumulation run
        position = self.vmc_markovchain.random_walk(steps // mpi_comm.size, self.decorr_period)
        logger.info(
            ' Optimization start\n'
            ' =================='
        )
        scale = self.wfn.get_parameters_scale()

        def callback(x):
            """Log intermediate results"""
            # logger.info(x * scale)
            energy_mean = data['energy_mean']
            jacobian_norm = np.linalg.norm(self.energy_parameters_gradient(data) * scale)
            logger.info(f'     {callback.iteration:3d}            {callback.nfev:3d}        {energy_mean:.6e}         {jacobian_norm:.5e}')
            # Sorry, but we need a pointer!
            callback.iteration += 1

        def fun(x, *args):
            """For Nelder-Mead, Powell, COBYLA and those listed in jac and hess methods."""
            callback.nfev += 1
            self.wfn.set_parameters(x * scale)
            data['energy'] = vmc_observable(position, self.wfn.energy)
            data['energy_mean'] = data['energy'].mean()
            return mpi_comm.allreduce(data['energy_mean']) / mpi_comm.size

        def jac(x, *args):
            """Only for CG, BFGS, L-BFGS-B, TNC, SLSQP and those listed in hess method."""
            self.wfn.set_parameters(x * scale)
            self.wfn.set_parameters_projector()
            data['energy'] = vmc_observable(position, self.wfn.energy)
            data['energy_mean'] = data['energy'].mean()
            data['wfn_gradient'] = vmc_observable(position, self.wfn.value_parameters_d1)
            data['wfn_gradient_mean'] = np.mean(data['wfn_gradient'], axis=0)
            return self.energy_parameters_gradient(data) * scale

        def hess(x, *args):
            """Only for Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr."""
            self.wfn.set_parameters(x * scale)
            self.wfn.set_parameters_projector()
            data['energy'] = vmc_observable(position, self.wfn.energy)
            data['energy_mean'] = data['energy'].mean()
            data['wfn_gradient'] = vmc_observable(position, self.wfn.value_parameters_d1)
            data['wfn_gradient_mean'] = np.mean(data['wfn_gradient'], axis=0)
            data['wfn_hessian'] = vmc_observable(position, self.wfn.value_parameters_d2)
            data['energy_gradient'] = vmc_observable(position, self.wfn.energy_parameters_d1)
            return self.energy_parameters_hessian(data) * np.outer(scale, scale)

        callback.nfev = 0
        callback.iteration = 0
        logger.info(f'Optimization method: {method}')
        logger.info('   Iteration     Total nfev        Energy             Grad norm')
        if method == 'TNC':
            options = dict(disp=self.root, scale=np.ones(shape=(x0.size, )), offset=np.zeros(shape=(x0.size, )), stepmx=1)
        elif method in ('dogleg', 'trust-ncg', 'trust-exact'):
            # default 1:1000:0.15:1e-4
            options = dict(initial_trust_radius=0.1, max_trust_radius=1, eta=0.15, gtol=1e-3)
        else:
            options = dict(disp=self.root)
        # Desired error not necessarily achieved due to precision loss.
        # https://github.com/scipy/scipy/issues/15643
        res = minimize(fun, x0=x0 / scale, method=method, jac=jac, hess=hess, callback=callback, options=options)
        logger.info(f'Norm of Jacobian at the solution: {np.linalg.norm(res.jac):.5e}\n')
        parameters = res.x * scale
        mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters)

    def vmc_energy_minimization_linear_method(self, steps):
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
        One can introduce following approximation of S and H:
            S = extended_wfn_gradient.T @ extended_wfn_gradient
            H = extended_wfn_gradient.T @ diag(energy) @ extended_wfn_gradient - extended_wfn_gradient.T @ extended_energy_gradient
        :param steps: number of configs
        """
        invert_S = False
        steps = steps // mpi_comm.size * mpi_comm.size
        start, stop = mpi_comm.rank * steps // mpi_comm.size, (mpi_comm.rank + 1) * steps // mpi_comm.size
        x0 = self.wfn.get_parameters()
        if x0.all():
            self.wfn.set_parameters(x0)
        else:
            # CASINO variant
            # self.wfn.jastrow.set_u_parameters_for_emin()
            # not starting from HF distribution
            # self.wfn.set_parameters(x0)
            # starting from HF distribution
            pass
        # FIXME: reuse from vmc_energy_accumulation run
        position = self.vmc_markovchain.random_walk(steps // mpi_comm.size, self.decorr_period)
        logger.info(
            ' Optimization start\n'
            ' =================='
        )
        self.wfn.set_parameters_projector()
        energy_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create energy numpy array whose data points to the shared buffer
        buffer, _ = energy_buffer.Shared_query(rank=0)
        energy = np.ndarray(buffer=buffer, shape=(steps, ))
        wfn_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create wfn_gradient numpy array whose data points to the shared buffer
        buffer, _ = wfn_gradient_buffer.Shared_query(rank=0)
        wfn_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))
        energy_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create energy_gradient numpy array whose data points to the shared buffer
        buffer, _ = energy_gradient_buffer.Shared_query(rank=0)
        energy_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))
        energy[start:stop] = vmc_observable(position, self.wfn.energy)
        wfn_gradient[start:stop] = vmc_observable(position, self.wfn.value_parameters_d1)
        if self.config.input.opt_fixnl:
            energy_gradient[start:stop] = vmc_observable(position, self.wfn.kinetic_energy_parameters_d1)
        else:
            energy_gradient[start:stop] = vmc_observable(position, self.wfn.energy_parameters_d1)
        mpi_comm.Barrier()
        dp = np.empty_like(x0)
        if self.root:
            energy_0 = energy.mean()
            sem = np.mean(correlated_sem(energy.reshape(mpi_comm.size, steps // mpi_comm.size))) / np.sqrt(mpi_comm.size)
            if invert_S:
                scale = 1
                S_inv_H = S_inv_H_matrix(wfn_gradient * scale, energy, energy_gradient * scale)
                stabilization = 1
                logger.info(f'Stabilization: {stabilization:.1f}')
                S_inv_H[1:, 1:] += stabilization * np.eye(x0.size)
                eigvals, eigvectors = sp.linalg.eig(S_inv_H)
            else:
                # rescale parameters so that S becomes the Pearson correlation matrix
                scale = 1 / np.std(wfn_gradient, axis=0)
                # FIXME: remove zero scale
                S = overlap_matrix(wfn_gradient * scale)
                H = hamiltonian_matrix(wfn_gradient * scale, energy, energy_gradient * scale)
                # logger.info(f'epsilon:\n{np.diag(H[1:, 1:]) / np.diag(S[1:, 1:]) - H[0, 0]}')
                stabilization = 1
                logger.info(f'Stabilization: {stabilization:.1f} SEM')
                H[1:, 1:] += stabilization * sem * np.eye(x0.size)
                eigvals, eigvectors = sp.linalg.eig(H, S)
            # since imaginary parts only arise from statistical noise, discard them
            eigvals, eigvectors = np.real(eigvals), np.real(eigvectors)
            idx = np.abs(eigvectors[0]).argmax()
            eigval, eigvector = eigvals[idx], eigvectors[:, idx]
            logger.info(f'E_0 {energy_0:.8f} E_lin {eigval:.8f} dE {eigval - energy_0:.8f}')
            logger.info(f'eigvector[0] {np.abs(eigvector[0]):.8f}')
            # uniform rescaling of normalized eigvector
            # in case ξ = 0 is equivalent to multiplying by eigvector[0]
            # in case ξ = 1 is equivalent to dividing by eigvector[0]
            # as 1 / (1 + Q) = eigvector[0] ** 2
            if x0.all():
                dp = eigvector[1:] * eigvector[0] * scale
            else:
                self.wfn.set_parameters(x0)
                dp = eigvector[1:] * eigvector[0] * scale

        mpi_comm.Bcast(dp)
        if x0.all():
            logger.info(f'delta p / p\n{dp / x0}\n')
        else:
            logger.info(f'delta p\n{dp}\n')
        self.wfn.set_parameters(x0 + dp)
        energy_buffer.Free()
        wfn_gradient_buffer.Free()
        energy_gradient_buffer.Free()

    def vmc_energy_minimization_stochastic_reconfiguration(self, steps, opt_jastrow, opt_backflow, opt_det_coeff):
        """Minimize vmc energy by stochastic reconfiguration.
        Stochastic Reconfiguration (SR) is a second-order optimization method. Instead of manipulating the gradients according to
        their history, the SR algorithm manipulates the gradients according to the curvature of the energy landscape. It can
        alternatively be viewed as stretching and squeezing the landscape itself, making it smoother or more isotropic in certain
        areas. SR provides a more favorable terrain for finding the global minimum and improves the exploration of the parameter space.
        SR reveals the following update rule for the parameters:
                                        p <= η * S(p)^−1 · energy_gradient(p) / epsilon
        as:
            epsilon = Hii/Sii - H0
            energy_gradient = wfn_gradient.T · energy
            S = wfn_gradient.T @ wfn_gradient
            diag(S) = np.std(wfn_gradient, axis=0) ** 2
            pinv(A) = (A.T · A)^-1 · A.T
                                        p <= η * pinv(wfn_gradient(p)) · energy(p)

        :param steps: number of configs
        """
        steps = steps // mpi_comm.size * mpi_comm.size
        start, stop = mpi_comm.rank * steps // mpi_comm.size, (mpi_comm.rank + 1) * steps // mpi_comm.size
        x0 = self.wfn.get_parameters()
        self.wfn.set_parameters(x0)
        # FIXME: reuse from vmc_energy_accumulation run
        position = self.vmc_markovchain.random_walk(steps // mpi_comm.size, self.decorr_period)
        logger.info(
            ' Optimization start\n'
            ' =================='
        )
        energy_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create energy numpy array whose data points to the shared buffer
        buffer, _ = energy_buffer.Shared_query(rank=0)
        energy = np.ndarray(buffer=buffer, shape=(steps, ))
        wfn_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create wfn_gradient numpy array whose data points to the shared buffer
        buffer, _ = wfn_gradient_buffer.Shared_query(rank=0)
        wfn_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))
        energy_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create energy_gradient numpy array whose data points to the shared buffer
        buffer, _ = energy_gradient_buffer.Shared_query(rank=0)
        energy_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))

        def fun(x, *args):
            self.wfn.set_parameters(x)
            energy[start:stop] = vmc_observable(position, self.wfn.energy)
            mpi_comm.Barrier()
            logger.info(f'energy: {energy.mean()}')
            return energy.mean()

        def jac(x, *args):
            self.wfn.set_parameters(x)
            self.wfn.set_parameters_projector()
            energy[start:stop] = vmc_observable(position, self.wfn.energy)
            wfn_gradient[start:stop] = vmc_observable(position, self.wfn.value_parameters_d1)
            mpi_comm.Barrier()
            if self.root:
                energy[:] -= np.mean(energy)
                wfn_gradient[:, :] -= np.mean(wfn_gradient, axis=0)
            mpi_comm.Barrier()
            return 2 * wfn_gradient.T @ energy / steps

        def hess(x, *args):
            self.wfn.set_parameters(x)
            self.wfn.set_parameters_projector()
            energy[start:stop] = vmc_observable(position, self.wfn.energy)
            wfn_gradient[start:stop] = vmc_observable(position, self.wfn.value_parameters_d1)
            if self.config.input.opt_fixnl:
                energy_gradient[start:stop] = vmc_observable(position, self.wfn.kinetic_energy_parameters_d1)
            else:
                energy_gradient[start:stop] = vmc_observable(position, self.wfn.energy_parameters_d1)
            mpi_comm.Barrier()
            if self.root:
                wfn_gradient[:, :] -= np.mean(wfn_gradient, axis=0)
            mpi_comm.Barrier()
            S_diag = np.var(wfn_gradient, axis=0)
            H_diag = np.mean(wfn_gradient * (np.expand_dims(energy, 1) * wfn_gradient), axis=0) + np.mean(wfn_gradient * energy_gradient, axis=0)
            epsilon = H_diag / S_diag
            logger.info(f'epsilon:\n{epsilon}')
            stabilization = 1
            logger.info(f'Stabilization: {stabilization:.1f}')
            return wfn_gradient.T @ wfn_gradient * (epsilon + stabilization) / steps

        options = dict(disp=self.root)
        res = minimize(fun, x0=x0, method='Newton-CG', jac=jac, hess=hess, options=options)
        logger.info('Jacobian matrix at the solution:')
        logger.info(res.jac)
        parameters = res.x
        energy_buffer.Free()
        wfn_gradient_buffer.Free()
        energy_gradient_buffer.Free()
        mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters)


def main():

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
