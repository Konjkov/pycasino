import argparse
import datetime
import importlib.metadata
import logging
import os
import sys
from timeit import default_timer

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy as sp
from mpi4py import MPI
from scipy.optimize import least_squares, minimize
from scipy.special import erfinv, ndtri
from statsmodels.tsa.stattools import pacf

from .backflow import Backflow
from .cusp import CuspFactory
from .dmc import DMC
from .geminal import Geminal
from .gjastrow import Gjastrow
from .jastrow import Jastrow
from .ppotential import PPotential
from .readers import CasinoConfig
from .sem import correlated_sem
from .slater import Slater
from .vmc import VMC
from .wfn import Wfn

__version__ = importlib.metadata.version('casino')
__author__ = 'Vladimir Konkov'


# created with art python package
logo = f"""
 ------------------------------------------------------------------------------
 ########::'##:::'##::'######:::::'###:::::'######::'####:'##::: ##::'#######::
 ##.... ##:. ##:'##::'##... ##:::'## ##:::'##... ##:. ##:: ###:: ##:'##.... ##:
 ##:::: ##::. ####::: ##:::..:::'##:. ##:: ##:::..::: ##:: ####: ##: ##:::: ##:
 ########::::. ##:::: ##:::::::'##:::. ##:. ######::: ##:: ## ## ##: ##:::: ##:
 ##.....:::::: ##:::: ##::::::: #########::..... ##:: ##:: ##. ####: ##:::: ##:
 ##::::::::::: ##:::: ##::: ##: ##.... ##:'##::: ##:: ##:: ##:. ###: ##:::: ##:
 ##::::::::::: ##::::. ######:: ##:::: ##:. ######::'####: ##::. ##:. #######::
 .::::::::::::..::::::......:::..:::::..:::......:::....::..::::..:::.......:::

                     Python Quantum Monte Carlo Package
                        v {__version__} [{__author__}]

    Main Author : {__author__}
 ------------------------------------------------------------------------------
 Started {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}

 Python {sys.version}
 Numba {nb.__version__}
 Numpy {np.__version__}
 Scipy {sp.__version__}
"""


logger = logging.getLogger(__name__)

mpi_comm = MPI.COMM_WORLD
double_size = MPI.DOUBLE.Get_size()


def configure_logging():
    # pycasino.log is written to the current directory, as correlation.out.* are
    logging.basicConfig(level=logging.INFO, filename='pycasino.log', filemode='w', format='%(message)s')
    logger.info(logo)
    if MPI.COMM_WORLD.size > 1:
        logger.info(' Running in parallel using %i MPI processes.\n', MPI.COMM_WORLD.size)
    else:
        logger.info(' Sequential run: not using MPI.\n')
        logger.info(' Using %i OpenMP threads on %s threading layer.\n', nb.config.NUMBA_NUM_THREADS, nb.config.THREADING_LAYER)

    if MPI.COMM_WORLD.rank == 0:
        # to redirect scipy.optimize stdout to log-file
        from casino.loggers import StreamToLogger

        sys.stdout = StreamToLogger(logger, logging.INFO)
        # sys.stderr = StreamToLogger(self.logger, logging.ERROR)
    else:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False


@nb.njit(nogil=True, parallel=False, cache=True)
def expand(np_array):
    """Set NaN to previous value"""
    for i in range(1, np_array.shape[0]):
        if np.isnan(np_array[i]).all():
            np_array[i] = np_array[i - 1]


@nb.njit(nogil=True, parallel=False, cache=True)
def overlap_matrix(wfn_gradient):
    """Overlap matrix S.
    <(X-<X>)(Y-<Y>)> = <XY> - <X><Y> = (X - <X>).T @ (Y - <Y>) / size
    As any covariance matrix it is symmetric and positive semi-definite.
    Cov(x1 + x2, y1 + y2) = Cov(x1, y1) + Cov(x1, y2) + Cov(x2, y1) + Cov(x2, y2)
    """
    size_0 = wfn_gradient.shape[0]
    size_1 = wfn_gradient.shape[1] + 1
    extended_wfn_gradient = np.ones(shape=(size_0, size_1))
    extended_wfn_gradient[:, 1:] = wfn_gradient
    return extended_wfn_gradient.T @ extended_wfn_gradient / size_0


@nb.njit(nogil=True, parallel=False, cache=True)
def hamiltonian_matrix(wfn_gradient, energy, energy_gradient):
    """Hamiltonian matrix H.
    <(X-<X>)(Y-<Y>)> = <XY> - <X><Y> = (X - <X>).T @ (Y - <Y>) / size
    <(X-<X>)Z(Y-<Y>)> = <XYZ> - <XZ><Y> - <X><YZ> + <X><Y><Z> = (X - <X>).T @ diag(Z) @ (Y - <Y>) / size
    """
    size_0 = wfn_gradient.shape[0]
    size_1 = wfn_gradient.shape[1] + 1
    extended_wfn_gradient = np.ones(shape=(size_0, size_1))
    extended_wfn_gradient[:, 1:] = wfn_gradient
    extended_energy_gradient = np.zeros(shape=(size_0, size_1))
    extended_energy_gradient[:, 1:] = energy_gradient
    return extended_wfn_gradient.T @ (np.expand_dims(energy, 1) * extended_wfn_gradient + extended_energy_gradient) / size_0


@nb.njit(nogil=True, parallel=False, cache=True)
def hamiltonian_v_matrix(v0, energy_variance_gradient, energy_variance_hessian):
    """Hamiltonian variance matrix H.
    """
    size = energy_variance_gradient.shape[0] + 1
    res = np.empty(shape=(size, size))
    res[0, 0] = v0
    res[0, 1:] = energy_variance_gradient
    res[1:, 0] = energy_variance_gradient
    res[1:, 1:] = energy_variance_hessian
    return res


@nb.njit(nogil=True, parallel=False, cache=True)
def S_inv_H_matrix(wfn_gradient, energy, energy_gradient):
    """S^-1 @ H"""
    size_0 = wfn_gradient.shape[0]
    size_1 = wfn_gradient.shape[1] + 1
    extended_wfn_gradient = np.ones(shape=(size_0, size_1))
    extended_wfn_gradient[:, 1:] = wfn_gradient
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
            cusp_factory = CuspFactory(self.config)
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
            ppotential = PPotential(self.config)
        else:
            ppotential = None

        slater = Slater(self.config, cusp)

        if self.config.geminal:
            geminal = Geminal(self.config)
        else:
            geminal = None

        jastrow = None
        if self.config.jastrow:
            if self.config.input.use_jastrow:
                jastrow = Jastrow(self.config)
            elif self.config.input.use_gjastrow:
                jastrow = Gjastrow(self.config)

        if self.config.backflow:
            backflow = Backflow(self.config)
        else:
            backflow = None

        self.wfn = Wfn(self.config, slater, geminal, jastrow, backflow, ppotential)

        self.vmc = VMC(
            self.initial_position(self.config.wfn.atom_positions, self.config.wfn.atom_charges),
            self.approximate_step_size,
            self.wfn,
            self.config.input.vmc_method,
        )
        self.auto_decorr_period = 3

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
    def atom_kinetic_energy(self):
        """Share of <T> carried by each nucleus, before any sampling has been done."""
        atom_charges = self.config.wfn.atom_charges
        is_pseudoatom = self.config.wfn.is_pseudoatom
        # all-electron atom: <T> = |E| by the virial theorem, which holds whatever the net charge,
        # and the Thomas-Fermi expansion with its Scott and Dirac terms reproduces the Hartree-Fock
        # energies to 1.8% with no free constant. Neutrality is assumed by that expansion and not
        # by the virial theorem, so it must not be rewritten through the electron count: that would
        # leave Be(2+) 80% low in <T>. Fed the nuclear charge it returns the neutral atom of the
        # same nucleus instead, 1.0% low on Li(+) and 4.7% high on Be(2+).
        charge = atom_charges[~is_pseudoatom]
        all_electron = 0.7687 * charge ** (7 / 3) - 0.5 * charge**2 + 0.2699 * charge ** (5 / 3)
        # pseudoatom: Thomas-Fermi is a semiclassical theory of many electrons and has nothing to
        # say about the handful of valence ones a pseudopotential leaves behind, applied to the
        # pseudo charge it overshoots fourfold on carbon. Those are hydrogenic instead, carrying
        # zeta**2/(2 * n**2) each, with the principal number of the row and Slater screening by
        # the other valence electrons. Exact on hydrogen and 11 to 55% high across the second row,
        # a pseudo-orbital being smoother than the hydrogenic one it is modelled on.
        charge = atom_charges[is_pseudoatom]
        row = np.searchsorted([2, 10, 18, 36, 54, 86], self.config.wfn.atom_numbers[is_pseudoatom]) + 1
        pseudo = charge * (0.65 * charge + 0.35) ** 2 / (2 * row**2)
        return np.append(all_electron, pseudo)

    @property
    def approximate_step_size(self):
        """Approximation to VMC step size. A uniform displacement by ts gives
        Var(ln(psi'**2/psi**2)) = 8/3 * ts**2 * <T>, and detailed balance pins the mean of that log
        ratio to minus half its variance, so a gaussian one is accepted with probability
        2 * Phi(-sigma/2) and the 50% point sits at ts * sqrt(<T>) = sqrt(3) * erfinv(1/2).
        """
        atom_kinetic_energy = self.atom_kinetic_energy
        if atom_kinetic_energy.size == 0:
            # no nuclei to take a scale from
            return 1.0
        # the gaussian assumption is the remaining error, and it falls off with the number of
        # nuclei sharing <T>, since |grad ln psi| ~ Z at a nucleus puts the tails in the cusps and
        # leaves outer shells contributing almost nothing. The participation ratio is 1 for a
        # single heavy atom and the number of equivalent nuclei for a symmetric molecule, and never
        # divides by zero for a hydrogen-only system. 0.9% rms over the seventeen systems in
        # examples/time_step/CBCS, atoms, ions, hydrides and hydrocarbons alike.
        nuclei = atom_kinetic_energy.sum() ** 2 / (atom_kinetic_energy**2).sum()
        step_size = np.sqrt(3) * erfinv(1 / 2) * (1 + 0.045 / nuclei) / np.sqrt(atom_kinetic_energy.sum())
        if self.config.input.vmc_method == 1:
            # EBES moves one electron, so its share of the sum rule is <T> over their number. The
            # non-gaussian correction that a three term sum needs is left to optimize_vmc_step.
            return step_size * np.sqrt(self.neu + self.ned)
        elif self.config.input.vmc_method == 2:
            # DBDS moves one spin determinant, which carries half of <T> in a closed shell
            return step_size * np.sqrt(2)
        else:
            return step_size

    def acceptance_ratio(self, steps):
        """Probability of accepting a move at the current step size."""
        position = self.vmc.random_walk(steps, 1)
        moved = np.isfinite(position[:, 0, 0]).mean()
        if self.config.input.vmc_method == 1:
            # EBES marks a step accepted if at least one of the electrons moved, so the
            # per-electron probability is recovered by inverting 1 - (1 - acceptance)**electrons
            return 1 - (1 - moved) ** (1 / (self.neu + self.ned))
        else:
            return moved

    def vmc_step_graph(self, steps=1000000):
        """Acceptance probability and the moments behind it vs step size to plot a graph.
        The acceptance alone measures the sum rule and the gaussian shape assumed for the log
        probability density ratio at once and cannot separate them. The moments of that ratio can:
        the sum rule is exact in leading order and says nothing about shape, while detailed balance
        fixes two of its moments whatever the shape is, so each column tests one statement on its
        own. They come from the same walk, so there is nothing to reconcile between them.
        """
        # the informative part of the curve is where the acceptance moves: a grid in step size
        # spends most of its points in the saturated tails, and where it spends them depends on the
        # system, so nothing is comparable between files. Placing the points equally in acceptance
        # instead, through the inverse of the law being measured, gives every system the same grid
        # in the only variable that law knows about, and the step sizes are still written out in
        # atomic units, so the data stay unnormalized by the law they are measured to establish.
        approximate_step_size = self.approximate_step_size
        self.optimize_vmc_step(steps // 10)
        step_size_50 = self.vmc.step_size
        if self.config.input.vmc_method == 1:
            # EBES moves one electron, so its share of the sum rule is <T> over their number
            electrons = self.neu + self.ned
        else:
            electrons = 1
        # <T> of the very wave function being sampled, so that the file is self-contained: it sets
        # the scale the correction column below is measured against, and is the only thing that
        # makes that column comparable between systems. Both estimators are reported because they
        # have the same mean by parts and wildly different variances, so their agreement is the
        # check that the walk is long enough; the drift form is the reference, the laplacian is KEI.
        position = self.vmc.random_walk(steps, self.decorr_period)
        laplacian_form = self.vmc.observable(self.wfn.kinetic_energy, position)
        drift_form = self.vmc.observable(self.wfn.drift_kinetic_energy, position)
        kinetic_energy = drift_form.mean()
        logger.info(
            f' electrons = {self.neu + self.ned}\n'
            f' approximate step size = {approximate_step_size:.5f}\n'
            f' optimized step size = {step_size_50:.5f}\n'
            f' kinetic energy = {kinetic_energy:.5f} +/- {drift_form.std() / np.sqrt(drift_form.size):.5f}\n'
            f' kinetic energy KEI = {laplacian_form.mean():.5f} +/- {laplacian_form.std() / np.sqrt(laplacian_form.size):.5f}\n'
            f' step_size   target  acc_ratio  correction  sum_rule  gaussian  exp_mean  kurtosis'
        )  # fmt: skip
        for target in np.linspace(0.95, 0.05, 19):
            self.vmc.step_size = step_size_50 * ndtri(1 - target / 2) / ndtri(3 / 4)
            x = self.vmc.log_ratio_walk(steps)
            variance = x.var()
            # the probability a proposal is accepted, averaged over the proposals themselves rather
            # than counted from the coin flips that follow, which has the same mean and less noise
            acceptance = np.minimum(np.exp(x), 1).mean()
            # the step size that produced the acceptance actually measured, against the one the law
            # would set for it. This is the factor approximate_step_size carries as 1 + 0.045 /
            # nuclei, in the same units, so the file gives the number to fit rather than an energy
            # to convert, and its value at the 50% target is that constant measured on this system
            law = ndtri(1 - acceptance / 2) * np.sqrt(3 * electrons / (2 * kinetic_energy))
            correction = self.vmc.step_size / law
            # the sum rule alone, with no assumption about the shape of the distribution: it is
            # one wherever Var(x) = 8/3 * step_size**2 * <T> holds, and its departure at large
            # step size is the O(step_size**2) term the leading order leaves out
            sum_rule = 3 * variance / (8 * self.vmc.step_size**2 * kinetic_energy)
            # gaussianity, independent of the acceptance: <exp(x)> = 1 exactly for a stationary
            # walk with a symmetric proposal, and it forces <x> = -Var(x)/2 only if x is gaussian
            gaussian = x.mean() + variance / 2
            logger.info(
                '%10.5f %8.2f %10.5f %11.5f %9.5f %9.5f %9.5f %9.3f',
                self.vmc.step_size,
                target,
                acceptance,
                correction,
                sum_rule,
                gaussian,
                np.exp(x).mean(),
                ((x - x.mean()) ** 4).mean() / variance**2 - 3,
            )

    def optimize_vmc_step(self, steps):
        """Optimize vmc step size to 50% acceptance.
        A measurement at one step size already fixes the whole curve: sigma is proportional to the
        step size by the sum rule, so inverting the gaussian acceptance law A = 2 * Phi(-sigma/2)
        at the measured A and reading it back at 1/2 lands on the target in a single shot whenever
        that law is exact. Its fixed point is A = 1/2 for any monotone acceptance curve, the law
        setting only the rate of convergence, so it cannot converge to the wrong answer where the
        gaussian fails. Measured on examples/time_step/CBCS the map takes a 40% error to 2%, then
        to 0.2%, then to 0.02%: three iterations cost a third of the eleven point scan they replace
        and are limited by the noise of the acceptance rather than by the model.
        """
        logger.info(' Performing time-step optimization.')
        for _ in range(3):
            # a rank that accepts everything or nothing carries no scale of its own
            acceptance = np.clip(mpi_comm.allreduce(self.acceptance_ratio(steps)) / mpi_comm.size, 0.05, 0.95)
            self.vmc.step_size *= ndtri(3 / 4) / ndtri(1 - acceptance / 2)

    @property
    def decorr_period(self):
        """Decorr period"""
        if self.config.input.vmc_decorr_period == 0:
            return self.auto_decorr_period
        else:
            return self.config.input.vmc_decorr_period

    def optimize_decorr_period(self, correlation, time_move, time_energy):
        """Optimize decorr period to maximize the efficiency of the run, i.e. to minimize the
        product of the residual correlation time and the wall time spent per stored configuration.
        :param correlation: correlation time of the series already thinned by the current decorr period
        :param time_move: wall time of one configuration move
        :param time_energy: wall time of one stored configuration apart from the moves
        """
        if np.isfinite(correlation) and correlation > 1:
            # a Metropolis walk decorrelates exponentially, so thinning by d turns rho into rho**d.
            # Inverting that recovers the correlation of the unthinned walk from the production
            # block itself, which is orders of magnitude longer than any dedicated calibration run.
            rho = ((correlation - 1) / (correlation + 1)) ** (1 / self.decorr_period)
            period = np.arange(1, 101)
            thinned = (1 + rho**period) / (1 - rho**period)
            self.auto_decorr_period = int(period[np.argmin(thinned * (period * time_move + time_energy))])
        else:
            self.auto_decorr_period = 1
        logger.info(
            f' Optimized vmc_decorr_period: {self.auto_decorr_period}\n'
        )  # fmt: skip

    def run(self):
        """Run Casino workflow."""
        start = default_timer()
        if self.config.input.testrun:
            logger.info(' TEST RUN only.\n' ' Quitting.\n')
        elif self.config.input.runtype == 'vmc':
            logger.info(
                ' ====================================\n'
                ' PERFORMING A SINGLE VMC CALCULATION.\n'
                ' ====================================\n\n'
            )  # fmt: skip
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
                )  # fmt: skip
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
            )  # fmt: skip
            position = self.vmc_energy_accumulation()
            r_e_list = position[-self.config.input.vmc_nconfig_write // mpi_comm.size :]
            expand(r_e_list)
            self.dmc = DMC(
                r_e_list,
                self.config.input.alimit,
                self.config.input.nucleus_gf_mods,
                self.config.input.use_tmove,
                self.config.input.dtdmc,
                self.config.input.dmc_target_weight,
                self.wfn,
                self.config.input.dmc_method,
            )
            self.dmc_energy_equilibration()
            self.dmc_energy_accumulation()

        stop = default_timer()
        logger.info(
            f' =========================================================================\n\n'
            f' Total PyCasino real time : : :    {stop - start:.4f}'
        )  # fmt: skip

    def equilibrate(self, steps):
        """Burn-in.
        :param steps: burn-in period
        :return:
        """
        self.vmc.random_walk(steps, self.decorr_period)
        logger.info(
            f' Running VMC equilibration ({steps} moves).'
        )  # fmt: skip

    def vmc_energy_accumulation(self):
        """VMC energy accumulation"""
        logger.info(
            ' BEGIN VMC CALCULATION\n'
            ' =====================\n'
        )  # fmt: skip
        self.equilibrate(self.config.input.vmc_equil_nstep)

        if self.config.input.opt_dtvmc == 0:
            self.vmc.step_size = np.sqrt(3 * self.config.input.dtvmc)
        elif self.config.input.opt_dtvmc == 1:
            # to achieve an acceptance ratio of (roughly) 50% (EBES default). Three iterations of
            # 3000 steps cost less than the ten of 1000 they replace, and the noise of the last
            # acceptance is what is left over, 2.33 of it reaching the step size
            self.optimize_vmc_step(3000)
        elif self.config.input.opt_dtvmc == 2:
            # to maximize the diffusion constant with respect to dtvmc (CBCS default).
            raise NotImplementedError

        logger.info(
            f' Optimized step size: {self.vmc.step_size:.5f}\n'
            f' DTVMC: {(self.vmc.step_size**2)/3:.5f}\n'
        )  # fmt: skip

        nblock = self.config.input.vmc_nblock
        steps = self.config.input.vmc_nstep // nblock // mpi_comm.size * nblock * mpi_comm.size
        nblock_steps = steps // nblock // mpi_comm.size

        logger.info(
            ' Starting VMC.\n'
        )  # fmt: skip
        energy_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create energy numpy array whose data points to the shared buffer
        buffer, _ = energy_buffer.Shared_query(rank=0)
        energy = np.ndarray(buffer=buffer, shape=(mpi_comm.size, nblock, nblock_steps))

        for i in range(nblock):
            block_start = default_timer()
            position = self.vmc.random_walk(nblock_steps, self.decorr_period)
            walk_stop = default_timer()
            energy[mpi_comm.rank, i] = self.vmc.observable(self.wfn.energy, position)
            # timings of the last block only, so that JIT compilation of the first one is left out
            time_move = (walk_stop - block_start) / (nblock_steps * self.decorr_period)
            time_energy = (default_timer() - walk_stop) / nblock_steps
            # wait until all processes have written to the array
            mpi_comm.Barrier()
            if self.root:
                energy_block_mean = np.mean(energy[:, i, :])
                energy_block_var = np.var(energy[:, i, :])
                energy_block_sem = np.std(energy[:, i, :]) / np.sqrt(mpi_comm.size * nblock_steps - 1)
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
            energy = energy.reshape(mpi_comm.size, nblock * nblock_steps)
            energy_mean = energy.mean()
            energy_std = energy.std() / np.sqrt(steps - 1)
            energy_cor = 0
            for i in range(mpi_comm.size):
                energy_cor += (2 * np.sum(pacf(energy[i], method='burg')) - 1)
            energy_cor *= energy_std / mpi_comm.size
            energy_sem = np.mean(correlated_sem(energy.reshape(mpi_comm.size, nblock * nblock_steps))) / np.sqrt(mpi_comm.size)
            logger.info(
                f' =========================================================================\n'
                f' FINAL RESULT:\n\n'
                f'  VMC energy (au)    Standard error      Correction for serial correlation\n\n'
                f' {energy_mean:.12f} +/- {energy_std:.12f}      No correction\n'
                f' {energy_mean:.12f} +/- {energy_cor:.12f}      Correlation time method\n'
                f' {energy_mean:.12f} +/- {energy_sem:.12f}      On-the-fly reblocking method\n\n'
                f' Sample variance of E_L (au^2/sim.cell) : {energy.var():.12f}\n\n'
            )
            if self.config.input.vmc_decorr_period == 0:
                self.optimize_decorr_period((energy_sem / energy_std) ** 2, time_move, time_energy)
        self.auto_decorr_period = mpi_comm.bcast(self.auto_decorr_period)
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
            f' EBEST = {self.dmc.best_estimate_energy} (au/prim cell inc. N-N)\n'
            f' EREF  = {self.dmc.energy_t}\n\n'
        )

        steps = self.config.input.dmc_equil_nstep
        nblock = self.config.input.dmc_equil_nblock

        for i in range(nblock):
            block_start = default_timer()
            energy = self.dmc.random_walk(steps // nblock)
            block_stop = default_timer()
            logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n\n'
                f' Number of moves in block                 : {steps // nblock}\n'
                f' Load-balancing efficiency (%)            : {100 * np.mean(self.dmc.efficiency_list):.3f}\n'
                f' Acceptance ratio (%)                     : {100 * self.dmc.step_eff / self.dmc.step_size:.3f}\n'
                f' Number of config transfers               : {self.dmc.ntransfers_tot}\n'
                f' New best estimate of DMC energy (au)     : {energy.mean():.8f}\n'
                f' New best estimate of effective time step : {self.dmc.step_eff:.8f}\n\n'
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
            f' EBEST = {self.dmc.best_estimate_energy} (au/prim cell inc. N-N)\n'
            f' EREF  = {self.dmc.energy_t}\n\n'
            f' Number of previous DMC stats accumulation moves : 0\n'
        )

        steps = self.config.input.dmc_stats_nstep
        nblock = self.config.input.dmc_stats_nblock
        block_steps = steps // nblock
        energy = np.zeros(shape=(steps,))

        for i in range(nblock):
            block_start = default_timer()
            energy[block_steps * i : block_steps * (i + 1)] = self.dmc.random_walk(block_steps)
            energy_mean = energy[: block_steps * (i + 1)].mean()
            block_stop = default_timer()
            logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n\n'
                f' Number of moves in block                 : {block_steps}\n'
                f' Load-balancing efficiency (%)            : {100 * np.mean(self.dmc.efficiency_list):.3f}\n'
                f' Acceptance ratio (%)                     : {100 * self.dmc.step_eff / self.dmc.step_size:.3f}\n'
                f' Number of config transfers               : {self.dmc.ntransfers_tot}\n'
                f' New best estimate of DMC energy (au)     : {energy_mean:.8f}\n'
                f' New best estimate of effective time step : {self.dmc.step_eff:.8f}\n\n'
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
        plt.hist(energy, bins='auto', range=(energy.mean() - 5 * energy.std(), energy.mean() + 5 * energy.std()), density=True)
        plt.savefig('hist.png')
        plt.clf()

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
        position = self.vmc.random_walk(steps // mpi_comm.size, self.decorr_period)
        logger.info(
            ' Optimization start\n'
            ' =================='
        )  # fmt: skip
        energy_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create energy numpy array whose data points to the shared buffer
        buffer, _ = energy_buffer.Shared_query(rank=0)
        energy = np.ndarray(buffer=buffer, shape=(steps,))
        energy_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create energy_gradient numpy array whose data points to the shared buffer
        buffer, _ = energy_gradient_buffer.Shared_query(rank=0)
        energy_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))

        def fun(x, *args, **kwargs):
            self.wfn.set_parameters(x)
            energy[start:stop] = self.vmc.observable(self.wfn.energy, position)
            mpi_comm.Barrier()
            return scale * (energy - energy.mean())

        def jac(x, *args, **kwargs):
            self.wfn.set_parameters(x)
            self.wfn.set_parameters_projector()
            if self.config.input.opt_fixnl:
                energy_gradient[start:stop] = self.vmc.observable(self.wfn.kinetic_energy_parameters_d1, position)
            else:
                energy_gradient[start:stop] = self.vmc.observable(self.wfn.energy_parameters_d1, position)
            mpi_comm.Barrier()
            return scale * (energy_gradient - energy_gradient.mean(axis=0))

        def trigger_fun(x, *args, **kwargs):
            mpi_comm.bcast(('fun', x, args, kwargs))
            return fun(x, *args, **kwargs)

        def trigger_jac(x, *args, **kwargs):
            mpi_comm.bcast(('jac', x, args, kwargs))
            return jac(x, *args, **kwargs)

        if self.root:
            res = least_squares(
                trigger_fun, x0=x0, jac=trigger_jac, method='trf', ftol=2/np.sqrt(steps-1), x_scale='jac',
                tr_solver='exact', max_nfev=self.config.input.opt_maxeval, verbose=self.root and verbose
            )
            mpi_comm.bcast(('break', 0, 0, 0))
            parameters = res.x
            norm = np.linalg.norm(res.jac.mean(axis=0))
        else:
            while True:
                command, x, args, kwargs = mpi_comm.bcast(None)
                if command == 'fun':
                    fun(x, *args, **kwargs)
                if command == 'jac':
                    jac(x, *args, **kwargs)
                if command == 'break':
                    break
            parameters = np.empty_like(x0)
            norm = 0

        energy_buffer.Free()
        energy_gradient_buffer.Free()
        mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters)
        logger.info(f' Norm of Jacobian at the solution: {norm:.5e}\n')

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
        position = self.vmc.random_walk(steps // mpi_comm.size, self.decorr_period)
        logger.info(
            ' Optimization start\n'
            ' =================='
        )  # fmt: skip
        wfn_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create wfn numpy array whose data points to the shared buffer
        buffer, _ = wfn_buffer.Shared_query(rank=0)
        wfn = np.ndarray(buffer=buffer, shape=(steps,))
        wfn_0_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create wfn_0 numpy array whose data points to the shared buffer
        buffer, _ = wfn_0_buffer.Shared_query(rank=0)
        wfn_0 = np.ndarray(buffer=buffer, shape=(steps,))
        energy_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create energy numpy array whose data points to the shared buffer
        buffer, _ = energy_buffer.Shared_query(rank=0)
        energy = np.ndarray(buffer=buffer, shape=(steps,))
        wfn_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create wfn_gradient numpy array whose data points to the shared buffer
        buffer, _ = wfn_gradient_buffer.Shared_query(rank=0)
        wfn_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))
        energy_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create energy_gradient numpy array whose data points to the shared buffer
        buffer, _ = energy_gradient_buffer.Shared_query(rank=0)
        energy_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))
        wfn_0[start:stop] = self.vmc.observable(self.wfn.value, position)
        mpi_comm.Barrier()

        def fun(x, *args, **kwargs):
            self.wfn.set_parameters(x)
            wfn[start:stop] = self.vmc.observable(self.wfn.value, position)
            energy[start:stop] = self.vmc.observable(self.wfn.energy, position)
            mpi_comm.Barrier()
            weights = (wfn / wfn_0) ** 2
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
            # wfn[start:stop] = self.vmc.observable(self.wfn.value, position)
            # energy[start:stop] = self.vmc.observable(self.wfn.energy, position)
            wfn_gradient[start:stop] = self.vmc.observable(self.wfn.value_parameters_d1, position)
            if self.config.input.opt_fixnl:
                energy_gradient[start:stop] = self.vmc.observable(self.wfn.kinetic_energy_parameters_d1, position)
            else:
                energy_gradient[start:stop] = self.vmc.observable(self.wfn.energy_parameters_d1, position)
            mpi_comm.Barrier()
            weights = (wfn / wfn_0) ** 2
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

        def trigger_fun(x, *args, **kwargs):
            mpi_comm.bcast(('fun', x, args, kwargs))
            return fun(x, *args, **kwargs)

        def trigger_jac(x, *args, **kwargs):
            mpi_comm.bcast(('jac', x, args, kwargs))
            return jac(x, *args, **kwargs)

        if self.root:
            res = least_squares(
                trigger_fun, x0=x0, jac=trigger_jac, method='trf', ftol=2/np.sqrt(steps-1),
                tr_solver='exact', max_nfev=self.config.input.opt_maxeval, verbose=self.root and verbose
            )
            mpi_comm.bcast(('break', 0, 0, 0))
            parameters = res.x
            norm = np.linalg.norm(res.jac.mean(axis=0))
        else:
            while True:
                command, x, args, kwargs = mpi_comm.bcast(None)
                if command == 'fun':
                    fun(x, *args, **kwargs)
                if command == 'jac':
                    jac(x, *args, **kwargs)
                if command == 'break':
                    break
            parameters = np.empty_like(x0)
            norm = 0

        wfn_buffer.Free()
        wfn_0_buffer.Free()
        energy_buffer.Free()
        wfn_gradient_buffer.Free()
        energy_gradient_buffer.Free()
        mpi_comm.Bcast(parameters)
        self.wfn.set_parameters(parameters)
        logger.info(f' Norm of Jacobian at the solution: {norm:.5e}\n')

    @staticmethod
    def energy_parameters_gradient(data):
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

    @staticmethod
    def energy_parameters_hessian(data):
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
        position = self.vmc.random_walk(steps // mpi_comm.size, self.decorr_period)
        logger.info(
            ' Optimization start\n'
            ' =================='
        )  # fmt: skip
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
            data['energy'] = self.vmc.observable(self.wfn.energy, position)
            data['energy_mean'] = data['energy'].mean()
            return mpi_comm.allreduce(data['energy_mean']) / mpi_comm.size

        def jac(x, *args):
            """Only for CG, BFGS, L-BFGS-B, TNC, SLSQP and those listed in hess method."""
            self.wfn.set_parameters(x * scale)
            self.wfn.set_parameters_projector()
            data['energy'] = self.vmc.observable(self.wfn.energy, position)
            data['energy_mean'] = data['energy'].mean()
            data['wfn_gradient'] = self.vmc.observable(self.wfn.value_parameters_d1, position)
            data['wfn_gradient_mean'] = np.mean(data['wfn_gradient'], axis=0)
            return self.energy_parameters_gradient(data) * scale

        def hess(x, *args):
            """Only for Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr."""
            self.wfn.set_parameters(x * scale)
            self.wfn.set_parameters_projector()
            data['energy'] = self.vmc.observable(self.wfn.energy, position)
            data['energy_mean'] = data['energy'].mean()
            data['wfn_gradient'] = self.vmc.observable(self.wfn.value_parameters_d1, position)
            data['wfn_gradient_mean'] = np.mean(data['wfn_gradient'], axis=0)
            data['wfn_hessian'] = self.vmc.observable(self.wfn.value_parameters_d2, position)
            data['energy_gradient'] = self.vmc.observable(self.wfn.energy_parameters_d1, position)
            return self.energy_parameters_hessian(data) * np.outer(scale, scale)

        callback.nfev = 0
        callback.iteration = 0
        logger.info(f'Optimization method: {method}')
        logger.info('   Iteration     Total nfev        Energy             Grad norm')
        if method == 'TNC':
            options = dict(disp=self.root, scale=np.ones(shape=(x0.size,)), offset=np.zeros(shape=(x0.size,)), stepmx=1)
        elif method in ('dogleg', 'trust-ncg', 'trust-exact'):
            # default 1:1000:0.15:1e-4
            options = dict(initial_trust_radius=0.1, max_trust_radius=1, eta=0.15, gtol=1e-3)
        else:
            options = dict(disp=self.root)
        # Desired error not necessarily achieved due to precision loss.
        # https://github.com/scipy/scipy/issues/15643
        res = minimize(fun, x0=x0 / scale, method=method, jac=jac, hess=hess, callback=callback, options=options)
        logger.info(f' Norm of Jacobian at the solution: {np.linalg.norm(res.jac):.5e}\n')
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
        The method is stabilized by 'level-shifting', i.e. by adding a positive constant L to the diagonal of H except
        for its first element. As L grows, Δp shrinks and rotates from the Newtonian direction to the steepest descent
        one. Once the shift dominates, Δp falls off as 1/L, so the smallest L keeping every parameter variation within
        the trust radius is found by bisection, which only solves the eigenvalue problem and costs nothing, unlike a
        correlated sampling pass. The four shifts starting from that one are then
        compared by the target function energy + 3 * error, estimated by correlated sampling on the very same set of
        configurations, so that no extra random walk is needed. Δp = 0 always takes part in the comparison, hence a
        cycle can never make the target function worse. Eigenvalues below emin_min_energy are discarded, as poor
        candidate wave functions produce spurious low energies.
        :param steps: number of configs
        """
        steps = steps // mpi_comm.size * mpi_comm.size
        start, stop = mpi_comm.rank * steps // mpi_comm.size, (mpi_comm.rank + 1) * steps // mpi_comm.size
        x0 = self.wfn.get_parameters()
        parameters_scale = self.wfn.get_parameters_scale()
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
        position = self.vmc.random_walk(steps // mpi_comm.size, self.decorr_period)
        logger.info(
            ' Optimization start\n'
            ' =================='
        )  # fmt: skip
        energy_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create energy numpy array whose data points to the shared buffer
        buffer, _ = energy_buffer.Shared_query(rank=0)
        energy = np.ndarray(buffer=buffer, shape=(steps,))
        wfn_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create wfn numpy array whose data points to the shared buffer
        buffer, _ = wfn_buffer.Shared_query(rank=0)
        wfn = np.ndarray(buffer=buffer, shape=(steps,))
        wfn_0_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create wfn_0 numpy array whose data points to the shared buffer
        buffer, _ = wfn_0_buffer.Shared_query(rank=0)
        wfn_0 = np.ndarray(buffer=buffer, shape=(steps,))
        wfn_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create wfn_gradient numpy array whose data points to the shared buffer
        buffer, _ = wfn_gradient_buffer.Shared_query(rank=0)
        wfn_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))
        energy_gradient_buffer = MPI.Win.Allocate_shared(steps * x0.size * double_size if self.root else 0, comm=mpi_comm)
        # create energy_gradient numpy array whose data points to the shared buffer
        buffer, _ = energy_gradient_buffer.Shared_query(rank=0)
        energy_gradient = np.ndarray(buffer=buffer, shape=(steps, x0.size))
        # wfn_0 is the wave function the configurations are distributed with, so it is sampled before anything else
        wfn_0[start:stop] = self.vmc.observable(self.wfn.value, position)
        energy[start:stop] = self.vmc.observable(self.wfn.energy, position)
        self.wfn.set_parameters_projector()
        wfn_gradient[start:stop] = self.vmc.observable(self.wfn.value_parameters_d1, position)
        if self.config.input.opt_fixnl:
            energy_gradient[start:stop] = self.vmc.observable(self.wfn.kinetic_energy_parameters_d1, position)
        else:
            energy_gradient[start:stop] = self.vmc.observable(self.wfn.energy_parameters_d1, position)
        mpi_comm.Barrier()
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        # thinning by decorr_period does not decorrelate the configurations completely, so the naive error is
        # optimistic. Measured on the very sample the target function is estimated on, which needs no reference
        # to the preceding VMC block and stays valid for the first optimization cycle.
        correlation = correlated_sem(energy) * np.sqrt(steps - 1) / energy_std
        if self.config.input.emin_min_energy is None:
            # local energy distributions of two wave functions for the same system usually overlap significantly
            min_energy = energy_mean - 4 * energy_std
        else:
            min_energy = self.config.input.emin_min_energy
        var_prefactor = self.config.input.emin_var_prefactor
        xi = self.config.input.emin_xi_value
        # Trust radius: largest absolute parameter variation allowed. It only places the scan window, whose
        # other end is a step small enough to change nothing. Measured on He, Be and N, a variation above 0.3
        # already costs several mHa, so scanning from 0.2 down wastes no correlated sampling pass.
        dp_max = 0.2

        def penalty(variance, weights):
            """Penalty added to the mean energy to keep the target function universal across systems.
            By default it is 3 times the error of the correlated sampling estimate, whose effective sample size
            (sum(w))**2 / sum(w**2) punishes candidates the configurations are no longer representative of.
            Serial correlation is independent of the weights, so the two corrections multiply.
            """
            if var_prefactor > 0:
                return var_prefactor * np.sqrt(variance)
            else:
                return 3 * correlation * np.sqrt(variance * (weights**2).sum()) / weights.sum()

        def target(dp):
            """Correlated sampling estimate of the target function for parameters x0 + dp.
            Reuses the energy buffer, so it must not be called before H is built.
            """
            self.wfn.set_parameters(x0 + dp)
            wfn[start:stop] = self.vmc.observable(self.wfn.value, position)
            energy[start:stop] = self.vmc.observable(self.wfn.energy, position)
            mpi_comm.Barrier()
            weights = (wfn / wfn_0) ** 2
            # a candidate far enough from the current wave function makes exp(J) over- or underflow
            if not np.isfinite(weights).all() or not weights.sum() > 0:
                return np.nan, np.nan
            mean = np.average(energy, weights=weights)
            variance = np.average((energy - mean) ** 2, weights=weights)
            return mean, penalty(variance, weights)

        def trigger_target(dp):
            mpi_comm.bcast(('target', dp))
            return target(dp)

        if self.root:
            try:
                wfn_gradient -= np.mean(wfn_gradient, axis=0)
                energy_gradient -= np.mean(energy_gradient, axis=0)
                # parameters the wave function does not depend on for this sample would make S singular
                active = np.std(wfn_gradient, axis=0) > 0
                # rescale parameters so that S becomes the Pearson correlation matrix
                scale = 1 / np.std(wfn_gradient[:, active], axis=0)
                S = overlap_matrix(wfn_gradient[:, active] * scale)
                H = hamiltonian_matrix(wfn_gradient[:, active] * scale, energy, energy_gradient[:, active] * scale)
                # logger.info(f'epsilon:\n{np.diag(H[1:, 1:]) / np.diag(S[1:, 1:]) - H[0, 0]}')
                level_shift = np.eye(S.shape[0])
                level_shift[0, 0] = 0

                def solve(shift):
                    """Δp of the generalized eigenvalue problem stabilized by a level shift."""
                    eigvals, eigvectors = sp.linalg.eig(H + shift * level_shift, S)
                    # since imaginary parts only arise from statistical noise, discard them
                    eigvals, eigvectors = np.real(eigvals), np.real(eigvectors)
                    allowed = np.flatnonzero(eigvals > min_energy)
                    if not allowed.size:
                        return np.nan, np.full_like(x0, np.inf)
                    idx = allowed[np.abs(eigvectors[0, allowed]).argmax()]
                    eigval, eigvector = eigvals[idx], eigvectors[:, idx]
                    # from "Implementation of the Linear Method for the optimization of Jastrow-Feenberg
                    # and Backflow Correlations" M. Motta, G. Bertaina, D. E. Galli, E. Vitali using (24)
                    # and eigvector is normalized solutions of H · Δp = E(p) * S · Δp
                    # and (1, Δp_i) = eigvector/eigvector[0] is properly rescaled Δp_i
                    # in case ξ = 0; Δp_i = eigvector[1:] * eigvector[0]
                    # in case ξ = 1; Δp_i = eigvector[1:] / eigvector[0]
                    dp = eigvector[1:] / eigvector[0]
                    Q = dp @ S[1:, 1:] @ dp
                    dp /= 1 + (1 - xi) * Q / (1 - xi + xi * np.sqrt(1 + Q))
                    res = np.zeros_like(x0)
                    res[active] = dp * scale
                    if not np.isfinite(res).all():
                        return np.nan, np.full_like(x0, np.inf)
                    return eigval, res

                if not x0.all():
                    self.wfn.set_parameters(x0)
                logger.info(
                    f' E_0 {energy_mean:.8f} minimal allowed energy {min_energy:.8f}'
                    f' serial correlation factor {correlation:.2f}'
                )  # fmt: skip
                # Once the shift dominates, Δp falls off as 1/L, so the smallest shift keeping every parameter
                # variation within the trust radius is found by bisection. The measure is not monotonic where the
                # shift is negligible and the selected root switches, but it is far above dp_max there. Only the
                # eigenvalue problem is solved here, which costs nothing next to a correlated sampling pass.
                lo, hi = -8.0, 8.0
                if np.max(np.abs(solve(10**lo)[1])) < dp_max:
                    # a converged wave function stays within the radius however small the shift is
                    hi = lo
                    bound = False
                else:
                    for _ in range(20):
                        mid = (lo + hi) / 2
                        if np.max(np.abs(solve(10**mid)[1])) < dp_max:
                            hi = mid
                        else:
                            lo = mid
                    bound = True
                logger.info(f' Trust radius max |delta p| < {dp_max} reached at level shift {10**hi:.2e}')
                # the L -> inf limit, i.e. Δp = 0, is the candidate every other one is compared with
                best_penalty = penalty(energy_std**2, np.ones(steps))
                best_shift, best_dp, best_target = np.inf, np.zeros_like(x0), energy_mean + best_penalty
                logger.info(f' level shift        inf  E_corr {energy_mean:.8f}  penalty {best_penalty:.8f}  target {best_target:.8f}')
                shifts = 10.0 ** (hi + np.arange(4))
                for shift in shifts:
                    eigval, dp = solve(shift)
                    if np.isnan(eigval):
                        logger.info(f' level shift {shift:.2e}  no eigenvalue above the minimal allowed energy')
                        continue
                    logger.info(
                        f' level shift {shift:.2e}  E_lin {eigval:.8f}'
                        f'  max |delta p| {np.max(np.abs(dp)):.4e}  max |delta p / scale| {np.max(np.abs(dp / parameters_scale)):.4e}'
                    )  # fmt: skip
                    mean, pen = trigger_target(dp)
                    if not np.isfinite(mean) or mean < min_energy:
                        logger.info(f'                       E_corr {mean:.8f}  rejected')
                        continue
                    logger.info(f'                       E_corr {mean:.8f}  penalty {pen:.8f}  target {mean + pen:.8f}')
                    if mean + pen < best_target:
                        best_shift, best_dp, best_target = shift, dp, mean + pen
                if best_shift == shifts[0] and bound:
                    logger.info(' Best level shift is the smallest scanned one, the trust radius may be too large')
                elif best_shift == shifts[-1]:
                    logger.info(' Best level shift is the largest scanned one, the optimum may lie above the scan')
            finally:
                # the other ranks wait in bcast, so they must be released even if anything above raises
                mpi_comm.bcast(('break', None))
            logger.info(f' Chosen level shift {best_shift:.2e}, target {best_target:.8f}')
            dp = best_dp
        else:
            while True:
                command, value = mpi_comm.bcast(None)
                if command == 'target':
                    target(value)
                if command == 'break':
                    break
            dp = np.zeros_like(x0)

        mpi_comm.Bcast(dp)
        if x0.all():
            logger.info(f' delta p / p\n{dp / x0}\n')
        else:
            logger.info(f' delta p\n{dp}\n')
        self.wfn.set_parameters(x0 + dp)
        energy_buffer.Free()
        wfn_buffer.Free()
        wfn_0_buffer.Free()
        wfn_gradient_buffer.Free()
        energy_gradient_buffer.Free()

    def vmc_energy_minimization_stochastic_reconfiguration(self, steps):
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
        position = self.vmc.random_walk(steps // mpi_comm.size, self.decorr_period)
        logger.info(
            ' Optimization start\n'
            ' =================='
        )  # fmt: skip
        energy_buffer = MPI.Win.Allocate_shared(steps * double_size if self.root else 0, comm=mpi_comm)
        # create energy numpy array whose data points to the shared buffer
        buffer, _ = energy_buffer.Shared_query(rank=0)
        energy = np.ndarray(buffer=buffer, shape=(steps,))
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
            energy[start:stop] = self.vmc.observable(self.wfn.energy, position)
            mpi_comm.Barrier()
            logger.info(f'energy: {energy.mean()}')
            return energy.mean()

        def jac(x, *args):
            self.wfn.set_parameters(x)
            self.wfn.set_parameters_projector()
            energy[start:stop] = self.vmc.observable(self.wfn.energy, position)
            wfn_gradient[start:stop] = self.vmc.observable(self.wfn.value_parameters_d1, position)
            mpi_comm.Barrier()
            if self.root:
                energy[:] -= np.mean(energy)
                wfn_gradient[:, :] -= np.mean(wfn_gradient, axis=0)
            mpi_comm.Barrier()
            return 2 * wfn_gradient.T @ energy / steps

        def hess(x, *args):
            self.wfn.set_parameters(x)
            self.wfn.set_parameters_projector()
            energy[start:stop] = self.vmc.observable(self.wfn.energy, position)
            wfn_gradient[start:stop] = self.vmc.observable(self.wfn.value_parameters_d1, position)
            if self.config.input.opt_fixnl:
                energy_gradient[start:stop] = self.vmc.observable(self.wfn.kinetic_energy_parameters_d1, position)
            else:
                energy_gradient[start:stop] = self.vmc.observable(self.wfn.energy_parameters_d1, position)
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
        configure_logging()
        Casino(args.config_path).run()
    else:
        print(f'File {args.config_path}input not found...')
        sys.exit(1)
