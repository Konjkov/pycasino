#!/usr/bin/env python3

import os
from timeit import default_timer
from cusp import CuspFactory, TestCuspFactory
from slater import Slater
from jastrow import Jastrow
from backflow import Backflow
from wfn import Wfn

os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import numpy as np
import numba as nb
from scipy.optimize import least_squares, minimize, root, Bounds
# import scipy as sp

from readers.casino import CasinoConfig
from sem import correlated_sem
from logger import logging

np.random.seed(31415926)

logger = logging.getLogger('vmc')

r_e_type = nb.types.Array(dtype=nb.float64, ndim=2, layout="C")

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('r_e', nb.types.ListType(r_e_type)),
    ('step', nb.float64),
    ('atom_positions', nb.float64[:, :]),
    ('atom_charges', nb.float64[:]),
    ('wfn', Wfn.class_type.instance_type),
]


@nb.jit(nopython=True, nogil=True, parallel=False)
def sum_typed_list(x):
    """Mixed estimator of energy
    Для проверки утечек пямяти
    """
    sum_e = 0.0
    for e in x:
        sum_e += e
    return sum_e


@nb.jit(forceobj=True)
def initial_position(ne, atom_positions, atom_charges):
    """Initial positions of electrons."""
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
    return r_e + np.random.laplace(0, 1, ne * 3).reshape((ne, 3))


@nb.experimental.jitclass(spec)
class MarkovChain:

    def __init__(self, neu, ned, step, atom_positions, atom_charges, wfn):
        """Markov chain Monte Carlo.
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param step: time step
        :param atom_positions: atomic positions
        :param atom_charges: atomic charges
        :param wfn: instance of Wfn class
        :return:
        """
        self.neu = neu
        self.ned = ned
        self.step = step
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.wfn = wfn

    def limiting_factor(self, v, a=1):
        """A significant source of error in DMC calculations comes from sampling electronic
        configurations near the nodal surface. Here both the drift velocity and local
        energy diverge, causing large time step errors and increasing the variance of
        energy estimates respectively. To reduce these undesirable effects it is necessary
        to limit the magnitude of both quantities.
        :param v: drift velocity
        :param a: strength of the limiting
        :return:
        """
        square_mod_v = np.sum(v**2)
        return (np.sqrt(1 + 2 * a * square_mod_v * self.step) - 1) / (a * square_mod_v * self.step)

    def simple_random_walker(self, r_e, decorr_period):
        """Simple random walker with random N-dim square proposal density in
        configuration-by-configuration sampling (CBCS).
        :param steps: number of steps to walk
        :param decorr_period: decorrelation period
        :return: is step accepted, next electron`s position
        """
        # FIXME: yield statement yields leaky memory
        #  https://github.com/numba/numba/issues/6993
        ne = self.neu + self.ned
        e_vectors, n_vectors = self.wfn.relative_coordinates(r_e)
        probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
        while True:
            cond = False
            for _ in range(decorr_period):
                next_state = r_e + self.step * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
                e_vectors, n_vectors = self.wfn.relative_coordinates(next_state)
                next_probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
                partial_cond = next_probability_density / probability_density > np.random.random()
                if partial_cond:
                    r_e, probability_density = next_state, next_probability_density
                    cond = True
            yield cond, r_e

    def gibbs_random_walker(self, r_e, decorr_period):
        """Simple random walker with electron-by-electron sampling (EBES)
        :param r_e: initial position
        :param decorr_period: decorrelation period
        :return: is step accepted, next step position
        """
        ne = self.neu + self.ned
        e_vectors, n_vectors = self.wfn.relative_coordinates(r_e)
        probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
        while True:
            cond = False
            for _ in range(decorr_period):
                next_r_e = np.copy(r_e)
                next_r_e[np.random.randint(ne)] += self.step * np.random.uniform(-1, 1, 3)
                e_vectors, n_vectors = self.wfn.relative_coordinates(next_r_e)
                next_probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
                partial_cond = next_probability_density / probability_density > np.random.random()
                if partial_cond:
                    r_e, p = next_r_e, next_probability_density
                    cond = True
            yield cond, r_e

    def biased_random_walker(self, r_e, decorr_period):
        """Biased random walker with diffusion-drift proposed step
        diffusion step s proportional to sqrt(2*D*dt)
        drift step is proportional to D*F*dt
        where D is diffusion constant = 1/2
        :param r_e: initial position
        :param decorr_period: decorrelation period
        :return: is step accepted, next step position
        """
        ne = self.neu + self.ned
        e_vectors, n_vectors = self.wfn.relative_coordinates(r_e)
        probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
        while True:
            cond = False
            for _ in range(decorr_period):
                v_forth = self.drift_velocity(r_e)
                move = np.sqrt(self.step) * np.random.normal(0, 1, ne * 3) + self.step * v_forth
                next_r_e = r_e + move.reshape((ne, 3))
                e_vectors, n_vectors = self.wfn.relative_coordinates(next_r_e)
                next_probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
                green_forth = np.exp(-np.sum((next_r_e.ravel() - r_e.ravel() - self.step * v_forth) ** 2) / 2 / self.step)
                green_back = np.exp(-np.sum((r_e.ravel() - next_r_e.ravel() - self.step * self.drift_velocity(next_r_e)) ** 2) / 2 / self.step)
                partial_cond = (green_back * next_probability_density) / (green_forth * probability_density) > np.random.random()
                if partial_cond:
                    r_e, probability_density = next_r_e, next_probability_density
                    cond = True
            yield cond, r_e

    def bbk_random_walker(self, r_e, decorr_period):
        """Brünger–Brooks–Karplus (13 B. Brünger, C. L. Brooks, and M. Karplus, Chem. Phys. Lett. 105, 495 1984).
        :param r_e: initial position
        :param decorr_period: decorrelation period
        :return: is step accepted, next step position
        """
        while True:
            cond = False
            for _ in range(decorr_period):
                pass
            yield cond, r_e

    def force_interpolation_random_walker(self, r_e, decorr_period):
        """M. P. Allen and D. J. Tildesley, Computer Simulation of Liquids Oxford University Press, Oxford, 1989 and references in Sec. 9.3.
        :param r_e: initial position
        :param decorr_period: decorrelation period
        :return: is step accepted, next step position
        """
        while True:
            cond = False
            for _ in range(decorr_period):
                pass
            yield cond, r_e

    def splitting_random_walker(self, r_e, decorr_period):
        """J. A. Izaguirre, D. P. Catarello, J. M. Wozniak, and R. D. Skeel, J. Chem. Phys. 114, 2090 2001.
        :param r_e: initial position
        :param decorr_period: decorrelation period
        :return: is step accepted, next step position
        """
        while True:
            cond = False
            for _ in range(decorr_period):
                pass
            yield cond, r_e

    def ricci_ciccottid_random_walker(self, r_e, decorr_period):
        """A. Ricci and G. Ciccotti, Mol. Phys. 101, 1927 2003.
        :param r_e: initial position
        :param decorr_period: decorrelation period
        :return: is step accepted, next step position
        """
        while True:
            cond = False
            for _ in range(decorr_period):
                pass
            yield cond, r_e

    def dmc_random_walker(self, positions, target_weight):
        """Collection of walkers representing the instantaneous wfn.
        C. J. Umrigar, M. P. Nightingale, K. J. Runge. A diffusion Monte Carlo algorithm with very small time-step errors.
        :param steps: number of steps to walk
        :param positions: initial positions of walkers
        :param target_weight: target weight of walkers
        :return: best estimate of energy, next position
        """
        ne = self.neu + self.ned
        # TODO: change to numpy.array
        r_e_list = nb.typed.List()
        energy_list = nb.typed.List()
        velocity_list = nb.typed.List()
        wfn_value_list = nb.typed.List()
        branching_energy_list = nb.typed.List()
        for r_e in positions:
            e_vectors, n_vectors = self.wfn.relative_coordinates(r_e)
            wfn_value_list.append(self.wfn.value(e_vectors, n_vectors))
            r_e_list.append(r_e)
            energy_list.append(self.wfn.energy(r_e))
            branching_energy_list.append(self.wfn.energy(r_e))
            velocity = self.wfn.drift_velocity(r_e)
            limiting_factor = self.limiting_factor(velocity)
            velocity_list.append(limiting_factor * velocity)
        step_eff = self.step
        best_estimate_energy = sum_typed_list(energy_list) / len(energy_list)
        energy_t = best_estimate_energy - np.log(len(energy_list) / target_weight) / step_eff
        while True:
            sum_acceptance_probability = 0
            next_r_e_list = nb.typed.List()
            next_energy_list = nb.typed.List()
            next_velocity_list = nb.typed.List()
            next_wfn_value_list = nb.typed.List()
            next_branching_energy_list = nb.typed.List()
            for r_e, wfn_value, velocity, energy, branching_energy in zip(r_e_list, wfn_value_list, velocity_list, energy_list, branching_energy_list):
                next_r_e = r_e + (np.sqrt(self.step) * np.random.normal(0, 1, ne * 3) + self.step * velocity).reshape((ne, 3))
                e_vectors, n_vectors = self.wfn.relative_coordinates(next_r_e)
                next_wfn_value = self.wfn.value(e_vectors, n_vectors)
                # prevent crossing nodal surface
                cond = np.sign(wfn_value) == np.sign(next_wfn_value)
                next_velocity = self.wfn.drift_velocity(next_r_e)
                next_energy = self.wfn.energy(next_r_e)
                limiting_factor = self.limiting_factor(next_velocity)
                next_velocity *= limiting_factor
                next_branching_energy = best_estimate_energy - (best_estimate_energy - next_energy) * limiting_factor
                p = 0
                if cond:
                    # Green`s functions
                    green_forth = np.exp(-np.sum((next_r_e.ravel() - r_e.ravel() - self.step * velocity) ** 2) / 2 / self.step)
                    green_back = np.exp(-np.sum((r_e.ravel() - next_r_e.ravel() - self.step * next_velocity) ** 2) / 2 / self.step)
                    # condition
                    p = min(1, (green_back * next_wfn_value ** 2) / (green_forth * wfn_value ** 2))
                    cond = p >= np.random.random()
                # branching
                if cond:
                    weight = np.exp(-self.step * (next_branching_energy + branching_energy - 2 * energy_t) / 2)
                else:
                    weight = np.exp(-self.step * (branching_energy - energy_t))
                for _ in range(int(weight + np.random.uniform(0, 1))):
                    sum_acceptance_probability += p
                    if cond:
                        next_r_e_list.append(next_r_e)
                        next_energy_list.append(next_energy)
                        next_velocity_list.append(next_velocity)
                        next_wfn_value_list.append(next_wfn_value)
                        next_branching_energy_list.append(next_branching_energy)
                    else:
                        next_r_e_list.append(r_e)
                        next_energy_list.append(energy)
                        next_velocity_list.append(velocity)
                        next_wfn_value_list.append(wfn_value)
                        next_branching_energy_list.append(branching_energy)
            r_e_list = next_r_e_list
            energy_list = next_energy_list
            velocity_list = next_velocity_list
            wfn_value_list = next_wfn_value_list
            branching_energy_list = next_branching_energy_list
            step_eff = sum_acceptance_probability / len(energy_list) * self.step
            best_estimate_energy = sum_typed_list(energy_list) / len(energy_list)
            energy_t = best_estimate_energy - np.log(len(energy_list) / target_weight) * self.step / step_eff
            yield best_estimate_energy, r_e_list

    walker = simple_random_walker

    def vmc_random_walk(self, steps, decorr_period):
        """Metropolis-Hastings random walk.
        """
        r_e = self.r_e[0]
        condition = np.zeros((steps, ), nb.boolean)
        position = np.zeros((steps, r_e.shape[0], r_e.shape[1]))
        walker = self.walker(r_e, decorr_period)

        for i in range(steps):
            cond, r_e = next(walker)
            condition[i] = cond
            position[i] = r_e

        self.r_e[0] = r_e
        return condition, position

    def dmc_random_walk(self, steps, target_weight):
        """DMC
        :param steps: number of steps to walk
        :param target_weight: target weight
        :return:
        """
        r_e = self.r_e
        energy = np.zeros(shape=(steps, ))
        walker = self.dmc_random_walker(r_e, target_weight)

        for i in range(steps):
            energy_t, r_e = next(walker)
            energy[i] = energy_t

        self.r_e = r_e
        return energy

    def local_energy(self, condition, position):
        """VMC local energy estimator.
        :param condition: accept/reject condition
        :param position: random walk positions
        :return:
        """
        energy = np.zeros(shape=(condition.size, ))
        for i, (c, p) in enumerate(zip(condition, position)):
            if i == 0 or c:
                energy[i] = self.wfn.energy(p)
            else:
                energy[i] = energy[i-1]
        return energy

    def jastrow_gradient(self, condition, position):
        """Jastrow gradient with respect to jastrow parameters.
        :param position: random walk positions
        :return:
        """
        first_res = self.wfn.jastrow_parameters_numerical_d1(position[0])
        res = np.zeros(shape=(position.shape[0], ) + first_res.shape)
        res[0] = first_res

        for i in range(1, position.shape[0]):
            res[i] = self.wfn.jastrow_parameters_numerical_d1(position[i])
        return res

    def jastrow_hessian(self, condition, position):
        """Jastrow hessian with respect to jastrow parameters.
        :param position: random walk positions
        :return:
        """
        first_res = self.wfn.jastrow_parameters_numerical_d2(position[0])
        res = np.zeros(shape=(position.shape[0], ) + first_res.shape)
        res[0] = first_res

        for i in range(1, position.shape[0]):
            res[i] = self.wfn.jastrow_parameters_numerical_d2(position[i])
        return res


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

    def __init__(self, path):
        """Casino workflow.
        :param path: path to config
        """
        self.config = CasinoConfig(path)

        if self.config.input.cusp_correction:
            cusp = CuspFactory(
                self.config.input.neu, self.config.input.ned, self.config.mdet.mo_up, self.config.mdet.mo_down,
                self.config.wfn.nbasis_functions, self.config.wfn.first_shells, self.config.wfn.shell_moments, self.config.wfn.primitives,
                self.config.wfn.coefficients, self.config.wfn.exponents, self.config.wfn.atom_positions, self.config.wfn.atom_charges
            ).create()
            # cusp = TestCuspFactory(self.config.input.neu, self.config.input.ned).create()
        else:
            cusp = None

        slater = Slater(
            self.config.input.neu, self.config.input.ned,
            self.config.wfn.nbasis_functions, self.config.wfn.first_shells, self.config.wfn.orbital_types, self.config.wfn.shell_moments,
            self.config.wfn.slater_orders, self.config.wfn.primitives, self.config.wfn.coefficients, self.config.wfn.exponents,
            self.config.mdet.mo_up, self.config.mdet.mo_down, self.config.mdet.coeff, cusp
        )
        jastrow = self.config.jastrow and Jastrow(
            self.config.input.neu, self.config.input.ned,
            self.config.jastrow.trunc, self.config.jastrow.u_parameters, self.config.jastrow.u_mask, self.config.jastrow.u_cutoff, self.config.jastrow.u_cusp_const,
            self.config.jastrow.chi_parameters, self.config.jastrow.chi_mask, self.config.jastrow.chi_cutoff, self.config.jastrow.chi_labels,
            self.config.jastrow.f_parameters, self.config.jastrow.f_mask, self.config.jastrow.f_cutoff, self.config.jastrow.f_labels,
            self.config.jastrow.no_dup_u_term, self.config.jastrow.no_dup_chi_term, self.config.jastrow.chi_cusp
        )
        backflow = self.config.backflow and Backflow(
            self.config.input.neu, self.config.input.ned,
            self.config.backflow.trunc, self.config.backflow.eta_parameters, self.config.backflow.eta_cutoff,
            self.config.backflow.mu_parameters, self.config.backflow.mu_cutoff, self.config.backflow.mu_labels,
            self.config.backflow.phi_parameters, self.config.backflow.theta_parameters, self.config.backflow.phi_cutoff,
            self.config.backflow.phi_labels, self.config.backflow.phi_irrotational, self.config.backflow.ae_cutoff
        )
        self.wfn = Wfn(
            self.config.input.neu, self.config.input.ned, self.config.wfn.atom_positions, self.config.wfn.atom_charges, slater, jastrow, backflow
        )
        self.neu, self.ned = self.config.input.neu, self.config.input.ned

        if self.config.input.vmc_method == 1:
            # EBES
            step = 1 / np.log(self.neu + self.ned)
        elif self.config.input.vmc_method == 3:
            # CBCS
            step = 1 / (self.neu + self.ned)
        else:
            # wrong method
            step = 0
        self.markovchain = MarkovChain(self.neu, self.ned, step, self.config.wfn.atom_positions, self.config.wfn.atom_charges, self.wfn)
        # FIXME: not supported by numba move to MarkovChain.__init__()
        self.markovchain.r_e = nb.typed.List.empty_list(r_e_type)
        self.markovchain.r_e.append(initial_position(self.neu + self.ned, self.markovchain.atom_positions, self.markovchain.atom_charges))

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
            if self.config.input.vmc_decorr_period == 0:
                decorr_period = self.optimize_decorr_period()
            else:
                decorr_period = self.config.input.vmc_decorr_period
            self.vmc_energy_accumulation(decorr_period)
            stop = default_timer()
            logger.info(
                f' =========================================================================\n\n'
                f' Total PyCasino real time : : :    {stop - start:.4f}'
            )
        elif self.config.input.runtype == 'vmc_opt':
            if self.config.input.opt_method == 'varmin':
                start = default_timer()
                self.optimize_vmc_step(10000)
                self.vmc_energy_accumulation(1)
                for i in range(self.config.input.opt_cycles):
                    res = self.vmc_variance_minimization(self.config.input.vmc_nconfig_write)
                    # unload to file
                    self.wfn.jastrow.set_parameters(res.x)
                    print(res.x / self.wfn.jastrow.get_bounds()[1])
                    # self.wfn.jastrow.write(f'./correlation.out.{i+1}')
                    self.optimize_vmc_step(10000)
                    self.vmc_energy_accumulation(1)
                stop = default_timer()
                logger.info(
                    f' =========================================================================\n\n'
                    f' Total PyCasino real time : : :    {stop - start:.4f}'
                )
            elif self.config.input.opt_method == 'emin':
                start = default_timer()
                self.optimize_vmc_step(10000)
                self.vmc_energy_accumulation(1)
                for i in range(self.config.input.opt_cycles):
                    res = self.vmc_energy_minimization(self.config.input.vmc_nconfig_write)
                    # unload to file
                    self.wfn.jastrow.set_parameters(res.x)
                    print(res.x / self.wfn.jastrow.get_bounds()[1])
                    # self.wfn.jastrow.write(f'./correlation.out.{i+1}')
                    self.optimize_vmc_step(10000)
                    self.vmc_energy_accumulation(1)
                stop = default_timer()
                logger.info(
                    f' =========================================================================\n\n'
                    f' Total PyCasino real time : : :    {stop - start:.4f}'
                )
        elif self.config.input.runtype == 'vmc_dmc':
            self.optimize_vmc_step(10000)
            # FIXME: decorr_period for dmc?
            cond, position = self.markovchain.vmc_random_walk(self.config.input.vmc_nstep, 1)
            energy = self.markovchain.local_energy(cond, position) + self.markovchain.wfn.nuclear_repulsion
            logger.info('VMC energy %.5f', energy.mean())
            position = position[-self.config.input.vmc_nconfig_write:]

            # FIXME: local variables?
            self.markovchain.step = self.config.input.dtdmc
            self.markovchain.r_e = nb.typed.List.empty_list(r_e_type)
            for p in position:
                self.markovchain.r_e.append(p)

            self.dmc_energy_equilibration()

            self.dmc_energy_accumulation()

    def equilibrate(self, steps):
        """
        :param steps: burn-in period
        :return:
        """
        condition, _ = self.markovchain.vmc_random_walk(steps, 1)
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
                condition, _ = self.markovchain.vmc_random_walk(steps, 1)
                acc_ration = condition.mean()
            else:
                acc_ration = 1
            return acc_ration - acceptance_rate

        options = dict(jac_options=dict(alpha=1))
        res = root(f, [self.markovchain.step], method='diagbroyden', tol=1/np.sqrt(steps), callback=callback, options=options)
        self.markovchain.step = np.abs(res.x[0])

    def optimize_decorr_period(self):
        """Optimize decorr period"""
        # TODO:
        return 3

    def vmc_energy_accumulation(self, decorr_period):
        """VMC energy accumulation"""
        steps = self.config.input.vmc_nstep
        nblock = self.config.input.vmc_nblock

        energy_block_mean = np.zeros(shape=(nblock, ))
        energy_block_sem = np.zeros(shape=(nblock, ))
        energy_block_var = np.zeros(shape=(nblock,))
        logger.info(
            f'Starting VMC.'
        )
        for i in range(nblock):
            block_start = default_timer()
            condition, position = self.markovchain.vmc_random_walk(steps // nblock, decorr_period)
            energy = self.markovchain.local_energy(condition, position) + self.markovchain.wfn.nuclear_repulsion
            energy_block_mean[i] = energy.mean()
            energy_block_sem[i] = correlated_sem(energy)
            energy_block_var[i] = np.var(energy)
            block_stop = default_timer()
            logger.info(
                f' =========================================================================\n'
                f' In block : {i + 1}\n'
                f'  Number of VMC steps           = {steps // nblock}\n\n'
                f'  Block average energies (au)\n\n'
                f'  Total energy                       (au) =       {energy_block_mean[i]:18.12f}\n'
                f'  Standard error                        +/-       {energy_block_sem[i]:18.12f}\n\n'
                f' Time taken in block    : : :       {block_stop - block_start:.4f}\n'
            )
        logger.info(
            f' =========================================================================\n'
            f' FINAL RESULT:\n\n'
            f'  VMC energy (au)    Standard error      Correction for serial correlation\n'
            f' {energy_block_mean.mean():.12f} +/- {energy_block_sem.mean() / np.sqrt(nblock):.12f}      On-the-fly reblocking method\n\n'
            f' Sample variance of E_L (au^2/sim.cell) : {energy_block_var.mean():.12f}\n\n'
        )

    def dmc_energy_equilibration(self):
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
            energy = self.markovchain.dmc_random_walk(steps // nblock, self.config.input.dmc_target_weight)
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

    def dmc_energy_accumulation(self):
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
            energy = self.markovchain.dmc_random_walk(steps // nblock, self.config.input.dmc_target_weight)
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

    def normal_test(self, energy):
        """Test whether energy distribution differs from a normal one."""
        from scipy import stats
        logger.info(f'skew = {stats.skewtest(energy)}, kurtosis = {stats.kurtosistest(energy)}')

    def vmc_variance_minimization(self, steps, loss='linear', f_scale=1):
        """Minimise vmc variance by jastrow parameters optimization.
        https://github.com/scipy/scipy/issues/10634
        """
        condition, position = self.markovchain.vmc_random_walk(steps, 1)

        # if loss == 'arctan':
        #     energy = self.markovchain.local_energy(condition, position)
        #     f_scale = 3 * energy.std()

        def fun(x, *args, **kwargs):
            self.wfn.jastrow.set_parameters(x)
            energy = self.markovchain.local_energy(condition, position)
            # median energy is lower than mean, but maybe np.min(np.median(energy), np.mean(energy))
            return energy - np.median(energy)

        return least_squares(
            fun, x0=self.wfn.jastrow.get_parameters(), jac='2-point', method='trf',
            max_nfev=7, x_scale=self.wfn.jastrow.get_x_scale(), loss=loss, f_scale=f_scale,
            tr_solver='lsmr', tr_options=dict(regularize=False), verbose=2
        )

    def vmc_energy_minimization(self, steps):
        """Minimise vmc energy by jastrow parameters optimization.
        Gradient only for : CG, BFGS, L-BFGS-B, TNC, SLSQP
        Gradient and Hessian is required for: Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact, trust-constr
        Constraints definition only for: COBYLA, SLSQP and trust-constr.
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods.

        SciPy, оптимизация с условиями - https://habr.com/ru/company/ods/blog/448054/
        """
        bounds = Bounds(*self.wfn.jastrow.get_bounds(), keep_feasible=True)
        condition, position = self.markovchain.vmc_random_walk(steps, 1)

        def fun(x, *args):
            self.wfn.jastrow.set_parameters(x)
            energy = self.markovchain.local_energy(condition, position)
            mean_energy = energy.mean()
            energy_gradient = self.markovchain.jastrow_gradient(condition, position)
            mean_energy_gradient = jastrow_parameters_gradient(energy, energy_gradient)
            return mean_energy, mean_energy_gradient

        def hess(x, *args):
            self.wfn.jastrow.set_parameters(x)
            energy = self.markovchain.local_energy(condition, position)
            energy_gradient = self.markovchain.jastrow_gradient(condition, position)
            energy_hessian = self.markovchain.jastrow_hessian(condition, position)
            mean_energy_hessian = jastrow_parameters_hessian(energy, energy_gradient, energy_hessian)
            logger.info('hessian = %s', mean_energy_hessian)
            return mean_energy_hessian

        parameters = self.wfn.jastrow.get_parameters()
        res = minimize(fun, parameters, method='TNC', jac=True, bounds=bounds, options=dict(disp=True, maxfun=10))
        # res = minimize(f, parameters, method='trust-ncg', jac=True, hess=hess, options=dict(disp=True)
        return res


if __name__ == '__main__':
    """Tests
    """
    # path = 'test/gwfn/He/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Be/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/N/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Ne/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Ar/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Kr/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/O3/HF/cc-pVQZ/CBCS/Slater/'

    # path = 'test/gwfn/Be/HF/ano-pVDZ/CBCS/Slater/'
    # path = 'test/gwfn/Ne/HF/ano-pVDZ/CBCS/Slater/'

    # path = 'test/gwfn/He/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/Be/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/N/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/Ne/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/Ar/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/Kr/HF/cc-pVQZ/CBCS/Jastrow/'
    # path = 'test/gwfn/O3/HF/cc-pVQZ/CBCS/Jastrow/'

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
