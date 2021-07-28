#!/usr/bin/env python3

import os
from timeit import default_timer
from slater import Slater
from jastrow import Jastrow
from backflow import Backflow
from wfn import Wfn

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
from readers.casino import CasinoConfig
from logger import logging

np.random.seed(31415926)

logger = logging.getLogger('vmc')
numba_logger = logging.getLogger('numba')

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
        """
        A significant source of error in DMC calculations comes from sampling electronic
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

    def simple_random_walker(self, steps, r_e):
        """Simple random walker with random N-dim square proposal density in
        configuration-by-configuration sampling (CBCS).
        In general, the proposal step may depend on the current position or
        have a different proposal density.
        It is believed that the algorithm works best if the proposal density
        matches the shape of the target distribution which is in case of
        Quantum Monte Carlo is squared modulus of WFN.
        :param steps: number of steps to walk
        :param r_e: initial position
        :return: is step accept, next step position
        """
        ne = self.neu + self.ned
        e_vectors, n_vectors = self.wfn.relative_coordinates(r_e)
        probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
        for _ in range(steps):
            new_r_e = r_e + self.step * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
            e_vectors, n_vectors = self.wfn.relative_coordinates(new_r_e)
            new_probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
            cond = new_probability_density / probability_density > np.random.random()
            if cond:
                r_e, probability_density = new_r_e, new_probability_density
            yield cond, r_e

    def gibbs_random_walker(self, steps, r_e):
        """Simple random walker with electron-by-electron sampling (EBES)
        :param steps: number of steps to walk
        :param r_e: initial position
        :return: is step accept, next step position
        """
        ne = self.neu + self.ned
        e_vectors, n_vectors = self.wfn.relative_coordinates(r_e)
        probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
        for _ in range(steps):
            new_r_e = np.copy(r_e)
            new_r_e[np.random.randint(ne)] += self.step * np.random.uniform(-1, 1, 3)
            e_vectors, n_vectors = self.wfn.relative_coordinates(new_r_e)
            new_probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
            cond = new_probability_density / probability_density > np.random.random()
            if cond:
                r_e, p = new_r_e, new_probability_density
            yield cond, r_e

    def biased_random_walker(self, steps, r_e):
        """Biased random walker with diffusion-drift proposed step
        diffusion step s proportional to sqrt(2*D*dt)
        drift step is proportional to D*F*dt
        where D is diffusion constant = 1/2
        :param steps: number of steps to walk
        :param r_e: initial position
        :return: is step accept, next step position
        """
        ne = self.neu + self.ned
        e_vectors, n_vectors = self.wfn.relative_coordinates(r_e)
        probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
        for _ in range(steps):
            v_forth = self.drift_velocity(r_e)
            move = np.sqrt(self.step) * np.random.normal(0, 1, ne * 3) + self.step * v_forth
            new_r_e = r_e + move.reshape((ne, 3))
            e_vectors, n_vectors = self.wfn.relative_coordinates(new_r_e)
            new_probability_density = self.wfn.value(e_vectors, n_vectors) ** 2
            green_forth = np.exp(-np.sum((new_r_e.ravel() - r_e.ravel() - self.step * v_forth) ** 2) / 2 / self.step)
            green_back = np.exp(-np.sum((r_e.ravel() - new_r_e.ravel() - self.step * self.drift_velocity(new_r_e)) ** 2) / 2 / self.step)
            cond = (green_back * new_probability_density) / (green_forth * probability_density) > np.random.random()
            if cond:
                r_e, probability_density = new_r_e, new_probability_density
            yield cond, r_e

    def bbk_random_walker(self, steps, r_e):
        """Brünger–Brooks–Karplus (13 B. Brünger, C. L. Brooks, and M. Karplus, Chem. Phys. Lett. 105, 495 1984).
        :param steps: number of steps to walk
        :param r_e: initial position
        :return: is step accept, next step position
        """
        for _ in range(steps):
            cond = False
            yield cond, r_e

    def force_interpolation_random_walker(self, steps, r_e):
        """M. P. Allen and D. J. Tildesley, Computer Simulation of Liquids Oxford University Press, Oxford, 1989 and references in Sec. 9.3.
        :param steps: number of steps to walk
        :param r_e: initial position
        :return: is step accept, next step position
        """
        for _ in range(steps):
            cond = False
            yield cond, r_e

    def splitting_random_walker(self, steps, r_e):
        """J. A. Izaguirre, D. P. Catarello, J. M. Wozniak, and R. D. Skeel, J. Chem. Phys. 114, 2090 2001.
        :param steps: number of steps to walk
        :param r_e: initial position
        :return: is step accept, next step position
        """
        for _ in range(steps):
            cond = False
            yield cond, r_e

    def ricci_ciccottid_random_walker(self, steps, r_e):
        """A. Ricci and G. Ciccotti, Mol. Phys. 101, 1927 2003.
        :param steps: number of steps to walk
        :param r_e: initial position
        :return: is step accept, next step position
        """
        for _ in range(steps):
            cond = False
            yield cond, r_e

    def dmc_random_walker(self, steps, positions, target_weight):
        """Collection of walkers representing wfn.
        :param steps: number of steps to walk
        :param positions: initial positions of walkers
        :param target_weight: target weight of walkers
        :return: best estimate of energy
        """
        def average_weight(weight):
            """average weigt"""
            sum_w = 0.0
            for w in weight:
                sum_w += w
            return sum_w / target_weight

        def dmc_energy(energy, weight):
            """Mixed estimator of energy"""
            sum_ew = sum_w = 0.0
            for e, w in zip(energy, weight):
                sum_w += w
                sum_ew += e * w
            return sum_ew / sum_w

        ne = self.neu + self.ned
        p_list = nb.typed.List()
        r_e_list = nb.typed.List()
        energy_list = nb.typed.List()
        weight_list = nb.typed.List()
        velocity_list = nb.typed.List()
        branching_energy_list = nb.typed.List()
        for r_e in positions:
            e_vectors, n_vectors = self.wfn.relative_coordinates(r_e)
            p_list.append(self.wfn.value(e_vectors, n_vectors))
            r_e_list.append(r_e)
            energy_list.append(self.wfn.energy(r_e))
            branching_energy_list.append(self.wfn.energy(r_e))
            weight_list.append(1.0)
            v = self.wfn.drift_velocity(r_e)
            l = self.limiting_factor(v)
            velocity_list.append(l * v)
        best_estimate_energy = dmc_energy(energy_list, weight_list)
        energy_t = best_estimate_energy - np.log(len(energy_list)/target_weight) / self.step
        for step in range(steps):
            new_p_list = nb.typed.List()
            new_r_e_list = nb.typed.List()
            new_weight_list = nb.typed.List()
            new_energy_list = nb.typed.List()
            new_velocity_list = nb.typed.List()
            new_branching_energy_list = nb.typed.List()
            for r_e, p, velocity, energy, branching_energy in zip(r_e_list, p_list, velocity_list, energy_list, branching_energy_list):
                new_r_e = r_e + (np.sqrt(self.step) * np.random.normal(0, 1, ne * 3) + self.step * velocity).reshape((ne, 3))
                e_vectors, n_vectors = self.wfn.relative_coordinates(new_r_e)
                new_p = self.wfn.value(e_vectors, n_vectors)
                # prevent crossing nodal surface
                cond = np.sign(p) * np.sign(new_p) > 0
                new_velocity = self.wfn.drift_velocity(new_r_e)
                new_energy = self.wfn.energy(new_r_e)
                limiting_factor = self.limiting_factor(new_velocity)
                new_velocity *= limiting_factor
                new_branching_energy = best_estimate_energy - (best_estimate_energy - new_energy) * limiting_factor
                if cond:
                    # Green`s functions
                    green_forth = np.exp(-np.sum((new_r_e.ravel() - r_e.ravel() - self.step * velocity) ** 2) / 2 / self.step)
                    green_back = np.exp(-np.sum((r_e.ravel() - new_r_e.ravel() - self.step * new_velocity) ** 2) / 2 / self.step)
                    # condition
                    cond = (green_back * new_p ** 2) / (green_forth * p ** 2) > np.random.random()
                # branching
                if cond:
                    weight = np.exp(-self.step * (new_branching_energy + branching_energy - 2 * energy_t) / 2)
                else:
                    weight = np.exp(-self.step * (branching_energy - energy_t))
                new_weight_list.append(weight)
                for _ in range(int(weight + np.random.uniform(0, 1))):
                    if cond:
                        new_p_list.append(new_p)
                        new_r_e_list.append(new_r_e)
                        new_energy_list.append(new_energy)
                        new_velocity_list.append(new_velocity)
                        new_branching_energy_list.append(new_branching_energy)
                    else:
                        new_p_list.append(p)
                        new_r_e_list.append(r_e)
                        new_energy_list.append(energy)
                        new_velocity_list.append(velocity)
                        new_branching_energy_list.append(branching_energy)
            p_list = new_p_list
            r_e_list = new_r_e_list
            energy_list = new_energy_list
            velocity_list = new_velocity_list
            branching_energy_list = new_branching_energy_list
            best_estimate_energy = dmc_energy(energy_list, new_weight_list)
            energy_t = best_estimate_energy - np.log(average_weight(new_weight_list)) / self.step
            yield best_estimate_energy, r_e_list

    walker = simple_random_walker

    def vmc_random_walk(self, steps):
        """Metropolis-Hastings random walk.
        """
        r_e = self.r_e[0]
        weight = np.ones((steps, ), np.int64)
        position = np.zeros((steps, r_e.shape[0], r_e.shape[1]))
        walker = self.walker(steps, r_e)
        _, position[0] = next(walker)

        i = 0
        for cond, r_e in walker:
            if cond:
                i += 1
                position[i] = r_e
            else:
                weight[i] += 1

        self.r_e[0] = r_e
        return weight[:i+1], position[:i+1]

    def dmc_random_walk(self, steps, target_weight):
        """DMC
        :param steps: number of steps to walk
        :param target_weight: target weight
        :return:
        """
        r_e = self.r_e
        walker = self.dmc_random_walker(steps, r_e, target_weight)
        energy = np.zeros((steps, ))
        for i, (energy_t, r_e) in enumerate(walker):
            energy[i] = energy_t

        # FIXME: class attribute is always any-array?
        self.r_e = r_e
        return energy

    def local_energy(self, position):
        """
        :param position: random walk positions
        :return:
        """
        return np.array([self.wfn.energy(p) for p in position])

    def jastrow_gradient(self, position):
        """Jastrow gradient with respect to jastrow parameters.
        :param position: random walk positions
        :return:
        """
        first_res = self.wfn.jastrow_parameters_numerical_d1(position[0])
        res = np.zeros((position.shape[0], ) + first_res.shape)
        res[0] = first_res

        for i in range(1, position.shape[0]):
            res[i] = self.wfn.jastrow_parameters_numerical_d1(position[i])
        return res

    def jastrow_hessian(self, position):
        """Jastrow hessian with respect to jastrow parameters.
        :param position: random walk positions
        :return:
        """
        first_res = self.wfn.jastrow_parameters_numerical_d2(position[0])
        res = np.zeros((position.shape[0], ) + first_res.shape)
        res[0] = first_res

        for i in range(1, position.shape[0]):
            res[i] = self.wfn.jastrow_parameters_numerical_d2(position[i])
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


class Casino:

    def __init__(self, path):
        """Casino workflow.
        :param path: path to config
        """
        self.config = CasinoConfig(path)

        self.slater = Slater(
            self.config.input.neu, self.config.input.ned,
            self.config.wfn.nbasis_functions, self.config.wfn.first_shells, self.config.wfn.orbital_types, self.config.wfn.shell_moments,
            self.config.wfn.slater_orders, self.config.wfn.primitives, self.config.wfn.coefficients, self.config.wfn.exponents,
            self.config.mdet.mo_up, self.config.mdet.mo_down, self.config.mdet.coeff
        )
        self.jastrow = self.config.jastrow and Jastrow(
            self.config.input.neu, self.config.input.ned,
            self.config.jastrow.trunc, self.config.jastrow.u_parameters, self.config.jastrow.u_mask, self.config.jastrow.u_cutoff, self.config.jastrow.u_cusp_const,
            self.config.jastrow.chi_parameters, self.config.jastrow.chi_mask, self.config.jastrow.chi_cutoff, self.config.jastrow.chi_labels,
            self.config.jastrow.f_parameters, self.config.jastrow.f_mask, self.config.jastrow.f_cutoff, self.config.jastrow.f_labels,
            self.config.jastrow.no_dup_u_term, self.config.jastrow.no_dup_chi_term, self.config.jastrow.chi_cusp
        )
        self.backflow = self.config.backflow and Backflow(
            self.config.input.neu, self.config.input.ned,
            self.config.backflow.trunc, self.config.backflow.eta_parameters, self.config.backflow.eta_cutoff,
            self.config.backflow.mu_parameters, self.config.backflow.mu_cutoff, self.config.backflow.mu_labels,
            self.config.backflow.phi_parameters, self.config.backflow.theta_parameters, self.config.backflow.phi_cutoff,
            self.config.backflow.phi_labels, self.config.backflow.phi_irrotational, self.config.backflow.ae_cutoff
        )
        self.wfn = Wfn(
            self.config.input.neu, self.config.input.ned, self.config.wfn.atom_positions, self.config.wfn.atom_charges, self.slater, self.jastrow, self.backflow
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
            # FIXME: in EBEC nstep = vmc_nstep * (neu + ned)
            self.optimize_vmc_step(10000)
            self.energy(self.config.input.vmc_nstep, self.config.input.vmc_nblock)
        elif self.config.input.runtype == 'vmc_opt':
            if self.config.input.opt_method == 'varmin':
                for _ in range(self.config.input.opt_cycles):
                    self.optimize_vmc_step(10000)
                    res = self.vmc_variance_minimization(self.config.input.vmc_nstep)
                    # unload to file
                    # logger.info('x = %s', res.x)
                    self.jastrow.set_parameters(res.x)
            elif self.config.input.opt_method == 'emin':
                for _ in range(self.config.input.opt_cycles):
                    self.optimize_vmc_step(10000)
                    res = self.vmc_energy_minimization(self.config.input.vmc_nstep)
                    # unload to file
                    # logger.info('x = %s', res.x)
                    self.jastrow.set_parameters(res.x)
        elif self.config.input.runtype == 'vmc_dmc':
            self.optimize_vmc_step(10000)
            weight, position = self.markovchain.vmc_random_walk(self.config.input.vmc_nstep)
            energy = self.markovchain.local_energy(position) + self.markovchain.wfn.nuclear_repulsion
            energy_average = np.average(energy, weights=weight)
            logger.info('VMC energy %.5f', energy_average)
            expanded_position = expand(weight, position)
            expanded_position = expanded_position[-self.config.input.vmc_nconfig_write:]
            self.markovchain.step = self.config.input.dtdmc
            self.markovchain.r_e = nb.typed.List.empty_list(r_e_type)
            for p in expanded_position:
                self.markovchain.r_e.append(p)
            start = default_timer()

            energy = self.markovchain.dmc_random_walk(self.config.input.dmc_equil_nstep, self.config.input.dmc_target_weight)
            reblock_data = pyblock.blocking.reblock(energy)
            opt = pyblock.blocking.find_optimal_block(self.config.input.dmc_equil_nstep, reblock_data)
            if opt[0]:
                opt_data = reblock_data[opt[0]]
                logger.info(opt_data)
                logger.info('{} +/- {}'.format(np.mean(opt_data.mean), np.mean(opt_data.std_err) / np.sqrt(opt_data.std_err.size)))
            stop = default_timer()
            logger.info('total time {}'.format(stop - start))

            energy = self.markovchain.dmc_random_walk(self.config.input.dmc_equil_nstep, self.config.input.dmc_target_weight)
            reblock_data = pyblock.blocking.reblock(energy)
            opt = pyblock.blocking.find_optimal_block(self.config.input.dmc_equil_nstep, reblock_data)
            if opt[0]:
                opt_data = reblock_data[opt[0]]
                logger.info(opt_data)
                logger.info('{} +/- {}'.format(np.mean(opt_data.mean), np.mean(opt_data.std_err) / np.sqrt(opt_data.std_err.size)))
            stop = default_timer()
            logger.info('total time {}'.format(stop - start))

    def equilibrate(self, steps):
        """Burn-in period"""
        weight, _ = self.markovchain.vmc_random_walk(steps)
        logger.info('dr * electrons = 1.00000, acc_ration = %.5f', weight.size / steps)

    def optimize_vmc_step(self, steps, acceptance_rate=0.5):
        """Optimize vmc step size."""

        def callback(tau, acc_ration):
            """dr = sqrt(3*dtvmc)"""
            logger.info('dr * electrons = %.5f, acc_ration = %.5f', tau[0] * (self.neu + self.ned), acc_ration[0] + acceptance_rate)

        def f(tau):
            self.markovchain.step = tau[0]
            logger.info('dr * electrons = %.5f', tau[0] * (self.neu + self.ned))
            if tau[0] > 0:
                weight, _ = self.markovchain.vmc_random_walk(steps)
                acc_ration = weight.size / steps
            else:
                acc_ration = 1
            return acc_ration - acceptance_rate

        options = dict(jac_options=dict(alpha=1))
        res = sp.optimize.root(f, [self.markovchain.step], method='diagbroyden', tol=1/np.sqrt(steps), callback=callback, options=options)
        self.markovchain.step = np.abs(res.x[0])

    def energy(self, steps, nblock):
        """Energy accumulation"""
        start = default_timer()
        expanded_energy = np.zeros((nblock, steps // nblock))
        for i in range(nblock):
            weight, position = self.markovchain.vmc_random_walk(steps // nblock)
            energy = self.markovchain.local_energy(position) + self.markovchain.wfn.nuclear_repulsion
            stop = default_timer()
            logger.info('total time {}'.format(stop - start))
            expanded_energy[i] = expand(weight, energy)

        reblock_data = pyblock.blocking.reblock(expanded_energy)
        opt = pyblock.blocking.find_optimal_block(steps, reblock_data)
        opt_data = reblock_data[opt[0]]
        logger.info(opt_data)
        logger.info('{} +/- {}'.format(np.mean(opt_data.mean), np.mean(opt_data.std_err) / np.sqrt(opt_data.std_err.size)))

    def normal_test(self, weight, energy):
        """Test whether energy distribution differs from a normal one."""
        from scipy import stats
        E = expand(weight, energy)
        logger.info('skew = %s, kurtosis = %s', stats.skewtest(E), stats.kurtosistest(E))

    def vmc_variance_minimization(self, steps):
        """Minimise vmc variance by jastrow parameters optimization.
        https://github.com/scipy/scipy/issues/10634
        """
        bounds = self.jastrow.get_bounds()
        weight, position = self.markovchain.vmc_random_walk(steps)

        def f(x, *args, **kwargs):
            self.jastrow.set_parameters(x)
            energy = self.markovchain.local_energy(position) + self.markovchain.wfn.nuclear_repulsion
            energy_average = np.average(energy, weights=weight)
            energy_variance = np.average((energy - energy_average) ** 2, weights=weight)
            std_err = np.sqrt(energy_variance / weight.sum())
            logger.info('energy = %.8f +- %.8f, variance = %.8f', energy_average, std_err, energy_variance)
            return expand(weight, energy) - energy_average

        def jac(x, *args, **kwargs):
            self.jastrow.set_parameters(x)
            return expand(weight, self.markovchain.jastrow_gradient(position))

        parameters = self.jastrow.get_parameters()
        res = sp.optimize.least_squares(
            f, parameters, jac=jac, bounds=bounds, method='trf', xtol=1e-4, max_nfev=20,
            x_scale='jac', loss='linear', tr_solver='exact', tr_options=dict(show=False, regularize=False),
            verbose=2
        )
        return res

    def vmc_energy_minimization(self, steps):
        """Minimise vmc energy by jastrow parameters optimization.
        Gradient only for : CG, BFGS, L-BFGS-B, TNC, SLSQP
        Gradient and Hessian is required for: Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact, trust-constr
        Constraints definition only for: COBYLA, SLSQP and trust-constr
        """
        bounds = self.jastrow.get_bounds()
        weight, position = self.markovchain.vmc_random_walk(steps)

        def callback(x, *args):
            logger.info('inner iteration x = %s', x)
            self.jastrow.set_parameters(x)
            weight, position = self.markovchain.vmc_random_walk(steps)

        def f(x, *args):
            self.jastrow.set_parameters(x)
            energy = self.markovchain.local_energy(position) + self.markovchain.wfn.nuclear_repulsion
            energy_gradient = self.markovchain.jastrow_gradient(position)
            mean_energy = np.average(energy, weights=weight)
            mean_energy_gradient = jastrow_parameters_gradient(weight, energy, energy_gradient)
            energy_average = np.average(energy, weights=weight)
            energy_variance = np.average((energy - energy_average) ** 2, weights=weight)
            std_err = np.sqrt(energy_variance / weight.sum())
            logger.info('energy = %.8f +- %.8f, variance = %.8f', energy_average, std_err, energy_variance)
            return mean_energy, mean_energy_gradient

        def hess(x, *args):
            self.jastrow.set_parameters(x)
            energy = self.markovchain.local_energy(position) + self.markovchain.wfn.nuclear_repulsion
            energy_gradient = self.markovchain.jastrow_gradient(position)
            energy_hessian = self.markovchain.jastrow_hessian(position)
            mean_energy_hessian = jastrow_parameters_hessian(weight, energy, energy_gradient, energy_hessian)
            logger.info('hessian = %s', mean_energy_hessian)
            return mean_energy_hessian

        parameters = self.jastrow.get_parameters()
        options = dict(disp=True, maxfun=50)
        res = sp.optimize.minimize(f, parameters, method='TNC', jac=True, bounds=list(zip(*bounds)), options=options, callback=callback)
        # options = dict(disp=True)
        # res = sp.optimize.minimize(f, parameters, method='trust-ncg', jac=True, hess=hess, options=options, callback=callback)
        return res


if __name__ == '__main__':
    """Tests
    """

    # path = 'test/gwfn/He/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Be/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Ne/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Ae/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Kr/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/O3/HF/cc-pVQZ/CBCS/Slater/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Slater/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Jastrow_opt/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Jastrow_opt/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Jastrow_opt/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Jastrow_opt/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Jastrow_opt/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Jastrow_opt/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Jastrow/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Backflow/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Backflow/'

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Jastrow_dmc/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Jastrow_dmc/'

    Casino(path).run()
