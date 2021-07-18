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
from readers.casino import Casino
from overload import subtract_outer
from logger import logging

np.random.seed(31415926)

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
    return r_e + np.random.laplace(0, 1, ne * 3).reshape((ne, 3))


@nb.experimental.jitclass(spec)
class MarkovChain:

    def __init__(self, neu, ned, atom_positions, atom_charges, wfn):
        """Markov chain Monte Carlo.
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
        # self.step = 1 / np.log(neu + ned)
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.wfn = wfn

    def gibbs_proposal_step(self, r_e):
        """Random N-dim square distributed step
        electron-by-electron sampling (EBES)
        """
        r_e[np.random.randint(self.neu + self.ned)] += self.step * np.random.uniform(-1, 1, 3)
        return r_e

    def simple_random_step(self, p, r_e):
        """Propose step with random N-dim square proposal density in
        configuration-by-configuration sampling (CBCS).
        In general, the proposal step may depend on the current position or
        have a different proposal density.
        It is believed that the algorithm works best if the proposal density
        matches the shape of the target distribution which is in case of
        Quantum Monte Carlo is squared modulus of WFN.
        """
        ne = self.neu + self.ned
        new_r_e = r_e + self.step * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
        e_vectors = subtract_outer(new_r_e, new_r_e)
        n_vectors = -subtract_outer(self.atom_positions, new_r_e)
        new_p = self.wfn.value(e_vectors, n_vectors)
        cond = new_p ** 2 / p ** 2 > np.random.random()
        return cond and (new_r_e, new_p, cond) or (r_e, p, cond)

    def biased_random_step(self, p, r_e):
        """Diffusion-drift proposed step
        diffusion is proportional to sqrt(2*D*dt)
        drift is proportional to D*F*dt
        where D is diffusion constant = 1/2
        """
        ne = self.neu + self.ned
        v_forth = self.wfn.drift_velocity(r_e)
        res = np.sqrt(self.step) * np.random.normal(0, 1, ne * 3) + self.step * v_forth
        new_r_e = r_e + res.reshape((ne, 3))
        e_vectors = subtract_outer(new_r_e, new_r_e)
        n_vectors = -subtract_outer(self.atom_positions, new_r_e)
        new_p = self.wfn.value(e_vectors, n_vectors)
        g_forth = np.exp(-np.sum((new_r_e.ravel() - r_e.ravel() - self.step * v_forth) ** 2) / 2 / self.step)
        g_back = np.exp(-np.sum((r_e.ravel() - new_r_e.ravel() - self.step * self.wfn.drift_velocity(new_r_e)) ** 2) / 2 / self.step)
        cond = (g_back * new_p ** 2) / (g_forth * p ** 2) > np.random.random()
        return cond and (new_r_e, new_p, cond) or (r_e, p, cond)

    make_step = simple_random_step

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
            casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_mask, casino.jastrow.u_cutoff, casino.jastrow.u_cusp_const,
            casino.jastrow.chi_parameters, casino.jastrow.chi_mask, casino.jastrow.chi_cutoff, casino.jastrow.chi_labels,
            casino.jastrow.f_parameters, casino.jastrow.f_mask, casino.jastrow.f_cutoff, casino.jastrow.f_labels,
            casino.jastrow.no_dup_u_term, casino.jastrow.no_dup_chi_term, casino.jastrow.chi_cusp
        )
        self.backflow = casino.backflow and Backflow(
            casino.input.neu, casino.input.ned,
            casino.backflow.trunc, casino.backflow.eta_parameters, casino.backflow.eta_cutoff,
            casino.backflow.mu_parameters, casino.backflow.mu_cutoff, casino.backflow.mu_labels,
            casino.backflow.phi_parameters, casino.backflow.theta_parameters, casino.backflow.phi_cutoff,
            casino.backflow.phi_labels, casino.backflow.phi_irrotational, casino.backflow.ae_cutoff
        )
        self.wfn = Wfn(casino.input.neu, casino.input.ned, casino.wfn.atom_positions, casino.wfn.atom_charges, self.slater, self.jastrow, self.backflow)
        self.neu, self.ned = casino.input.neu, casino.input.ned
        self.markovchain = MarkovChain(self.neu, self.ned, casino.wfn.atom_positions, casino.wfn.atom_charges, self.wfn)
        # FIXME: not supported by numba move to MarkovChain.__init__()
        self.markovchain.r_e = initial_position(self.neu + self.ned, self.markovchain.atom_positions, self.markovchain.atom_charges)

    def equilibrate(self, steps):
        """Burn-in period"""
        weight, _, _ = self.markovchain.random_walk(steps)
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
                weight, _, _ = self.markovchain.random_walk(steps)
                acc_ration = weight.size / steps
            else:
                acc_ration = 1
            return acc_ration - acceptance_rate

        options = dict(jac_options=dict(alpha=1))
        res = sp.optimize.root(f, [self.markovchain.step], method='diagbroyden', tol=1/np.sqrt(steps), callback=callback, options=options)
        self.markovchain.step = np.abs(res.x[0])

    def energy(self, steps, nblock):
        self.optimize_vmc_step(10000)

        start = default_timer()
        expanded_energy = np.zeros((nblock, steps // nblock))
        for i in range(nblock):
            weights, position, _ = self.markovchain.random_walk(steps // nblock)
            energy = self.markovchain.local_energy(position) + self.markovchain.wfn.nuclear_repulsion
            stop = default_timer()
            logger.info('total time {}'.format(stop - start))
            expanded_energy[i] = expand(weights, energy)

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
        weight, position, _ = self.markovchain.random_walk(steps)

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
        weight, position, _ = self.markovchain.random_walk(steps)

        def callback(x, *args):
            logger.info('inner iteration x = %s', x)
            self.jastrow.set_parameters(x)
            weight, position, _ = self.markovchain.random_walk(steps)

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

    def varmin(self, steps, opt_cycles):
        for _ in range(opt_cycles):
            self.optimize_vmc_step(10000)
            res = self.vmc_variance_minimization(steps)
            # unload to file
            # logger.info('x = %s', res.x)
            self.jastrow.set_parameters(res.x)

    def emin(self, steps, opt_cycles):
        for _ in range(opt_cycles):
            self.optimize_vmc_step(10000)
            res = self.vmc_energy_minimization(steps)
            # unload to file
            # logger.info('x = %s', res.x)
            self.jastrow.set_parameters(res.x)


def main(casino):
    """Configuration-by-configuration sampling (CBCS)
    Should be pure python function.
    """

    vmc = VMC(casino)
    vmc.equilibrate(casino.input.vmc_equil_nstep)
    if casino.input.runtype == 'vmc':
        vmc.energy(casino.input.vmc_nstep, casino.input.vmc_nblock)
    elif casino.input.runtype == 'vmc_opt':
        if casino.input.opt_method == 'varmin':
            vmc.varmin(casino.input.vmc_opt_nstep, 5)
        elif casino.input.opt_method == 'emin':
            vmc.emin(casino.input.vmc_opt_nstep, 5)


if __name__ == '__main__':
    """
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

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Jastrow_optimization/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Jastrow_optimization/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Jastrow_optimization/'

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

    casino = Casino(path)
    main(casino)
