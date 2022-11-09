
import os
from wfn import Wfn

os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import numpy as np
import numba as nb

vmc_spec = [
    ('r_e', nb.float64[:, :]),
    ('cond', nb.boolean),
    ('step_size', nb.float64),
    ('wfn', Wfn.class_type.instance_type),
    ('probability_density', nb.float64),
]


@nb.experimental.jitclass(vmc_spec)
class VMCMarkovChain:

    def __init__(self, r_e, step_size, wfn):
        """Markov chain Monte Carlo.
        :param r_e: initial position
        :param step_size: time step size
        :param wfn: instance of Wfn class
        :return:
        """
        self.r_e = r_e
        self.cond = False
        self.step_size = step_size
        self.wfn = wfn
        self.probability_density = self.wfn.value(self.r_e) ** 2

    def simple_random_step(self):
        """Simple random walker with random N-dim square proposal density in
        configuration-by-configuration sampling (CBCS).
        """
        ne = self.wfn.neu + self.wfn.ned

        next_state = self.r_e + self.step_size * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
        next_probability_density = self.wfn.value(next_state) ** 2
        self.cond = next_probability_density / self.probability_density > np.random.random()
        if self.cond:
            self.r_e, self.probability_density = next_state, next_probability_density

    def gibbs_random_step(self):
        """Simple random walker with electron-by-electron sampling (EBES)
        """
        ne = self.wfn.neu + self.wfn.ned

        next_r_e = np.copy(self.r_e)
        next_r_e[np.random.randint(ne)] += self.step_size * np.random.uniform(-1, 1, 3)
        next_probability_density = self.wfn.value(next_r_e) ** 2
        self.cond = next_probability_density / self.probability_density > np.random.random()
        if self.cond:
            self.r_e, self.probability_density = next_r_e, next_probability_density

    def biased_random_step(self):
        """Biased random walker with diffusion-drift proposed step
        diffusion step s proportional to sqrt(2*D*dt)
        drift step is proportional to D*F*dt
        where D is diffusion constant = 1/2
        """
        ne = self.wfn.neu + self.wfn.ned

        v_forth = self.wfn.drift_velocity(self.r_e)
        move = np.sqrt(self.step) * np.random.normal(0, 1, ne * 3) + self.step_size * v_forth
        next_r_e = self.r_e + move.reshape((ne, 3))
        next_probability_density = self.wfn.value(next_r_e) ** 2
        green_forth = np.exp(-np.sum((next_r_e.ravel() - self.r_e.ravel() - self.step_size * v_forth) ** 2) / 2 / self.step_size)
        green_back = np.exp(-np.sum((self.r_e.ravel() - next_r_e.ravel() - self.step_size * self.drift_velocity(next_r_e)) ** 2) / 2 / self.step_size)
        self.cond = (green_back * next_probability_density) / (green_forth * self.probability_density) > np.random.random()
        if self.cond:
            self.r_e, self.probability_density = next_r_e, next_probability_density

    def bbk_random_step(self):
        """Brünger–Brooks–Karplus (13 B. Brünger, C. L. Brooks, and M. Karplus, Chem. Phys. Lett. 105, 495 1984).
        :return: is step accepted, next step position
        """
        raise NotImplementedError

    def force_interpolation_random_step(self):
        """M. P. Allen and D. J. Tildesley, Computer Simulation of Liquids Oxford University Press, Oxford, 1989 and references in Sec. 9.3.
        :return: is step accepted, next step position
        """
        raise NotImplementedError

    def splitting_random_step(self):
        """J. A. Izaguirre, D. P. Catarello, J. M. Wozniak, and R. D. Skeel, J. Chem. Phys. 114, 2090 2001.
        :return: is step accepted, next step position
        """
        raise NotImplementedError

    def ricci_ciccottid_random_step(self):
        """A. Ricci and G. Ciccotti, Mol. Phys. 101, 1927 2003.
        :return: is step accepted, next step position
        """
        raise NotImplementedError

    def vmc_random_walk(self, steps, decorr_period):
        """Metropolis-Hastings random walk.
        :param r_e: initial electron configuration
        :param step_size: time step size
        :param steps: number of steps to walk
        :param decorr_period: number of steps to walk
        :param wfn: instance of Wfn class
        :return:
        """
        condition = np.empty(shape=(steps, ), dtype=np.bool_)
        position = np.empty(shape=(steps, ) + self.r_e.shape)

        for i in range(steps):
            for _ in range(decorr_period):
                self.simple_random_step()
            condition[i], position[i] = self.cond, self.r_e

        return condition, position


wfn_type = nb.types.float64
energy_type = nb.types.float64
r_e_type = nb.types.float64[:, :]
velocity_type = nb.types.float64[:]
dmc_spec = [
    ('r_e_list', nb.types.ListType(r_e_type)),
    ('step_size', nb.float64),
    ('wfn', Wfn.class_type.instance_type),
]


@nb.experimental.jitclass(dmc_spec)
class DMCMarkovChain:

    def __init__(self, r_e_list, step_size, wfn):
        """Markov chain Monte Carlo.
        :param r_e_list: initial positions of walkers
        :param step_size: time step size
        :param wfn: instance of Wfn class
        :return:
        """
        self.r_e_list = nb.typed.List.empty_list(r_e_type)
        [self.r_e_list.append(r_e) for r_e in r_e_list]
        self.step_size = step_size
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
        return (np.sqrt(1 + 2 * a * square_mod_v * self.step_size) - 1) / (a * square_mod_v * self.step_size)

    def dmc_random_walker(self, target_weight):
        """Collection of walkers representing the instantaneous wfn.
        C. J. Umrigar, M. P. Nightingale, K. J. Runge. A diffusion Monte Carlo algorithm with very small time-step errors.
        :param target_weight: target weight of walkers
        :return: best estimate of energy, next position
        """
        ne = self.wfn.neu + self.wfn.ned
        energy_list = []
        velocity_list = []
        wfn_value_list = []
        branching_energy_list = []
        for r_e in self.r_e_list:
            wfn_value_list.append(self.wfn.value(r_e))
            energy_list.append(self.wfn.energy(r_e))
            branching_energy_list.append(self.wfn.energy(r_e))
            velocity = self.wfn.drift_velocity(r_e)
            limiting_factor = self.limiting_factor(velocity)
            velocity_list.append(limiting_factor * velocity)
        step_eff = self.step_size
        best_estimate_energy = sum(energy_list) / len(energy_list)
        energy_t = best_estimate_energy - np.log(len(energy_list) / target_weight) / step_eff
        while True:
            sum_acceptance_probability = 0
            next_r_e_list = nb.typed.List.empty_list(r_e_type)
            next_energy_list = []
            next_velocity_list = []
            next_wfn_value_list = []
            next_branching_energy_list = []
            for r_e, wfn_value, velocity, energy, branching_energy in zip(self.r_e_list, wfn_value_list, velocity_list, energy_list, branching_energy_list):
                next_r_e = r_e + (np.sqrt(self.step_size) * np.random.normal(0, 1, ne * 3) + self.step_size * velocity).reshape((ne, 3))
                next_wfn_value = self.wfn.value(next_r_e)
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
                    green_forth = np.exp(-np.sum((next_r_e.ravel() - r_e.ravel() - self.step_size * velocity) ** 2) / 2 / self.step_size)
                    green_back = np.exp(-np.sum((r_e.ravel() - next_r_e.ravel() - self.step_size * next_velocity) ** 2) / 2 / self.step_size)
                    # condition
                    p = min(1, (green_back * next_wfn_value ** 2) / (green_forth * wfn_value ** 2))
                    cond = p >= np.random.random()
                # branching
                if cond:
                    weight = np.exp(-self.step_size * (next_branching_energy + branching_energy - 2 * energy_t) / 2)
                else:
                    weight = np.exp(-self.step_size * (branching_energy - energy_t))
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
            self.r_e_list = next_r_e_list
            energy_list = next_energy_list
            velocity_list = next_velocity_list
            wfn_value_list = next_wfn_value_list
            branching_energy_list = next_branching_energy_list
            step_eff = sum_acceptance_probability / len(energy_list) * self.step_size
            best_estimate_energy = sum(energy_list) / len(energy_list)
            energy_t = best_estimate_energy - np.log(len(energy_list) / target_weight) * self.step_size / step_eff
            yield best_estimate_energy

    def dmc_random_walk(self, steps, target_weight):
        """DMC
        :param steps: number of steps to walk
        :param target_weight: target weight
        :return:
        """
        energy = np.empty(shape=(steps,))
        walker = self.dmc_random_walker(target_weight)

        for i in range(steps):
            energy[i] = next(walker)

        return energy


# @nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def vmc_observable(condition, position, observable):
    """VMC observable.
    :param observable: observable quantity
    :param condition: accept/reject conditions
    :param position: random walk positions
    :return:
    """
    first_res = observable(position[0])
    res = np.empty(shape=condition.shape + np.shape(first_res))
    res[0] = first_res

    for i in range(1, condition.shape[0]):
        if condition[i]:
            res[i] = observable(position[i])
        else:
            res[i] = res[i-1]
    return res
