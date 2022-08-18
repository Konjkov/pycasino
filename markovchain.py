
import os
from wfn import Wfn

os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import numpy as np
import numba as nb

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('step', nb.float64),
    ('atom_positions', nb.float64[:, :]),
    ('atom_charges', nb.float64[:]),
    ('wfn', Wfn.class_type.instance_type),
]


@nb.jit(nopython=True, nogil=True, parallel=False)
def sum_typed_list(x):
    """Mixed estimator of energy
    Для проверки утечек памяти
    """
    sum_e = 0.0
    for e in x:
        sum_e += e
    return sum_e


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

    def vmc_random_walk(self, r_e, steps, decorr_period):
        """Metropolis-Hastings random walk.
        :param r_e: initial electron configuration
        :param steps: number of steps to walk
        :param decorr_period: number of steps to walk
        :return:
        """
        condition = np.zeros(shape=(steps, ), dtype=nb.boolean)
        position = np.zeros(shape=(steps, ) + r_e.shape)
        walker = self.walker(r_e, decorr_period)

        for i in range(steps):
            cond, r_e = next(walker)
            condition[i] = cond
            position[i] = r_e

        return condition, position

    def dmc_random_walk(self, r_e, steps, target_weight):
        """DMC
        :param r_e: initial electron configuration
        :param steps: number of steps to walk
        :param target_weight: target weight
        :return:
        """
        energy = np.zeros(shape=(steps, ))
        walker = self.dmc_random_walker(r_e, target_weight)

        for i in range(steps):
            energy_t, r_e = next(walker)
            energy[i] = energy_t

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

    def profiling_simple_random_walk(self, steps, r_initial, decorr_period):
        walker = self.simple_random_walker(r_initial, decorr_period)
        for _ in range(steps):
            next(walker)
