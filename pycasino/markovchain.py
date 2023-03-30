from numpy_config import np
from math import erfc
import numba as nb
import numba_mpi as nb_mpi

from pycasino.overload import subtract_outer
from wfn import Wfn

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
        """
        raise NotImplementedError

    def force_interpolation_random_step(self):
        """M. P. Allen and D. J. Tildesley, Computer Simulation of Liquids Oxford University Press, Oxford, 1989 and references in Sec. 9.3.
        """
        raise NotImplementedError

    def splitting_random_step(self):
        """J. A. Izaguirre, D. P. Catarello, J. M. Wozniak, and R. D. Skeel, J. Chem. Phys. 114, 2090 2001.
        """
        raise NotImplementedError

    def ricci_ciccotti_random_step(self):
        """A. Ricci and G. Ciccotti, Mol. Phys. 101, 1927 2003.
        """
        raise NotImplementedError

    def random_walk(self, steps, decorr_period=1):
        """Metropolis-Hastings random walk.
        :param steps: number of steps to walk
        :param decorr_period: decorrelation period
        :return:
        """
        self.probability_density = self.wfn.value(self.r_e) ** 2
        condition = np.empty(shape=(steps, ), dtype=np.bool_)
        position = np.empty(shape=(steps, ) + self.r_e.shape)

        for i in range(steps):
            cond = False
            for _ in range(decorr_period):
                self.simple_random_step()
                cond |= self.cond
            condition[i], position[i] = cond, self.r_e

        return condition, position


wfn_value_type = nb.types.float64
energy_type = nb.types.float64
r_e_type = nb.types.float64[:, :]
velocity_type = nb.types.float64[:]
dmc_spec = [
    ('alimit', nb.float64),
    ('step_size', nb.float64),
    ('target_weight', nb.float64),
    ('nucleus_gf_mods', nb.boolean),
    ('r_e_list', nb.types.ListType(r_e_type)),
    ('wfn_value_list', nb.types.ListType(wfn_value_type)),
    ('velocity_list', nb.types.ListType(velocity_type)),
    ('energy_list', nb.types.ListType(energy_type)),
    ('branching_energy_list', nb.types.ListType(energy_type)),
    ('best_estimate_energy', energy_type),
    ('energy_t', energy_type),
    ('wfn', Wfn.class_type.instance_type),
]


@nb.experimental.jitclass(dmc_spec)
class DMCMarkovChain:

    def __init__(self, r_e_list, alimit, nucleus_gf_mods, step_size, target_weight, wfn):
        """Markov chain Monte Carlo.
        :param r_e_list: initial positions of walkers
        :param alimit: parameter required by DMC drift-velocity- and energy-limiting schemes
        :param step_size: time step size
        :param target_weight: target weight of walkers
        :param wfn: instance of Wfn class
        :return:
        """
        self.wfn = wfn
        self.alimit = alimit
        self.step_size = step_size
        self.target_weight = target_weight
        self.nucleus_gf_mods = nucleus_gf_mods
        self.r_e_list = nb.typed.List.empty_list(r_e_type)
        self.wfn_value_list = nb.typed.List.empty_list(wfn_value_type)
        self.velocity_list = nb.typed.List.empty_list(velocity_type)
        self.energy_list = nb.typed.List.empty_list(energy_type)
        self.branching_energy_list = nb.typed.List.empty_list(energy_type)
        for r_e in r_e_list:
            self.r_e_list.append(r_e)
            self.wfn_value_list.append(self.wfn.value(r_e))
            drift_velocity = self.wfn.drift_velocity(r_e)
            self.velocity_list.append(self.limiting_velocity(r_e, drift_velocity))
            self.energy_list.append(self.wfn.energy(r_e))
            self.branching_energy_list.append(self.wfn.energy(r_e))
        energy_list_len = np.empty(1, dtype=np.int64)
        energy_list_sum = np.empty(1, dtype=np.float64)
        nb_mpi.allreduce(len(self.energy_list), energy_list_len)
        nb_mpi.allreduce(sum(self.energy_list), energy_list_sum)
        step_eff = self.step_size  # first guess
        self.best_estimate_energy = energy_list_sum[0] / energy_list_len[0]
        self.energy_t = self.best_estimate_energy - np.log(energy_list_len[0] / self.target_weight) / step_eff

    def alimit_vector(self, r_e, velocity):
        """Parameter required by DMC drift-velocity- and energy-limiting schemes
        :param r_e: electrons positions
        :param velocity: drift velocity
        :return:
        """
        ne = self.wfn.neu + self.wfn.ned
        n_vectors = -subtract_outer(self.wfn.atom_positions, r_e)
        # FIXME: multiple nuclei
        e = n_vectors[0]
        v = velocity.reshape(ne, 3)
        res = np.empty(shape=(ne, 3))
        for i in range(ne):
            Z2_z2 = (self.wfn.atom_charges[0] * np.linalg.norm(e[i])) ** 2
            res[i] = (1 + (v[i] @ e[i]) / np.linalg.norm(v[i]) / np.linalg.norm(e[i])) / 2 + Z2_z2 / 10 / (4 + Z2_z2)
        return res.ravel()

    def limiting_velocity(self, r_e, velocity):
        """A significant source of error in DMC calculations comes from sampling electronic
        configurations near the nodal surface. Here both the drift velocity and local
        energy diverge, causing large time step errors and increasing the variance of
        energy estimates respectively. To reduce these undesirable effects it is necessary
        to limit the magnitude of both quantities.
        :param r_e: position of walker
        :param velocity: drift velocity
        :return: limiting_factor * velocity
        """
        if self.nucleus_gf_mods:
            a_v_t = np.sum(velocity**2) * self.step_size * self.alimit_vector(r_e, velocity)
            res = velocity * (np.sqrt(1 + 2 * a_v_t) - 1) / a_v_t
        else:
            a_v_t = np.sum(velocity**2) * self.step_size * self.alimit
            res = velocity * (np.sqrt(1 + 2 * a_v_t) - 1) / a_v_t
        return res

    def drift_diffusion_step(self, r_e, velocity):
        """Drift-diffusion step."""
        ne = self.wfn.neu + self.wfn.ned
        v = np.ascontiguousarray(velocity).reshape(ne, 3)
        # v = velocity.reshape(ne, 3)
        if self.nucleus_gf_mods:
            # FIXME: multiple nuclei
            n_vectors = -subtract_outer(self.wfn.atom_positions, r_e)
            e = n_vectors[0]
            drift = np.zeros(shape=(ne, 3))
            diffusion = np.zeros(shape=(ne, 3))
            for i in range(ne):
                z = np.linalg.norm(e[i])
                e_z = e[i] / z
                v_z = v[i] @ e_z
                v_rho_vec = v[i] - v_z * e_z
                v_rho = np.linalg.norm(v_rho_vec)
                e_rho = v_rho_vec / v_rho
                z_stroke = max(z + v_z * self.step_size, 0)
                rho_stroke = 2 * v_rho * self.step_size * z_stroke / (z + z_stroke)
                drift[i] -= e[i]
                if erfc((z + v_z * self.step_size) / np.sqrt(2 * self.step_size)) / 2 < np.random.random():
                    # probability p = 1 - q
                    drift[i] = z_stroke * e_z + rho_stroke * e_rho
                    diffusion[i] = np.random.normal(0, np.sqrt(self.step_size), 3)
                else:
                    # probability q = erfc((z + v_z * self.step_size) / np.sqrt(2 * self.step_size)) / 2
                    zeta = np.sqrt(self.wfn.atom_charges[0] ** 2 + 1/self.step_size)
                    diffusion[i] = np.random.laplace(0, 1/(2 * zeta), 3)
        else:
            drift = self.step_size * v
            diffusion = np.random.normal(0, np.sqrt(self.step_size), ne * 3).reshape(ne, 3)
        return drift, diffusion

    def unr_random_step(self):
        """DMC random step according to
        C. J. Umrigar, M. P. Nightingale, K. J. Runge. A diffusion Monte Carlo algorithm with very small time-step errors.
        """
        sum_acceptance_probability = 0
        next_r_e_list = nb.typed.List.empty_list(r_e_type)
        next_wfn_value_list = nb.typed.List.empty_list(wfn_value_type)
        next_velocity_list = nb.typed.List.empty_list(velocity_type)
        next_energy_list = nb.typed.List.empty_list(energy_type)
        next_branching_energy_list = nb.typed.List.empty_list(energy_type)
        for r_e, wfn_value, velocity, energy, branching_energy in zip(self.r_e_list, self.wfn_value_list, self.velocity_list, self.energy_list, self.branching_energy_list):
            drift, diffusion = self.drift_diffusion_step(r_e, velocity)
            next_r_e = r_e + drift + diffusion
            next_wfn_value = self.wfn.value(next_r_e)
            # prevent crossing nodal surface
            cond = np.sign(wfn_value) == np.sign(next_wfn_value)
            next_energy = self.wfn.energy(next_r_e)
            next_drift_velocity = self.wfn.drift_velocity(next_r_e)
            next_velocity = self.limiting_velocity(r_e, next_drift_velocity)
            next_branching_energy = self.best_estimate_energy - (self.best_estimate_energy - next_energy) * np.linalg.norm(next_velocity) / np.linalg.norm(next_drift_velocity)
            p = 0
            if cond:
                # Green`s functions
                next_drift, _ = self.drift_diffusion_step(next_r_e, next_velocity)
                green_forth = np.exp(-np.sum((next_r_e - r_e - drift) ** 2) / 2 / self.step_size)
                green_back = np.exp(-np.sum((r_e - next_r_e - next_drift) ** 2) / 2 / self.step_size)
                # condition
                p = min(1, (green_back * next_wfn_value ** 2) / (green_forth * wfn_value ** 2))
                cond = p >= np.random.random()
            # branching
            if cond:
                weight = np.exp(-self.step_size * (next_branching_energy + branching_energy - 2 * self.energy_t) / 2)
            else:
                weight = np.exp(-self.step_size * (branching_energy - self.energy_t))
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
        self.energy_list = next_energy_list
        self.velocity_list = next_velocity_list
        self.wfn_value_list = next_wfn_value_list
        self.branching_energy_list = next_branching_energy_list
        energy_list_len = np.empty(1, dtype=np.int64)
        energy_list_sum = np.empty(1, dtype=np.float64)
        nb_mpi.allreduce(len(self.energy_list), energy_list_len)
        nb_mpi.allreduce(sum(self.energy_list), energy_list_sum)
        step_eff = sum_acceptance_probability / energy_list_len[0] * self.step_size
        self.best_estimate_energy = energy_list_sum[0] / energy_list_len[0]
        self.energy_t = self.best_estimate_energy - np.log(energy_list_len[0] / self.target_weight) * self.step_size / step_eff

    def redistribute_walker(self, from_rank, to_rank, count):
        """Redistribute count walkers from MPI from_rank to to_rank"""
        rank = nb_mpi.rank()
        if rank in (from_rank, to_rank):
            ne = self.wfn.neu + self.wfn.ned
            r_e = np.empty(shape=(count, ne, 3))
            energy = np.empty(shape=(count,))
            velocity = np.empty(shape=(count, ne * 3))
            wfn_value = np.empty(shape=(count,))
            branching_energy = np.empty(shape=(count, ))
            if rank == from_rank:
                for i in range(count):
                    r_e[i] = self.r_e_list.pop()
                    energy[i] = self.energy_list.pop()
                    velocity[i] = self.velocity_list.pop()
                    wfn_value[i] = self.wfn_value_list.pop()
                    branching_energy[i] = self.branching_energy_list.pop()
                nb_mpi.send(r_e, dest=to_rank)
                nb_mpi.send(energy, dest=to_rank)
                nb_mpi.send(velocity, dest=to_rank)
                nb_mpi.send(wfn_value, dest=to_rank)
                nb_mpi.send(branching_energy, dest=to_rank)
            elif rank == to_rank:
                nb_mpi.recv(r_e, source=from_rank)
                nb_mpi.recv(energy, source=from_rank)
                nb_mpi.recv(velocity, source=from_rank)
                nb_mpi.recv(wfn_value, source=from_rank)
                nb_mpi.recv(branching_energy, source=from_rank)
                for i in range(count):
                    self.r_e_list.append(r_e[i])
                    self.energy_list.append(energy[i])
                    self.velocity_list.append(velocity[i])
                    self.wfn_value_list.append(wfn_value[i])
                    self.branching_energy_list.append(branching_energy[i])

    def load_balancing(self):
        """Redistribute walkers across processes."""
        if nb_mpi.size() == 1:
            return
        rank = nb_mpi.rank()
        walkers = np.zeros(shape=(nb_mpi.size(),), dtype=np.int64)
        walkers[rank] = len(self.energy_list)
        if rank == 0:
            for source in range(1, nb_mpi.size()):
                nb_mpi.recv(walkers[source:source+1], source=source)
        else:
            nb_mpi.send(walkers[rank:rank+1], dest=0)
        nb_mpi.bcast(walkers, root=0)

        walkers = (walkers - walkers.mean()).astype(np.int64)
        rank_1 = 0
        rank_2 = 1
        while rank_2 < nb_mpi.size():
            count = min(abs(walkers[rank_1]), abs(walkers[rank_2]))
            if walkers[rank_1] > 0 > walkers[rank_2]:
                self.redistribute_walker(rank_1, rank_2, count)
                walkers[rank_1] -= count
                walkers[rank_2] += count
            elif walkers[rank_2] > 0 > walkers[rank_1]:
                self.redistribute_walker(rank_2, rank_1, count)
                walkers[rank_2] -= count
                walkers[rank_1] += count
            else:
                rank_2 += 1
            if walkers[rank_1] == 0:
                rank_1 += 1
            if walkers[rank_2] == 0:
                rank_2 += 1

    def random_walk(self, steps):
        """DMC random walk.
        :param steps: number of steps to walk
        :return: energy
        """
        energy = np.empty(shape=(steps,))

        for i in range(steps):
            self.unr_random_step()
            energy[i] = self.best_estimate_energy
            if i % 500 == 0:
                self.load_balancing()

        return energy


# @nb.jit(nopython=True, nogil=True, parallel=False, cache=True)
def vmc_observable(condition, position, observable, *args):
    """VMC observable.
    :param observable: observable quantity
    :param condition: accept/reject conditions
    :param position: random walk positions
    :return:
    """
    first_res = observable(position[0], *args)
    res = np.empty(shape=condition.shape + np.shape(first_res))
    res[0] = first_res

    for i in range(1, condition.shape[0]):
        if condition[i]:
            res[i] = observable(position[i])
        else:
            res[i] = res[i-1]
    return res
