import numpy as np
import numba as nb
import numba_mpi as nb_mpi

from math import erfc
from casino.wfn import Wfn

vmc_spec = [
    ('r_e', nb.float64[:, :]),
    ('cond', nb.int64),
    ('step_size', nb.float64),
    ('wfn', Wfn.class_type.instance_type),
    ('method', nb.int64),
    ('probability_density', nb.float64),
]


@nb.njit(nogil=True, parallel=False, cache=True)
def laplace_multivariate_distribution(zeta):
    """Sample from ζ³/π * exp(−2ζw).
    In order to sample w from ζ³/π * exp(−2ζw), we sample the cosine of the polar angle uniformly on [−1, 1],
    the azimuthal angle uniformly on [0, 2π] and the magnitude w from 4ζ³ * w² * exp(−2ζw).
    This is achieved by sampling r1, r2 and r3 uniformly on [0, 1] and setting w = − log(r1*r2*r3)/2ζ

    copy-past from CASINO: R.J. Needs, M.D. Towler, N.D. Drummond, and P. Lopez Rios
    SUBROUTINE g2_dist(zeta,xi)
      USE constants, ONLY : twopi
      IMPLICIT NONE
      REAL(dp),INTENT(in) :: zeta
      REAL(dp),INTENT(out) :: xi(3)
      REAL(dp) r1,r2,r3,mod_xi,cos_theta_xi,phi_xi,mod_xi_sin_theta_xi
      r1=ranx() ; r2=ranx() ; r3=ranx()
      mod_xi=-log(r1*r2*r3)/(2._dp*zeta)
      cos_theta_xi=1._dp-2._dp*ranx() ; phi_xi=ranx()*twopi
      mod_xi_sin_theta_xi=mod_xi*sqrt(1._dp-cos_theta_xi**2)
      xi(1)=mod_xi_sin_theta_xi*cos(phi_xi)
      xi(2)=mod_xi_sin_theta_xi*sin(phi_xi)
      xi(3)=mod_xi*cos_theta_xi
    END SUBROUTINE g2_dist
    """
    mod_xi = -np.log(np.prod(np.random.random(3))) / (2 * zeta)
    cos_theta_xi = 1 - 2 * np.random.random()
    phi_xi = 2 * np.pi * np.random.random()
    mod_xi_sin_theta_xi = mod_xi * np.sqrt(1 - cos_theta_xi ** 2)
    return np.array([
        mod_xi_sin_theta_xi * np.cos(phi_xi),
        mod_xi_sin_theta_xi * np.sin(phi_xi),
        mod_xi * cos_theta_xi
    ])


@nb.experimental.jitclass(vmc_spec)
class VMCMarkovChain:

    def __init__(self, r_e, step_size, wfn, method):
        """Markov chain Monte Carlo.
        :param r_e: initial position
        :param step_size: time step size
        :param wfn: instance of Wfn class
        :param method: vmc method: (1) - EBES (work in progress), (3) - CBCS.
        :return:
        """
        self.r_e = r_e
        self.cond = 0
        self.step_size = step_size
        self.wfn = wfn
        self.method = method
        self.probability_density = self.wfn.value(self.r_e) ** 2

    def random_step(self):
        """Wrapper"""
        if self.method == 1:
            self.gibbs_random_step()
        elif self.method == 3:
            self.simple_random_step()

    def simple_random_step(self):
        """Simple random walker with random N-dim square proposal density in
        configuration-by-configuration sampling (CBCS).
        """
        ne = self.wfn.neu + self.wfn.ned
        next_r_e = self.r_e + self.step_size * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
        next_probability_density = self.wfn.value(next_r_e) ** 2
        self.cond = next_probability_density / self.probability_density > np.random.random()
        if self.cond:
            self.r_e, self.probability_density = next_r_e, next_probability_density

    def gibbs_random_step(self):
        """Simple random walker with electron-by-electron sampling (EBES)
        """
        ne = self.wfn.neu + self.wfn.ned
        self.cond = 0
        for i in range(ne):
            next_r_e = np.copy(self.r_e)
            next_r_e[i] += self.step_size * np.random.uniform(-1, 1, 3)
            next_probability_density = self.wfn.value(next_r_e) ** 2
            cond = next_probability_density / self.probability_density > np.random.random()
            self.cond += cond
            if cond:
                self.r_e, self.probability_density = next_r_e, next_probability_density

    def biased_random_step(self):
        """Biased random walker with diffusion-drift proposed step
        diffusion step s proportional to sqrt(2*D*dt)
        drift step is proportional to D*F*dt
        where D is diffusion constant = 1/2
        """
        ne = self.wfn.neu + self.wfn.ned
        v_forth = self.wfn.drift_velocity(self.r_e)
        move = np.sqrt(self.step_size) * np.random.normal(0, 1, ne * 3) + self.step_size * v_forth
        next_r_e = self.r_e + move.reshape((ne, 3))
        next_probability_density = self.wfn.value(next_r_e) ** 2
        green_forth = np.exp(-np.sum((next_r_e.ravel() - self.r_e.ravel() - self.step_size * v_forth) ** 2) / 2 / self.step_size)
        green_back = np.exp(-np.sum((self.r_e.ravel() - next_r_e.ravel() - self.step_size * self.wfn.drift_velocity(next_r_e)) ** 2) / 2 / self.step_size)
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

    def random_walk(self, steps, decorr_period):
        """Metropolis-Hastings random walk.
        :param steps: number of steps to walk
        :param decorr_period: decorrelation period
        :return:
        """
        # reset if parameters changed
        self.probability_density = self.wfn.value(self.r_e) ** 2
        condition = np.empty(shape=(steps, ), dtype=np.int_)
        position = np.empty(shape=(steps, ) + self.r_e.shape)

        for i in range(steps):
            cond = 0
            for _ in range(decorr_period):
                self.random_step()
                cond += self.cond
            condition[i], position[i] = cond, self.r_e

        return condition, position

    def vmc_energy(self, condition, position):
        """VMC energy.
        :param condition: accept/reject conditions
        :param position: random walk positions
        :return:
        """
        # FIXME: very slow
        first_res = self.wfn.energy(position[0])
        res = np.empty(shape=condition.shape + np.shape(first_res))
        res[0] = first_res

        for i in range(1, condition.shape[0]):
            if condition[i]:
                res[i] = self.wfn.energy(position[i])
            else:
                res[i] = res[i-1]
        return res


efficiency_type = nb.types.float64
wfn_value_type = nb.types.float64
energy_type = nb.types.float64
age_type = nb.types.int64
r_e_type = nb.types.float64[:, :]
velocity_type = nb.types.float64[:, :]
dmc_spec = [
    ('method', nb.int64),
    ('alimit', nb.float64),
    ('step_size', nb.float64),
    ('step_eff', nb.float64),
    ('target_weight', nb.float64),
    ('nucleus_gf_mods', nb.boolean),
    ('use_tmove', nb.boolean),
    ('age_list', nb.types.ListType(age_type)),
    ('r_e_list', nb.types.ListType(r_e_type)),
    ('wfn_value_list', nb.types.ListType(wfn_value_type)),
    ('velocity_list', nb.types.ListType(velocity_type)),
    ('energy_list', nb.types.ListType(energy_type)),
    ('branching_energy_list', nb.types.ListType(energy_type)),
    ('best_estimate_energy', energy_type),
    ('energy_t', energy_type),
    ('ntransfers_tot', nb.int64),
    ('efficiency_list', nb.types.ListType(efficiency_type)),
    ('wfn', Wfn.class_type.instance_type),
]


@nb.experimental.jitclass(dmc_spec)
class DMCMarkovChain:

    def __init__(self, r_e_list, alimit, nucleus_gf_mods, use_tmove, step_size, target_weight, wfn, method):
        """Markov chain Monte Carlo.
        :param r_e_list: initial positions of walkers
        :param alimit: parameter required by DMC drift-velocity- and energy-limiting schemes
        :param nucleus_gf_mods:
        :param use_tmove: use T-move
        :param step_size: time step size
        :param target_weight: target weight of walkers
        :param wfn: instance of Wfn class
        :param method: dmc method: (1) - EBES, (2) - CBCS.
        :return:
        """
        self.wfn = wfn
        self.method = method
        self.alimit = alimit
        self.step_size = step_size
        self.target_weight = target_weight
        self.nucleus_gf_mods = nucleus_gf_mods and not self.wfn.ppotential.is_pseudoatom.any()
        self.use_tmove = use_tmove
        self.age_list = nb.typed.List.empty_list(age_type)
        self.r_e_list = nb.typed.List.empty_list(r_e_type)
        self.wfn_value_list = nb.typed.List.empty_list(wfn_value_type)
        self.velocity_list = nb.typed.List.empty_list(velocity_type)
        self.energy_list = nb.typed.List.empty_list(energy_type)
        self.branching_energy_list = nb.typed.List.empty_list(energy_type)
        for r_e in r_e_list:
            self.age_list.append(0)
            self.r_e_list.append(r_e)
            self.wfn_value_list.append(self.wfn.value(r_e))
            self.velocity_list.append(self.limiting_velocity(r_e)[0])
            self.energy_list.append(self.wfn.energy(r_e))
            self.branching_energy_list.append(self.wfn.energy(r_e))
        energy_list_len = np.empty(1, dtype=np.int_)
        energy_list_sum = np.empty(1, dtype=np.float_)
        nb_mpi.allreduce(len(self.energy_list), energy_list_len)
        nb_mpi.allreduce(sum(self.energy_list), energy_list_sum)
        self.step_eff = self.step_size  # first guess
        self.best_estimate_energy = energy_list_sum[0] / energy_list_len[0]
        self.energy_t = self.best_estimate_energy - np.log(energy_list_len[0] / self.target_weight) / self.step_eff
        self.ntransfers_tot = 0
        self.efficiency_list = nb.typed.List.empty_list(efficiency_type)

    def drift_diffusion(self, age, r_e, wfn_value, velocity, energy, branching_energy):
        """Wrapper for drift-diffusion step."""
        if self.method == 1:
            return self.ebe_drift_diffusion(age, r_e, wfn_value, velocity, energy, branching_energy)
        elif self.method == 2:
            return self.cbc_drift_diffusion(age, r_e, wfn_value, velocity, energy, branching_energy)

    def alimit_vector(self, r_e, velocity):
        """Parameter required by DMC drift-velocity- and energy-limiting schemes
        :param r_e: electrons positions
        :param velocity: drift velocity
        :return:
        """
        ne = self.wfn.neu + self.wfn.ned
        n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(self.wfn.atom_positions, 1)
        r = np.sqrt(np.sum(n_vectors ** 2, axis=2))
        # find the nearest nucleus for each electron
        idx = np.argmin(r, axis=0)
        res = np.empty(shape=(ne, 3))
        for i in range(ne):
            Z2_z2 = (self.wfn.atom_charges[idx[i]] * r[idx[i], i]) ** 2
            res[i] = (1 + (velocity[i] @ n_vectors[idx[i], i]) / np.linalg.norm(velocity[i]) / r[idx[i], i]) / 2 + Z2_z2 / 10 / (4 + Z2_z2)
        return res

    def limiting_velocity(self, r_e):
        """A significant source of error in DMC calculations comes from sampling electronic
        configurations near the nodal surface. Here both the drift velocity and local
        energy diverge, causing large time step errors and increasing the variance of
        energy estimates respectively. To reduce these undesirable effects it is necessary
        to limit the magnitude of both quantities.
        :param r_e: position of walker
        :return: velocity, limiting_factor
        """
        ne = self.wfn.neu + self.wfn.ned
        drift_velocity = self.wfn.drift_velocity(r_e).reshape(ne, 3)
        if self.nucleus_gf_mods:
            a_v_t = np.sum(drift_velocity**2) * self.step_size * self.alimit_vector(r_e, drift_velocity)
            velocity = drift_velocity * (np.sqrt(1 + 2 * a_v_t) - 1) / a_v_t
        else:
            a_v_t = np.sum(drift_velocity**2) * self.step_size * self.alimit
            velocity = drift_velocity * (np.sqrt(1 + 2 * a_v_t) - 1) / a_v_t
        return velocity, drift_velocity

    def t_move(self, r_e, wfn_value, velocity, energy, branching_energy):
        """T-move."""
        moved, r_e = self.wfn.t_move(r_e, self.step_size)
        if moved:
            wfn_value = self.wfn.value(r_e)
            velocity, drift_velocity = self.limiting_velocity(r_e)
            energy = self.wfn.energy(r_e)
            branching_energy = self.branching(energy, velocity, drift_velocity)
            return r_e, wfn_value, velocity, energy, branching_energy
        return r_e, wfn_value, velocity, energy, branching_energy

    def ebe_drift_diffusion(self, age, r_e, wfn_value, velocity, energy, branching_energy):
        """EBES drift-diffusion step."""
        p = 0
        age_p = 1.1 ** max(0, age - 20)
        moved = False
        ne = self.wfn.neu + self.wfn.ned
        _, drift_velocity = self.limiting_velocity(r_e)
        if self.nucleus_gf_mods:
            # random step according to
            # C. J. Umrigar, M. P. Nightingale, K. J. Runge. A diffusion Monte Carlo algorithm with very small time-step errors.
            next_r_e = np.copy(r_e)
            next_wfn_value = wfn_value
            next_velocity = np.copy(velocity)
            diffuse_step = np.zeros(shape=(ne, 3))
            for i in range(ne):
                n_vectors = np.expand_dims(next_r_e, 0) - np.expand_dims(self.wfn.atom_positions, 1)
                r = np.sqrt(np.sum(n_vectors ** 2, axis=2))
                # find the closest nucleus for each electron
                idx = np.argmin(r, axis=0)
                z = r[idx[i], i]
                e_z = n_vectors[idx[i], i] / z
                v_z = next_velocity[i] @ e_z
                v_rho_vec = next_velocity[i] - v_z * e_z
                z_stroke = max(z + v_z * self.step_size, 0)
                drift_to = z_stroke * (e_z + 2 * v_rho_vec * self.step_size / (z + z_stroke)) + self.wfn.atom_positions[idx[i]]
                q = erfc((z + v_z * self.step_size) / np.sqrt(2 * self.step_size)) / 2
                zeta = np.sqrt(self.wfn.atom_charges[idx[i]] ** 2 + 1 / self.step_size)
                interim_r_e = np.copy(next_r_e)
                if q > np.random.random():
                    diffuse_step[i] = laplace_multivariate_distribution(zeta)
                    interim_r_e[i] = diffuse_step[i] + self.wfn.atom_positions[idx[i]]
                else:
                    diffuse_step[i] = np.random.normal(0, np.sqrt(self.step_size), 3)
                    interim_r_e[i] = diffuse_step[i] + drift_to
                # prevent crossing nodal surface
                interim_wfn_value = self.wfn.value(interim_r_e)
                if np.sign(wfn_value) == np.sign(interim_wfn_value):
                    gf_forth = (
                        (1 - q) * np.exp(-np.sum((interim_r_e[i] - drift_to) ** 2) / 2 / self.step_size) / (2 * np.pi * self.step_size) ** 1.5 +
                        q * zeta ** 3 / np.pi * np.exp(-2 * zeta * np.linalg.norm(interim_r_e[i] - self.wfn.atom_positions[idx[i]]))
                    )
                    interim_velocity, interim_drift_velocity = self.limiting_velocity(interim_r_e)
                    n_vectors = np.expand_dims(interim_r_e, 0) - np.expand_dims(self.wfn.atom_positions, 1)
                    r = np.sqrt(np.sum(n_vectors ** 2, axis=2))
                    # find the closest nucleus for each electron
                    idx = np.argmin(r, axis=0)
                    z = r[idx[i], i]
                    e_z = n_vectors[idx[i], i] / z
                    v_z = interim_velocity[i] @ e_z
                    v_rho_vec = interim_velocity[i] - v_z * e_z
                    z_stroke = max(z + v_z * self.step_size, 0)
                    drift_to = z_stroke * (e_z + 2 * v_rho_vec * self.step_size / (z + z_stroke)) + self.wfn.atom_positions[idx[i]]
                    q = erfc((z + v_z * self.step_size) / np.sqrt(2 * self.step_size)) / 2
                    zeta = np.sqrt(self.wfn.atom_charges[idx[i]] ** 2 + 1 / self.step_size)
                    gf_back = (
                        (1 - q) * np.exp(-np.sum((next_r_e[i] - drift_to) ** 2) / 2 / self.step_size) / (2 * np.pi * self.step_size) ** 1.5 +
                        q * zeta ** 3 / np.pi * np.exp(-2 * zeta * np.linalg.norm(next_r_e[i] - self.wfn.atom_positions[idx[i]]))
                    )
                    p_i = min(1, age_p * (gf_back * interim_wfn_value ** 2) / (gf_forth * next_wfn_value ** 2))
                    if p_i >= np.random.random():
                        moved = True
                        next_r_e = interim_r_e
                        next_velocity = interim_velocity
                        next_wfn_value = interim_wfn_value
                        drift_velocity[i] = interim_drift_velocity[i]
                    # Casino manual (62)
                    p += p_i * np.sum(diffuse_step[i] ** 2)
                else:
                    return 0, False, r_e, wfn_value, velocity, energy, branching_energy
        else:
            # simple random step
            next_r_e = np.copy(r_e)
            next_wfn_value = wfn_value
            next_velocity = np.copy(velocity)
            diffuse_step = np.random.normal(0, np.sqrt(self.step_size), ne * 3).reshape(ne, 3)
            for i in range(ne):
                interim_r_e = np.copy(next_r_e)
                interim_r_e[i] += diffuse_step[i] + self.step_size * next_velocity[i]
                interim_wfn_value = self.wfn.value(interim_r_e)
                # prevent crossing nodal surface
                if np.sign(wfn_value) == np.sign(interim_wfn_value):
                    gf_forth = np.exp(-np.sum((interim_r_e[i] - next_r_e[i] - self.step_size * next_velocity[i]) ** 2) / 2 / self.step_size)
                    interim_velocity, interim_drift_velocity = self.limiting_velocity(interim_r_e)
                    gf_back = np.exp(-np.sum((next_r_e[i] - interim_r_e[i] - self.step_size * interim_velocity[i]) ** 2) / 2 / self.step_size)
                    p_i = min(1, age_p * (gf_back * interim_wfn_value ** 2) / (gf_forth * next_wfn_value ** 2))
                    if p_i >= np.random.random():
                        moved = True
                        next_r_e = interim_r_e
                        next_velocity = interim_velocity
                        next_wfn_value = interim_wfn_value
                        drift_velocity[i] = interim_drift_velocity[i]
                    # Casino manual (62)
                    p += p_i * np.sum(diffuse_step[i] ** 2)
                else:
                    return 0, False, r_e, wfn_value, velocity, energy, branching_energy
        next_energy = self.wfn.energy(next_r_e)
        next_branching_energy = self.branching(next_energy, next_velocity, drift_velocity)
        return p / np.sum(diffuse_step ** 2), moved, next_r_e, next_wfn_value, next_velocity, next_energy, next_branching_energy

    def cbc_drift_diffusion(self, age, r_e, wfn_value, velocity, energy, branching_energy):
        """CBCS drift-diffusion step."""
        p = 0
        age_p = 1.1 ** max(0, age - 50)
        ne = self.wfn.neu + self.wfn.ned
        if self.nucleus_gf_mods:
            # random step according to
            # C. J. Umrigar, M. P. Nightingale, K. J. Runge. A diffusion Monte Carlo algorithm with very small time-step errors.
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(self.wfn.atom_positions, 1)
            r = np.sqrt(np.sum(n_vectors ** 2, axis=2))
            # find the closest nucleus for each electron
            idx = np.argmin(r, axis=0)
            gf_forth = 1
            next_r_e = np.zeros(shape=(ne, 3))
            for i in range(ne):
                z = r[idx[i], i]
                e_z = n_vectors[idx[i], i] / z
                v_z = velocity[i] @ e_z
                v_rho_vec = velocity[i] - v_z * e_z
                z_stroke = max(z + v_z * self.step_size, 0)
                drift_to = z_stroke * (e_z + 2 * v_rho_vec * self.step_size / (z + z_stroke)) + self.wfn.atom_positions[idx[i]]
                q = erfc((z + v_z * self.step_size) / np.sqrt(2 * self.step_size)) / 2
                zeta = np.sqrt(self.wfn.atom_charges[idx[i]] ** 2 + 1 / self.step_size)
                if q > np.random.random():
                    next_r_e[i] = laplace_multivariate_distribution(zeta) + self.wfn.atom_positions[idx[i]]
                else:
                    next_r_e[i] = np.random.normal(0, np.sqrt(self.step_size), 3) + drift_to
                gf_forth *= (
                    (1 - q) * np.exp(-np.sum((next_r_e[i] - drift_to) ** 2) / 2 / self.step_size) / (2 * np.pi * self.step_size) ** 1.5 +
                    q * zeta ** 3 / np.pi * np.exp(-2 * zeta * np.linalg.norm(next_r_e[i] - self.wfn.atom_positions[idx[i]]))
                )
            # prevent crossing nodal surface
            next_wfn_value = self.wfn.value(next_r_e)
            if np.sign(wfn_value) == np.sign(next_wfn_value):
                next_velocity, drift_velocity = self.limiting_velocity(next_r_e)
                n_vectors = np.expand_dims(next_r_e, 0) - np.expand_dims(self.wfn.atom_positions, 1)
                r = np.sqrt(np.sum(n_vectors ** 2, axis=2))
                # find the closest nucleus for each electron
                idx = np.argmin(r, axis=0)
                gf_back = 1
                for i in range(ne):
                    z = r[idx[i], i]
                    e_z = n_vectors[idx[i], i] / z
                    v_z = next_velocity[i] @ e_z
                    v_rho_vec = next_velocity[i] - v_z * e_z
                    z_stroke = max(z + v_z * self.step_size, 0)
                    drift_to = z_stroke * (e_z + 2 * v_rho_vec * self.step_size / (z + z_stroke)) + self.wfn.atom_positions[idx[i]]
                    q = erfc((z + v_z * self.step_size) / np.sqrt(2 * self.step_size)) / 2
                    zeta = np.sqrt(self.wfn.atom_charges[idx[i]] ** 2 + 1 / self.step_size)
                    gf_back *= (
                        (1 - q) * np.exp(-np.sum((r_e[i] - drift_to) ** 2) / 2 / self.step_size) / (2 * np.pi * self.step_size) ** 1.5 +
                        q * zeta ** 3 / np.pi * np.exp(-2 * zeta * np.linalg.norm(r_e[i] - self.wfn.atom_positions[idx[i]]))
                    )
                p = min(1, age_p * (gf_back * next_wfn_value ** 2) / (gf_forth * wfn_value ** 2))
                if p >= np.random.random():
                    next_energy = self.wfn.energy(next_r_e)
                    next_branching_energy = self.branching(next_energy, next_velocity, drift_velocity)
                    return p, True, next_r_e, next_wfn_value, next_velocity, next_energy, next_branching_energy
        else:
            # simple random step
            next_r_e = np.random.normal(0, np.sqrt(self.step_size), ne * 3).reshape(ne, 3) + self.step_size * velocity + r_e
            # prevent crossing nodal surface
            next_wfn_value = self.wfn.value(next_r_e)
            if np.sign(wfn_value) == np.sign(next_wfn_value):
                gf_forth = np.exp(-np.sum((next_r_e - r_e - self.step_size * velocity) ** 2) / 2 / self.step_size)
                next_velocity, drift_velocity = self.limiting_velocity(next_r_e)
                gf_back = np.exp(-np.sum((r_e - next_r_e - self.step_size * next_velocity) ** 2) / 2 / self.step_size)
                p = min(1, age_p * (gf_back * next_wfn_value ** 2) / (gf_forth * wfn_value ** 2))
                if p >= np.random.random():
                    next_energy = self.wfn.energy(next_r_e)
                    next_branching_energy = self.branching(next_energy, next_velocity, drift_velocity)
                    return p, True, next_r_e, next_wfn_value, next_velocity, next_energy, next_branching_energy
        return p, False, r_e, wfn_value, velocity, energy, branching_energy

    def branching(self, energy, next_velocity, drift_velocity):
        """Branching step."""
        ne = self.wfn.neu + self.wfn.ned
        limiting_factor = np.linalg.norm(next_velocity) / np.linalg.norm(drift_velocity)
        # Andrea Zen, Sandro Sorella, Michael J. Gillan, Angelos Michaelides and Dario Alfè
        # Boosting the accuracy and speed of quantum Monte Carlo: Size consistency and time step
        if np.abs(self.best_estimate_energy - energy) > 0.2 * np.sqrt(ne / self.step_size):
            E_cut = np.sign(self.best_estimate_energy - energy) * 0.2 * np.sqrt(ne / self.step_size)
        else:
            E_cut = self.best_estimate_energy - energy
        # branching UNR (39)
        return (self.energy_t - self.best_estimate_energy) + E_cut * limiting_factor

    def random_step(self):
        """Random step"""
        sum_acceptance_probability = 0
        next_age_list = nb.typed.List.empty_list(age_type)
        next_r_e_list = nb.typed.List.empty_list(r_e_type)
        next_wfn_value_list = nb.typed.List.empty_list(wfn_value_type)
        next_velocity_list = nb.typed.List.empty_list(velocity_type)
        next_energy_list = nb.typed.List.empty_list(energy_type)
        next_branching_energy_list = nb.typed.List.empty_list(energy_type)
        for age, r_e, wfn_value, velocity, energy, branching_energy in zip(self.age_list, self.r_e_list, self.wfn_value_list, self.velocity_list, self.energy_list, self.branching_energy_list):
            p, moved, next_r_e, next_wfn_value, next_velocity, next_energy, next_branching_energy = self.drift_diffusion(age, r_e, wfn_value, velocity, energy, branching_energy)
            # branching UNR (23)
            weight = np.exp(self.step_eff * (next_branching_energy + branching_energy) / 2)
            for _ in range(int(weight + np.random.random())):
                sum_acceptance_probability += p
                next_age_list.append(0 if moved else age + 1)
                next_r_e_list.append(next_r_e)
                next_energy_list.append(next_energy)
                next_velocity_list.append(next_velocity)
                next_wfn_value_list.append(next_wfn_value)
                next_branching_energy_list.append(next_branching_energy)
        if self.wfn.ppotential.is_pseudoatom.any() and self.use_tmove:
            for i in range(len(next_r_e_list)):
                next_r_e_list[i], next_wfn_value_list[i], next_velocity_list[i], next_energy_list[i], next_branching_energy_list[i] = self.t_move(
                    next_r_e_list[i], next_wfn_value_list[i], next_velocity_list[i], next_energy_list[i], next_branching_energy_list[i])
        self.age_list = next_age_list
        self.r_e_list = next_r_e_list
        self.energy_list = next_energy_list
        self.velocity_list = next_velocity_list
        self.wfn_value_list = next_wfn_value_list
        self.branching_energy_list = next_branching_energy_list
        energy_list_len = np.empty(1, dtype=np.int_)
        energy_list_sum = np.empty(1, dtype=np.float_)
        total_sum_acceptance_probability = np.empty(1, dtype=np.float_)
        nb_mpi.allreduce(len(self.energy_list), energy_list_len)
        nb_mpi.allreduce(sum(self.energy_list), energy_list_sum)
        nb_mpi.allreduce(sum_acceptance_probability, total_sum_acceptance_probability)
        self.step_eff = total_sum_acceptance_probability[0] / energy_list_len[0] * self.step_size
        self.best_estimate_energy = energy_list_sum[0] / energy_list_len[0]
        # UNR (11)
        self.energy_t = self.best_estimate_energy - np.log(energy_list_len[0] / self.target_weight) * self.step_size / self.step_eff

    def redistribute_walker(self, from_rank, to_rank, count):
        """Redistribute count walkers from MPI from_rank to to_rank"""
        rank = nb_mpi.rank()
        if rank in (from_rank, to_rank):
            ne = self.wfn.neu + self.wfn.ned
            r_e = np.empty(shape=(count, ne, 3))
            energy = np.empty(shape=(count,))
            velocity = np.empty(shape=(count, ne, 3))
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
            self.efficiency_list.append(1)
            return
        rank = nb_mpi.rank()
        walkers = np.zeros(shape=(nb_mpi.size(),), dtype=np.int_)
        walkers[rank] = len(self.energy_list)
        # FIXME: use MPI_IN_PLACE
        nb_mpi.allgather(walkers[rank:rank+1], walkers, 1)
        self.efficiency_list.append(walkers.mean() / np.max(walkers))
        # round down
        walkers = (walkers - walkers.mean()).astype(np.int_)
        self.ntransfers_tot += np.abs(walkers).sum() // 2
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
        :return: energy, number of config transfers
        """
        self.ntransfers_tot = 0
        self.efficiency_list = nb.typed.List.empty_list(efficiency_type)
        energy = np.empty(shape=(steps,))

        for i in range(steps):
            self.random_step()
            energy[i] = self.best_estimate_energy
            if i % 500 == 0:
                self.load_balancing()

        return energy


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
            res[i] = observable(position[i], *args)
        else:
            res[i] = res[i-1]
    return res
