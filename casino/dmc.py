from math import erfc

import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino.mpi import Comm, Comm_t
from casino.wfn import Wfn_t


@nb.njit(nogil=True, parallel=False, cache=True)
def laplace_multivariate_distribution(zeta):
    """Sample from ζ³/π * exp(−2ζw).
    In order to sample w from ζ³/π * exp(−2ζw), we sample the cosine of the polar angle uniformly on [−1, 1],
    the azimuthal angle uniformly on [0, 2π] and the magnitude w from 4ζ³ * w² * exp(−2ζw).
    This is achieved by sampling r1, r2 and r3 uniformly on [0, 1] and setting w = − log(r1*r2*r3)/2ζ

    copy-paste from CASINO: R.J. Needs, M.D. Towler, N.D. Drummond, and P. Lopez Rios
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
    mod_xi_sin_theta_xi = mod_xi * np.sqrt(1 - cos_theta_xi**2)
    return np.array([
        mod_xi_sin_theta_xi * np.cos(phi_xi),
        mod_xi_sin_theta_xi * np.sin(phi_xi),
        mod_xi * cos_theta_xi
    ])  # fmt: skip


@structref.register
class DMC_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


efficiency_type = nb.float64
wfn_value_type = nb.float64
energy_type = nb.float64
age_type = nb.int64
weight_type = nb.float64
r_e_type = nb.float64[:, :]
velocity_type = nb.float64[:, ::1]

DMC_t = DMC_class_t(
    [
        ('mpi_size', nb.int64),
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
        ('weight_list', nb.types.ListType(weight_type)),
        ('best_estimate_energy', energy_type),
        ('energy_t', energy_type),
        ('ntransfers_tot', nb.int64),
        ('sum_acceptance_probability', nb.float64),
        ('efficiency_list', nb.types.ListType(efficiency_type)),
        ('wfn', Wfn_t),
        ('mpi_comm', Comm_t),
    ]
)


class DMC(structref.StructRefProxy):
    def __new__(cls, *args, **kwargs):
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

        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(mpi_comm, r_e_list, alimit, nucleus_gf_mods, use_tmove, step_size, target_weight, wfn, method):
            self = structref.new(DMC_t)
            self.mpi_comm = mpi_comm
            self.mpi_size = mpi_comm.Get_size()
            self.wfn = wfn
            self.method = method
            self.alimit = alimit
            self.step_size = step_size
            self.target_weight = target_weight
            self.nucleus_gf_mods = nucleus_gf_mods
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
            if self.mpi_size == 1:
                walkers = len(self.energy_list)
                energy = sum(self.energy_list) / walkers
            else:
                walkers = self.mpi_comm.allreduce(len(self.energy_list))
                energy = self.mpi_comm.allreduce(sum(self.energy_list)) / walkers
            self.step_eff = self.step_size  # first guess
            self.best_estimate_energy = energy
            self.energy_t = self.best_estimate_energy - np.log(walkers / self.target_weight) / self.step_eff
            self.ntransfers_tot = 0
            self.sum_acceptance_probability = 0
            self.efficiency_list = nb.typed.List.empty_list(efficiency_type)
            return self

        mpi_comm = Comm()
        return init(mpi_comm, *args, **kwargs)

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def best_estimate_energy(self) -> float:
        return self.best_estimate_energy

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def energy_t(self) -> float:
        return self.energy_t

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def step_eff(self) -> float:
        return self.step_eff

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def step_size(self) -> float:
        return self.step_size

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def ntransfers_tot(self) -> float:
        return self.ntransfers_tot

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def efficiency_list(self) -> list:
        return self.efficiency_list

    @nb.njit(nogil=True, parallel=False, cache=True)
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


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 'drift_diffusion')
def dmc_drift_diffusion(self):
    """Wrapper for drift-diffusion step."""

    def impl(self):
        if self.method == 1:
            self.ebe_drift_diffusion()
        elif self.method == 2:
            self.cbc_drift_diffusion()

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 'alimit_vector')
def dmc_alimit_vector(self, r_e, velocity):
    """Parameter required by DMC drift-velocity- and energy-limiting schemes
    :param r_e: electrons positions
    :param velocity: drift velocity
    :return:
    """

    def impl(self, r_e, velocity):
        ne = self.wfn.neu + self.wfn.ned
        res = np.ones(shape=(ne, 3))
        if self.nucleus_gf_mods and self.wfn.ppotential is None:
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(self.wfn.atom_positions, 1)
            r = np.sqrt(np.sum(n_vectors**2, axis=2))
            # find the nearest nucleus for each electron
            idx = np.argmin(r, axis=0)
            for i in range(ne):
                Z2_z2 = (self.wfn.atom_charges[idx[i]] * r[idx[i], i]) ** 2
                res[i] = (1 + (velocity[i] @ n_vectors[idx[i], i]) / np.linalg.norm(velocity[i]) / r[idx[i], i]) / 2 + Z2_z2 / 10 / (4 + Z2_z2)
        else:
            res *= self.alimit
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 'limiting_velocity')
def dmc_limiting_velocity(self, r_e):
    """A significant source of error in DMC calculations comes from sampling electronic
    configurations near the nodal surface. Here both the drift velocity and local
    energy diverge, causing large time step errors and increasing the variance of
    energy estimates respectively. To reduce these undesirable effects it is necessary
    to limit the magnitude of both quantities.
    :param r_e: position of walker
    :return: velocity, limiting_factor
    """

    def impl(self, r_e):
        ne = self.wfn.neu + self.wfn.ned
        drift_velocity = self.wfn.drift_velocity(r_e).reshape(ne, 3)
        a_v_t = np.sum(drift_velocity**2) * self.step_size * self.alimit_vector(r_e, drift_velocity)
        limit = (np.sqrt(1 + 2 * a_v_t) - 1) / a_v_t
        return drift_velocity * limit, drift_velocity

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 'branching_energy')
def dmc_branching_energy(self, energy, next_velocity, drift_velocity):
    """Branching energy."""

    def impl(self, energy, next_velocity, drift_velocity):
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

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 'ebe_drift_diffusion')
def dmc_ebe_drift_diffusion(self):
    """EBES drift-diffusion step."""

    def impl(self):
        for i in range(len(self.r_e_list)):
            p = 0
            moved = False
            r_e = self.r_e_list[i]
            velocity = self.velocity_list[i]
            wfn_value = self.wfn_value_list[i]
            age_p = 1.1 ** max(0, self.age_list[i] - 20)
            ne = self.wfn.neu + self.wfn.ned
            drift_velocity = self.wfn.drift_velocity(r_e).reshape(ne, 3)
            if self.nucleus_gf_mods and self.wfn.ppotential is None:
                # random step according to
                # C. J. Umrigar, M. P. Nightingale, K. J. Runge. A diffusion Monte Carlo algorithm with very small time-step errors.
                next_r_e = np.copy(r_e)
                next_wfn_value = wfn_value
                next_velocity = np.copy(velocity)
                diffuse_step = np.zeros(shape=(ne, 3))
                for e1 in range(ne):
                    n_vectors = np.expand_dims(next_r_e, 0) - np.expand_dims(self.wfn.atom_positions, 1)
                    r = np.sqrt(np.sum(n_vectors**2, axis=2))
                    # find the closest nucleus for each electron
                    idx = np.argmin(r, axis=0)
                    z = r[idx[e1], e1]
                    e_z = n_vectors[idx[e1], e1] / z
                    v_z = next_velocity[e1] @ e_z
                    v_rho_vec = next_velocity[e1] - v_z * e_z
                    z_stroke = max(z + v_z * self.step_size, 0)
                    drift_to = z_stroke * (e_z + 2 * v_rho_vec * self.step_size / (z + z_stroke)) + self.wfn.atom_positions[idx[e1]]
                    q = erfc((z + v_z * self.step_size) / np.sqrt(2 * self.step_size)) / 2
                    zeta = np.sqrt(self.wfn.atom_charges[idx[e1]] ** 2 + 1 / self.step_size)
                    interim_r_e = np.copy(next_r_e)
                    if q > np.random.random():
                        diffuse_step[e1] = laplace_multivariate_distribution(zeta)
                        interim_r_e[e1] = diffuse_step[e1] + self.wfn.atom_positions[idx[e1]]
                    else:
                        diffuse_step[e1] = np.random.normal(0, np.sqrt(self.step_size), 3)
                        interim_r_e[e1] = diffuse_step[e1] + drift_to
                    # prevent crossing nodal surface
                    interim_wfn_value = self.wfn.value(interim_r_e)
                    if np.sign(wfn_value) == np.sign(interim_wfn_value):
                        gf_forth = (1 - q) * np.exp(-np.sum((interim_r_e[e1] - drift_to) ** 2) / 2 / self.step_size) / (
                            2 * np.pi * self.step_size
                        ) ** 1.5 + q * zeta**3 / np.pi * np.exp(-2 * zeta * np.linalg.norm(interim_r_e[e1] - self.wfn.atom_positions[idx[e1]]))
                        interim_velocity, interim_drift_velocity = self.limiting_velocity(interim_r_e)
                        n_vectors = np.expand_dims(interim_r_e, 0) - np.expand_dims(self.wfn.atom_positions, 1)
                        r = np.sqrt(np.sum(n_vectors**2, axis=2))
                        # find the closest nucleus for each electron
                        idx = np.argmin(r, axis=0)
                        z = r[idx[e1], e1]
                        e_z = n_vectors[idx[e1], e1] / z
                        v_z = interim_velocity[e1] @ e_z
                        v_rho_vec = interim_velocity[e1] - v_z * e_z
                        z_stroke = max(z + v_z * self.step_size, 0)
                        drift_to = z_stroke * (e_z + 2 * v_rho_vec * self.step_size / (z + z_stroke)) + self.wfn.atom_positions[idx[e1]]
                        q = erfc((z + v_z * self.step_size) / np.sqrt(2 * self.step_size)) / 2
                        zeta = np.sqrt(self.wfn.atom_charges[idx[e1]] ** 2 + 1 / self.step_size)
                        gf_back = (1 - q) * np.exp(-np.sum((next_r_e[e1] - drift_to) ** 2) / 2 / self.step_size) / (
                            2 * np.pi * self.step_size
                        ) ** 1.5 + q * zeta**3 / np.pi * np.exp(-2 * zeta * np.linalg.norm(next_r_e[e1] - self.wfn.atom_positions[idx[e1]]))
                        p_i = min(1, age_p * (gf_back * interim_wfn_value**2) / (gf_forth * next_wfn_value**2))
                        if p_i >= np.random.random():
                            moved = True
                            next_r_e = interim_r_e
                            next_velocity = interim_velocity
                            next_wfn_value = interim_wfn_value
                            drift_velocity[e1] = interim_drift_velocity[e1]
                        # Casino manual (62)
                        p += p_i * np.sum(diffuse_step[e1] ** 2)
                    else:
                        # branching UNR (23)
                        self.age_list[i] += 1
                        self.weight_list.append(np.exp(self.step_eff * self.branching_energy_list[i]))
                        continue
            else:
                # simple random step
                next_r_e = np.copy(r_e)
                next_wfn_value = wfn_value
                next_velocity = np.copy(velocity)
                diffuse_step = np.random.normal(0, np.sqrt(self.step_size), ne * 3).reshape(ne, 3)
                for e1 in range(ne):
                    interim_r_e = np.copy(next_r_e)
                    interim_r_e[e1] += diffuse_step[e1] + self.step_size * next_velocity[e1]
                    interim_wfn_value = self.wfn.value(interim_r_e)
                    # prevent crossing nodal surface
                    if np.sign(wfn_value) == np.sign(interim_wfn_value):
                        gf_forth = np.exp(-np.sum((interim_r_e[e1] - next_r_e[e1] - self.step_size * next_velocity[e1]) ** 2) / 2 / self.step_size)
                        interim_velocity, interim_drift_velocity = self.limiting_velocity(interim_r_e)
                        gf_back = np.exp(-np.sum((next_r_e[e1] - interim_r_e[e1] - self.step_size * interim_velocity[e1]) ** 2) / 2 / self.step_size)
                        p_i = min(1, age_p * (gf_back * interim_wfn_value**2) / (gf_forth * next_wfn_value**2))
                        if p_i >= np.random.random():
                            moved = True
                            next_r_e = interim_r_e
                            next_velocity = interim_velocity
                            next_wfn_value = interim_wfn_value
                            drift_velocity[e1] = interim_drift_velocity[e1]
                        # Casino manual (62)
                        p += p_i * np.sum(diffuse_step[e1] ** 2)
                    else:
                        # branching UNR (23)
                        self.age_list[i] += 1
                        self.weight_list.append(np.exp(self.step_eff * self.branching_energy_list[i]))
                        continue
            next_energy = self.wfn.energy(next_r_e)
            next_branching_energy = self.branching_energy(next_energy, next_velocity, drift_velocity)
            # branching UNR (23)
            weight = np.exp(self.step_eff * (next_branching_energy + self.branching_energy_list[i]) / 2)
            p /= np.sum(diffuse_step**2)
            self.weight_list.append(weight)
            self.sum_acceptance_probability += p * weight
            if moved:
                self.age_list[i] = 0
                self.r_e_list[i] = next_r_e
                self.energy_list[i] = next_energy
                self.velocity_list[i] = next_velocity
                self.wfn_value_list[i] = next_wfn_value
                self.branching_energy_list[i] = next_branching_energy
            else:
                self.age_list[i] += 1

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 'cbc_drift_diffusion')
def dmc_cbc_drift_diffusion(self):
    """CBCS drift-diffusion step."""

    def impl(self):
        for i in range(len(self.r_e_list)):
            p = 0
            r_e = self.r_e_list[i]
            velocity = self.velocity_list[i]
            wfn_value = self.wfn_value_list[i]
            age_p = 1.1 ** max(0, self.age_list[i] - 50)
            ne = self.wfn.neu + self.wfn.ned
            if self.nucleus_gf_mods and self.wfn.ppotential is None:
                # random step according to
                # C. J. Umrigar, M. P. Nightingale, K. J. Runge. A diffusion Monte Carlo algorithm with very small time-step errors.
                n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(self.wfn.atom_positions, 1)
                r = np.sqrt(np.sum(n_vectors**2, axis=2))
                # find the closest nucleus for each electron
                idx = np.argmin(r, axis=0)
                gf_forth = 1
                next_r_e = np.zeros(shape=(ne, 3))
                for e1 in range(ne):
                    z = r[idx[e1], e1]
                    e_z = n_vectors[idx[e1], e1] / z
                    v_z = velocity[e1] @ e_z
                    v_rho_vec = velocity[e1] - v_z * e_z
                    z_stroke = max(z + v_z * self.step_size, 0)
                    drift_to = z_stroke * (e_z + 2 * v_rho_vec * self.step_size / (z + z_stroke)) + self.wfn.atom_positions[idx[e1]]
                    q = erfc((z + v_z * self.step_size) / np.sqrt(2 * self.step_size)) / 2
                    zeta = np.sqrt(self.wfn.atom_charges[idx[e1]] ** 2 + 1 / self.step_size)
                    if q > np.random.random():
                        next_r_e[e1] = laplace_multivariate_distribution(zeta) + self.wfn.atom_positions[idx[e1]]
                    else:
                        next_r_e[e1] = np.random.normal(0, np.sqrt(self.step_size), 3) + drift_to
                    gf_forth *= (1 - q) * np.exp(-np.sum((next_r_e[e1] - drift_to) ** 2) / 2 / self.step_size) / (
                        2 * np.pi * self.step_size
                    ) ** 1.5 + q * zeta**3 / np.pi * np.exp(-2 * zeta * np.linalg.norm(next_r_e[e1] - self.wfn.atom_positions[idx[e1]]))
                # prevent crossing nodal surface
                next_wfn_value = self.wfn.value(next_r_e)
                if np.sign(wfn_value) == np.sign(next_wfn_value):
                    next_velocity, drift_velocity = self.limiting_velocity(next_r_e)
                    n_vectors = np.expand_dims(next_r_e, 0) - np.expand_dims(self.wfn.atom_positions, 1)
                    r = np.sqrt(np.sum(n_vectors**2, axis=2))
                    # find the closest nucleus for each electron
                    idx = np.argmin(r, axis=0)
                    gf_back = 1
                    for e1 in range(ne):
                        z = r[idx[e1], e1]
                        e_z = n_vectors[idx[e1], e1] / z
                        v_z = next_velocity[e1] @ e_z
                        v_rho_vec = next_velocity[e1] - v_z * e_z
                        z_stroke = max(z + v_z * self.step_size, 0)
                        drift_to = z_stroke * (e_z + 2 * v_rho_vec * self.step_size / (z + z_stroke)) + self.wfn.atom_positions[idx[e1]]
                        q = erfc((z + v_z * self.step_size) / np.sqrt(2 * self.step_size)) / 2
                        zeta = np.sqrt(self.wfn.atom_charges[idx[e1]] ** 2 + 1 / self.step_size)
                        gf_back *= (1 - q) * np.exp(-np.sum((r_e[e1] - drift_to) ** 2) / 2 / self.step_size) / (
                            2 * np.pi * self.step_size
                        ) ** 1.5 + q * zeta**3 / np.pi * np.exp(-2 * zeta * np.linalg.norm(r_e[e1] - self.wfn.atom_positions[idx[e1]]))
                    p = min(1, age_p * (gf_back * next_wfn_value**2) / (gf_forth * wfn_value**2))
                    if p >= np.random.random():
                        next_energy = self.wfn.energy(next_r_e)
                        next_branching_energy = self.branching_energy(next_energy, next_velocity, drift_velocity)
                        # branching UNR (23)
                        weight = np.exp(self.step_eff * (next_branching_energy + self.branching_energy_list[i]) / 2)
                        self.age_list[i] = 0
                        self.r_e_list[i] = next_r_e
                        self.energy_list[i] = next_energy
                        self.velocity_list[i] = next_velocity
                        self.wfn_value_list[i] = next_wfn_value
                        self.branching_energy_list[i] = next_branching_energy
                        self.weight_list.append(weight)
                        self.sum_acceptance_probability += p * weight
                        continue
            else:
                # simple random step
                next_r_e = np.random.normal(0, np.sqrt(self.step_size), ne * 3).reshape(ne, 3) + self.step_size * velocity + r_e
                # prevent crossing nodal surface
                next_wfn_value = self.wfn.value(next_r_e)
                if np.sign(wfn_value) == np.sign(next_wfn_value):
                    gf_forth = np.exp(-np.sum((next_r_e - r_e - self.step_size * velocity) ** 2) / 2 / self.step_size)
                    next_velocity, drift_velocity = self.limiting_velocity(next_r_e)
                    gf_back = np.exp(-np.sum((r_e - next_r_e - self.step_size * next_velocity) ** 2) / 2 / self.step_size)
                    p = min(1, age_p * (gf_back * next_wfn_value**2) / (gf_forth * wfn_value**2))
                    if p >= np.random.random():
                        next_energy = self.wfn.energy(next_r_e)
                        next_branching_energy = self.branching_energy(next_energy, next_velocity, drift_velocity)
                        # branching UNR (23)
                        weight = np.exp(self.step_eff * (next_branching_energy + self.branching_energy_list[i]) / 2)
                        self.age_list[i] = 0
                        self.r_e_list[i] = next_r_e
                        self.energy_list[i] = next_energy
                        self.velocity_list[i] = next_velocity
                        self.wfn_value_list[i] = next_wfn_value
                        self.branching_energy_list[i] = next_branching_energy
                        self.weight_list.append(weight)
                        self.sum_acceptance_probability += p * weight
                        continue
            # branching UNR (23)
            weight = np.exp(self.step_eff * self.branching_energy_list[i])
            self.age_list[i] += 1
            self.weight_list.append(weight)
            self.sum_acceptance_probability += p * weight

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 'branching')
def dmc_branching(self):
    """Branching step."""

    def impl(self):
        deleted = 0
        for i in range(len(self.weight_list)):
            weight = int(self.weight_list[i] + np.random.random())
            if weight == 0:
                self.age_list.pop(i - deleted)
                self.r_e_list.pop(i - deleted)
                self.energy_list.pop(i - deleted)
                self.velocity_list.pop(i - deleted)
                self.wfn_value_list.pop(i - deleted)
                self.branching_energy_list.pop(i - deleted)
                deleted += 1
            for _ in range(1, weight):
                self.age_list.append(self.age_list[i - deleted])
                self.r_e_list.append(self.r_e_list[i - deleted])
                self.energy_list.append(self.energy_list[i - deleted])
                self.velocity_list.append(self.velocity_list[i - deleted])
                self.wfn_value_list.append(self.wfn_value_list[i - deleted])
                self.branching_energy_list.append(self.branching_energy_list[i - deleted])

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 't_move')
def dmc_t_move(self):
    """T-move."""

    def impl(self):
        if self.wfn.ppotential is not None and self.use_tmove:
            for i in range(len(self.r_e_list)):
                moved, r_e = self.wfn.t_move(self.r_e_list[i], self.step_size)
                if moved:
                    self.r_e_list[i] = r_e
                    self.wfn_value_list[i] = self.wfn.value(r_e)
                    velocity, drift_velocity = self.limiting_velocity(r_e)
                    energy = self.wfn.energy(r_e)
                    self.energy_list[i] = energy
                    self.velocity_list[i] = velocity
                    self.branching_energy_list[i] = self.branching_energy(energy, velocity, drift_velocity)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 'random_step')
def dmc_random_step(self):
    """DMC random step"""

    def impl(self):
        self.sum_acceptance_probability = 0
        self.weight_list = nb.typed.List.empty_list(weight_type)
        self.drift_diffusion()
        self.branching()
        self.t_move()

        if self.mpi_size == 1:
            walkers = len(self.energy_list)
            energy = sum(self.energy_list) / walkers
            acceptance_probability = self.sum_acceptance_probability / walkers
        else:
            walkers = self.mpi_comm.allreduce(len(self.energy_list))
            energy = self.mpi_comm.allreduce(sum(self.energy_list)) / walkers
            acceptance_probability = self.mpi_comm.allreduce(self.sum_acceptance_probability) / walkers
        self.best_estimate_energy = energy
        self.step_eff = acceptance_probability * self.step_size
        # UNR (11)
        self.energy_t = self.best_estimate_energy - np.log(walkers / self.target_weight) * self.step_size / self.step_eff

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 'redistribute_walker')
def dmc_redistribute_walker(self, from_rank, to_rank, count):
    """Redistribute count walkers from MPI from_rank to to_rank"""

    def impl(self, from_rank, to_rank, count):
        rank = self.mpi_comm.Get_rank()
        if rank in (from_rank, to_rank):
            ne = self.wfn.neu + self.wfn.ned
            age = np.empty(shape=(count,), dtype=np.int_)
            r_e = np.empty(shape=(count, ne, 3))
            energy = np.empty(shape=(count,))
            velocity = np.empty(shape=(count, ne, 3))
            wfn_value = np.empty(shape=(count,))
            branching_energy = np.empty(shape=(count,))
            if rank == from_rank:
                for i in range(count):
                    age[i] = self.age_list.pop()
                    r_e[i] = self.r_e_list.pop()
                    energy[i] = self.energy_list.pop()
                    velocity[i] = self.velocity_list.pop()
                    wfn_value[i] = self.wfn_value_list.pop()
                    branching_energy[i] = self.branching_energy_list.pop()
                self.mpi_comm.Send(age, dest=to_rank)
                self.mpi_comm.Send(r_e, dest=to_rank)
                self.mpi_comm.Send(energy, dest=to_rank)
                self.mpi_comm.Send(velocity, dest=to_rank)
                self.mpi_comm.Send(wfn_value, dest=to_rank)
                self.mpi_comm.Send(branching_energy, dest=to_rank)
            elif rank == to_rank:
                self.mpi_comm.Recv(age, source=from_rank)
                self.mpi_comm.Recv(r_e, source=from_rank)
                self.mpi_comm.Recv(energy, source=from_rank)
                self.mpi_comm.Recv(velocity, source=from_rank)
                self.mpi_comm.Recv(wfn_value, source=from_rank)
                self.mpi_comm.Recv(branching_energy, source=from_rank)
                for i in range(count):
                    self.age_list.append(age[i])
                    self.r_e_list.append(r_e[i])
                    self.energy_list.append(energy[i])
                    self.velocity_list.append(velocity[i])
                    self.wfn_value_list.append(wfn_value[i])
                    self.branching_energy_list.append(branching_energy[i])

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(DMC_class_t, 'load_balancing')
def dmc_load_balancing(self):
    """Redistribute walkers across processes."""

    def impl(self):
        if self.mpi_size == 1:
            self.efficiency_list.append(1)
        else:
            rank = self.mpi_comm.Get_rank()
            walkers = np.zeros(shape=(self.mpi_size,), dtype=np.int_)
            walkers[rank] = len(self.energy_list)
            # FIXME: use MPI.IN_PLACE
            self.mpi_comm.Allgather(walkers[rank : rank + 1], walkers)
            self.efficiency_list.append(walkers.mean() / np.max(walkers))
            # round down
            walkers = (walkers - walkers.mean()).astype(np.int_)
            self.ntransfers_tot += np.abs(walkers).sum() // 2
            rank_1 = 0
            rank_2 = 1
            while rank_2 < self.mpi_size:
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

    return impl


structref.define_boxing(DMC_class_t, DMC)
