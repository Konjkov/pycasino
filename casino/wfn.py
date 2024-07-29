import numpy as np
import numba as nb

from casino.abstract import AbstractWfn
from casino.slater import Slater, Slater_t
from casino.jastrow import Jastrow
from casino.backflow import Backflow
from casino.overload import block_diag
from casino.ppotential import PPotential

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('atom_positions', nb.float64[:, :]),
    ('atom_charges', nb.float64[:]),
    ('nuclear_repulsion', nb.float64),
    ('slater', Slater_t),
    ('jastrow', nb.optional(Jastrow.class_type.instance_type)),
    ('backflow', nb.optional(Backflow.class_type.instance_type)),
    ('ppotential', nb.optional(PPotential.class_type.instance_type)),
]


@nb.experimental.jitclass(spec)
class Wfn(AbstractWfn):

    def __init__(self, neu, ned, atom_positions, atom_charges, slater, jastrow, backflow, ppotential):
        """Wave function in general form.
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param atom_positions: atomic positions
        :param atom_charges: atomic charges
        :param slater: instance of Slater class
        :param jastrow: instance of Jastrow class
        :param backflow: instance of Backflow class
        :param ppotential: instance of Pseudopotential class
        :return:
        """
        self.neu = neu
        self.ned = ned
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.nuclear_repulsion = self._get_nuclear_repulsion()
        self.slater = slater
        self.jastrow = jastrow
        self.backflow = backflow
        self.ppotential = ppotential

    def _relative_coordinates(self, r_e):
        """Get relative electron coordinates
        :param r_e: electron positions
        :return: e-e vectors - array(nelec, nelec, 3), e-n vectors - array(natom, nelec, 3)
        """
        e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
        n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(self.atom_positions, 1)
        return e_vectors, n_vectors

    def _get_nuclear_repulsion(self) -> float:
        """Value of n-n repulsion."""
        res = 0.0
        for atom1 in range(self.atom_positions.shape[0] - 1):
            for atom2 in range(atom1 + 1, self.atom_positions.shape[0]):
                res += self.atom_charges[atom1] * self.atom_charges[atom2] / np.linalg.norm(self.atom_positions[atom1] - self.atom_positions[atom2])
        return res

    def coulomb(self, r_e) -> float:
        """Value of e-e and e-n coulomb interaction."""
        res = 0.0
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        # e-e coulomb interaction
        for e1 in range(e_vectors.shape[0] - 1):
            for e2 in range(e1 + 1, e_vectors.shape[1]):
                res += 1 / np.linalg.norm(e_vectors[e1, e2])
        # e-n coulomb interaction
        for atom in range(n_vectors.shape[0]):
            for e1 in range(n_vectors.shape[1]):
                res -= self.atom_charges[atom] / np.linalg.norm(n_vectors[atom, e1])
        # local channel pseudopotential
        if self.ppotential is not None:
            potential = self.ppotential.get_ppotential(n_vectors)
            for atom in range(n_vectors.shape[0]):
                if self.ppotential.is_pseudoatom[atom]:
                    for e1 in range(self.neu + self.ned):
                        res += potential[atom][e1, 2]
        return res

    def value(self, r_e) -> float:
        """Value of wave function.
        :param r_e: electron positions
        :return:
        """
        res = 1
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None:
            res *= np.exp(self.jastrow.value(e_vectors, n_vectors))
        if self.backflow is not None:
            n_vectors = self.backflow.value(e_vectors, n_vectors) + n_vectors
        res *= self.slater.value(n_vectors)
        return res

    def drift_velocity(self, r_e):
        """Drift velocity
        drift velocity = 1/2 * 'drift or quantum force'
        where D is diffusion constant = 1/2
        """
        e_vectors, n_vectors = self._relative_coordinates(r_e)

        if self.backflow is not None:
            b_g, b_v = self.backflow.gradient(e_vectors, n_vectors)
            s_g = self.slater.gradient(b_v + n_vectors)
            s_g = s_g @ b_g
            if self.jastrow is not None:
                j_g = self.jastrow.gradient(e_vectors, n_vectors)
                return s_g + j_g
            else:
                return s_g
        else:
            s_g = self.slater.gradient(n_vectors)
            if self.jastrow is not None:
                j_g = self.jastrow.gradient(e_vectors, n_vectors)
                return s_g + j_g
            else:
                return s_g

    def t_move(self, r_e, step_size):
        """T-move
        :param r_e: electron positions - array(nelec, 3)
        :param step_size: DMC step size
        :return: next electrons positions
        """
        if self.ppotential is not None:
            moved = False
            next_r_e = r_e.copy()
            for e1 in range(self.neu + self.ned):
                t_prob = [1.0]
                t_grid = [next_r_e]
                value = self.value(next_r_e)
                e_vectors, n_vectors = self._relative_coordinates(next_r_e)
                grid = self.ppotential.integration_grid(n_vectors)
                potential = self.ppotential.get_ppotential(n_vectors)
                for atom in range(n_vectors.shape[0]):
                    if self.ppotential.is_pseudoatom[atom]:
                        if potential[atom][e1, 0] or potential[atom][e1, 1]:
                            for q in range(grid.shape[2]):
                                cos_theta = (grid[atom, e1, q] @ n_vectors[atom, e1]) / (n_vectors[atom, e1] @ n_vectors[atom, e1])
                                r_e_q = next_r_e.copy()
                                r_e_q[e1] = grid[atom, e1, q] + self.atom_positions[atom]
                                value_ratio = self.value(r_e_q) / value
                                weight = self.ppotential.weight[atom][q]
                                v = 0
                                for l in range(2):
                                    v += potential[atom][e1, l] * self.ppotential.legendre(l, cos_theta) * weight * value_ratio
                                # negative probability is not possible
                                if v < 0:
                                    t_prob.append(-step_size * v)
                                    t_grid.append(r_e_q)
                t_prob = np.array(t_prob)
                i = np.searchsorted(np.cumsum(t_prob / np.sum(t_prob)), np.random.random())
                if i > 0:
                    moved = True
                    next_r_e = t_grid[i]
            return moved, next_r_e
        return False, r_e


    def local_potential(self, r_e) -> float:
        """Local potential.
        :param r_e: electron positions - array(nelec, 3)
        """
        return self.coulomb(r_e) + self.nuclear_repulsion

    def nonlocal_potential(self, r_e) -> float:
        """Nonlocal (pseudopotential) energy Wφ/φ.
        :param r_e: electron positions - array(nelec, 3)
        """
        res = 0.0
        if self.ppotential is not None:
            value = self.value(r_e)
            e_vectors, n_vectors = self._relative_coordinates(r_e)
            grid = self.ppotential.integration_grid(n_vectors)
            potential = self.ppotential.get_ppotential(n_vectors)
            for atom in range(n_vectors.shape[0]):
                if self.ppotential.is_pseudoatom[atom]:
                    for e1 in range(self.neu + self.ned):
                        if potential[atom][e1, 0] or potential[atom][e1, 1]:
                            for q in range(grid.shape[2]):
                                cos_theta = (grid[atom, e1, q] @ n_vectors[atom, e1]) / (n_vectors[atom, e1] @ n_vectors[atom, e1])
                                r_e_q = r_e.copy()
                                r_e_q[e1] = grid[atom, e1, q] + self.atom_positions[atom]
                                value_ratio = self.value(r_e_q) / value
                                weight = self.ppotential.weight[atom][q]
                                for l in range(2):
                                    res += potential[atom][e1, l] * self.ppotential.legendre(l, cos_theta) * weight * value_ratio
        return res

    def kinetic_energy(self, r_e) -> float:
        """Kinetic energy.
        :param r_e: electron coordinates - array(nelec, 3)

        if f is a scalar multivariable function and a is a vector multivariable function then:

        gradient composition rule:
        ∇(f ○ a) = (∇f ○ a) • ∇a
        where ∇a is a Jacobian matrix.
        https://towardsdatascience.com/step-by-step-the-math-behind-neural-networks-d002440227fb

        laplacian composition rule
        Δ(f ○ a) = ∇((∇f ○ a) • ∇a) = ∇(∇f ○ a) • ∇a + (∇f ○ a) • ∇²a = tr(transpose(∇a) • (∇²f ○ a) • ∇a) + (∇f ○ a) • Δa
        where ∇²f is a hessian
        tr(transpose(∇a) • (∇²f ○ a) • ∇a) - can be further simplified since cyclic property of trace
        and np.trace(A @ B) = np.sum(A * B.T) and (A @ A.T).T = A @ A.T
        :return: local energy
        """
        with_F_and_T = True
        e_vectors, n_vectors = self._relative_coordinates(r_e)

        if self.backflow is not None:
            b_l, b_g, b_v = self.backflow.laplacian(e_vectors, n_vectors)
            s_h, s_g = self.slater.hessian(b_v + n_vectors)
            s_l = np.sum(s_h * (b_g @ b_g.T)) + s_g @ b_l
            if self.jastrow is not None:
                j_g = self.jastrow.gradient(e_vectors, n_vectors)
                j_l = self.jastrow.laplacian(e_vectors, n_vectors)
                s_g = s_g @ b_g
                F = np.sum((s_g + j_g)**2) / 2
                T = (np.sum(s_g**2) - s_l - j_l) / 4
                return 2 * T - F
            else:
                return - s_l / 2
        else:
            s_l = self.slater.laplacian(n_vectors)
            if self.jastrow is not None:
                j_g = self.jastrow.gradient(e_vectors, n_vectors)
                j_l = self.jastrow.laplacian(e_vectors, n_vectors)
                s_g = self.slater.gradient(n_vectors)
                F = np.sum((s_g + j_g)**2) / 2
                T = (np.sum(s_g**2) - s_l - j_l) / 4
                return 2 * T - F
            elif with_F_and_T:
                s_g = self.slater.gradient(n_vectors)
                F = np.sum(s_g**2) / 2
                T = (np.sum(s_g**2) - s_l) / 4
                return 2 * T - F
            else:
                return - s_l / 2

    def energy(self, r_e) -> float:
        """Local energy.
        :param r_e: electron coordinates - array(nelec, 3)
        """
        return self.kinetic_energy(r_e) + self.local_potential(r_e) + self.nonlocal_potential(r_e)

    def get_parameters(self, opt_jastrow=True, opt_backflow=True, opt_det_coeff=True, all_parameters=False):
        """Get WFN parameters to be optimized
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param opt_det_coeff: optimize coefficients of the determinants
        :param all_parameters: optimize all parameters or only independent
        """
        res = np.zeros(0)
        if self.jastrow is not None and opt_jastrow:
            res = np.concatenate((
                res, self.jastrow.get_parameters(all_parameters)
            ))
        if self.backflow is not None and opt_backflow:
            res = np.concatenate((
                res, self.backflow.get_parameters(all_parameters)
            ))
        if self.slater.det_coeff.size > 1 and opt_det_coeff:
            res = np.concatenate((
                res, self.slater.get_parameters(all_parameters)
            ))
        return res

    def set_parameters(self, parameters, opt_jastrow=True, opt_backflow=True, opt_det_coeff=True, all_parameters=False):
        """Update optimized parameters
        :param parameters: parameters to update
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param opt_det_coeff: optimize coefficients of the determinants
        :param all_parameters: optimize all parameters or only independent
        """
        if self.jastrow is not None and opt_jastrow:
            parameters = self.jastrow.set_parameters(parameters, all_parameters=all_parameters)
        if self.backflow is not None and opt_backflow:
            parameters = self.backflow.set_parameters(parameters, all_parameters=all_parameters)
        if self.slater.det_coeff.size > 1 and opt_det_coeff:
            self.slater.set_parameters(parameters, all_parameters=all_parameters)

    def set_parameters_projector(self, opt_jastrow=True, opt_backflow=True, opt_det_coeff=True):
        """Update parameters projector
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param opt_det_coeff: optimize coefficients of the determinants
        """
        if self.jastrow is not None and opt_jastrow:
            self.jastrow.set_parameters_projector()
        if self.backflow is not None and opt_backflow:
            self.backflow.set_parameters_projector()
        if self.slater.det_coeff.size > 1 and opt_det_coeff:
            self.slater.set_parameters_projector()

    def get_parameters_scale(self, opt_jastrow=True, opt_backflow=True, opt_det_coeff=True, all_parameters=False):
        """Characteristic scale of each optimized parameter.
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param opt_det_coeff: optimize coefficients of the determinants
        :param all_parameters: optimize all parameters or only independent
        """
        res = np.zeros(0)
        if self.jastrow is not None and opt_jastrow:
            res = np.concatenate((
                res, self.jastrow.get_parameters_scale(all_parameters)
            ))
        if self.backflow is not None and opt_backflow:
            res = np.concatenate((
                res, self.backflow.get_parameters_scale(all_parameters)
            ))
        if self.slater.det_coeff.size > 1 and opt_det_coeff:
            res = np.concatenate((
                res, self.slater.get_parameters_scale(all_parameters)
            ))
        return res

    def value_parameters_d1(self, r_e, opt_jastrow=True, opt_backflow=True, opt_det_coeff=True):
        """First-order derivatives of the wave function value w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param opt_det_coeff: optimize coefficients of the determinants
        :return:
        """
        res = np.zeros(0)
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None and opt_jastrow:
            res = np.concatenate((
                res, self.jastrow.value_parameters_d1(e_vectors, n_vectors)
            ))
        if self.backflow is not None and opt_backflow:
            b_v = self.backflow.value(e_vectors, n_vectors) + n_vectors
            res = np.concatenate((
                res, self.backflow.value_parameters_d1(e_vectors, n_vectors) @ self.slater.gradient(b_v)
            ))
        if self.slater.det_coeff.size > 1 and opt_det_coeff:
            if self.backflow is not None:
                n_vectors = self.backflow.value(e_vectors, n_vectors) + n_vectors
            res = np.concatenate((
                res, self.slater.value_parameters_d1(n_vectors)
            ))
        return res

    def value_parameters_d2(self, r_e, opt_jastrow=True, opt_backflow=True, opt_det_coeff=True):
        """Second-order derivatives of the wave function value w.r.t parameters.
        1/wfn * d²wfn/dp² - 1/wfn * dwfn/dp * 1/wfn * dwfn/dp
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param opt_det_coeff: optimize coefficients of the determinants
        :return:
        """
        res = []
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None and opt_jastrow:
            res.append(self.jastrow.value_parameters_d2(e_vectors, n_vectors))
        if self.backflow is not None and opt_backflow:
            raise NotImplementedError
        if self.slater.det_coeff.size > 1 and opt_det_coeff:
            raise NotImplementedError
        return block_diag(res)

    def nonlocal_energy_parameters_d1(self, r_e, opt_jastrow=True, opt_backflow=True, opt_det_coeff=True):
        """First-order derivatives of pseudopotential energy w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param opt_det_coeff: optimize coefficients of the determinants
        :return:
        """
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        grid = self.ppotential.integration_grid(n_vectors)
        potential = self.ppotential.get_ppotential(n_vectors)
        value_parameters_d1 = self.value_parameters_d1(r_e, opt_jastrow, opt_backflow, opt_det_coeff)
        res = np.zeros(shape=(value_parameters_d1.size,))
        for atom in range(n_vectors.shape[0]):
            if self.ppotential.is_pseudoatom[atom]:
                for e1 in range(self.neu + self.ned):
                    if potential[atom][e1, 0] or potential[atom][e1, 1]:
                        for q in range(grid.shape[2]):
                            cos_theta = (grid[atom, e1, q] @ n_vectors[atom, e1]) / (n_vectors[atom, e1] @ n_vectors[atom, e1])
                            r_e_q = r_e.copy()
                            r_e_q[e1] = grid[atom, e1, q] + self.atom_positions[atom]
                            value_q = self.value(r_e_q)
                            value_parameters_d1_q = self.value_parameters_d1(r_e_q, opt_jastrow, opt_backflow, opt_det_coeff)
                            weight = self.ppotential.weight[atom][q]
                            for l in range(2):
                                res += potential[atom][e1, l] * self.ppotential.legendre(l, cos_theta) * weight * value_q * (value_parameters_d1_q - value_parameters_d1)
        return res / self.value(r_e)

    def energy_parameters_d1(self, r_e, opt_jastrow=True, opt_backflow=True, opt_det_coeff=True):
        """First-order derivatives of local energy w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param opt_det_coeff: optimize coefficients of the determinants
        :return:
        """
        res = np.zeros(0)
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None and opt_jastrow:
            # Jastrow parameters part
            j_g = self.jastrow.gradient(e_vectors, n_vectors)
            j_g_d1 = self.jastrow.gradient_parameters_d1(e_vectors, n_vectors)
            j_l_d1 = self.jastrow.laplacian_parameters_d1(e_vectors, n_vectors)
            if self.backflow is not None and opt_backflow:
                b_g, b_v = self.backflow.gradient(e_vectors, n_vectors)
                s_g = self.slater.gradient(b_v + n_vectors) @ b_g
            else:
                s_g = self.slater.gradient(n_vectors)
            j_d1 = j_g_d1 @ (s_g + j_g) + j_l_d1 / 2
            res = np.concatenate((res, j_d1))
        if self.backflow is not None and opt_backflow:
            # backflow parameters part
            b_l, b_g, b_v = self.backflow.laplacian(e_vectors, n_vectors)
            b_l_d1, b_g_d1, b_v_d1 = self.backflow.laplacian_parameters_d1(e_vectors, n_vectors)
            s_t, s_h, s_g = self.slater.tressian(b_v + n_vectors)
            s_g_d1 = b_v_d1 @ (s_h - np.outer(s_g, s_g))  # as hessian is d²ln(phi)/dxdy
            s_h_coordinates_d1 = s_t - np.expand_dims(np.expand_dims(s_g, 1), 2) * s_h  # d(d²ln(phi)/dydz)/dx
            s_h_d1 = (
                b_v_d1 @ s_h_coordinates_d1.reshape(s_h_coordinates_d1.shape[0], -1)
            ).reshape(b_v_d1.shape[0], s_h_coordinates_d1.shape[1], s_h_coordinates_d1.shape[2])

            parameters = self.backflow.get_parameters(all_parameters=True)
            bf_d1 = np.zeros(shape=parameters.shape)
            for i in range(parameters.size):
                bf_d1[i] += np.sum(s_h_d1[i] * (b_g @ b_g.T)) / 2
                # d(b_g @ b_g.T) = b_g_d1 @ b_g.T + b_g @ b_g_d1.T = b_g_d1 @ b_g.T + (b_g_d1 @ b_g.T).T
                # and s_h is symmetric matrix
                bf_d1[i] += np.sum(s_h * (b_g_d1[i] @ b_g.T))
                bf_d1[i] += (s_g_d1[i] @ b_l + s_g @ b_l_d1[i]) / 2
                if self.jastrow is not None:
                    j_g = self.jastrow.gradient(e_vectors, n_vectors)
                    bf_d1[i] += (s_g_d1[i] @ b_g + s_g @ b_g_d1[i]) @ j_g
            res = np.concatenate((res, bf_d1 @ self.backflow.parameters_projector))
        if self.slater.det_coeff.size > 1 and opt_det_coeff:
            # determinants coefficients part
            if self.backflow is not None:
                b_l, b_g, b_v = self.backflow.laplacian(e_vectors, n_vectors)
                s_g_d1 = self.slater.gradient_parameters_d1(b_v + n_vectors)
                s_h_d1 = self.slater.hessian_parameters_d1(b_v + n_vectors)
                s_g_d1 = s_g_d1 @ b_g
                parameters = self.slater.get_parameters(all_parameters=False)
                sl_d1 = np.zeros(shape=parameters.shape)
                for i in range(parameters.size):
                    sl_d1[i] = (np.sum(s_h_d1[i] * (b_g @ b_g.T)) + s_g_d1[i] @ b_l) / 2
            else:
                s_g_d1 = self.slater.gradient_parameters_d1(n_vectors)
                sl_d1 = self.slater.laplacian_parameters_d1(n_vectors) / 2
            if self.jastrow is not None:
                j_g = self.jastrow.gradient(e_vectors, n_vectors)
                sl_d1 += s_g_d1 @ j_g
            res = np.concatenate((res, sl_d1))
        if self.ppotential is not None:
            # pseudopotential part
            res -= self.nonlocal_energy_parameters_d1(r_e, opt_jastrow, opt_backflow, opt_det_coeff)
        return -res
