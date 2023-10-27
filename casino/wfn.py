import numpy as np
import numba as nb

from casino import delta
from casino.slater import Slater
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
    ('slater', Slater.class_type.instance_type),
    ('jastrow', nb.optional(Jastrow.class_type.instance_type)),
    ('backflow', nb.optional(Backflow.class_type.instance_type)),
    ('ppotential', nb.optional(PPotential.class_type.instance_type)),
]


@nb.experimental.jitclass(spec)
class Wfn:

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
        for i in range(self.atom_positions.shape[0] - 1):
            for j in range(i + 1, self.atom_positions.shape[0]):
                res += self.atom_charges[i] * self.atom_charges[j] / np.linalg.norm(self.atom_positions[i] - self.atom_positions[j])
        return res

    def coulomb(self, r_e) -> float:
        """Value of e-e and e-n coulomb interaction."""
        res = 0.0
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        for i in range(n_vectors.shape[0]):
            for j in range(n_vectors.shape[1]):
                res -= self.atom_charges[i] / np.linalg.norm(n_vectors[i, j])
        for i in range(e_vectors.shape[0] - 1):
            for j in range(i + 1, e_vectors.shape[1]):
                res += 1 / np.linalg.norm(e_vectors[i, j])
        if self.ppotential is not None:
            value = self.value(r_e)
            grid = self.ppotential.grid(n_vectors)
            pp_value = self.ppotential.pp_value(n_vectors)
            for atom in range(n_vectors.shape[0]):
                for i in range(self.neu + self.ned):
                    for q in range(4):
                        r_e_hatch = grid[atom, i, q, 0] + self.atom_positions[0]
                        cos_theta = r_e_hatch[i] @ r_e[i] / np.linalg.norm(r_e_hatch) / np.linalg.norm(r_e)
                        value_ratio = self.value(r_e_hatch) / value
                        for l in range(pp_value.shape[0]):
                            legendre_polynomial = 1
                            if l == 1:
                                legendre_polynomial = cos_theta
                            elif l == 2:
                                legendre_polynomial = (3 * cos_theta**2 - 1) / 2
                            res += pp_value[atom, i, l] * (2 * l + 1) * legendre_polynomial / 4 * value_ratio
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

    def energy(self, r_e) -> float:
        """Local energy.
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

        res = self.coulomb(r_e) + self.nuclear_repulsion

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
                res += 2 * T - F
            else:
                res -= s_l / 2
        else:
            s_l = self.slater.laplacian(n_vectors)
            if self.jastrow is not None:
                j_g = self.jastrow.gradient(e_vectors, n_vectors)
                j_l = self.jastrow.laplacian(e_vectors, n_vectors)
                s_g = self.slater.gradient(n_vectors)
                F = np.sum((s_g + j_g)**2) / 2
                T = (np.sum(s_g**2) - s_l - j_l) / 4
                res += 2 * T - F
            elif with_F_and_T:
                s_g = self.slater.gradient(n_vectors)
                F = np.sum(s_g**2) / 2
                T = (np.sum(s_g**2) - s_l) / 4
                res += 2 * T - F
            else:
                res -= s_l / 2
        return res

    def get_parameters(self, opt_jastrow=True, opt_backflow=True, all_parameters=False):
        """Get WFN parameters to be optimized"""
        res = np.zeros(0)
        if self.jastrow is not None and opt_jastrow:
            res = np.concatenate((
                res, self.jastrow.get_parameters(all_parameters)
            ))
        if self.backflow is not None and opt_backflow:
            res = np.concatenate((
                res, self.backflow.get_parameters(all_parameters)
            ))
        if self.slater.det_coeff.size > 1:
            res = np.concatenate((
                res, self.slater.get_parameters(all_parameters)
            ))
        return res

    def set_parameters(self, parameters, opt_jastrow=True, opt_backflow=True, all_parameters=False):
        """Update optimized parameters"""
        if self.jastrow is not None and opt_jastrow:
            parameters = self.jastrow.set_parameters(parameters, all_parameters=all_parameters)
        if self.backflow is not None and opt_backflow:
            parameters = self.backflow.set_parameters(parameters, all_parameters=all_parameters)
        if self.slater.det_coeff.size > 1:
            self.slater.set_parameters(parameters, all_parameters=all_parameters)

    def set_parameters_projector(self, opt_jastrow=True, opt_backflow=True):
        """Update optimized parameters"""
        if self.jastrow is not None and opt_jastrow:
            self.jastrow.set_parameters_projector()
        if self.backflow is not None and opt_backflow:
            self.backflow.set_parameters_projector()
        if self.slater.det_coeff.size > 1:
            self.slater.set_parameters_projector()

    def get_parameters_scale(self, opt_jastrow=True, opt_backflow=True, all_parameters=False):
        """Characteristic scale of each optimized parameter."""
        res = np.zeros(0)
        if self.jastrow is not None and opt_jastrow:
            res = np.concatenate((
                res, self.jastrow.get_parameters_scale(all_parameters)
            ))
        if self.backflow is not None and opt_backflow:
            res = np.concatenate((
                res, self.backflow.get_parameters_scale(all_parameters)
            ))
        if self.slater.det_coeff.size > 1:
            res = np.concatenate((
                res, self.slater.get_parameters_scale(all_parameters)
            ))
        return res

    def value_parameters_d1(self, r_e, opt_jastrow=True, opt_backflow=True):
        """First-order derivatives of the wave function value w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
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
        if self.slater.det_coeff.size > 1:
            if self.backflow is not None:
                n_vectors = self.backflow.value(e_vectors, n_vectors) + n_vectors
            res = np.concatenate((
                res, self.slater.value_parameters_d1(n_vectors)
            ))
        return res

    def value_parameters_d2(self, r_e, opt_jastrow=True, opt_backflow=True):
        """Second-order derivatives of the wave function value w.r.t parameters.
        1/wfn * d²wfn/dp² - 1/wfn * dwfn/dp * 1/wfn * dwfn/dp
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :return:
        """
        res = []
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None and opt_jastrow:
            res.append(self.jastrow.value_parameters_d2(e_vectors, n_vectors))
        if self.backflow is not None and opt_backflow:
            raise NotImplementedError
        return block_diag(res)

    def energy_parameters_d1(self, r_e, opt_jastrow=True, opt_backflow=True):
        """First-order derivatives of local energy w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
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
        if self.slater.det_coeff.size > 1:
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
        return -res

    def value_parameters_numerical_d1(self, r_e, opt_jastrow, opt_backflow, all_parameters=False):
        """First-order derivatives of log wfn value w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param all_parameters: optimize all parameters or only independent
        :return:
        """
        scale = self.get_parameters_scale(opt_jastrow, opt_backflow)
        parameters = self.get_parameters(opt_jastrow, opt_backflow, all_parameters)
        res = np.zeros(shape=parameters.shape)
        for i in range(parameters.size):
            parameters[i] -= delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, all_parameters)
            res[i] -= self.value(r_e) / scale[i]
            parameters[i] += 2 * delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, all_parameters)
            res[i] += self.value(r_e) / scale[i]
            parameters[i] -= delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, all_parameters)

        return res / delta / 2 / self.value(r_e)

    def energy_parameters_numerical_d1(self, r_e, opt_jastrow, opt_backflow, all_parameters=False):
        """First-order derivatives of local energy w.r.t. parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param all_parameters: optimize all parameters or only independent
        :return:
        """
        scale = self.get_parameters_scale(opt_jastrow, opt_backflow)
        parameters = self.get_parameters(opt_jastrow, opt_backflow, all_parameters)
        res = np.zeros(shape=parameters.shape)
        for i in range(parameters.size):
            parameters[i] -= delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, all_parameters)
            res[i] -= self.energy(r_e) / scale[i]
            parameters[i] += 2 * delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, all_parameters)
            res[i] += self.energy(r_e) / scale[i]
            parameters[i] -= delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, all_parameters)

        return res / delta / 2

    def numerical_gradient(self, r_e):
        """Numerical gradient of log wfn value w.r.t e-coordinates
        :param r_e: electron coordinates - array(nelec, 3)
        """
        val = self.value(r_e)
        res = np.zeros((self.neu + self.ned, 3))
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res[i, j] -= self.value(r_e)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.value(r_e)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res.ravel() / delta / 2 / val

    def numerical_laplacian(self, r_e):
        """Numerical laplacian  of log wfn value w.r.t. e-coordinates
        :param r_e: electron coordinates - array(nelec, 3)
        """
        val = self.value(r_e)
        res = - 6 * (self.neu + self.ned) * self.value(r_e)
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res += self.value(r_e)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res += self.value(r_e)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res / delta / delta / val
