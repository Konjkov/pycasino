from numpy_config import np
import numba as nb

from slater import Slater
from jastrow import Jastrow
from backflow import Backflow
from overload import subtract_outer

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('atom_positions', nb.float64[:, :]),
    ('atom_charges', nb.float64[:]),
    ('nuclear_repulsion', nb.float64),
    ('slater', Slater.class_type.instance_type),
    ('jastrow', nb.optional(Jastrow.class_type.instance_type)),
    ('backflow', nb.optional(Backflow.class_type.instance_type)),
]


@nb.experimental.jitclass(spec)
class Wfn:

    def __init__(self, neu, ned, atom_positions, atom_charges, slater, jastrow, backflow):
        """Wave function in general form.
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param atom_positions: atomic positions
        :param atom_charges: atomic charges
        :param slater: instance of Slater class
        :param jastrow: instance of Jastrow class
        :param backflow: instance of Backflow class
        :return:
        """
        self.neu = neu
        self.ned = ned
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.nuclear_repulsion = self.get_nuclear_repulsion()
        self.slater = slater
        self.jastrow = jastrow
        self.backflow = backflow

    def _relative_coordinates(self, r_e):
        """Get relative electron coordinates
        :param r_e: electron positions
        :return: e-e vectors - array(nelec, nelec, 3), e-n vectors - array(natom, nelec, 3)
        """
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)
        return e_vectors, n_vectors

    def get_nuclear_repulsion(self) -> float:
        """Value of n-n repulsion."""
        res = 0.0
        for i in range(self.atom_positions.shape[0] - 1):
            for j in range(i + 1, self.atom_positions.shape[0]):
                res += self.atom_charges[i] * self.atom_charges[j] / np.linalg.norm(self.atom_positions[i] - self.atom_positions[j])
        return res

    def coulomb(self, e_vectors, n_vectors) -> float:
        """Value of e-e and e-n coulomb interaction."""
        res = 0.0
        for i in range(n_vectors.shape[0]):
            for j in range(n_vectors.shape[1]):
                res -= self.atom_charges[i] / np.linalg.norm(n_vectors[i, j])
        for i in range(e_vectors.shape[0] - 1):
            for j in range(i + 1, e_vectors.shape[1]):
                res += 1 / np.linalg.norm(e_vectors[i, j])
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
            n_vectors = self.backflow.value(e_vectors, n_vectors)
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
            s_g = self.slater.gradient(b_v)
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

        res = self.coulomb(e_vectors, n_vectors)

        if self.backflow is not None:
            b_l, b_g, b_v = self.backflow.laplacian(e_vectors, n_vectors)
            s_g = self.slater.gradient(b_v)
            s_h = self.slater.hessian(b_v)
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

    def get_parameters(self, opt_jastrow=True, opt_backflow=True):
        """Get WFN parameters to be optimized"""
        res = np.zeros(0)
        if self.jastrow is not None and opt_jastrow:
            res = np.concatenate((
                res, self.jastrow.get_parameters()
            ))
        if self.backflow is not None and opt_backflow:
            res = np.concatenate((
                res, self.backflow.get_parameters()
            ))
        return res

    def set_parameters(self, parameters, opt_jastrow=True, opt_backflow=True):
        """Update optimized parameters"""
        if self.jastrow is not None and opt_jastrow:
            parameters = self.jastrow.set_parameters(parameters)
        if self.backflow is not None and opt_backflow:
            self.backflow.set_parameters(parameters)

    def get_parameters_scale(self, opt_jastrow=True, opt_backflow=True):
        """Characteristic scale of each optimized parameter."""
        res = np.zeros(0)
        if self.jastrow is not None and opt_jastrow:
            res = np.concatenate((
                res, np.concatenate(self.jastrow.get_parameters_scale())
            ))
        if self.backflow is not None and opt_backflow:
            res = np.concatenate((
                res, self.backflow.get_parameters_scale()
            ))
        return res

    def energy_parameters_numerical_d1(self, r_e, opt_jastrow=True, opt_backflow=True):
        """First-order derivatives of energy with respect to the parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :return:
        """
        delta = 0.000001  # (1/2**52)**(1/3)
        scale = self.get_parameters_scale()
        parameters = self.get_parameters(opt_jastrow, opt_backflow)
        res = np.zeros(shape=parameters.shape)
        for i in range(parameters.size):
            parameters[i] -= delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow)
            res[i] -= self.energy(r_e) / scale[i]
            parameters[i] += 2 * delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow)
            res[i] += self.energy(r_e) / scale[i]
            parameters[i] -= delta * scale[i]

        self.set_parameters(parameters, opt_jastrow, opt_backflow)
        return res / delta / 2

    def energy_parameters_numerical_d2(self, r_e, opt_jastrow=True, opt_backflow=True):
        """Second-order derivatives of energy with respect to the parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :return:
        """
        delta = 0.000001  # (1/2**52)**(1/3)
        scale = self.get_parameters_scale()
        parameters = self.get_parameters(opt_jastrow, opt_backflow)
        res = -2 * self.energy(r_e) * np.eye(parameters.size)
        for i in range(parameters.size):
            res[i, i] /= scale[i] * scale[i]

        for i in range(parameters.size):
            parameters[i] -= 2 * delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow)
            res[i, i] += self.energy(r_e) / scale[i] / scale[i]
            parameters[i] += 4 * delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow)
            res[i, i] += self.energy(r_e) / scale[i] / scale[i]
            parameters[i] -= 2 * delta * scale[i]

        for i in range(parameters.size):
            for j in range(parameters.size):
                if i >= j:
                    continue
                parameters[i] -= delta * scale[i]
                parameters[j] -= delta * scale[j]
                self.set_parameters(parameters, opt_jastrow, opt_backflow)
                res[i, j] += self.energy(r_e) / scale[i] / scale[j]
                parameters[i] += 2 * delta * scale[i]
                self.set_parameters(parameters, opt_jastrow, opt_backflow)
                res[i, j] -= self.energy(r_e) / scale[i] / scale[j]
                parameters[j] += 2 * delta * scale[j]
                self.set_parameters(parameters, opt_jastrow, opt_backflow)
                res[i, j] += self.energy(r_e) / scale[i] / scale[j]
                parameters[i] -= 2 * delta * scale[i]
                self.set_parameters(parameters, opt_jastrow, opt_backflow)
                res[i, j] -= self.energy(r_e) / scale[i] / scale[j]
                parameters[i] += delta * scale[i]
                parameters[j] -= delta * scale[j]
                res[j, i] = res[i, j]

        self.set_parameters(parameters, opt_jastrow, opt_backflow)
        return res / delta / delta / 4

    def value_parameters_d1(self, r_e, opt_jastrow=True, opt_backflow=True):
        """First-order derivatives of the wave function with respect to the parameters divided by wfn.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :return:
        """
        res = np.zeros(0)
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None and opt_jastrow:
            res = np.concatenate((
                res, self.jastrow.parameters_numerical_d1(e_vectors, n_vectors)
            ))
        if self.backflow is not None and opt_backflow:
            b_v = self.backflow.value(e_vectors, n_vectors)
            res = np.concatenate((
                res, self.backflow.parameters_numerical_d1(e_vectors, n_vectors) @ self.slater.gradient(b_v).ravel()
            ))
        return res

    def value_parameters_d2(self, r_e, opt_jastrow=True, opt_backflow=True):
        """Second-order derivatives of the wave function with respect to the parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :return: wfn_hessian - np.outer(wfn_gradient, wfn_gradient)
        """
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        return self.jastrow.parameters_numerical_d2(e_vectors, n_vectors)

    def numerical_gradient(self, r_e):
        """Numerical gradient with respect to e-coordinates
        :param r_e: electron coordinates - array(nelec, 3)
        """
        # https://scicomp.stackexchange.com/questions/14355/choosing-epsilons
        delta = 0.00001  # (1/2**52)**(1/3)

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
        """Numerical laplacian with respect to e-coordinates
        :param r_e: electron coordinates - array(nelec, 3)
        """
        delta = 0.00001

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
