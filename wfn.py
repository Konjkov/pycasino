import os
from slater import Slater
from jastrow import Jastrow
from backflow import Backflow

os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import numpy as np
import numba as nb

from overload import subtract_outer
from logger import logging

logger = logging.getLogger('vmc')

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
        self.nuclear_repulsion = self._nuclear_repulsion()
        self.slater = slater
        self.jastrow = jastrow
        self.backflow = backflow

    def relative_coordinates(self, r_e):
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)
        return e_vectors, n_vectors

    def _nuclear_repulsion(self) -> float:
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

    def value(self, e_vectors, n_vectors) -> float:
        """Value of wave function.
        :param e_vectors: e-e vectors - array(nelec, nelec, 3)
        :param n_vectors: e-n vectors - array(nelec, natom, 3)
        :return:
        """
        res = 1
        if self.jastrow is not None:
            res *= np.exp(self.jastrow.value(e_vectors, n_vectors))
        if self.backflow is not None:
            n_vectors += self.backflow.value(e_vectors, n_vectors)
        res *= self.slater.value(n_vectors)
        return res

    def drift_velocity(self, r_e):
        """Drift velocity
        drift velocity = 1/2 * 'drift or quantum force'
        where D is diffusion constant = 1/2
        """
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)

        if self.backflow is not None:
            b_v = self.backflow.value(e_vectors, n_vectors) + n_vectors
            b_g = self.backflow.gradient(e_vectors, n_vectors) + np.eye((self.neu + self.ned) * 3)
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

        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)

        res = self.coulomb(e_vectors, n_vectors)

        if self.backflow is not None:
            b_v = self.backflow.value(e_vectors, n_vectors) + n_vectors
            b_g = self.backflow.gradient(e_vectors, n_vectors) + np.eye((self.neu + self.ned) * 3)
            b_l = self.backflow.laplacian(e_vectors, n_vectors)
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
                res, self.jastrow.get_x_scale()
            ))
        if self.backflow is not None and opt_backflow:
            res = np.concatenate((
                res, self.backflow.get_x_scale()
            ))
        return res

    def slater_numerical_gradient(self, e_vectors, n_vectors):
        """Numerical gradient with respect to e-coordinates
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        delta = 0.00001

        val = self.slater.value(n_vectors + self.backflow.value(e_vectors, n_vectors))
        res = np.zeros((self.neu + self.ned, 3))
        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res[i, j] -= self.slater.value(n_vectors + self.backflow.value(e_vectors, n_vectors))
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.slater.value(n_vectors + self.backflow.value(e_vectors, n_vectors))
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res.ravel() / delta / 2 / val

    def slater_numerical_laplacian(self, e_vectors, n_vectors):
        """Numerical laplacian with respect to e-coordinates
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        delta = 0.00001

        val = self.slater.value(n_vectors + self.backflow.value(e_vectors, n_vectors))
        res = - 6 * (self.neu + self.ned) * self.slater.value(n_vectors + self.backflow.value(e_vectors, n_vectors))
        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res += self.slater.value(n_vectors + self.backflow.value(e_vectors, n_vectors))
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res += self.slater.value(n_vectors + self.backflow.value(e_vectors, n_vectors))
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res / delta / delta / val

    def jastrow_parameters_numerical_d1(self, r_e):
        """"""
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)
        return self.jastrow.parameters_numerical_d1(e_vectors, n_vectors)

    def jastrow_parameters_numerical_d2(self, r_e):
        """"""
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)
        return self.jastrow.parameters_numerical_d2(e_vectors, n_vectors)
