import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import numba as nb

# np.show_config()

from logger import logging
from readers.wfn import GAUSSIAN_TYPE, SLATER_TYPE
from cusp import Cusp, CuspFactory
from harmonics import angular_part, gradient_angular_part, hessian_angular_part
from overload import subtract_outer, random_step

logger = logging.getLogger('vmc')


slater_spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('nbasis_functions', nb.int64),
    ('first_shells', nb.int64[:]),
    ('orbital_types', nb.int64[:]),
    ('shell_moments', nb.int64[:]),
    ('slater_orders', nb.int64[:]),
    ('primitives', nb.int64[:]),
    ('coefficients', nb.float64[:]),
    ('exponents', nb.float64[:]),
    ('permutation_up', nb.int64[:, :]),
    ('permutation_down', nb.int64[:, :]),
    ('mo_up', nb.float64[:, :, :]),
    ('mo_down', nb.float64[:, :, :]),
    ('coeff', nb.float64[:]),
    ('cusp', nb.optional(Cusp.class_type.instance_type)),
    ('norm', nb.float64),
]


@nb.experimental.jitclass(slater_spec)
class Slater:

    def __init__(
            self, neu, ned,
            nbasis_functions, first_shells, orbital_types, shell_moments, slater_orders, primitives, coefficients, exponents,
            mo_up, mo_down, permutation_up, permutation_down, coeff, cusp
    ):
        """
        Slater
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param nbasis_functions:
        :param first_shells:
        :param orbital_types:
        :param shell_moments:
        :param slater_orders:
        :param primitives:
        :param coefficients:
        :param exponents:
        :param mo_up:
        :param mo_down:
        :param coeff:
        """
        self.neu = neu
        self.ned = ned
        self.nbasis_functions = nbasis_functions
        self.first_shells = first_shells
        self.orbital_types = orbital_types
        self.shell_moments = shell_moments
        self.slater_orders = slater_orders
        self.primitives = primitives
        self.coefficients = coefficients
        self.exponents = exponents
        self.permutation_up = permutation_up
        self.permutation_down = permutation_down
        self.mo_up = np.zeros(shape=(permutation_up.shape[0], neu, mo_up.shape[1]))
        self.mo_down = np.zeros(shape=(permutation_down.shape[0], ned, mo_down.shape[1]))
        for i in range(permutation_up.shape[0]):
            self.mo_up[i] = mo_up[permutation_up[i]]
            self.mo_down[i] = mo_down[permutation_down[i]]
        self.coeff = coeff
        self.cusp = cusp
        self.norm = np.exp(-(np.math.lgamma(self.neu + 1) + np.math.lgamma(self.ned + 1)) / (self.neu + self.ned) / 2)

    def ao_value(self, n_vectors: np.ndarray) -> np.ndarray:
        """AO value.
        Atomic orbitals for every electron
        :param n_vectors: electron-nuclei array(nelec, natom, 3)
        :return: AO array(nelec, nbasis_functions)
        """
        orbital = np.zeros(shape=(self.neu + self.ned, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = x * x + y * y + z * z
                angular_1 = angular_part(x, y, z)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            radial_1 += self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r2)
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        for primitive in range(self.primitives[nshell]):
                            radial_1 += r**self.slater_orders[nshell] * self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r)
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital[i, ao+m] = angular_1[l*l+m] * radial_1
                    ao += 2*l+1
        return self.norm * orbital

    def ao_gradient(self, n_vectors: np.ndarray) -> np.ndarray:
        """AO gradient.
        :param n_vectors: electron-nuclei - array(natom, nelec, 3)
        :return: AO gradient - array(3, nelec, nbasis_functions)
        """
        orbital = np.zeros(shape=(self.neu + self.ned, 3, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = x * x + y * y + z * z
                angular_1 = angular_part(x, y, z)
                angular_2 = gradient_angular_part(x, y, z)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    radial_2 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            exponent = self.coefficients[p + primitive] * np.exp(-alpha * r2)
                            radial_1 -= 2 * alpha * exponent
                            radial_2 += exponent
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        n = self.slater_orders[nshell]
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            exponent = r**self.slater_orders[nshell] * self.coefficients[p + primitive] * np.exp(-alpha * r)
                            radial_1 -= (alpha*r - n)/r2 * exponent
                            radial_2 += exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital[i, 0, ao+m] = x * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 0] * radial_2
                        orbital[i, 1, ao+m] = y * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 1] * radial_2
                        orbital[i, 2, ao+m] = z * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 2] * radial_2
                    ao += 2*l+1
        return self.norm * orbital.reshape((self.neu + self.ned) * 3, self.nbasis_functions)

    def ao_laplacian(self, n_vectors: np.ndarray) -> np.ndarray:
        """AO Laplacian.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        :return: AO laplacian - array(nelec, nbasis_functions)
        """
        orbital = np.zeros(shape=(self.neu + self.ned, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = x * x + y * y + z * z
                angular_1 = angular_part(x, y, z)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            radial_1 += 2 * alpha * (2 * alpha * r2 - 2 * l - 3) * self.coefficients[p + primitive] * np.exp(-alpha * r2)
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        n = self.slater_orders[nshell]
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            exponent = r**n * self.coefficients[p + primitive] * np.exp(-alpha * r)
                            radial_1 += (alpha**2 - 2*(l+n+1)*alpha/r + (2*l+n+1)*n/r2) * exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital[i, ao+m] = angular_1[l*l+m] * radial_1
                    ao += 2*l+1
        return self.norm * orbital

    def ao_hessian(self, n_vectors: np.ndarray) -> np.ndarray:
        """AO hessian.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        :return: AO hessian - array(6, nelec, nbasis_functions)
        """
        orbital = np.zeros(shape=(self.neu + self.ned, 9, self.nbasis_functions))

        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = x * x + y * y + z * z
                angular_1 = angular_part(x, y, z)
                angular_2 = gradient_angular_part(x, y, z)
                angular_3 = hessian_angular_part(x, y, z)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    radial_2 = 0.0
                    radial_3 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            exponent = self.coefficients[p + primitive] * np.exp(-alpha * r2)
                            c = -2 * alpha
                            radial_1 += c**2 * exponent
                            radial_2 += c * exponent
                            radial_3 += exponent
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        for primitive in range(self.primitives[nshell]):
                            n = self.slater_orders[nshell]
                            alpha = self.exponents[p + primitive]
                            exponent = r**self.slater_orders[nshell] * self.coefficients[p + primitive] * np.exp(-alpha * r)
                            c = -(alpha*r - n)/r2
                            d = c**2 + (alpha*r - 2*n)/r2**2
                            radial_1 += d * exponent
                            radial_2 += c * exponent
                            radial_3 += exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital[i, 0, ao+m] = x*x * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * x * angular_2[l*l+m, 0]) * radial_2 + angular_3[l*l+m, 0] * radial_3
                        orbital[i, 1, ao+m] = x*y * angular_1[l*l+m] * radial_1 + (y * angular_2[l*l+m, 0] + x * angular_2[l*l+m, 1]) * radial_2 + angular_3[l*l+m, 1] * radial_3
                        orbital[i, 2, ao+m] = x*z * angular_1[l*l+m] * radial_1 + (z * angular_2[l*l+m, 0] + x * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 3] * radial_3
                        orbital[i, 3, ao+m] = orbital[i, 1, ao+m]
                        orbital[i, 4, ao+m] = y*y * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * y * angular_2[l*l+m, 1]) * radial_2 + angular_3[l*l+m, 2] * radial_3
                        orbital[i, 5, ao+m] = y*z * angular_1[l*l+m] * radial_1 + (z * angular_2[l*l+m, 1] + y * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 4] * radial_3
                        orbital[i, 6, ao+m] = orbital[i, 2, ao+m]
                        orbital[i, 7, ao+m] = orbital[i, 5, ao+m]
                        orbital[i, 8, ao+m] = z*z * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * z * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 5] * radial_3
                    ao += 2*l+1

        return self.norm * orbital.reshape((self.neu + self.ned) * 9, self.nbasis_functions)

    def value(self, n_vectors: np.ndarray) -> float:
        """Multideterminant wave function value.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao_value = self.ao_value(n_vectors)
        if self.cusp is not None:
            cusp_value_u, cusp_value_d = self.cusp.value(n_vectors)

        val = 0.0
        for i in range(self.coeff.shape[0]):
            if self.cusp is not None:
                wfn_u = np.where(cusp_value_u[self.permutation_up[i]], cusp_value_u[self.permutation_up[i]], self.mo_up[i] @ ao_value[:self.neu].T)
                wfn_d = np.where(cusp_value_d[self.permutation_down[i]], cusp_value_d[self.permutation_down[i]], self.mo_down[i] @ ao_value[self.neu:].T)
            else:
                wfn_u = self.mo_up[i] @ ao_value[:self.neu].T
                wfn_d = self.mo_down[i] @ ao_value[self.neu:].T
            val += self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
        return val

    def gradient(self, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient ∇(phi).
        ∇ln(det(A)) = tr(A^-1 * ∇A)
        where matrix ∇A is column-wise gradient of A
        then using np.trace(A @ B) = np.sum(A * B.T)
        Read for details:
        "Simple formalism for efficient derivatives and multi-determinant expansions in quantum Monte Carlo"
        C. Filippi, R. Assaraf, S. Moroni
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao_value = self.ao_value(n_vectors)
        ao_gradient = self.ao_gradient(n_vectors)
        if self.cusp is not None:
            cusp_value_u, cusp_value_d = self.cusp.value(n_vectors)
            cusp_gradient_u, cusp_gradient_d = self.cusp.gradient(n_vectors)

        val = 0.0
        grad = np.zeros(shape=(self.neu + self.ned, 3))
        for i in range(self.coeff.shape[0]):

            if self.cusp is not None:
                wfn_u = np.where(cusp_value_u[self.permutation_up[i]], cusp_value_u[self.permutation_up[i]], self.mo_up[i] @ ao_value[:self.neu].T)
                wfn_d = np.where(cusp_value_d[self.permutation_down[i]], cusp_value_d[self.permutation_down[i]], self.mo_down[i] @ ao_value[self.neu:].T)
                grad_u = np.where(cusp_gradient_u[self.permutation_up[i]], cusp_gradient_u[self.permutation_up[i]], (self.mo_up[i] @ ao_gradient[:self.neu * 3].T).reshape(self.neu, self.neu, 3))
                grad_d = np.where(cusp_gradient_d[self.permutation_down[i]], cusp_gradient_d[self.permutation_down[i]], (self.mo_down[i] @ ao_gradient[self.neu * 3:].T).reshape(self.ned, self.ned, 3))
            else:
                wfn_u = self.mo_up[i] @ ao_value[:self.neu].T
                wfn_d = self.mo_down[i] @ ao_value[self.neu:].T
                grad_u = (self.mo_up[i] @ ao_gradient[:self.neu * 3].T).reshape(self.neu, self.neu, 3)
                grad_d = (self.mo_down[i] @ ao_gradient[self.neu * 3:].T).reshape(self.ned, self.ned, 3)

            res_u = (np.linalg.inv(wfn_u) * grad_u.T).T.sum(axis=0)
            res_d = (np.linalg.inv(wfn_d) * grad_d.T).T.sum(axis=0)

            c = self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
            val += c
            grad += c * np.concatenate((res_u, res_d))

        return grad.ravel() / val

    def laplacian(self, n_vectors: np.ndarray) -> float:
        """Scalar laplacian Δ(phi).
        Δln(det(A)) = sum(tr(slater^-1 * B(n)) over n
        where matrix B(n) is zero with exception of the n-th column
        as tr(A) + tr(B) = tr(A + B)
        Δln(det(A)) = tr(slater^-1 * B)
        where the matrix Bij = ∆phi i (rj)
        then using np.trace(A @ B) = np.sum(A * B.T)
        Read for details:
        "Simple formalism for efficient derivatives and multi-determinant expansions in quantum Monte Carlo"
        C. Filippi, R. Assaraf, S. Moroni
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao_value = self.ao_value(n_vectors)
        ao_laplacian = self.ao_laplacian(n_vectors)
        if self.cusp is not None:
            cusp_value_u, cusp_value_d = self.cusp.value(n_vectors)
            cusp_laplacian_u, cusp_laplacian_d = self.cusp.laplacian(n_vectors)

        val = lap = 0
        for i in range(self.coeff.shape[0]):

            if self.cusp is not None:
                wfn_u = np.where(cusp_value_u[self.permutation_up[i]], cusp_value_u[self.permutation_up[i]], self.mo_up[i] @ ao_value[:self.neu].T)
                wfn_d = np.where(cusp_value_d[self.permutation_down[i]], cusp_value_d[self.permutation_down[i]], self.mo_down[i] @ ao_value[self.neu:].T)
                lap_u = np.where(cusp_laplacian_u[self.permutation_up[i]], cusp_laplacian_u[self.permutation_up[i]], self.mo_up[i] @ ao_laplacian[:self.neu].T)
                lap_d = np.where(cusp_laplacian_d[self.permutation_down[i]], cusp_laplacian_d[self.permutation_down[i]], self.mo_down[i] @ ao_laplacian[self.neu:].T)
            else:
                wfn_u = self.mo_up[i] @ ao_value[:self.neu].T
                wfn_d = self.mo_down[i] @ ao_value[self.neu:].T
                lap_u = self.mo_up[i] @ ao_laplacian[:self.neu].T
                lap_d = self.mo_down[i] @ ao_laplacian[self.neu:].T

            res_u = np.sum(np.linalg.inv(wfn_u) * lap_u.T)
            res_d = np.sum(np.linalg.inv(wfn_d) * lap_d.T)

            c = self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
            val += c
            lap += c * (res_u + res_d)

        return lap / val

    def hessian(self, n_vectors: np.ndarray) -> np.ndarray:
        """Hessian.
        d²ln(det(A))/dxdy = (
            tr(A^-1 * d²A/dxdy) +
            tr(A^-1 * dA/dx) * tr(A^-1 * dA/dy) -
            tr(A^-1 * dA/dx * A^-1 * dA/dy)
        )
        https://math.stackexchange.com/questions/2325807/second-derivative-of-a-determinant
        in case of x and y is a coordinates of different electrons first term is zero
        in other case a sum of last two terms is zero.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao_value = self.ao_value(n_vectors)
        ao_gradient = self.ao_gradient(n_vectors)
        ao_hessian = self.ao_hessian(n_vectors)
        if self.cusp is not None:
            cusp_value_u, cusp_value_d = self.cusp.value(n_vectors)
            cusp_gradient_u, cusp_gradient_d = self.cusp.gradient(n_vectors)
            cusp_hessian_u, cusp_hessian_d = self.cusp.hessian(n_vectors)

        val = 0
        hass = np.zeros(shape=(self.neu + self.ned, 3, self.neu + self.ned, 3))
        for i in range(self.coeff.shape[0]):

            if self.cusp is not None:
                wfn_u = np.where(cusp_value_u[self.permutation_up[i]], cusp_value_u[self.permutation_up[i]], self.mo_up[i] @ ao_value[:self.neu].T)
                wfn_d = np.where(cusp_value_d[self.permutation_down[i]], cusp_value_d[self.permutation_down[i]], self.mo_down[i] @ ao_value[self.neu:].T)
                grad_u = np.where(cusp_gradient_u[self.permutation_up[i]], cusp_gradient_u[self.permutation_up[i]], (self.mo_up[i] @ ao_gradient[:self.neu * 3].T).reshape(self.neu, self.neu, 3))
                grad_d = np.where(cusp_gradient_d[self.permutation_down[i]], cusp_gradient_d[self.permutation_down[i]], (self.mo_down[i] @ ao_gradient[self.neu * 3:].T).reshape(self.ned, self.ned, 3))
                hess_u = np.where(cusp_hessian_u[self.permutation_up[i]], cusp_hessian_u[self.permutation_up[i]], (self.mo_up[i] @ ao_hessian[:self.neu * 9].T).reshape(self.neu, self.neu, 3, 3))
                hess_d = np.where(cusp_hessian_d[self.permutation_down[i]], cusp_hessian_d[self.permutation_down[i]], (self.mo_down[i] @ ao_hessian[self.neu * 9:].T).reshape(self.neu, self.neu, 3, 3))
            else:
                wfn_u = self.mo_up[i] @ ao_value[:self.neu].T
                wfn_d = self.mo_down[i] @ ao_value[self.neu:].T
                grad_u = (self.mo_up[i] @ ao_gradient[:self.neu * 3].T).reshape(self.neu, self.neu, 3)
                grad_d = (self.mo_down[i] @ ao_gradient[self.neu * 3:].T).reshape(self.neu, self.neu, 3)
                hess_u = (self.mo_up[i] @ ao_hessian[:self.neu * 9].T).reshape(self.neu, self.neu, 3, 3)
                hess_d = (self.mo_down[i] @ ao_hessian[self.neu * 9:].T).reshape(self.ned, self.ned, 3, 3)

            inv_wfn_u = np.linalg.inv(wfn_u)
            inv_wfn_d = np.linalg.inv(wfn_d)

            temp_grad_u = (inv_wfn_u @ grad_u.reshape(self.neu, self.neu * 3)).reshape(self.neu, self.neu, 3)
            dx = temp_grad_u[:, :, 0]
            dy = temp_grad_u[:, :, 1]
            dz = temp_grad_u[:, :, 2]

            temp_hess_u = (inv_wfn_u * hess_u.T).T.sum(axis=0)

            # tr(A^-1 * d²A/dxdy) - tr(A^-1 * dA/dx * A^-1 * dA/dy)
            res_u = np.zeros((self.neu, 3, self.neu, 3))
            res_u[:, 0, :, 0] = np.diag(temp_hess_u[:, 0, 0]) - dx.T * dx
            res_u[:, 0, :, 1] = np.diag(temp_hess_u[:, 0, 1]) - dx.T * dy
            res_u[:, 0, :, 2] = np.diag(temp_hess_u[:, 0, 2]) - dx.T * dz
            res_u[:, 1, :, 0] = np.diag(temp_hess_u[:, 1, 0]) - dy.T * dx
            res_u[:, 1, :, 1] = np.diag(temp_hess_u[:, 1, 1]) - dy.T * dy
            res_u[:, 1, :, 2] = np.diag(temp_hess_u[:, 1, 2]) - dy.T * dz
            res_u[:, 2, :, 0] = np.diag(temp_hess_u[:, 2, 0]) - dz.T * dx
            res_u[:, 2, :, 1] = np.diag(temp_hess_u[:, 2, 1]) - dz.T * dy
            res_u[:, 2, :, 2] = np.diag(temp_hess_u[:, 2, 2]) - dz.T * dz

            temp_grad_d = (inv_wfn_d @ grad_d.reshape(self.ned, self.ned * 3)).reshape(self.ned, self.ned, 3)
            dx = temp_grad_d[:, :, 0]
            dy = temp_grad_d[:, :, 1]
            dz = temp_grad_d[:, :, 2]

            temp_hess_d = (inv_wfn_d * hess_d.T).T.sum(axis=0)

            # tr(A^-1 * d²A/dxdy) - tr(A^-1 * dA/dx * A^-1 * dA/dy)
            res_d = np.zeros((self.ned, 3, self.ned, 3))
            # FIXME: res_d[:, ::(n + 1)] = temp_hess_d - https://stackoverflow.com/questions/67241824/3d-tensor-of-diagonal-matrices
            res_d[:, 0, :, 0] = np.diag(temp_hess_d[:, 0, 0]) - dx.T * dx
            res_d[:, 0, :, 1] = np.diag(temp_hess_d[:, 0, 1]) - dx.T * dy
            res_d[:, 0, :, 2] = np.diag(temp_hess_d[:, 0, 2]) - dx.T * dz
            res_d[:, 1, :, 0] = np.diag(temp_hess_d[:, 1, 0]) - dy.T * dx
            res_d[:, 1, :, 1] = np.diag(temp_hess_d[:, 1, 1]) - dy.T * dy
            res_d[:, 1, :, 2] = np.diag(temp_hess_d[:, 1, 2]) - dy.T * dz
            res_d[:, 2, :, 0] = np.diag(temp_hess_d[:, 2, 0]) - dz.T * dx
            res_d[:, 2, :, 1] = np.diag(temp_hess_d[:, 2, 1]) - dz.T * dy
            res_d[:, 2, :, 2] = np.diag(temp_hess_d[:, 2, 2]) - dz.T * dz

            res_grad_u = (np.linalg.inv(wfn_u) * grad_u.reshape(self.neu, self.neu, 3).T).T.sum(axis=0)
            res_grad_d = (np.linalg.inv(wfn_d) * grad_d.reshape(self.ned, self.ned, 3).T).T.sum(axis=0)

            c = self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
            val += c
            res_grad = np.concatenate((res_grad_u.ravel(), res_grad_d.ravel()))
            # tr(A^-1 * dA/dx) * tr(A^-1 * dA/dy)
            hass += c * np.outer(res_grad, res_grad).reshape((self.neu + self.ned), 3, (self.neu + self.ned), 3)
            hass[:self.neu, :, :self.neu, :] += c * res_u
            hass[self.neu:, :, self.neu:, :] += c * res_d

        return hass.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3) / val

    def numerical_gradient(self, n_vectors: np.ndarray) -> float:
        """Numerical gradient with respect to e-coordinates
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        delta = 0.00001

        val = self.value(n_vectors)
        res = np.zeros(shape=(self.neu + self.ned, 3))
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= delta
                res[i, j] -= self.value(n_vectors)
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.value(n_vectors)
                n_vectors[:, i, j] -= delta

        return res.ravel() / delta / 2 / val

    def numerical_laplacian(self, n_vectors: np.ndarray) -> float:
        """Numerical laplacian with respect to e-coordinates
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        delta = 0.00001

        val = self.value(n_vectors)
        res = - 6 * (self.neu + self.ned) * val
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= delta
                res += self.value(n_vectors)
                n_vectors[:, i, j] += 2 * delta
                res += self.value(n_vectors)
                n_vectors[:, i, j] -= delta

        return res / delta / delta / val

    def numerical_hessian(self, n_vectors: np.ndarray) -> np.ndarray:
        """Numerical hessian with respect to e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        delta = 0.00001

        val = self.value(n_vectors)
        res = -2 * val * np.eye((self.neu + self.ned) * 3).reshape(self.neu + self.ned, 3, self.neu + self.ned, 3)
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= 2 * delta
                res[i, j, i, j] += self.value(n_vectors)
                n_vectors[:, i, j] += 4 * delta
                res[i, j, i, j] += self.value(n_vectors)
                n_vectors[:, i, j] -= 2 * delta

        for i1 in range(self.neu + self.ned):
            for j1 in range(3):
                for i2 in range(i1 + 1):
                    for j2 in range(3):
                        if i1 == i2 and j1 >= j2:
                            continue
                        n_vectors[:, i1, j1] -= delta
                        n_vectors[:, i2, j2] -= delta
                        res[i1, j1, i2, j2] += self.value(n_vectors)
                        n_vectors[:, i1, j1] += 2 * delta
                        res[i1, j1, i2, j2] -= self.value(n_vectors)
                        n_vectors[:, i2, j2] += 2 * delta
                        res[i1, j1, i2, j2] += self.value(n_vectors)
                        n_vectors[:, i1, j1] -= 2 * delta
                        res[i1, j1, i2, j2] -= self.value(n_vectors)
                        n_vectors[:, i1, j1] += delta
                        n_vectors[:, i2, j2] -= delta
                        res[i2, j2, i1, j1] = res[i1, j1, i2, j2]

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3) / delta / delta / 4 / val

    def profile_value(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = subtract_outer(atom_positions, r_e)
            self.value(n_vectors)

    def profile_gradient(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = subtract_outer(atom_positions, r_e)
            self.gradient(n_vectors)

    def profile_laplacian(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = subtract_outer(atom_positions, r_e)
            self.laplacian(n_vectors)

    def profile_hessian(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = subtract_outer(atom_positions, r_e)
            self.hessian(n_vectors)
