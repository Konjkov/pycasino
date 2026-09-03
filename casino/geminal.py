import math

import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino.abstract import AbstractSlater
from casino.cusp import Cusp_t
from casino.harmonics import Harmonics, Harmonics_t
from casino.readers.wfn import GAUSSIAN_TYPE, SLATER_TYPE

log_10 = np.log(10)


@structref.register
class Geminal_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


Geminal_t = Geminal_class_t(
    [
        ('neu', nb.int64),
        ('ned', nb.int64),
        ('nunpaired', nb.int64),
        ('norb', nb.int64),
        ('nbasis_functions', nb.int64),
        ('first_shells', nb.int64[::1]),
        ('orbital_types', nb.int64[::1]),
        ('shell_moments', nb.int64[::1]),
        ('slater_orders', nb.int64[::1]),
        ('primitives', nb.int64[::1]),
        ('coefficients', nb.float64[::1]),
        ('exponents', nb.float64[::1]),
        ('gautol', nb.float64),
        ('mo_up', nb.float64[:, ::1]),
        ('mo_down', nb.float64[:, ::1]),
        ('c', nb.float64[::1]),
        ('g', nb.float64[:, :, ::1]),
        ('u', nb.float64[:, :, ::1]),
        ('c_mask', nb.boolean[::1]),
        ('g_mask', nb.boolean[:, :, ::1]),
        ('u_mask', nb.boolean[:, :, ::1]),
        ('g_available', nb.boolean[:, :, ::1]),
        ('u_available', nb.boolean[:, :, ::1]),
        ('c_ties', nb.int64[:, ::1]),
        ('g_ties', nb.int64[:, ::1]),
        ('u_ties', nb.int64[:, ::1]),
        ('cusp', nb.optional(Cusp_t)),
        ('norm', nb.float64),
        ('parameters_projector', nb.float64[:, ::1]),
        ('harmonics', Harmonics_t),
    ]
)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'pool_matrix')
def geminal_pool_matrix(self, n_vectors: np.ndarray):
    """Orbital pool value matrices.
    :param n_vectors: electron-nuclei array(natom, nelec, 3)
    :return: array(norb, up_electrons), array(norb, down_electrons)
    """

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        orbitals = np.zeros(shape=(self.neu + self.ned, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = n_vectors[atom, i] @ n_vectors[atom, i]
                angular_1 = self.harmonics.get_value(x, y, z)
                for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            if alpha * r2 < log_10 * self.gautol:
                                radial_1 += self.coefficients[p + primitive] * np.exp(-alpha * r2)
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        r_n = r ** self.slater_orders[nshell]
                        for primitive in range(self.primitives[nshell]):
                            minus_alpha_r = -self.exponents[p + primitive] * r
                            radial_1 += r_n * self.coefficients[p + primitive] * np.exp(minus_alpha_r)
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbitals[i, ao + m] = angular_1[l * l + m] * radial_1
                    ao += 2 * l + 1

        ao_value = self.norm * orbitals
        pool_u = self.mo_up @ ao_value[: self.neu].T
        pool_d = self.mo_down @ ao_value[self.neu :].T
        if self.cusp is not None:
            cusp_value_u, cusp_value_d = self.cusp.value(n_vectors)
            pool_u += cusp_value_u
            pool_d += cusp_value_d
        return pool_u, pool_d

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'pool_gradient_matrix')
def geminal_pool_gradient_matrix(self, n_vectors: np.ndarray):
    """Orbital pool value and gradient matrices in one pass over the basis."""

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        orbital_value = np.zeros(shape=(self.neu + self.ned, self.nbasis_functions))
        orbital = np.zeros(shape=(self.neu + self.ned, 3, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = n_vectors[atom, i] @ n_vectors[atom, i]
                angular_1 = self.harmonics.get_value(x, y, z)
                angular_2 = self.harmonics.get_gradient(x, y, z)
                for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    radial_2 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            if alpha * r2 < log_10 * self.gautol:
                                exponent = self.coefficients[p + primitive] * np.exp(-alpha * r2)
                                radial_1 -= 2 * alpha * exponent
                                radial_2 += exponent
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        n = self.slater_orders[nshell]
                        r_n = r**n
                        for primitive in range(self.primitives[nshell]):
                            minus_alpha_r = -self.exponents[p + primitive] * r
                            exponent = r_n * self.coefficients[p + primitive] * np.exp(minus_alpha_r)
                            radial_1 += (minus_alpha_r + n) / r2 * exponent
                            radial_2 += exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital_value[i, ao + m] = angular_1[l * l + m] * radial_2
                        orbital[i, 0, ao + m] = x * angular_1[l * l + m] * radial_1 + angular_2[l * l + m, 0] * radial_2
                        orbital[i, 1, ao + m] = y * angular_1[l * l + m] * radial_1 + angular_2[l * l + m, 1] * radial_2
                        orbital[i, 2, ao + m] = z * angular_1[l * l + m] * radial_1 + angular_2[l * l + m, 2] * radial_2
                    ao += 2 * l + 1

        ao_value = self.norm * orbital_value
        pool_u = self.mo_up @ ao_value[: self.neu].T
        pool_d = self.mo_down @ ao_value[self.neu :].T
        ao_gradient = self.norm * orbital.reshape((self.neu + self.ned) * 3, self.nbasis_functions)
        pool_grad_u = (self.mo_up @ ao_gradient[: self.neu * 3].T).reshape(self.norb, self.neu, 3)
        pool_grad_d = (self.mo_down @ ao_gradient[self.neu * 3 :].T).reshape(self.norb, self.ned, 3)
        if self.cusp is not None:
            cusp_value_u, cusp_value_d = self.cusp.value(n_vectors)
            cusp_gradient_u, cusp_gradient_d = self.cusp.gradient(n_vectors)
            pool_u += cusp_value_u
            pool_d += cusp_value_d
            pool_grad_u += cusp_gradient_u
            pool_grad_d += cusp_gradient_d
        return pool_u, pool_d, pool_grad_u, pool_grad_d

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'pool_laplacian_matrix')
def geminal_pool_laplacian_matrix(self, n_vectors: np.ndarray):
    """Orbital pool value and laplacian matrices in one pass over the basis."""

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        orbital_value = np.zeros(shape=(self.neu + self.ned, self.nbasis_functions))
        orbital = np.zeros(shape=(self.neu + self.ned, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = n_vectors[atom, i] @ n_vectors[atom, i]
                angular_1 = self.harmonics.get_value(x, y, z)
                for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    radial_2 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            if alpha * r2 < log_10 * self.gautol:
                                exponent = self.coefficients[p + primitive] * np.exp(-alpha * r2)
                                radial_1 += 2 * alpha * (2 * alpha * r2 - 2 * l - 3) * exponent
                                radial_2 += exponent
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        n = self.slater_orders[nshell]
                        r_n = r**n
                        for primitive in range(self.primitives[nshell]):
                            minus_alpha_r = -self.exponents[p + primitive] * r
                            exponent = r_n * self.coefficients[p + primitive] * np.exp(minus_alpha_r)
                            radial_1 += (minus_alpha_r**2 + 2 * (l + n + 1) * minus_alpha_r + (2 * l + n + 1) * n) / r2 * exponent
                            radial_2 += exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital_value[i, ao + m] = angular_1[l * l + m] * radial_2
                        orbital[i, ao + m] = angular_1[l * l + m] * radial_1
                    ao += 2 * l + 1

        ao_value = self.norm * orbital_value
        pool_u = self.mo_up @ ao_value[: self.neu].T
        pool_d = self.mo_down @ ao_value[self.neu :].T
        ao_laplacian = self.norm * orbital
        pool_lap_u = self.mo_up @ ao_laplacian[: self.neu].T
        pool_lap_d = self.mo_down @ ao_laplacian[self.neu :].T
        if self.cusp is not None:
            cusp_value_u, cusp_value_d = self.cusp.value(n_vectors)
            cusp_laplacian_u, cusp_laplacian_d = self.cusp.laplacian(n_vectors)
            pool_u += cusp_value_u
            pool_d += cusp_value_d
            pool_lap_u += cusp_laplacian_u
            pool_lap_d += cusp_laplacian_d
        return pool_u, pool_d, pool_lap_u, pool_lap_d

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'pool_hessian_matrix')
def geminal_pool_hessian_matrix(self, n_vectors: np.ndarray):
    """Orbital pool value, gradient and hessian matrices in one pass over the basis."""

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        orbital_value = np.zeros(shape=(self.neu + self.ned, self.nbasis_functions))
        orbital_gradient = np.zeros(shape=(self.neu + self.ned, 3, self.nbasis_functions))
        orbital = np.zeros(shape=(self.neu + self.ned, 3, 3, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = n_vectors[atom, i] @ n_vectors[atom, i]
                angular_1 = self.harmonics.get_value(x, y, z)
                angular_2 = self.harmonics.get_gradient(x, y, z)
                angular_3 = self.harmonics.get_hessian(x, y, z)
                for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    radial_2 = 0.0
                    radial_3 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            if alpha * r2 < log_10 * self.gautol:
                                exponent = self.coefficients[p + primitive] * np.exp(-alpha * r2)
                                c = -2 * alpha
                                radial_1 += c**2 * exponent
                                radial_2 += c * exponent
                                radial_3 += exponent
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        n = self.slater_orders[nshell]
                        r_n = r**n
                        for primitive in range(self.primitives[nshell]):
                            minus_alpha_r = -self.exponents[p + primitive] * r
                            exponent = r_n * self.coefficients[p + primitive] * np.exp(minus_alpha_r)
                            c = (minus_alpha_r + n) / r2
                            d = c**2 - c / r2 - n / r2**2
                            radial_1 += d * exponent
                            radial_2 += c * exponent
                            radial_3 += exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital_value[i, ao + m] = angular_1[l * l + m] * radial_3
                        orbital_gradient[i, 0, ao + m] = x * angular_1[l * l + m] * radial_2 + angular_2[l * l + m, 0] * radial_3
                        orbital_gradient[i, 1, ao + m] = y * angular_1[l * l + m] * radial_2 + angular_2[l * l + m, 1] * radial_3
                        orbital_gradient[i, 2, ao + m] = z * angular_1[l * l + m] * radial_2 + angular_2[l * l + m, 2] * radial_3
                        orbital[i, 0, 0, ao + m] = x*x * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * x * angular_2[l*l+m, 0]) * radial_2 + angular_3[l*l+m, 0] * radial_3  # fmt: skip
                        orbital[i, 0, 1, ao + m] = x*y * angular_1[l*l+m] * radial_1 + (y * angular_2[l*l+m, 0] + x * angular_2[l*l+m, 1]) * radial_2 + angular_3[l*l+m, 1] * radial_3  # fmt: skip
                        orbital[i, 0, 2, ao + m] = x*z * angular_1[l*l+m] * radial_1 + (z * angular_2[l*l+m, 0] + x * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 2] * radial_3  # fmt: skip
                        orbital[i, 1, 0, ao + m] = orbital[i, 0, 1, ao + m]
                        orbital[i, 1, 1, ao + m] = y*y * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * y * angular_2[l*l+m, 1]) * radial_2 + angular_3[l*l+m, 3] * radial_3  # fmt: skip
                        orbital[i, 1, 2, ao + m] = y*z * angular_1[l*l+m] * radial_1 + (z * angular_2[l*l+m, 1] + y * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 4] * radial_3  # fmt: skip
                        orbital[i, 2, 0, ao + m] = orbital[i, 0, 2, ao + m]
                        orbital[i, 2, 1, ao + m] = orbital[i, 1, 2, ao + m]
                        orbital[i, 2, 2, ao + m] = z*z * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * z * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 5] * radial_3  # fmt: skip
                    ao += 2 * l + 1

        ao_value = self.norm * orbital_value
        pool_u = self.mo_up @ ao_value[: self.neu].T
        pool_d = self.mo_down @ ao_value[self.neu :].T
        ao_gradient = self.norm * orbital_gradient.reshape((self.neu + self.ned) * 3, self.nbasis_functions)
        pool_grad_u = (self.mo_up @ ao_gradient[: self.neu * 3].T).reshape(self.norb, self.neu, 3)
        pool_grad_d = (self.mo_down @ ao_gradient[self.neu * 3 :].T).reshape(self.norb, self.ned, 3)
        ao_hessian = self.norm * orbital.reshape((self.neu + self.ned) * 9, self.nbasis_functions)
        pool_hess_u = (self.mo_up @ ao_hessian[: self.neu * 9].T).reshape(self.norb, self.neu, 3, 3)
        pool_hess_d = (self.mo_down @ ao_hessian[self.neu * 9 :].T).reshape(self.norb, self.ned, 3, 3)
        if self.cusp is not None:
            cusp_value_u, cusp_value_d = self.cusp.value(n_vectors)
            cusp_gradient_u, cusp_gradient_d = self.cusp.gradient(n_vectors)
            cusp_hessian_u, cusp_hessian_d = self.cusp.hessian(n_vectors)
            pool_u += cusp_value_u
            pool_d += cusp_value_d
            pool_grad_u += cusp_gradient_u
            pool_grad_d += cusp_gradient_d
            pool_hess_u += cusp_hessian_u
            pool_hess_d += cusp_hessian_d
        return pool_u, pool_d, pool_grad_u, pool_grad_d, pool_hess_u, pool_hess_d

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'pool_tressian_matrix')
def geminal_pool_tressian_matrix(self, n_vectors: np.ndarray):
    """Orbital pool value, gradient, hessian and tressian matrices in one pass over the basis."""

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        orbital_value = np.zeros(shape=(self.neu + self.ned, self.nbasis_functions))
        orbital_gradient = np.zeros(shape=(self.neu + self.ned, 3, self.nbasis_functions))
        orbital_hessian = np.zeros(shape=(self.neu + self.ned, 3, 3, self.nbasis_functions))
        orbital = np.zeros(shape=(self.neu + self.ned, 3, 3, 3, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = n_vectors[atom, i] @ n_vectors[atom, i]
                angular_1 = self.harmonics.get_value(x, y, z)
                angular_2 = self.harmonics.get_gradient(x, y, z)
                angular_3 = self.harmonics.get_hessian(x, y, z)
                angular_4 = self.harmonics.get_tressian(x, y, z)
                for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    radial_2 = 0.0
                    radial_3 = 0.0
                    radial_4 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            if alpha * r2 < log_10 * self.gautol:
                                exponent = self.coefficients[p + primitive] * np.exp(-alpha * r2)
                                c = -2 * alpha
                                radial_1 += c**3 * exponent
                                radial_2 += c**2 * exponent
                                radial_3 += c * exponent
                                radial_4 += exponent
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        n = self.slater_orders[nshell]
                        r_n = r**n
                        for primitive in range(self.primitives[nshell]):
                            minus_alpha_r = -self.exponents[p + primitive] * r
                            exponent = r_n * self.coefficients[p + primitive] * np.exp(minus_alpha_r)
                            c = (minus_alpha_r + n) / r2
                            d = c**2 - c / r2 - n / r2**2
                            e = c**3 - 3 * c**2 / r2 - 3 * (n - 1) * c / r2**2 + 5 * n / r2**3
                            radial_1 += e * exponent
                            radial_2 += d * exponent
                            radial_3 += c * exponent
                            radial_4 += exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        # orbital[i, :, :, ao+m] = (
                        #     np.prod(np.ix_(n_vectors[atom, i], n_vectors[atom, i], n_vectors[atom, i])) * angular_1[l*l+m] * radial_1 +
                        #     ...
                        # )
                        orbital_value[i, ao + m] = angular_1[l * l + m] * radial_4
                        orbital_gradient[i, 0, ao + m] = x * angular_1[l * l + m] * radial_3 + angular_2[l * l + m, 0] * radial_4
                        orbital_gradient[i, 1, ao + m] = y * angular_1[l * l + m] * radial_3 + angular_2[l * l + m, 1] * radial_4
                        orbital_gradient[i, 2, ao + m] = z * angular_1[l * l + m] * radial_3 + angular_2[l * l + m, 2] * radial_4
                        orbital_hessian[i, 0, 0, ao + m] = x*x * angular_1[l*l+m] * radial_2 + (angular_1[l*l+m] + 2 * x * angular_2[l*l+m, 0]) * radial_3 + angular_3[l*l+m, 0] * radial_4  # fmt: skip
                        orbital_hessian[i, 0, 1, ao + m] = x*y * angular_1[l*l+m] * radial_2 + (y * angular_2[l*l+m, 0] + x * angular_2[l*l+m, 1]) * radial_3 + angular_3[l*l+m, 1] * radial_4  # fmt: skip
                        orbital_hessian[i, 0, 2, ao + m] = x*z * angular_1[l*l+m] * radial_2 + (z * angular_2[l*l+m, 0] + x * angular_2[l*l+m, 2]) * radial_3 + angular_3[l*l+m, 2] * radial_4  # fmt: skip
                        orbital_hessian[i, 1, 0, ao + m] = orbital_hessian[i, 0, 1, ao + m]
                        orbital_hessian[i, 1, 1, ao + m] = y*y * angular_1[l*l+m] * radial_2 + (angular_1[l*l+m] + 2 * y * angular_2[l*l+m, 1]) * radial_3 + angular_3[l*l+m, 3] * radial_4  # fmt: skip
                        orbital_hessian[i, 1, 2, ao + m] = y*z * angular_1[l*l+m] * radial_2 + (z * angular_2[l*l+m, 1] + y * angular_2[l*l+m, 2]) * radial_3 + angular_3[l*l+m, 4] * radial_4  # fmt: skip
                        orbital_hessian[i, 2, 0, ao + m] = orbital_hessian[i, 0, 2, ao + m]
                        orbital_hessian[i, 2, 1, ao + m] = orbital_hessian[i, 1, 2, ao + m]
                        orbital_hessian[i, 2, 2, ao + m] = z*z * angular_1[l*l+m] * radial_2 + (angular_1[l*l+m] + 2 * z * angular_2[l*l+m, 2]) * radial_3 + angular_3[l*l+m, 5] * radial_4  # fmt: skip
                        orbital[i, 0, 0, 0, ao + m] = x*x*x * angular_1[l*l+m] * radial_1 + 3*x*(angular_1[l*l+m] + x*angular_2[l*l+m, 0]) * radial_2 + 3*(angular_2[l*l+m, 0] + x * angular_3[l*l+m, 0]) * radial_3 + angular_4[l*l+m, 0] * radial_4  # fmt: skip
                        orbital[i, 0, 0, 1, ao + m] = x*x*y * angular_1[l*l+m] * radial_1 + (y*angular_1[l*l+m] + 2*x*y*angular_2[l*l+m, 0] + x*x*angular_2[l*l+m, 1]) * radial_2 + (angular_2[l*l+m, 1] + 2*x*angular_3[l*l+m, 1] + y*angular_3[l*l+m, 0]) * radial_3 + angular_4[l*l+m, 1] * radial_4  # fmt: skip
                        orbital[i, 0, 0, 2, ao + m] = x*x*z * angular_1[l*l+m] * radial_1 + (z*angular_1[l*l+m] + 2*x*z*angular_2[l*l+m, 0] + x*x*angular_2[l*l+m, 2]) * radial_2 + (angular_2[l*l+m, 2] + 2*x*angular_3[l*l+m, 2] + z*angular_3[l*l+m, 0]) * radial_3 + angular_4[l*l+m, 2] * radial_4  # fmt: skip
                        orbital[i, 0, 1, 0, ao + m] = orbital[i, 0, 0, 1, ao + m]
                        orbital[i, 0, 1, 1, ao + m] = x*y*y * angular_1[l*l+m] * radial_1 + (x*angular_1[l*l+m] + 2*x*y*angular_2[l*l+m, 1] + y*y*angular_2[l*l+m, 0]) * radial_2 + (angular_2[l*l+m, 0] + 2*y*angular_3[l*l+m, 1] + x*angular_3[l*l+m, 3]) * radial_3 + angular_4[l*l+m, 3] * radial_4  # fmt: skip
                        orbital[i, 0, 1, 2, ao + m] = x*y*z * angular_1[l*l+m] * radial_1 + (y*z*angular_2[l*l+m, 0] + x*z*angular_2[l*l+m, 1] + x*y*angular_2[l*l+m, 2]) * radial_2 + (z*angular_3[l*l+m, 1] + y*angular_3[l*l+m, 2] + x*angular_3[l*l+m, 4]) * radial_3 + angular_4[l*l+m, 4] * radial_4  # fmt: skip
                        orbital[i, 0, 2, 0, ao + m] = orbital[i, 0, 0, 2, ao + m]
                        orbital[i, 0, 2, 1, ao + m] = orbital[i, 0, 1, 2, ao + m]
                        orbital[i, 0, 2, 2, ao + m] = x*z*z * angular_1[l*l+m] * radial_1 + (x*angular_1[l*l+m] + 2*x*z*angular_2[l*l+m, 2] + z*z*angular_2[l*l+m, 0]) * radial_2 + (angular_2[l*l+m, 0] + 2*z*angular_3[l*l+m, 2] + x*angular_3[l*l+m, 5]) * radial_3 + angular_4[l*l+m, 5] * radial_4  # fmt: skip
                        orbital[i, 1, 0, 0, ao + m] = orbital[i, 0, 0, 1, ao + m]
                        orbital[i, 1, 0, 1, ao + m] = orbital[i, 0, 1, 1, ao + m]
                        orbital[i, 1, 0, 2, ao + m] = orbital[i, 0, 1, 2, ao + m]
                        orbital[i, 1, 1, 0, ao + m] = orbital[i, 0, 1, 1, ao + m]
                        orbital[i, 1, 1, 1, ao + m] = y*y*y * angular_1[l*l+m] * radial_1 + 3*y*(angular_1[l*l+m] + y*angular_2[l*l+m, 1]) * radial_2 + 3*(angular_2[l*l+m, 1] + y * angular_3[l*l+m, 3]) * radial_3 + angular_4[l*l+m, 6] * radial_4  # fmt: skip
                        orbital[i, 1, 1, 2, ao + m] = y*y*z * angular_1[l*l+m] * radial_1 + (z*angular_1[l*l+m] + 2*y*z*angular_2[l*l+m, 1] + y*y*angular_2[l*l+m, 2]) * radial_2 + (angular_2[l*l+m, 2] + 2*y*angular_3[l*l+m, 4] + z*angular_3[l*l+m, 3]) * radial_3 + angular_4[l*l+m, 7] * radial_4  # fmt: skip
                        orbital[i, 1, 2, 0, ao + m] = orbital[i, 0, 1, 2, ao + m]
                        orbital[i, 1, 2, 1, ao + m] = orbital[i, 1, 1, 2, ao + m]
                        orbital[i, 1, 2, 2, ao + m] = y*z*z * angular_1[l*l+m] * radial_1 + (y*angular_1[l*l+m] + 2*y*z*angular_2[l*l+m, 2] + z*z*angular_2[l*l+m, 1]) * radial_2 + (angular_2[l*l+m, 1] + 2*z*angular_3[l*l+m, 4] + y*angular_3[l*l+m, 5]) * radial_3 + angular_4[l*l+m, 8] * radial_4  # fmt: skip
                        orbital[i, 2, 0, 0, ao + m] = orbital[i, 0, 0, 2, ao + m]
                        orbital[i, 2, 0, 1, ao + m] = orbital[i, 0, 1, 2, ao + m]
                        orbital[i, 2, 0, 2, ao + m] = orbital[i, 0, 2, 2, ao + m]
                        orbital[i, 2, 1, 0, ao + m] = orbital[i, 0, 1, 2, ao + m]
                        orbital[i, 2, 1, 1, ao + m] = orbital[i, 1, 1, 2, ao + m]
                        orbital[i, 2, 1, 2, ao + m] = orbital[i, 1, 2, 2, ao + m]
                        orbital[i, 2, 2, 0, ao + m] = orbital[i, 0, 2, 2, ao + m]
                        orbital[i, 2, 2, 1, ao + m] = orbital[i, 1, 2, 2, ao + m]
                        orbital[i, 2, 2, 2, ao + m] = z*z*z * angular_1[l*l+m] * radial_1 + 3*z*(angular_1[l*l+m] + z*angular_2[l*l+m, 2]) * radial_2 + 3*(angular_2[l*l+m, 2] + z * angular_3[l*l+m, 5]) * radial_3 + angular_4[l*l+m, 9] * radial_4  # fmt: skip
                    ao += 2 * l + 1

        ao_value = self.norm * orbital_value
        pool_u = self.mo_up @ ao_value[: self.neu].T
        pool_d = self.mo_down @ ao_value[self.neu :].T
        ao_gradient = self.norm * orbital_gradient.reshape((self.neu + self.ned) * 3, self.nbasis_functions)
        pool_grad_u = (self.mo_up @ ao_gradient[: self.neu * 3].T).reshape(self.norb, self.neu, 3)
        pool_grad_d = (self.mo_down @ ao_gradient[self.neu * 3 :].T).reshape(self.norb, self.ned, 3)
        ao_hessian = self.norm * orbital_hessian.reshape((self.neu + self.ned) * 9, self.nbasis_functions)
        pool_hess_u = (self.mo_up @ ao_hessian[: self.neu * 9].T).reshape(self.norb, self.neu, 3, 3)
        pool_hess_d = (self.mo_down @ ao_hessian[self.neu * 9 :].T).reshape(self.norb, self.ned, 3, 3)
        ao_tressian = self.norm * orbital.reshape((self.neu + self.ned) * 27, self.nbasis_functions)
        pool_tress_u = (self.mo_up @ ao_tressian[: self.neu * 27].T).reshape(self.norb, self.neu, 3, 3, 3)
        pool_tress_d = (self.mo_down @ ao_tressian[self.neu * 27 :].T).reshape(self.norb, self.ned, 3, 3, 3)
        if self.cusp is not None:
            cusp_value_u, cusp_value_d = self.cusp.value(n_vectors)
            cusp_gradient_u, cusp_gradient_d = self.cusp.gradient(n_vectors)
            cusp_hessian_u, cusp_hessian_d = self.cusp.hessian(n_vectors)
            cusp_tressian_u, cusp_tressian_d = self.cusp.tressian(n_vectors)
            pool_u += cusp_value_u
            pool_d += cusp_value_d
            pool_grad_u += cusp_gradient_u
            pool_grad_d += cusp_gradient_d
            pool_hess_u += cusp_hessian_u
            pool_hess_d += cusp_hessian_d
            pool_tress_u += cusp_tressian_u
            pool_tress_d += cusp_tressian_d
        return pool_u, pool_d, pool_grad_u, pool_grad_d, pool_hess_u, pool_hess_d, pool_tress_u, pool_tress_d

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'geminal_matrix')
def geminal_geminal_matrix(self, pool_u, pool_d, n):
    """The neu x neu matrix M of geminal n: [ Phi(up, down) | unpaired ]."""

    def impl(self, pool_u, pool_d, n) -> np.ndarray:
        matrix = np.empty(shape=(self.neu, self.neu))
        matrix[:, : self.ned] = pool_u.T @ (self.g[n] @ pool_d)
        if self.nunpaired:
            matrix[:, self.ned :] = pool_u.T @ self.u[n]
        return matrix

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'pool_value')
def geminal_pool_value(self, pool_u, pool_d):
    """Wave function value = sum_n c_n det(M_n) from an orbital pool that is already built.
    The pool does not depend on the geminal parameters, so a derivative w.r.t. them walks over
    the basis once and only redoes the determinants.
    """

    def impl(self, pool_u, pool_d) -> float:
        val = 0.0
        for n in range(self.c.size):
            val += self.c[n] * np.linalg.det(self.geminal_matrix(pool_u, pool_d, n))
        return val

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'value')
def geminal_value(self, n_vectors: np.ndarray):
    """Wave function value = sum_n c_n det(M_n)."""

    def impl(self, n_vectors: np.ndarray) -> float:
        pool_u, pool_d = self.pool_matrix(n_vectors)
        return self.pool_value(pool_u, pool_d)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'log_value')
def geminal_log_value(self, n_vectors: np.ndarray):
    """Logarithm of the absolute wave function value, and its sign."""

    def impl(self, n_vectors: np.ndarray) -> tuple[float, float]:
        pool_u, pool_d = self.pool_matrix(n_vectors)
        log_det = np.empty(shape=self.c.size)
        sign_det = np.empty(shape=self.c.size)
        for n in range(self.c.size):
            sign, log = np.linalg.slogdet(self.geminal_matrix(pool_u, pool_d, n))
            log_det[n] = log + np.log(np.abs(self.c[n]))
            sign_det[n] = sign * np.sign(self.c[n])
        shift = np.max(log_det)
        val = np.sum(sign_det * np.exp(log_det - shift))
        return shift + np.log(np.abs(val)), np.sign(val)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'pool_gradient')
def geminal_pool_gradient(self, pool_u, pool_d, pool_grad_u, pool_grad_d):
    """Gradient ∇ψ/ψ from an orbital pool that is already built."""

    def impl(self, pool_u, pool_d, pool_grad_u, pool_grad_d) -> np.ndarray:
        val = 0.0
        grad = np.zeros(shape=(self.neu + self.ned) * 3)
        single = self.c.size == 1
        for n in range(self.c.size):
            full_c = np.empty(shape=(self.norb, self.neu))
            full_c[:, : self.ned] = self.g[n] @ pool_d
            if self.nunpaired:
                full_c[:, self.ned :] = self.u[n]
            matrix = pool_u.T @ full_c
            inv = np.linalg.inv(matrix)
            a = pool_u.T @ self.g[n]
            tr_grad_u = np.zeros(shape=(self.neu, 3))
            tr_grad_d = np.zeros(shape=(self.ned, 3))
            for d in range(3):
                grad_matrix_u = pool_grad_u[:, :, d].T @ full_c
                tr_grad_u[:, d] = (inv.T * grad_matrix_u).sum(axis=1)
                grad_matrix_d = a @ pool_grad_d[:, :, d]
                tr_grad_d[:, d] = (inv[: self.ned] * grad_matrix_d.T).sum(axis=1)
            tr_grad = np.concatenate((tr_grad_u.ravel(), tr_grad_d.ravel()))
            c = 1.0 if single else self.c[n] * np.linalg.det(matrix)
            val += c
            grad += c * tr_grad
        return grad / val

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'gradient')
def geminal_gradient(self, n_vectors: np.ndarray):
    """Gradient ∇ψ/ψ w.r.t e-coordinates."""

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        pool_u, pool_d, pool_grad_u, pool_grad_d = self.pool_gradient_matrix(n_vectors)
        return self.pool_gradient(pool_u, pool_d, pool_grad_u, pool_grad_d)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'pool_laplacian')
def geminal_pool_laplacian(self, pool_u, pool_d, pool_lap_u, pool_lap_d):
    """Scalar laplacian Δψ/ψ from an orbital pool that is already built."""

    def impl(self, pool_u, pool_d, pool_lap_u, pool_lap_d) -> float:
        val = 0.0
        lap = 0.0
        single = self.c.size == 1
        for n in range(self.c.size):
            full_c = np.empty(shape=(self.norb, self.neu))
            full_c[:, : self.ned] = self.g[n] @ pool_d
            if self.nunpaired:
                full_c[:, self.ned :] = self.u[n]
            matrix = pool_u.T @ full_c
            inv = np.linalg.inv(matrix)
            lap_matrix_u = pool_lap_u.T @ full_c
            tr_lap = (inv.T * lap_matrix_u).sum()
            lap_matrix_d = (pool_u.T @ self.g[n]) @ pool_lap_d
            tr_lap += (inv[: self.ned] * lap_matrix_d.T).sum()
            c = 1.0 if single else self.c[n] * np.linalg.det(matrix)
            val += c
            lap += c * tr_lap
        return lap / val

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'laplacian')
def geminal_laplacian(self, n_vectors: np.ndarray):
    """Scalar laplacian Δψ/ψ w.r.t e-coordinates."""

    def impl(self, n_vectors: np.ndarray) -> float:
        pool_u, pool_d, pool_lap_u, pool_lap_d = self.pool_laplacian_matrix(n_vectors)
        return self.pool_laplacian(pool_u, pool_d, pool_lap_u, pool_lap_d)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'pool_hessian')
def geminal_pool_hessian(self, pool_u, pool_d, pool_grad_u, pool_grad_d, pool_hess_u, pool_hess_d):
    """Hessian H(ψ)/ψ and gradient ∇ψ/ψ from an orbital pool that is already built.
    An entry of the geminal matrix carries an up electron through its row and a down electron
    through its column, so unlike a slater determinant it has a mixed second derivative, and it
    is the up-down block below that a determinant does not have. With
        d²ln(det(M)) = tr(M^-1 • d²M) - tr(M^-1 • dM/dx • M^-1 • dM/dy)
    the first trace survives only where the two derivatives touch the same entry - both on the
    row of one up electron, both on the column of one down electron, or one of each - and the
    second is read off the three products the two derivative shapes, a row and a column, make.
    """

    def impl(self, pool_u, pool_d, pool_grad_u, pool_grad_d, pool_hess_u, pool_hess_d) -> tuple[np.ndarray, np.ndarray]:
        ne = self.neu + self.ned
        val = 0.0
        grad = np.zeros(shape=ne * 3)
        hess = np.zeros(shape=(ne * 3, ne * 3))
        single = self.c.size == 1
        for n in range(self.c.size):
            full_c = np.empty(shape=(self.norb, self.neu))
            full_c[:, : self.ned] = self.g[n] @ pool_d
            if self.nunpaired:
                full_c[:, self.ned :] = self.u[n]
            matrix = pool_u.T @ full_c
            inv = np.linalg.inv(matrix)
            a = pool_u.T @ self.g[n]
            flat_grad_u = pool_grad_u.reshape(self.norb, self.neu * 3)
            flat_grad_d = pool_grad_d.reshape(self.norb, self.ned * 3)

            # dM of an up electron fills one row of the matrix, dM of a down electron one column
            gu = flat_grad_u.T @ full_c
            gd = (a @ flat_grad_d).reshape(self.neu, self.ned, 3)
            # the entry the two of them share, the only second derivative across two electrons
            x = flat_grad_u.T @ (self.g[n] @ flat_grad_d)
            # and the second derivatives of one electron with itself
            hu = (pool_hess_u.reshape(self.norb, self.neu * 9).T @ full_c).reshape(self.neu, 3, 3, self.neu)
            hd = (a @ pool_hess_d.reshape(self.norb, self.ned * 9)).reshape(self.neu, self.ned, 3, 3)

            # M^-1 • dM is a column of the inverse times a row for an up electron, and a column
            # times a unit row for a down one, so their products need only these three
            p = gu @ inv
            w = inv @ gd.reshape(self.neu, self.ned * 3)
            q = gu @ w

            tr_grad_u = np.zeros(shape=(self.neu, 3))
            tr_grad_d = np.zeros(shape=(self.ned, 3))
            for i in range(self.neu):
                for d in range(3):
                    tr_grad_u[i, d] = gu[i * 3 + d] @ inv[:, i]
            for j in range(self.ned):
                for d in range(3):
                    tr_grad_d[j, d] = inv[j] @ gd[:, j, d]

            h = np.zeros(shape=(ne * 3, ne * 3))
            for i in range(self.neu):
                for d1 in range(3):
                    for i2 in range(self.neu):
                        for d2 in range(3):
                            h[i * 3 + d1, i2 * 3 + d2] = -p[i * 3 + d1, i2] * p[i2 * 3 + d2, i]
                    for d2 in range(3):
                        h[i * 3 + d1, i * 3 + d2] += inv[:, i] @ hu[i, d1, d2]
                    for j in range(self.ned):
                        for d2 in range(3):
                            v = inv[j, i] * (x[i * 3 + d1, j * 3 + d2] - q[i * 3 + d1, j * 3 + d2])
                            h[i * 3 + d1, (self.neu + j) * 3 + d2] = v
                            h[(self.neu + j) * 3 + d2, i * 3 + d1] = v
            for j in range(self.ned):
                for d1 in range(3):
                    for j2 in range(self.ned):
                        for d2 in range(3):
                            h[(self.neu + j) * 3 + d1, (self.neu + j2) * 3 + d2] = -w[j2, j * 3 + d1] * w[j, j2 * 3 + d2]
                    for d2 in range(3):
                        h[(self.neu + j) * 3 + d1, (self.neu + j) * 3 + d2] += inv[j] @ hd[:, j, d1, d2]

            tr_grad = np.concatenate((tr_grad_u.ravel(), tr_grad_d.ravel()))
            c = 1.0 if single else self.c[n] * np.linalg.det(matrix)
            val += c
            grad += c * tr_grad
            # d²ln(ψ) + ∇ln(ψ) ⊗ ∇ln(ψ) is the hessian over the value, as slater.hessian returns it
            hess += c * (h + np.outer(tr_grad, tr_grad))
        return hess / val, grad / val

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'hessian')
def geminal_hessian(self, n_vectors: np.ndarray):
    """Hessian H(ψ)/ψ and gradient ∇ψ/ψ w.r.t e-coordinates."""

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pool_u, pool_d, pool_grad_u, pool_grad_d, pool_hess_u, pool_hess_d = self.pool_hessian_matrix(n_vectors)
        return self.pool_hessian(pool_u, pool_d, pool_grad_u, pool_grad_d, pool_hess_u, pool_hess_d)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'pool_tressian_dot')
def geminal_pool_tressian_dot(self, pool_u, pool_d, pool_grad_u, pool_grad_d, pool_hess_u, pool_hess_d, pool_tress_u, pool_tress_d, bb):
    """Tressian contracted over its last two axes with a symmetric matrix bb, together with the
    hessian and the gradient, as slater.tressian_dot returns them.

    M^-1 • dM is of rank one whatever the coordinate is: an up electron fills a row of the
    geminal matrix, so it gives a column of the inverse times a row of orbital gradients, and a
    down electron fills a column, so it gives a column times a unit row. Every trace of a product
    of such factors therefore collapses into the one matrix
        s[a, b] = row(a) • col(b),
    with tr(M^-1 • dM/da) = s[a, a] and tr(M^-1 • dM/da • M^-1 • dM/db) = s[a, b] • s[b, a]. The
    two triple products of the third derivative are s[a,b]s[b,c]s[c,a] and s[a,c]s[c,b]s[b,a],
    which a symmetric bb makes equal, and both contract into the diagonal of s • (s∘bb) • s.
    The second derivatives are rank one as well - a row for two derivatives of one up electron, a
    column for two of one down electron, a single entry for one of each - which is what leaves
    the loop below over pairs rather than over triples.
    """

    def impl(self, pool_u, pool_d, pool_grad_u, pool_grad_d, pool_hess_u, pool_hess_d, pool_tress_u, pool_tress_d, bb):
        ne = self.neu + self.ned
        val = 0.0
        grad = np.zeros(shape=ne * 3)
        hess = np.zeros(shape=(ne * 3, ne * 3))
        tress_bb = np.zeros(shape=ne * 3)
        single = self.c.size == 1
        for n in range(self.c.size):
            full_c = np.empty(shape=(self.norb, self.neu))
            full_c[:, : self.ned] = self.g[n] @ pool_d
            if self.nunpaired:
                full_c[:, self.ned :] = self.u[n]
            matrix = pool_u.T @ full_c
            inv = np.linalg.inv(matrix)
            a = pool_u.T @ self.g[n]
            flat_grad_u = pool_grad_u.reshape(self.norb, self.neu * 3)
            flat_grad_d = pool_grad_d.reshape(self.norb, self.ned * 3)
            flat_hess_u = pool_hess_u.reshape(self.norb, self.neu * 9)
            flat_hess_d = pool_hess_d.reshape(self.norb, self.ned * 9)

            gu = flat_grad_u.T @ full_c
            gd = (a @ flat_grad_d).reshape(self.neu, self.ned, 3)
            hu = (flat_hess_u.T @ full_c).reshape(self.neu, 3, 3, self.neu)
            hd = (a @ flat_hess_d).reshape(self.neu, self.ned, 3, 3)
            tu = pool_tress_u.reshape(self.norb, self.neu * 27).T @ full_c
            td = (a @ pool_tress_d.reshape(self.norb, self.ned * 27)).reshape(self.neu, self.ned, 3, 3, 3)
            # the derivatives that fall on one up and one down electron at once
            xg = flat_grad_u.T @ (self.g[n] @ flat_grad_d)
            xhu = flat_hess_u.T @ (self.g[n] @ flat_grad_d)
            xhd = flat_grad_u.T @ (self.g[n] @ flat_hess_d)

            col = np.zeros(shape=(ne * 3, self.neu))
            row = np.zeros(shape=(ne * 3, self.neu))
            for i in range(self.neu):
                for d in range(3):
                    col[i * 3 + d] = inv[:, i]
                    row[i * 3 + d] = gu[i * 3 + d]
            w = inv @ gd.reshape(self.neu, self.ned * 3)
            for j in range(self.ned):
                for d in range(3):
                    col[(self.neu + j) * 3 + d] = w[:, j * 3 + d]
                    row[(self.neu + j) * 3 + d, j] = 1.0
            s = row @ col.T
            ri = row @ inv

            tr_grad = np.zeros(shape=ne * 3)
            for k in range(ne * 3):
                tr_grad[k] = s[k, k]

            # d²ln(det), the hessian of the logarithm
            h = -s * s.T
            for i in range(self.neu):
                for d1 in range(3):
                    for d2 in range(3):
                        h[i * 3 + d1, i * 3 + d2] += inv[:, i] @ hu[i, d1, d2]
                    for j in range(self.ned):
                        for d2 in range(3):
                            v = inv[j, i] * xg[i * 3 + d1, j * 3 + d2]
                            h[i * 3 + d1, (self.neu + j) * 3 + d2] += v
                            h[(self.neu + j) * 3 + d2, i * 3 + d1] += v
            for j in range(self.ned):
                for d1 in range(3):
                    for d2 in range(3):
                        h[(self.neu + j) * 3 + d1, (self.neu + j) * 3 + d2] += inv[j] @ hd[:, j, d1, d2]

            # tr(M^-1 • d³M), which needs the three coordinates to meet on the same entry
            t3 = np.zeros(shape=ne * 3)
            for i in range(self.neu):
                for p in range(3):
                    acc = 0.0
                    for q in range(3):
                        for r in range(3):
                            acc += bb[i * 3 + q, i * 3 + r] * (inv[:, i] @ tu[i * 27 + p * 9 + q * 3 + r])
                        for j in range(self.ned):
                            for r in range(3):
                                acc += 2 * bb[i * 3 + q, (self.neu + j) * 3 + r] * inv[j, i] * xhu[i * 9 + p * 3 + q, j * 3 + r]
                    for j in range(self.ned):
                        for q in range(3):
                            for r in range(3):
                                acc += bb[(self.neu + j) * 3 + q, (self.neu + j) * 3 + r] * inv[j, i] * xhd[i * 3 + p, j * 9 + q * 3 + r]
                    t3[i * 3 + p] += acc
            for j in range(self.ned):
                for p in range(3):
                    acc = 0.0
                    for q in range(3):
                        for r in range(3):
                            acc += bb[(self.neu + j) * 3 + q, (self.neu + j) * 3 + r] * (inv[j] @ td[:, j, p, q, r])
                    for i in range(self.neu):
                        for q in range(3):
                            for r in range(3):
                                acc += bb[i * 3 + q, i * 3 + r] * inv[j, i] * xhu[i * 9 + q * 3 + r, j * 3 + p]
                            for r in range(3):
                                acc += 2 * bb[i * 3 + q, (self.neu + j) * 3 + r] * inv[j, i] * xhd[i * 3 + q, j * 9 + p * 3 + r]
                    t3[(self.neu + j) * 3 + p] += acc

            # tr(M^-1 • d²M • M^-1 • dM), over the pairs whose second derivative does not vanish
            pair_1 = np.zeros(shape=ne * 3)
            pair_2 = np.zeros(shape=ne * 3)
            for i in range(self.neu):
                for d1 in range(3):
                    for d2 in range(3):
                        d_vec = (hu[i, d1, d2] @ col.T) * ri[:, i]
                        pair_1[i * 3 + d1] += bb[i * 3 + d2] @ d_vec
                        pair_2 += bb[i * 3 + d1, i * 3 + d2] * d_vec
                    for j in range(self.ned):
                        for d2 in range(3):
                            d_vec = col[:, j] * (xg[i * 3 + d1, j * 3 + d2] * ri[:, i])
                            pair_1[i * 3 + d1] += bb[(self.neu + j) * 3 + d2] @ d_vec
                            pair_1[(self.neu + j) * 3 + d2] += bb[i * 3 + d1] @ d_vec
                            pair_2 += 2 * bb[i * 3 + d1, (self.neu + j) * 3 + d2] * d_vec
            for j in range(self.ned):
                for d1 in range(3):
                    for d2 in range(3):
                        d_vec = col[:, j] * (row @ (inv @ hd[:, j, d1, d2]))
                        pair_1[(self.neu + j) * 3 + d1] += bb[(self.neu + j) * 3 + d2] @ d_vec
                        pair_2 += bb[(self.neu + j) * 3 + d1, (self.neu + j) * 3 + d2] * d_vec

            triple = np.zeros(shape=ne * 3)
            sks = s @ (bb * s) @ s
            for k in range(ne * 3):
                triple[k] = 2 * sks[k, k]

            partial_hess = h + np.outer(tr_grad, tr_grad) / 3
            c = 1.0 if single else self.c[n] * np.linalg.det(matrix)
            val += c
            grad += c * tr_grad
            hess += c * (partial_hess + 2 / 3 * np.outer(tr_grad, tr_grad))
            # Σ_bc bb[b,c] (g[c]·PH[a,b] + g[b]·PH[a,c] + g[a]·PH[b,c]), as slater.tressian_dot
            outer = 2 * (partial_hess @ (bb @ tr_grad)) + tr_grad * np.sum(partial_hess * bb)
            tress_bb += c * (t3 - 2 * pair_1 - pair_2 + triple + outer)
        return tress_bb / val, hess / val, grad / val

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'tressian_dot')
def geminal_tressian_dot(self, n_vectors: np.ndarray, bb: np.ndarray):
    """Tressian contracted with bb, hessian and gradient w.r.t e-coordinates."""

    def impl(self, n_vectors: np.ndarray, bb: np.ndarray):
        pool_u, pool_d, pool_grad_u, pool_grad_d, pool_hess_u, pool_hess_d, pool_tress_u, pool_tress_d = self.pool_tressian_matrix(n_vectors)
        return self.pool_tressian_dot(pool_u, pool_d, pool_grad_u, pool_grad_d, pool_hess_u, pool_hess_d, pool_tress_u, pool_tress_d, bb)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'fix_parameters')
def geminal_fix_parameters(self):
    """Restore the parameters that the constraints determine from the reference of their group."""

    def impl(self):
        for i in range(self.c_ties.shape[0]):
            self.c[self.c_ties[i, 0]] = self.c[self.c_ties[i, 1]]
        for i in range(self.g_ties.shape[0]):
            value = self.g[self.g_ties[i, 3], self.g_ties[i, 4], self.g_ties[i, 5]]
            self.g[self.g_ties[i, 0], self.g_ties[i, 1], self.g_ties[i, 2]] = value
            self.g[self.g_ties[i, 0], self.g_ties[i, 2], self.g_ties[i, 1]] = value
        for i in range(self.u_ties.shape[0]):
            self.u[self.u_ties[i, 0], self.u_ties[i, 1], self.u_ties[i, 2]] = self.u[self.u_ties[i, 3], self.u_ties[i, 4], self.u_ties[i, 5]]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'get_parameters_mask')
def geminal_get_parameters_mask(self):
    """Mask dependent parameters. Every element a constraint determines is dependent, only the
    reference of its group is varied.
    """

    def impl(self) -> np.ndarray:
        res = []
        for n in range(self.c.size):
            res.append(self.c_mask[n])
            for i in range(self.norb):
                for j in range(i, self.norb):
                    if self.g_available[n, i, j]:
                        res.append(self.g_mask[n, i, j])
            for i in range(self.norb):
                for k in range(self.nunpaired):
                    if self.u_available[n, i, k]:
                        res.append(self.u_mask[n, i, k])
        return np.array(res)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'get_parameters_scale')
def geminal_get_parameters_scale(self, all_parameters):
    """Characteristic scale of each variable. The geminal matrix is defined up to a common
    factor and its elements are of order one, so they are their own scale.
    """

    def impl(self, all_parameters) -> np.ndarray:
        return np.ones(shape=self.get_parameters(all_parameters).shape)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'get_parameters_constraints')
def geminal_get_parameters_constraints(self):
    """Constraints x_determined - x_reference = 0, one row per tied parameter, in the order the
    parameters are laid out by get_parameters.
    """

    def impl(self):
        index_c = np.zeros(shape=self.c.shape, dtype=np.int64)
        index_g = np.zeros(shape=self.g.shape, dtype=np.int64)
        index_u = np.zeros(shape=self.u.shape, dtype=np.int64)
        size = 0
        for n in range(self.c.size):
            index_c[n] = size
            size += 1
            for i in range(self.norb):
                for j in range(i, self.norb):
                    if self.g_available[n, i, j]:
                        index_g[n, i, j] = size
                        size += 1
            for i in range(self.norb):
                for k in range(self.nunpaired):
                    if self.u_available[n, i, k]:
                        index_u[n, i, k] = size
                        size += 1
        ties = self.c_ties.shape[0] + self.g_ties.shape[0] + self.u_ties.shape[0]
        # an all-zero row leaves the projector an identity when nothing is tied
        a = np.zeros(shape=(max(ties, 1), size))
        row = 0
        for i in range(self.c_ties.shape[0]):
            a[row, index_c[self.c_ties[i, 0]]] = 1
            a[row, index_c[self.c_ties[i, 1]]] = -1
            row += 1
        for i in range(self.g_ties.shape[0]):
            a[row, index_g[self.g_ties[i, 0], self.g_ties[i, 1], self.g_ties[i, 2]]] = 1
            a[row, index_g[self.g_ties[i, 3], self.g_ties[i, 4], self.g_ties[i, 5]]] = -1
            row += 1
        for i in range(self.u_ties.shape[0]):
            a[row, index_u[self.u_ties[i, 0], self.u_ties[i, 1], self.u_ties[i, 2]]] = 1
            a[row, index_u[self.u_ties[i, 3], self.u_ties[i, 4], self.u_ties[i, 5]]] = -1
            row += 1
        return a, np.zeros(shape=(a.shape[0],))

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'set_parameters_projector')
def geminal_set_parameters_projector(self):
    """Set Projector matrix"""

    def impl(self):
        a, b = self.get_parameters_constraints()
        p = np.eye(a.shape[1]) - a.T @ np.linalg.pinv(a.T)
        mask_idx = np.argwhere(self.get_parameters_mask()).ravel()
        inv_p = np.linalg.inv(p[:, mask_idx][mask_idx, :])
        self.parameters_projector = p[:, mask_idx] @ inv_p

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'get_parameters')
def geminal_get_parameters(self, all_parameters):
    """Returns parameters in the following order:
    for every geminal: its c coefficient, the upper triangle of g, the unpaired columns u.
    :param all_parameters:
    :return:
    """

    def impl(self, all_parameters):
        res = []
        for n in range(self.c.size):
            if self.c_mask[n] or all_parameters:
                res.append(self.c[n])
            for i in range(self.norb):
                for j in range(i, self.norb):
                    if (self.g_mask[n, i, j] or all_parameters) and self.g_available[n, i, j]:
                        res.append(self.g[n, i, j])
            for i in range(self.norb):
                for k in range(self.nunpaired):
                    if (self.u_mask[n, i, k] or all_parameters) and self.u_available[n, i, k]:
                        res.append(self.u[n, i, k])
        return np.array(res)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'set_parameters')
def geminal_set_parameters(self, parameters, all_parameters):
    """Set parameters in the following order:
    for every geminal: its c coefficient, the upper triangle of g, the unpaired columns u.
    :param parameters:
    :param all_parameters:
    :return:
    """

    def impl(self, parameters, all_parameters):
        n_par = 0
        for n in range(self.c.size):
            if self.c_mask[n] or all_parameters:
                self.c[n] = parameters[n_par]
                n_par += 1
            for i in range(self.norb):
                for j in range(i, self.norb):
                    if (self.g_mask[n, i, j] or all_parameters) and self.g_available[n, i, j]:
                        self.g[n, i, j] = self.g[n, j, i] = parameters[n_par]
                        n_par += 1
            for i in range(self.norb):
                for k in range(self.nunpaired):
                    if (self.u_mask[n, i, k] or all_parameters) and self.u_available[n, i, k]:
                        self.u[n, i, k] = parameters[n_par]
                        n_par += 1
        if not all_parameters:
            self.fix_parameters()
        return parameters[n_par:]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'parameters_weights')
def geminal_parameters_weights(self, pool_u, pool_d):
    """What every derivative w.r.t. the geminal parameters is built out of, per geminal:
    the weight c_n·det(M_n)/ψ it carries in the sum, the inverse of its matrix, and the traces
        tr(M^-1 • dM/dp)
    that give d(ln(c_n·det(M_n)))/dp. A parameter of g or of u changes the geminal matrix by a
    rank one term - a column of the orbital pool of the up electrons times a row of the pool of
    the down ones, or times a unit row for an unpaired column - so the trace of the product with
    the inverse is one element of a matrix product rather than a determinant of its own.
    :return: weights, the same over c so that a zero coefficient does not divide, and
        iu = M^-1 • pool_u.T and zg = pool_d(padded) • iu of every geminal, the two of which
        hold every trace the parameters need
    """

    def impl(self, pool_u, pool_d):
        ngem = self.c.size
        weights = np.zeros(shape=ngem)
        c_weights = np.zeros(shape=ngem)
        iu = np.zeros(shape=(ngem, self.neu, self.norb))
        zg = np.zeros(shape=(ngem, self.norb, self.norb))
        # the down-spin pool padded over the unpaired columns, which no down electron enters
        pool_d_full = np.zeros(shape=(self.norb, self.neu))
        pool_d_full[:, : self.ned] = pool_d
        val = 0.0
        for n in range(ngem):
            full_c = np.empty(shape=(self.norb, self.neu))
            full_c[:, : self.ned] = self.g[n] @ pool_d
            if self.nunpaired:
                full_c[:, self.ned :] = self.u[n]
            matrix = pool_u.T @ full_c
            iu[n] = np.linalg.inv(matrix) @ pool_u.T
            zg[n] = pool_d_full @ iu[n]
            c_weights[n] = np.linalg.det(matrix)
            weights[n] = self.c[n] * c_weights[n]
            val += weights[n]
        return weights / val, c_weights / val, iu, zg

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'value_parameters_d1')
def geminal_value_parameters_d1(self, n_vectors: np.ndarray):
    """First derivatives of logarithm wfn w.r.t. the parameters
    :param n_vectors: e-n vectors
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        pool_u, pool_d = self.pool_matrix(n_vectors)
        weights, c_weights, iu, zg = self.parameters_weights(pool_u, pool_d)
        res = np.zeros(shape=self.get_parameters(True).size)
        n_par = 0
        for n in range(self.c.size):
            res[n_par] = c_weights[n]
            n_par += 1
            for i in range(self.norb):
                for j in range(i, self.norb):
                    if self.g_available[n, i, j]:
                        # g is symmetric and its upper triangle is what a parameter is, so an
                        # off-diagonal one moves the matrix on both sides of the diagonal
                        if i == j:
                            res[n_par] = weights[n] * zg[n, i, i]
                        else:
                            res[n_par] = weights[n] * (zg[n, j, i] + zg[n, i, j])
                        n_par += 1
            for i in range(self.norb):
                for k in range(self.nunpaired):
                    if self.u_available[n, i, k]:
                        res[n_par] = weights[n] * iu[n, self.ned + k, i]
                        n_par += 1
        return self.parameters_projector.T @ res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'gradient_parameters_d1')
def geminal_gradient_parameters_d1(self, n_vectors: np.ndarray):
    """First derivatives of gradient w.r.t. the parameters
    :param n_vectors: e-n vectors
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        pool_u, pool_d, pool_grad_u, pool_grad_d = self.pool_gradient_matrix(n_vectors)
        ne = self.neu + self.ned
        ngem = self.c.size
        pool_d_full = np.zeros(shape=(self.norb, self.neu))
        pool_d_full[:, : self.ned] = pool_d
        flat_grad_u = pool_grad_u.reshape(self.norb, self.neu * 3)
        flat_grad_d = pool_grad_d.reshape(self.norb, self.ned * 3)

        weights = np.zeros(shape=ngem)
        c_weights = np.zeros(shape=ngem)
        iu = np.zeros(shape=(ngem, self.neu, self.norb))
        zg = np.zeros(shape=(ngem, self.norb, self.norb))
        inverses = np.zeros(shape=(ngem, self.neu, self.neu))
        # the derivative of a coordinate is rank one too, so the products it makes with the rank
        # one of a parameter are these three matrices and nothing else
        pdinv = np.zeros(shape=(ngem, self.norb, self.neu))
        pdw = np.zeros(shape=(ngem, self.norb, self.ned * 3))
        guiu = np.zeros(shape=(ngem, self.neu * 3, self.norb))
        w_col = np.zeros(shape=(ngem, self.neu, self.ned * 3))
        gr = np.zeros(shape=(ngem, ne * 3))
        val = 0.0
        for n in range(ngem):
            full_c = np.empty(shape=(self.norb, self.neu))
            full_c[:, : self.ned] = self.g[n] @ pool_d
            if self.nunpaired:
                full_c[:, self.ned :] = self.u[n]
            matrix = pool_u.T @ full_c
            inv = np.linalg.inv(matrix)
            gu = flat_grad_u.T @ full_c
            inverses[n] = inv
            iu[n] = inv @ pool_u.T
            zg[n] = pool_d_full @ iu[n]
            pdinv[n] = pool_d_full @ inv
            w_col[n] = inv @ ((pool_u.T @ self.g[n]) @ flat_grad_d)
            pdw[n] = pool_d_full @ w_col[n]
            guiu[n] = gu @ iu[n]
            for r in range(self.neu):
                for d in range(3):
                    gr[n, r * 3 + d] = gu[r * 3 + d] @ inv[:, r]
            for s in range(self.ned):
                for d in range(3):
                    gr[n, (self.neu + s) * 3 + d] = w_col[n, s, s * 3 + d]
            c_weights[n] = np.linalg.det(matrix)
            weights[n] = self.c[n] * c_weights[n]
            val += weights[n]
        weights /= val
        c_weights /= val
        total = np.zeros(shape=ne * 3)
        for n in range(ngem):
            total += weights[n] * gr[n]

        res = np.zeros(shape=(self.get_parameters(True).size, ne * 3))
        n_par = 0
        for n in range(ngem):
            inv = inverses[n]
            # a parameter of one geminal moves the weights of all of them, and that whole sum
            # collapses into d(ln(ψ))/dp · (gradient of this geminal - gradient of the sum)
            res[n_par] = c_weights[n] * (gr[n] - total)
            n_par += 1
            for i in range(self.norb):
                for j in range(i, self.norb):
                    if self.g_available[n, i, j]:
                        # an off-diagonal parameter moves g on both sides of the diagonal, so it
                        # contributes the term of the pair and the term of its transpose
                        if i == j:
                            terms = 1
                            trace = zg[n, i, i]
                        else:
                            terms = 2
                            trace = zg[n, j, i] + zg[n, i, j]
                        d_gr = np.zeros(shape=ne * 3)
                        for term in range(terms):
                            if term == 0:
                                i1, j1 = i, j
                            else:
                                i1, j1 = j, i
                            for r in range(self.neu):
                                for d in range(3):
                                    a = r * 3 + d
                                    d_gr[a] += pdinv[n, j1, r] * (pool_grad_u[i1, r, d] - guiu[n, a, i1])
                            for s in range(self.ned):
                                for d in range(3):
                                    a = (self.neu + s) * 3 + d
                                    d_gr[a] += iu[n, s, i1] * (pool_grad_d[j1, s, d] - pdw[n, j1, s * 3 + d])
                        res[n_par] = weights[n] * (trace * (gr[n] - total) + d_gr)
                        n_par += 1
            for i in range(self.norb):
                for k in range(self.nunpaired):
                    if self.u_available[n, i, k]:
                        d_gr = np.zeros(shape=ne * 3)
                        for r in range(self.neu):
                            for d in range(3):
                                a = r * 3 + d
                                d_gr[a] += inv[self.ned + k, r] * (pool_grad_u[i, r, d] - guiu[n, a, i])
                        for s in range(self.ned):
                            for d in range(3):
                                a = (self.neu + s) * 3 + d
                                d_gr[a] -= w_col[n, self.ned + k, s * 3 + d] * iu[n, s, i]
                        res[n_par] = weights[n] * (iu[n, self.ned + k, i] * (gr[n] - total) + d_gr)
                        n_par += 1
        return self.parameters_projector.T @ res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Geminal_class_t, 'laplacian_parameters_d1')
def geminal_laplacian_parameters_d1(self, n_vectors: np.ndarray):
    """First derivatives of laplacian w.r.t. the parameters
    :param n_vectors: e-n vectors
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        pool_u, pool_d, pool_lap_u, pool_lap_d = self.pool_laplacian_matrix(n_vectors)
        ngem = self.c.size
        pool_d_full = np.zeros(shape=(self.norb, self.neu))
        pool_d_full[:, : self.ned] = pool_d
        pool_lap_d_full = np.zeros(shape=(self.norb, self.neu))
        pool_lap_d_full[:, : self.ned] = pool_lap_d

        weights = np.zeros(shape=ngem)
        c_weights = np.zeros(shape=ngem)
        lp = np.zeros(shape=ngem)
        iu = np.zeros(shape=(ngem, self.neu, self.norb))
        zg = np.zeros(shape=(ngem, self.norb, self.norb))
        # the laplacian of the geminal matrix is a sum of rank one terms rather than one of them,
        # so what a parameter meets is the whole M^-1 • laplacian matrix, folded here into biu
        ilu = np.zeros(shape=(ngem, self.neu, self.norb))
        zlg = np.zeros(shape=(ngem, self.norb, self.norb))
        zgl = np.zeros(shape=(ngem, self.norb, self.norb))
        biu = np.zeros(shape=(ngem, self.neu, self.norb))
        pdbiu = np.zeros(shape=(ngem, self.norb, self.norb))
        val = 0.0
        for n in range(ngem):
            full_c = np.empty(shape=(self.norb, self.neu))
            full_c[:, : self.ned] = self.g[n] @ pool_d
            if self.nunpaired:
                full_c[:, self.ned :] = self.u[n]
            matrix = pool_u.T @ full_c
            inv = np.linalg.inv(matrix)
            dm_lap = pool_lap_u.T @ full_c
            dm_lap[:, : self.ned] += (pool_u.T @ self.g[n]) @ pool_lap_d
            b = inv @ dm_lap
            iu[n] = inv @ pool_u.T
            ilu[n] = inv @ pool_lap_u.T
            biu[n] = b @ iu[n]
            zg[n] = pool_d_full @ iu[n]
            zlg[n] = pool_d_full @ ilu[n]
            zgl[n] = pool_lap_d_full @ iu[n]
            pdbiu[n] = pool_d_full @ biu[n]
            lp[n] = np.trace(b)
            c_weights[n] = np.linalg.det(matrix)
            weights[n] = self.c[n] * c_weights[n]
            val += weights[n]
        weights /= val
        c_weights /= val
        total = 0.0
        for n in range(ngem):
            total += weights[n] * lp[n]

        res = np.zeros(shape=self.get_parameters(True).size)
        n_par = 0
        for n in range(ngem):
            res[n_par] = c_weights[n] * (lp[n] - total)
            n_par += 1
            for i in range(self.norb):
                for j in range(i, self.norb):
                    if self.g_available[n, i, j]:
                        if i == j:
                            terms = 1
                            trace = zg[n, i, i]
                        else:
                            terms = 2
                            trace = zg[n, j, i] + zg[n, i, j]
                        d_lp = 0.0
                        for term in range(terms):
                            if term == 0:
                                i1, j1 = i, j
                            else:
                                i1, j1 = j, i
                            d_lp += zlg[n, j1, i1] + zgl[n, j1, i1] - pdbiu[n, j1, i1]
                        res[n_par] = weights[n] * (trace * (lp[n] - total) + d_lp)
                        n_par += 1
            for i in range(self.norb):
                for k in range(self.nunpaired):
                    if self.u_available[n, i, k]:
                        d_lp = ilu[n, self.ned + k, i] - biu[n, self.ned + k, i]
                        res[n_par] = weights[n] * (iu[n, self.ned + k, i] * (lp[n] - total) + d_lp)
                        n_par += 1
        return self.parameters_projector.T @ res

    return impl


class Geminal(structref.StructRefProxy, AbstractSlater):
    def __new__(cls, config, cusp=None):
        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(
            neu,
            ned,
            norb,
            gautol,
            nbasis_functions,
            first_shells,
            orbital_types,
            shell_moments,
            slater_orders,
            primitives,
            coefficients,
            exponents,
            mo_up,
            mo_down,
            c,
            g,
            u,
            c_mask,
            g_mask,
            u_mask,
            g_available,
            u_available,
            c_ties,
            g_ties,
            u_ties,
            cusp,
            harmonics,
        ):
            self = structref.new(Geminal_t)
            self.neu = neu
            self.ned = ned
            self.nunpaired = neu - ned
            self.norb = norb
            self.nbasis_functions = nbasis_functions
            self.first_shells = first_shells
            self.orbital_types = orbital_types
            self.shell_moments = shell_moments
            self.slater_orders = slater_orders
            self.primitives = primitives
            self.coefficients = coefficients
            self.exponents = exponents
            self.gautol = gautol
            self.mo_up = mo_up[:norb]
            self.mo_down = mo_down[:norb]
            self.c = c
            self.g = g
            self.u = u
            self.c_mask = c_mask
            self.g_mask = g_mask
            self.u_mask = u_mask
            self.g_available = g_available
            self.u_available = u_available
            self.c_ties = c_ties
            self.g_ties = g_ties
            self.u_ties = u_ties
            self.cusp = cusp
            self.harmonics = harmonics
            self.norm = np.exp(-(math.lgamma(neu + 1) + math.lgamma(ned + 1)) / (neu + ned) / 2)
            self.parameters_projector = np.zeros(shape=(0, 0))
            return self

        return init(
            config.input.neu,
            config.input.ned,
            config.geminal.norb,
            config.input.gautol,
            config.wfn.nbasis_functions,
            config.wfn.first_shells,
            config.wfn.orbital_types,
            config.wfn.shell_moments,
            config.wfn.slater_orders,
            config.wfn.primitives,
            config.wfn.coefficients,
            config.wfn.exponents,
            config.wfn.mo_up,
            config.wfn.mo_down,
            config.geminal.c,
            config.geminal.g,
            config.geminal.u,
            config.geminal.c_mask,
            config.geminal.g_mask,
            config.geminal.u_mask,
            config.geminal.g_available,
            config.geminal.u_available,
            config.geminal.c_ties,
            config.geminal.g_ties,
            config.geminal.u_ties,
            cusp,
            Harmonics(np.max(config.wfn.shell_moments)),
        )

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def cusp(self):
        return self.cusp

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value(self, n_vectors):
        return self.value(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def log_value(self, n_vectors):
        return self.log_value(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def gradient(self, n_vectors):
        return self.gradient(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def laplacian(self, n_vectors):
        return self.laplacian(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def hessian(self, n_vectors):
        return self.hessian(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def tressian_dot(self, n_vectors, bb):
        return self.tressian_dot(n_vectors, bb)


structref.define_boxing(Geminal_class_t, Geminal)
