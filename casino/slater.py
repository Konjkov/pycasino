import math

import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from . import delta
from .abstract import AbstractSlater
from .cusp import Cusp_t
from .harmonics import gradient_angular_part, hessian_angular_part, tressian_angular_part, value_angular_part
from .readers.wfn import GAUSSIAN_TYPE, SLATER_TYPE

log_10 = np.log(10)


@structref.register
class Slater_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


Slater_t = Slater_class_t(
    [
        ('neu', nb.int64),
        ('ned', nb.int64),
        ('nbasis_functions', nb.int64),
        ('first_shells', nb.int64[::1]),
        ('orbital_types', nb.int64[::1]),
        ('shell_moments', nb.int64[::1]),
        ('slater_orders', nb.int64[::1]),
        ('primitives', nb.int64[::1]),
        ('coefficients', nb.float64[::1]),
        ('exponents', nb.float64[::1]),
        ('gautol', nb.float64),
        ('permutation_up', nb.int64[:, ::1]),
        ('permutation_down', nb.int64[:, ::1]),
        ('mo_up', nb.float64[:, ::1]),
        ('mo_down', nb.float64[:, ::1]),
        ('det_coeff', nb.float64[::1]),
        ('cusp', nb.optional(Cusp_t)),
        ('norm', nb.float64),
        ('parameters_projector', nb.float64[:, ::1]),
    ]
)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'value_matrix')
def slater_value_matrix(self, n_vectors: np.ndarray):
    """Value matrix.
    :param n_vectors: electron-nuclei array(natom, nelec, 3)
    :return: array(up_orbitals, up_electrons), array(down_orbitals, down_electrons)
    """

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        orbitals = np.zeros(shape=(self.neu + self.ned, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = n_vectors[atom, i] @ n_vectors[atom, i]
                angular_1 = value_angular_part(x, y, z)
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
        wfn_u = self.mo_up @ ao_value[: self.neu].T
        wfn_d = self.mo_down @ ao_value[self.neu :].T
        if self.cusp is not None:
            cusp_value_u, cusp_value_d = self.cusp.value(n_vectors)
            wfn_u += cusp_value_u
            wfn_d += cusp_value_d
        return wfn_u, wfn_d

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'gradient_matrix')
def slater_gradient_matrix(self, n_vectors: np.ndarray):
    """Gradient matrix.
    :param n_vectors: electron-nuclei - array(natom, nelec, 3)
    :return: array(up_orbitals, up_electrons, 3), array(down_orbitals, down_electrons, 3)
    """

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        orbital = np.zeros(shape=(self.neu + self.ned, 3, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = n_vectors[atom, i] @ n_vectors[atom, i]
                angular_1 = value_angular_part(x, y, z)
                angular_2 = gradient_angular_part(x, y, z)
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
                        orbital[i, 0, ao + m] = x * angular_1[l * l + m] * radial_1 + angular_2[l * l + m, 0] * radial_2
                        orbital[i, 1, ao + m] = y * angular_1[l * l + m] * radial_1 + angular_2[l * l + m, 1] * radial_2
                        orbital[i, 2, ao + m] = z * angular_1[l * l + m] * radial_1 + angular_2[l * l + m, 2] * radial_2
                    ao += 2 * l + 1

        ao_gradient = self.norm * orbital.reshape((self.neu + self.ned) * 3, self.nbasis_functions)
        grad_u = (self.mo_up @ ao_gradient[: self.neu * 3].T).reshape(self.mo_up.shape[0], self.neu, 3)
        grad_d = (self.mo_down @ ao_gradient[self.neu * 3 :].T).reshape(self.mo_down.shape[0], self.ned, 3)
        if self.cusp is not None:
            cusp_gradient_u, cusp_gradient_d = self.cusp.gradient(n_vectors)
            grad_u += cusp_gradient_u
            grad_d += cusp_gradient_d
        return grad_u, grad_d

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'laplacian_matrix')
def slater_laplacian_matrix(self, n_vectors: np.ndarray):
    """Laplacian matrix.
    :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
    :return: array(up_orbitals, up_electrons), array(down_orbitals, down_electrons)
    """

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        orbital = np.zeros(shape=(self.neu + self.ned, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = n_vectors[atom, i] @ n_vectors[atom, i]
                angular_1 = value_angular_part(x, y, z)
                for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            if alpha * r2 < log_10 * self.gautol:
                                radial_1 += 2 * alpha * (2 * alpha * r2 - 2 * l - 3) * self.coefficients[p + primitive] * np.exp(-alpha * r2)
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        n = self.slater_orders[nshell]
                        r_n = r**n
                        for primitive in range(self.primitives[nshell]):
                            minus_alpha_r = -self.exponents[p + primitive] * r
                            exponent = r_n * self.coefficients[p + primitive] * np.exp(minus_alpha_r)
                            radial_1 += (minus_alpha_r**2 + 2 * (l + n + 1) * minus_alpha_r + (2 * l + n + 1) * n) / r2 * exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital[i, ao + m] = angular_1[l * l + m] * radial_1
                    ao += 2 * l + 1

        ao_laplacian = self.norm * orbital
        lap_u = self.mo_up @ ao_laplacian[: self.neu].T
        lap_d = self.mo_down @ ao_laplacian[self.neu :].T
        if self.cusp is not None:
            cusp_laplacian_u, cusp_laplacian_d = self.cusp.laplacian(n_vectors)
            lap_u += cusp_laplacian_u
            lap_d += cusp_laplacian_d
        return lap_u, lap_d

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'hessian_matrix')
def slater_hessian_matrix(self, n_vectors: np.ndarray):
    """Hessian matrix.
    :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
    :return: array(up_orbitals, up_electrons, 3, 3), array(down_orbitals, down_electrons, 3, 3)
    """

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        orbital = np.zeros(shape=(self.neu + self.ned, 3, 3, self.nbasis_functions))

        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = n_vectors[atom, i] @ n_vectors[atom, i]
                angular_1 = value_angular_part(x, y, z)
                angular_2 = gradient_angular_part(x, y, z)
                angular_3 = hessian_angular_part(x, y, z)
                # angular_3 = hessian_angular_part_square(x, y, z)
                # n_vector_angular_2 = np.outer(angular_2, n_vectors[atom, i]).reshape(-1, 3, 3)
                # n_vector_outer = np.outer(n_vectors[atom, i], n_vectors[atom, i])
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
                        # orbital[i, :, :, ao+m] = (
                        #     n_vector_outer * angular_1[l*l+m] * radial_1 +
                        #     (np.eye(3) * angular_1[l*l+m] + n_vector_angular_2[l*l+m] + n_vector_angular_2[l*l+m].T) * radial_2 +
                        #     # convert hessian_angular_part to symmetric matrix a + a.T - np.diag(a.diagonal())
                        #     angular_3[l*l+m, :, :] * radial_3
                        # )
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

        ao_hessian = self.norm * orbital.reshape((self.neu + self.ned) * 9, self.nbasis_functions)
        hess_u = (self.mo_up @ ao_hessian[: self.neu * 9].T).reshape(self.mo_up.shape[0], self.neu, 3, 3)
        hess_d = (self.mo_down @ ao_hessian[self.neu * 9 :].T).reshape(self.mo_down.shape[0], self.ned, 3, 3)
        if self.cusp is not None:
            cusp_hessian_u, cusp_hessian_d = self.cusp.hessian(n_vectors)
            hess_u += cusp_hessian_u
            hess_d += cusp_hessian_d
        return hess_u, hess_d

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'tressian_matrix')
def slater_tressian_matrix(self, n_vectors: np.ndarray):
    """Tressian matrix.
    :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
    :return: array(up_orbitals, up_electrons, 3, 3, 3), array(down_orbitals, down_electrons, 3, 3, 3)
    """

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        orbital = np.zeros(shape=(self.neu + self.ned, 3, 3, 3, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = n_vectors[atom, i] @ n_vectors[atom, i]
                angular_1 = value_angular_part(x, y, z)
                angular_2 = gradient_angular_part(x, y, z)
                angular_3 = hessian_angular_part(x, y, z)
                angular_4 = tressian_angular_part(x, y, z)
                for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    radial_2 = 0.0
                    radial_3 = 0.0
                    radial_4 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            if alpha * r2 < 2.303 * self.gautol:
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
                        orbital[i, 1, 1, 1, ao + m] = y*y*y * angular_1[l*l+m] * radial_1 + 3*y*(angular_1[l*l+m] + y*angular_2[l*l+m, 0]) * radial_2 + 3*(angular_2[l*l+m, 1] + y * angular_3[l*l+m, 3]) * radial_3 + angular_4[l*l+m, 6] * radial_4  # fmt: skip
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
                        orbital[i, 2, 2, 2, ao + m] = z*z*z * angular_1[l*l+m] * radial_1 + 3*z*(angular_1[l*l+m] + z*angular_2[l*l+m, 0]) * radial_2 + 3*(angular_2[l*l+m, 2] + z * angular_3[l*l+m, 5]) * radial_3 + angular_4[l*l+m, 9] * radial_4  # fmt: skip
                    ao += 2 * l + 1

        ao_tressian = self.norm * orbital.reshape((self.neu + self.ned) * 27, self.nbasis_functions)
        tress_u = (self.mo_up @ ao_tressian[: self.neu * 27].T).reshape(self.mo_up.shape[0], self.neu, 3, 3, 3)
        tress_d = (self.mo_down @ ao_tressian[self.neu * 27 :].T).reshape(self.mo_down.shape[0], self.ned, 3, 3, 3)
        if self.cusp is not None:
            cusp_tressian_u, cusp_tressian_d = self.cusp.tressian(n_vectors)
            tress_u += cusp_tressian_u
            tress_d += cusp_tressian_d
        return tress_u, tress_d

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'value')
def slater_value(self, n_vectors: np.ndarray):
    """Wave function value.
    :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
    :return: float
    """

    def impl(self, n_vectors: np.ndarray) -> float:
        wfn_u, wfn_d = self.value_matrix(n_vectors)
        val = 0
        for i in range(self.det_coeff.size):
            val += self.det_coeff[i] * np.linalg.det(wfn_u[self.permutation_up[i]]) * np.linalg.det(wfn_d[self.permutation_down[i]])
        return val

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'gradient')
def slater_gradient(self, n_vectors: np.ndarray):
    """Gradient ∇φ/φ w.r.t e-coordinates.
    Derivative of determinant of symmetric matrix w.r.t a scalar
    dln(det(A))/dr = tr(A^-1 • dA/dr)
    then using np.trace(A • B) = np.sum(A * B.T) = np.tensordot(A, B.T)
    Read for details:
    "Simple formalism for efficient derivatives and multi-determinant expansions in quantum Monte Carlo"
    C. Filippi, R. Assaraf, S. Moroni
    :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
    :return: vectors shape = (nelec * 3,)
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        wfn_u, wfn_d = self.value_matrix(n_vectors)
        grad_u, grad_d = self.gradient_matrix(n_vectors)
        val = 0
        grad = np.zeros(shape=(self.neu + self.ned) * 3)
        single_det = self.det_coeff.size == 1
        for i in range(self.det_coeff.size):
            # einsum('ij,jik -> ik', np.linalg.inv(wfn_u[self.permutation_up[i]]), grad_u[self.permutation_up[i]])
            tr_grad_u = (np.linalg.inv(wfn_u[self.permutation_up[i]]) * grad_u[self.permutation_up[i]].T).T.sum(axis=0)
            tr_grad_d = (np.linalg.inv(wfn_d[self.permutation_down[i]]) * grad_d[self.permutation_down[i]].T).T.sum(axis=0)
            tr_grad = np.concatenate((tr_grad_u, tr_grad_d)).ravel()
            c = 1 if single_det else self.det_coeff[i] * np.linalg.det(wfn_u[self.permutation_up[i]]) * np.linalg.det(wfn_d[self.permutation_down[i]])
            val += c
            grad += c * tr_grad

        return grad / val

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'laplacian')
def slater_laplacian(self, n_vectors: np.ndarray):
    """Scalar laplacian Δφ/φ w.r.t e-coordinates.
    Δln(det(A)) = sum(tr(slater^-1 * B(n)) over n
    where matrix B(n) is zero with exception to the n-th column
    as tr(A) + tr(B) = tr(A + B)
    Δln(det(A)) = tr(slater^-1 • B)
    where the matrix Bij = ∆phi i (rj)
    then using np.trace(A • B) = np.sum(A * B.T) = np.tensordot(A, B.T)
    Read for details:
    "Simple formalism for efficient derivatives and multi-determinant expansions in quantum Monte Carlo"
    C. Filippi, R. Assaraf, S. Moroni
    :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
    :return: float
    """

    def impl(self, n_vectors: np.ndarray) -> float:
        wfn_u, wfn_d = self.value_matrix(n_vectors)
        lap_u, lap_d = self.laplacian_matrix(n_vectors)
        val = lap = 0
        single_det = self.det_coeff.size == 1
        for i in range(self.det_coeff.size):
            # einsum('ij,ji', np.linalg.inv(wfn_u[self.permutation_up[i]]), lap_u[self.permutation_up[i]])
            tr_lap_u = (np.linalg.inv(wfn_u[self.permutation_up[i]]) * lap_u[self.permutation_up[i]].T).sum()
            tr_lap_d = (np.linalg.inv(wfn_d[self.permutation_down[i]]) * lap_d[self.permutation_down[i]].T).sum()
            c = 1 if single_det else self.det_coeff[i] * np.linalg.det(wfn_u[self.permutation_up[i]]) * np.linalg.det(wfn_d[self.permutation_down[i]])
            val += c
            lap += c * (tr_lap_u + tr_lap_d)

        return lap / val

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'hessian')
def slater_hessian(self, n_vectors: np.ndarray):
    """Hessian H(φ)/φ w.r.t e-coordinates.
    d²ln(det(A))/dr² = (
        tr(A^-1 • d²A/dr²) +
        tr(A^-1 • dA/dr) ⊗ tr(A^-1 • dA/dr) -
        tr(A^-1 • dA/dr • A^-1 • dA/dr)
    )
    https://math.stackexchange.com/questions/2325807/second-derivative-of-a-determinant
    where ⊗ - outer product, r - vector shape = (nelec * 3)
    then using tr(A • B) = np.sum(A * B.T)
    in case of ri and rj is a coordinates of different electrons first term is zero
    in case of ri and rj is a coordinates of same electrons a sum of last two terms is zero.
    :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
    :return: vectors shape = (nelec * 3, nelec * 3)
    """

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ne = self.neu + self.ned
        wfn_u, wfn_d = self.value_matrix(n_vectors)
        grad_u, grad_d = self.gradient_matrix(n_vectors)
        hess_u, hess_d = self.hessian_matrix(n_vectors)
        val = 0
        grad = np.zeros(shape=ne * 3)
        hess = np.zeros(shape=(ne * 3, ne * 3))
        single_det = self.det_coeff.size == 1
        for i in range(self.det_coeff.size):
            inv_wfn_u = np.linalg.inv(wfn_u[self.permutation_up[i]])
            inv_wfn_d = np.linalg.inv(wfn_d[self.permutation_down[i]])
            tr_grad_u = (inv_wfn_u * grad_u[self.permutation_up[i]].T).T.sum(axis=0)
            tr_grad_d = (inv_wfn_d * grad_d[self.permutation_down[i]].T).T.sum(axis=0)
            tr_grad = np.concatenate((tr_grad_u, tr_grad_d)).ravel()
            tr_hess_u = (inv_wfn_u * hess_u[self.permutation_up[i]].T).T.sum(axis=0)
            tr_hess_d = (inv_wfn_d * hess_d[self.permutation_down[i]].T).T.sum(axis=0)
            matrix_grad_u = (inv_wfn_u @ grad_u[self.permutation_up[i]].reshape(self.neu, self.neu * 3)).reshape(self.neu, self.neu, 3)
            matrix_grad_d = (inv_wfn_d @ grad_d[self.permutation_down[i]].reshape(self.ned, self.ned * 3)).reshape(self.ned, self.ned, 3)
            c = 1 if single_det else self.det_coeff[i] * np.linalg.det(wfn_u[self.permutation_up[i]]) * np.linalg.det(wfn_d[self.permutation_down[i]])
            val += c
            grad += c * tr_grad
            # tr(A^-1 • d²A/dxdy) - tr(A^-1 • dA/dx • A^-1 • dA/dy)
            res_u = np.zeros(shape=(self.neu, 3, self.neu, 3))
            res_d = np.zeros(shape=(self.ned, 3, self.ned, 3))
            for r1 in range(3):
                for r2 in range(3):
                    res_u[:, r1, :, r2] = np.diag(tr_hess_u[:, r1, r2]) - matrix_grad_u[:, :, r1].T * matrix_grad_u[:, :, r2]
                    res_d[:, r1, :, r2] = np.diag(tr_hess_d[:, r1, r2]) - matrix_grad_d[:, :, r1].T * matrix_grad_d[:, :, r2]
            hess[: self.neu * 3, : self.neu * 3] += c * res_u.reshape(self.neu * 3, self.neu * 3)
            hess[self.neu * 3 :, self.neu * 3 :] += c * res_d.reshape(self.ned * 3, self.ned * 3)
            # tr(A^-1 • dA/dx) ⊗ tr(A^-1 • dA/dy)
            hess += c * np.outer(tr_grad, tr_grad)

        return hess / val, grad / val

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'tressian')
def slater_tressian(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tressian or numerical third partial derivatives w.r.t e-coordinates
    d³ln(det(A))/dxdydz = (
        tr(A^-1 • d³A/dxdydz)
        + tr(A^-1 • dA/dx) ⊗ Hessian_yz + tr(A^-1 • dA/dy) ⊗ Hessian_xz + tr(A^-1 • dA/dz) ⊗ Hessian_xy
        - tr(A^-1 • d²A/dxdy ⊗ A^-1 • dA/dz) - tr(A^-1 • d²A/dxdz ⊗ A^-1 • dA/dy) - tr(A^-1 • d²A/dydz ⊗ A^-1 • dA/dx)
        + tr(A^-1 • dA/dx ⊗ A^-1 • dA/dy ⊗ A^-1 • dA/dz) + tr(A^-1 • dA/dz ⊗ A^-1 • dA/dy ⊗ A^-1 • dA/dx)
        - 2 * tr(A^-1 • dA/dx) ⊗ tr(A^-1 • dA/dy) ⊗ tr(A^-1 • dA/dz)
    )
    Hessian = (
        tr(A^-1 • d²A/dr²)
        + tr(A^-1 • dA/dr) ⊗ tr(A^-1 • dA/dr)
        - tr(A^-1 • dA/dr ⊗ A^-1 • dA/dr)
    )
    where ⊗ - outer product, r - vector shape = (nelec * 3)
    :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
    :return: vectors shape = (nelec * 3, nelec * 3, nelec * 3)
    """

    def impl(self, n_vectors: np.ndarray):
        ne = self.neu + self.ned
        wfn_u, wfn_d = self.value_matrix(n_vectors)
        grad_u, grad_d = self.gradient_matrix(n_vectors)
        hess_u, hess_d = self.hessian_matrix(n_vectors)
        tress_u, tress_d = self.tressian_matrix(n_vectors)
        val = 0
        grad = np.zeros(shape=ne * 3)
        hess = np.zeros(shape=(ne * 3, ne * 3))
        tress = np.zeros(shape=(ne * 3, ne * 3, ne * 3))
        single_det = self.det_coeff.size == 1
        for i in range(self.det_coeff.size):
            inv_wfn_u = np.linalg.inv(wfn_u[self.permutation_up[i]])
            inv_wfn_d = np.linalg.inv(wfn_d[self.permutation_down[i]])
            tr_grad_u = (inv_wfn_u * grad_u[self.permutation_up[i]].T).T.sum(axis=0)
            tr_grad_d = (inv_wfn_d * grad_d[self.permutation_down[i]].T).T.sum(axis=0)
            tr_grad = np.concatenate((tr_grad_u, tr_grad_d)).ravel()
            tr_hess_u = (inv_wfn_u * hess_u[self.permutation_up[i]].T).T.sum(axis=0)
            tr_hess_d = (inv_wfn_d * hess_d[self.permutation_down[i]].T).T.sum(axis=0)
            tr_tress_u = (inv_wfn_u * tress_u[self.permutation_up[i]].T).T.sum(axis=0)
            tr_tress_d = (inv_wfn_d * tress_d[self.permutation_down[i]].T).T.sum(axis=0)
            matrix_grad_u = (inv_wfn_u @ grad_u[self.permutation_up[i]].reshape(self.neu, self.neu * 3)).reshape(self.neu, self.neu, 3)
            matrix_grad_d = (inv_wfn_d @ grad_d[self.permutation_down[i]].reshape(self.ned, self.ned * 3)).reshape(self.ned, self.ned, 3)
            matrix_hess_u = (inv_wfn_u @ hess_u[self.permutation_up[i]].reshape(self.neu, self.neu * 9)).reshape(self.neu, self.neu, 3, 3)
            matrix_hess_d = (inv_wfn_d @ hess_d[self.permutation_down[i]].reshape(self.ned, self.ned * 9)).reshape(self.ned, self.ned, 3, 3)

            # tr(A^-1 • dA/dx) ⊗ tr(A^-1 • dA/dy) + tr(A^-1 • dA/dx) ⊗ tr(A^-1 • dA/dy) / 3
            partial_hess = np.outer(tr_grad, tr_grad) / 3
            res_u = np.zeros(shape=(self.neu, 3, self.neu, 3))
            res_d = np.zeros(shape=(self.ned, 3, self.ned, 3))
            # tr(A^-1 • d²A/dxdy) - tr(A^-1 • dA/dx • A^-1 • dA/dy)
            for r1 in range(3):
                for r2 in range(3):
                    res_u[:, r1, :, r2] = np.diag(tr_hess_u[:, r1, r2]) - matrix_grad_u[:, :, r1].T * matrix_grad_u[:, :, r2]
                    res_d[:, r1, :, r2] = np.diag(tr_hess_d[:, r1, r2]) - matrix_grad_d[:, :, r1].T * matrix_grad_d[:, :, r2]
            partial_hess[: self.neu * 3, : self.neu * 3] += res_u.reshape(self.neu * 3, self.neu * 3)
            partial_hess[self.neu * 3 :, self.neu * 3 :] += res_d.reshape(self.ned * 3, self.ned * 3)

            c = 1 if single_det else self.det_coeff[i] * np.linalg.det(wfn_u[self.permutation_up[i]]) * np.linalg.det(wfn_d[self.permutation_down[i]])
            val += c
            grad += c * tr_grad
            hess += c * (partial_hess + 2 / 3 * np.outer(tr_grad, tr_grad))
            # tr(A^-1 • dA/dx) ⊗ Hessian_yz + tr(A^-1 • dA/dy) ⊗ Hessian_xz + tr(A^-1 • dA/dz) ⊗ Hessian_xy
            tress += c * (
                # (1, 1, self.ned * 3) * (self.ned * 3, self.ned * 3, 1)
                tr_grad * np.expand_dims(partial_hess, 2)
                # (1, self.ned * 3, 1) * (self.ned * 3, 1, self.ned * 3)
                + np.expand_dims(tr_grad, 1) * np.expand_dims(partial_hess, 1)
                # (self.ned * 3, 1, 1) * (1, self.ned * 3, self.ned * 3)
                + np.expand_dims(np.expand_dims(tr_grad, 1), 2) * partial_hess
            )
            res_u = np.zeros(shape=(self.neu, 3, self.neu, 3, self.neu, 3))
            res_d = np.zeros(shape=(self.ned, 3, self.ned, 3, self.ned, 3))
            for r1 in range(3):
                for r2 in range(3):
                    for r3 in range(3):
                        # tr(A^-1 • dA/dx • A^-1 • dA/dy • A^-1 • dA/dz) + tr(A^-1 • dA/dz • A^-1 • dA/dy • A^-1 • dA/dx)
                        res_u[:, r1, :, r2, :, r3] += (
                            # (1, 1, self.ned, 3, self.ned, 1)
                            np.expand_dims(matrix_grad_u[:, :, r2].T, 0)
                            # (self.ned, 1, 1, 1, self.ned, 3)
                            * np.expand_dims(matrix_grad_u[:, :, r3], 1)
                            # (self.ned, 3, self.ned, 1, 1, 1)
                            * np.expand_dims(matrix_grad_u[:, :, r1].T, 2)
                        ) + (
                            # (1, 1, self.ned, 1, self.ned, 3)
                            np.expand_dims(matrix_grad_u[:, :, r3], 0)
                            # (self.ned, 3, 1, 1, self.ned, 1)
                            * np.expand_dims(matrix_grad_u[:, :, r1].T, 1)
                            # (self.ned, 1, self.ned, 3, 1, 1)
                            * np.expand_dims(matrix_grad_u[:, :, r2], 2)
                        )
                        res_d[:, r1, :, r2, :, r3] += (
                            np.expand_dims(matrix_grad_d[:, :, r2].T, 0)
                            * np.expand_dims(matrix_grad_d[:, :, r3], 1)
                            * np.expand_dims(matrix_grad_d[:, :, r1].T, 2)
                        ) + (
                            np.expand_dims(matrix_grad_d[:, :, r3], 0)
                            * np.expand_dims(matrix_grad_d[:, :, r1].T, 1)
                            * np.expand_dims(matrix_grad_d[:, :, r2], 2)
                        )
                        # tr(A^-1 • d³A/dxdydz) - tr(A^-1 • d²A/dxdy • A^-1 * dA/dz) - tr(A^-1 • dA²/dxdz • A^-1 • dA/dy) - tr(A^-1 • d²A/dydz • A^-1 * dA/dx)
                        for e in range(self.neu):
                            res_u[e, r1, e, r2, e, r3] += tr_tress_u[e, r1, r2, r3]
                            res_u[e, r1, e, r2, :, r3] -= matrix_hess_u[:, e, r1, r2] * matrix_grad_u[e, :, r3]
                            res_u[e, r1, :, r2, e, r3] -= matrix_hess_u[:, e, r1, r3] * matrix_grad_u[e, :, r2]
                            res_u[:, r1, e, r2, e, r3] -= matrix_hess_u[:, e, r2, r3] * matrix_grad_u[e, :, r1]
                        for e in range(self.ned):
                            res_d[e, r1, e, r2, e, r3] += tr_tress_d[e, r1, r2, r3]
                            res_d[e, r1, e, r2, :, r3] -= matrix_hess_d[:, e, r1, r2] * matrix_grad_d[e, :, r3]
                            res_d[e, r1, :, r2, e, r3] -= matrix_hess_d[:, e, r1, r3] * matrix_grad_d[e, :, r2]
                            res_d[:, r1, e, r2, e, r3] -= matrix_hess_d[:, e, r2, r3] * matrix_grad_d[e, :, r1]
            tress[: self.neu * 3, : self.neu * 3, : self.neu * 3] += c * res_u.reshape(self.neu * 3, self.neu * 3, self.neu * 3)
            tress[self.neu * 3 :, self.neu * 3 :, self.neu * 3 :] += c * res_d.reshape(self.ned * 3, self.ned * 3, self.ned * 3)

        return tress / val, hess / val, grad / val

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'tressian_v2')
def slater_tressian_v2(self, n_vectors: np.ndarray):
    """Tressian or numerical third partial derivatives w.r.t. e-coordinates
    d³ln(det(A))/dxdydz
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, n_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hess, grad = self.hessian(n_vectors)
        # d(d²ln(phi)/dydz)/dx
        res = np.zeros(shape=(self.neu + self.ned, 3, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3))
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= delta
                res[i, j] -= self.hessian(n_vectors)[0]
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.hessian(n_vectors)[0]
                n_vectors[:, i, j] -= delta
        hess_div = res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3) / delta / 2
        tress = hess_div + np.expand_dims(np.expand_dims(grad, 1), 2) * hess
        return tress, hess, grad

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'fix_det_coeff_parameters')
def slater_fix_det_coeff_parameters(self):
    """Fix dependent parameters."""

    def impl(self):
        self.det_coeff[0] = np.sqrt(1 - np.sum(self.det_coeff[1:] ** 2))

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'get_parameters_mask')
def slater_get_parameters_mask(self):
    """Mask dependent parameters."""

    def impl(self) -> np.ndarray:
        res = np.ones_like(self.det_coeff, dtype=np.bool_)
        res[0] = False
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'get_parameters_scale')
def slater_get_parameters_scale(self, all_parameters):
    """Characteristic scale of each variable. Setting x_scale is equivalent
    to reformulating the problem in scaled variables xs = x / x_scale.
    An alternative view is that the size of a trust region along j-th
    dimension is proportional to x_scale[j].
    The purpose of this method is to reformulate the optimization problem
    with dimensionless variables having only one dimensional parameter - scale.
    """

    def impl(self, all_parameters) -> np.ndarray:
        if all_parameters:
            return 1 / self.det_coeff
        else:
            return 1 / self.det_coeff[1:]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'get_parameters_constraints')
def slater_get_parameters_constraints(self):
    """Returns det_coeff parameters
    :return:
    """

    def impl(self):
        return np.expand_dims(self.det_coeff, 0), np.ones(shape=(1,))

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'set_parameters_projector')
def slater_set_parameters_projector(self):
    """Get Projector matrix"""

    def impl(self):
        a, b = self.get_parameters_constraints()
        p = np.eye(a.shape[1]) - a.T @ np.linalg.pinv(a.T)
        mask_idx = np.argwhere(self.get_parameters_mask()).ravel()
        inv_p = np.linalg.inv(p[:, mask_idx][mask_idx, :])
        self.parameters_projector = p[:, mask_idx] @ inv_p

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'get_parameters')
def slater_get_parameters(self, all_parameters):
    """Returns parameters in the following order:
    determinant coefficients accept the first.
    :param all_parameters:
    :return:
    """

    def impl(self, all_parameters):
        if all_parameters:
            return self.det_coeff
        else:
            return self.det_coeff[1:]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'set_parameters')
def slater_set_parameters(self, parameters, all_parameters):
    """Set parameters in the following order:
    determinant coefficients accept the first.
    :param parameters:
    :param all_parameters:
    :return:
    """

    def impl(self, parameters, all_parameters):
        if all_parameters:
            self.det_coeff = parameters[: self.det_coeff.size]
            return parameters[self.det_coeff.size :]
        else:
            self.det_coeff[1:] = parameters[: self.det_coeff.size - 1]
            self.fix_det_coeff_parameters()
            return parameters[self.det_coeff.shape[0] - 1 :]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'value_parameters_d1')
def slater_value_parameters_d1(self, n_vectors: np.ndarray):
    """First derivatives of logarithm wfn w.r.t. the parameters
    :param n_vectors: e-n vectors
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=(self.det_coeff.size,))
        for i in range(self.det_coeff.size):
            self.det_coeff[i] -= delta
            res[i] -= self.value(n_vectors)
            self.det_coeff[i] += 2 * delta
            res[i] += self.value(n_vectors)
            self.det_coeff[i] -= delta
        return self.parameters_projector.T @ (res / delta / 2 / self.value(n_vectors))

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'gradient_parameters_d1')
def slater_gradient_parameters_d1(self, n_vectors: np.ndarray):
    """First derivatives of gradient w.r.t. the parameters
    :param n_vectors: e-n vectors
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=(self.det_coeff.size, (self.neu + self.ned) * 3))
        for i in range(self.det_coeff.size):
            self.det_coeff[i] -= delta
            res[i] -= self.gradient(n_vectors)
            self.det_coeff[i] += 2 * delta
            res[i] += self.gradient(n_vectors)
            self.det_coeff[i] -= delta
        return self.parameters_projector.T @ (res / delta / 2)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'laplacian_parameters_d1')
def slater_laplacian_parameters_d1(self, n_vectors: np.ndarray):
    """First derivatives of laplacian w.r.t. the parameters
    :param n_vectors: e-n vectors
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=(self.det_coeff.size,))
        for i in range(self.det_coeff.size):
            self.det_coeff[i] -= delta
            res[i] -= self.laplacian(n_vectors)
            self.det_coeff[i] += 2 * delta
            res[i] += self.laplacian(n_vectors)
            self.det_coeff[i] -= delta
        return self.parameters_projector.T @ (res / delta / 2)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Slater_class_t, 'hessian_parameters_d1')
def slater_hessian_parameters_d1(self, n_vectors: np.ndarray):
    """First derivatives of hessian w.r.t. the parameters
    :param n_vectors: e-n vectors
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=(self.det_coeff.size, (self.neu + self.ned) * 3 * (self.neu + self.ned) * 3))
        for i in range(self.det_coeff.size):
            self.det_coeff[i] -= delta
            res[i] -= self.hessian(n_vectors)[0].ravel()
            self.det_coeff[i] += 2 * delta
            res[i] += self.hessian(n_vectors)[0].ravel()
            self.det_coeff[i] -= delta
        return (self.parameters_projector.T @ (res / delta / 2)).reshape(-1, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


class Slater(structref.StructRefProxy, AbstractSlater):
    def __new__(cls, *args, **kwargs):
        """Slater multideterminant wavefunction.
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
        :param gautol:
        :param mo_up:
        :param mo_down:
        :param coeff: determinant coefficients
        """

        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(
            neu,
            ned,
            nbasis_functions,
            first_shells,
            orbital_types,
            shell_moments,
            slater_orders,
            primitives,
            coefficients,
            exponents,
            gautol,
            mo_up,
            mo_down,
            permutation_up,
            permutation_down,
            coeff,
            cusp,
        ):
            self = structref.new(Slater_t)
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
            self.gautol = gautol
            self.permutation_up = permutation_up
            self.permutation_down = permutation_down
            self.mo_up = mo_up[: np.max(permutation_up) + 1 if neu else 0]
            self.mo_down = mo_down[: np.max(permutation_down) + 1 if ned else 0]
            self.det_coeff = coeff
            self.cusp = cusp
            self.norm = np.exp(-(math.lgamma(neu + 1) + math.lgamma(ned + 1)) / (neu + ned) / 2)
            self.parameters_projector = np.zeros(shape=(0, 0))
            # self.const_eye_2d = np.eye(neu + ned)
            # self.const_eye_3d = np.zeros(shape=(neu + ned, neu + ned, neu + ned))
            # np.fill_diagonal(self.const_eye_3d, 1)
            return self

        return init(*args, **kwargs)

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def cusp(self):
        return self.cusp

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value_matrix(self, n_vectors):
        return self.value_matrix(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value(self, n_vectors):
        return self.value(n_vectors)

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
    def tressian(self, n_vectors):
        return self.tressian(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def tressian_v2(self, n_vectors):
        return self.tressian_v2(n_vectors)


structref.define_boxing(Slater_class_t, Slater)
