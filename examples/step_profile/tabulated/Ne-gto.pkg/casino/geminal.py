import math

import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino.abstract import AbstractSlater
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
        return pool_u, pool_d, pool_lap_u, pool_lap_d

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
@overload_method(Geminal_class_t, 'value')
def geminal_value(self, n_vectors: np.ndarray):
    """Wave function value = sum_n c_n det(M_n)."""

    def impl(self, n_vectors: np.ndarray) -> float:
        pool_u, pool_d = self.pool_matrix(n_vectors)
        val = 0.0
        for n in range(self.c.size):
            val += self.c[n] * np.linalg.det(self.geminal_matrix(pool_u, pool_d, n))
        return val

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
@overload_method(Geminal_class_t, 'gradient')
def geminal_gradient(self, n_vectors: np.ndarray):
    """Gradient ∇ψ/ψ w.r.t e-coordinates."""

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        pool_u, pool_d, pool_grad_u, pool_grad_d = self.pool_gradient_matrix(n_vectors)
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
@overload_method(Geminal_class_t, 'laplacian')
def geminal_laplacian(self, n_vectors: np.ndarray):
    """Scalar laplacian Δψ/ψ w.r.t e-coordinates."""

    def impl(self, n_vectors: np.ndarray) -> float:
        pool_u, pool_d, pool_lap_u, pool_lap_d = self.pool_laplacian_matrix(n_vectors)
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


class Geminal(structref.StructRefProxy, AbstractSlater):
    def __new__(cls, config):
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
            Harmonics(np.max(config.wfn.shell_moments)),
        )

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


structref.define_boxing(Geminal_class_t, Geminal)
