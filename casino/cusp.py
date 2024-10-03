#!/usr/bin/env python3

import logging
import math
import numpy as np
import numba as nb
from numba.experimental import structref
from numba.extending import overload_method

from scipy.optimize import minimize
from numpy.polynomial.polynomial import polyval
from casino.abstract import AbstractCusp
from casino.harmonics import value_angular_part
from casino.readers import CasinoConfig

logger = logging.getLogger(__name__)


@structref.register
class Cusp_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)

Cusp_t = Cusp_class_t([
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('orbitals_up', nb.int64),
    ('orbitals_down', nb.int64),
    ('rc', nb.float64[:, ::1]),
    ('shift', nb.float64[:, ::1]),
    ('orbital_sign', nb.int64[:, ::1]),
    ('alpha', nb.float64[:, :, ::1]),
    ('norm', nb.float64),
    ('mo', nb.float64[:, ::1]),
    ('first_shells', nb.int64[::1]),
    ('shell_moments', nb.int64[::1]),
    ('primitives', nb.int64[::1]),
    ('coefficients', nb.float64[::1]),
    ('exponents', nb.float64[::1]),
    ('is_pseudoatom', nb.boolean[::1]),
])

class Cusp(structref.StructRefProxy, AbstractCusp):
    """Scheme for adding electron–nucleus cusps to Gaussian orbitals
    A. Ma, M. D. Towler, N. D. Drummond and R. J. Needs

    An orbital, psi, expanded in a Gaussian basis set can be written as:

    psi = phi + eta

    where phi is the part of the orbital arising from the s-type Gaussian functions
    centered on the nucleus in question (which, for convenience is at r = 0)

    In our scheme we seek a corrected orbital, psi_tilde, which differs from psi
    only in the part arising from the s-type Gaussian functions centered on the nucleus,
    i.e., so that psi_tilde obeys the cusp condition at r=0

    psi_tilde = phi_tilde + eta

    We apply a cusp correction to each orbital at each nucleus at which it is nonzero.
    Inside some cusp correction radius rc we replace phi, the part of the orbital arising
    from s-type Gaussian functions centered on the nucleus in question, by:

    phi_tilde = C + sign[phi_tilde(0)] * exp(p(r))

    In this expression sign[phi_tilde(0)], reflecting the sign of tilde_phĩ at the nucleus,
    and C is a shift chosen so that phi_tilde − C is of one sign within rc. This shift is
    necessary since the uncorrected s-part of the orbital phi may have a node where it changes
    sign inside the cusp correction radius, and we wish to replace phi by an exponential
    function, which is necessarily of one sign everywhere. The polynomial p is given by:

        p = alpha_0 + alpha_1 * r + alpha_2 * r^2 + alpha_3 * r^3 + alpha_0 * r^4

    To get gaussian cusp information from CASINO output set the following settings in
    input:
        cusp_info         : T       #*! Print Gaussian cusp info
    and in gaussians.f90:
        POLYPRINT=.true. ! Include cusp polynomial coefficients in CUSP_INFO output.
    """

    def __new__(cls, *args, **kwargs):
        return cusp_init(*args, **kwargs)

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def orbital_sign(self):
        return self.orbital_sign

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def shift(self):
        return self.shift

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def rc(self):
        return self.rc

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def alpha(self):
        return self.alpha


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Cusp_class_t, 'exp')
def cusp_exp(self, atom, orbital, r):
    """Exponent part"""
    def impl(self, atom, orbital, r) -> float:
        return self.orbital_sign[atom, orbital] * np.exp(
            # FIXME: use polyval(r, self.alpha[:, atom, i])
            self.alpha[0, atom, orbital] +
            self.alpha[1, atom, orbital] * r +
            self.alpha[2, atom, orbital] * r ** 2 +
            self.alpha[3, atom, orbital] * r ** 3 +
            self.alpha[4, atom, orbital] * r ** 4
        )
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Cusp_class_t, 'diff_1')
def cusp_diff_1(self, atom, orbital, r):
    """f`(r) / r"""
    def impl(self, atom, orbital, r) -> float:
        return (
               self.alpha[1, atom, orbital] +
               2 * self.alpha[2, atom, orbital] * r +
               3 * self.alpha[3, atom, orbital] * r ** 2 +
               4 * self.alpha[4, atom, orbital] * r ** 3
           ) / r
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Cusp_class_t, 'diff_2')
def cusp_diff_2(self, atom, orbital, r):
    """f``(r) / r²"""
    def impl(self, atom, orbital, r) -> float:
        return (
               2 * self.alpha[2, atom, orbital] +
               6 * self.alpha[3, atom, orbital] * r +
               12 * self.alpha[4, atom, orbital] * r ** 2 +
               (
                   self.alpha[1, atom, orbital] +
                   2 * self.alpha[2, atom, orbital] * r +
                   3 * self.alpha[3, atom, orbital] * r ** 2 +
                   4 * self.alpha[4, atom, orbital] * r ** 3
               ) ** 2
           ) / r ** 2
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Cusp_class_t, 'diff_3')
def cusp_diff_3(self, atom, orbital, r):
    """f```(r) / r³"""
    def impl(self, atom, orbital, r) -> float:
        return (
               6 * self.alpha[3, atom, orbital] +
               24 * self.alpha[4, atom, orbital] * r +
               6 * (
                   self.alpha[2, atom, orbital] +
                   3 * self.alpha[3, atom, orbital] * r +
                   6 * self.alpha[4, atom, orbital] * r ** 2
               ) * (
                   self.alpha[1, atom, orbital] +
                   2 * self.alpha[2, atom, orbital] * r +
                   3 * self.alpha[3, atom, orbital] * r ** 2 +
                   4 * self.alpha[4, atom, orbital] * r ** 3
               ) + (
                   self.alpha[1, atom, orbital] +
                   2 * self.alpha[2, atom, orbital] * r +
                   3 * self.alpha[3, atom, orbital] * r ** 2 +
                   4 * self.alpha[4, atom, orbital] * r ** 3
               ) ** 3
           ) / r ** 3
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Cusp_class_t, 'value')
def cusp_value(self, n_vectors: np.ndarray):
    """Cusp correction for s-part of orbitals."""
    def impl(self, n_vectors: np.ndarray):
        value = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned))
        for i in range(self.orbitals_up):
            for j in range(self.neu):
                p = ao = 0
                for atom in range(n_vectors.shape[0]):
                    if not self.is_pseudoatom[atom]:
                        r = np.sqrt(n_vectors[atom, j] @ n_vectors[atom, j])
                        if r < self.rc[atom, i]:
                            value[i, j] = self.exp(atom, i, r) + self.shift[atom, i]
                        # FIXME: if s-орбитали contribution < cusp_threshold = 1e-7
                        s_part = 0.0
                        for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                                for primitive in range(self.primitives[nshell]):
                                    # look for Slater.value for s-radial gaussian function
                                    s_part += self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r * r) * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        # subtract uncusped s-part
                        value[i, j] -= s_part * self.norm

        for i in range(self.orbitals_up, self.orbitals_up + self.orbitals_down):
            for j in range(self.neu, self.neu + self.ned):
                p = ao = 0
                for atom in range(n_vectors.shape[0]):
                    if not self.is_pseudoatom[atom]:
                        r = np.sqrt(n_vectors[atom, j] @ n_vectors[atom, j])
                        if r < self.rc[atom, i]:
                            value[i, j] = self.exp(atom, i, r) + self.shift[atom, i]

                        s_part = 0.0
                        for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                                for primitive in range(self.primitives[nshell]):
                                    # look for Slater.value for s-radial gaussian function
                                    s_part += self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r * r) * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        # subtract uncusped s-part
                        value[i, j] -= s_part * self.norm

        return value[:self.orbitals_up, :self.neu], value[self.orbitals_up:, self.neu:]
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Cusp_class_t, 'gradient')
def cusp_gradient(self, n_vectors: np.ndarray):
    """Cusp part of gradient
    df(r)/dx = ri * f`(r) / r
    """
    def impl(self, n_vectors: np.ndarray):
        gradient = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned, 3))
        for i in range(self.orbitals_up):
            for j in range(self.neu):
                p = ao = 0
                for atom in range(n_vectors.shape[0]):
                    if not self.is_pseudoatom[atom]:
                        r = np.sqrt(n_vectors[atom, j] @ n_vectors[atom, j])
                        if r < self.rc[atom, i]:
                            gradient[i, j] = self.diff_1(atom, i, r) * self.exp(atom, i, r) * n_vectors[atom, j]

                        s_part = 0.0
                        for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                                for primitive in range(self.primitives[nshell]):
                                    # look for Slater.gradient for s-radial gaussian function
                                    alpha = self.exponents[p + primitive]
                                    exponent = self.coefficients[p + primitive] * np.exp(-alpha * r * r)
                                    s_part -= 2 * alpha * exponent * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        # subtract uncusped s-part
                        gradient[i, j] -= n_vectors[atom, j] * s_part * self.norm

        for i in range(self.orbitals_up, self.orbitals_up + self.orbitals_down):
            for j in range(self.neu, self.neu + self.ned):
                p = ao = 0
                for atom in range(n_vectors.shape[0]):
                    if not self.is_pseudoatom[atom]:
                        r = np.sqrt(n_vectors[atom, j] @ n_vectors[atom, j])
                        if r < self.rc[atom, i]:
                            gradient[i, j] = self.diff_1(atom, i, r) * self.exp(atom, i, r) * n_vectors[atom, j]

                        s_part = 0.0
                        for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                                for primitive in range(self.primitives[nshell]):
                                    # look for Slater.gradient for s-radial gaussian function
                                    alpha = self.exponents[p + primitive]
                                    exponent = self.coefficients[p + primitive] * np.exp(-alpha * r * r)
                                    s_part -= 2 * alpha * exponent * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        # subtract uncusped s-part
                        gradient[i, j] -= n_vectors[atom, j] * s_part * self.norm

        return gradient[:self.orbitals_up, :self.neu], gradient[self.orbitals_up:, self.neu:]
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Cusp_class_t, 'laplacian')
def cusp_laplacian(self, n_vectors: np.ndarray):
    """Cusp part of laplacian
    https://math.stackexchange.com/questions/1048973/laplacian-of-a-radial-function
    ∇²(f(r)) = f``(r) + 2 * f`(r) / r
    """
    def impl(self, n_vectors: np.ndarray):
        laplacian = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned))
        for i in range(self.orbitals_up):
            for j in range(self.neu):
                p = ao = 0
                for atom in range(n_vectors.shape[0]):
                    if not self.is_pseudoatom[atom]:
                        r = np.sqrt(n_vectors[atom, j] @ n_vectors[atom, j])
                        if r < self.rc[atom, i]:
                            laplacian[i, j] = (2 * self.diff_1(atom, i, r) + self.diff_2(atom, i, r) * r ** 2) * self.exp(atom, i, r)

                        s_part = 0.0
                        for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                                for primitive in range(self.primitives[nshell]):
                                    # look for Slater.laplacian for s-radial gaussian function
                                    alpha = self.exponents[p + primitive]
                                    exponent = self.coefficients[p + primitive] * np.exp(-alpha * r * r)
                                    s_part += 2 * alpha * (2 * alpha * r * r - 3) * exponent * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        # subtract uncusped s-part
                        laplacian[i, j] -= s_part * self.norm

        for i in range(self.orbitals_up, self.orbitals_up + self.orbitals_down):
            for j in range(self.neu, self.neu + self.ned):
                p = ao = 0
                for atom in range(n_vectors.shape[0]):
                    if not self.is_pseudoatom[atom]:
                        r = np.sqrt(n_vectors[atom, j] @ n_vectors[atom, j])
                        if r < self.rc[atom, i]:
                            laplacian[i, j] = (2 * self.diff_1(atom, i, r) + self.diff_2(atom, i, r) * r ** 2) * self.exp(atom, i, r)

                        s_part = 0.0
                        for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                                for primitive in range(self.primitives[nshell]):
                                    # look for Slater.laplacian for s-radial gaussian function
                                    alpha = self.exponents[p + primitive]
                                    exponent = self.coefficients[p + primitive] * np.exp(-alpha * r * r)
                                    s_part += 2 * alpha * (2 * alpha * r * r - 3) * exponent * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        # subtract uncusped s-part
                        laplacian[i, j] -= s_part * self.norm

        return laplacian[:self.orbitals_up, :self.neu], laplacian[self.orbitals_up:, self.neu:]
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Cusp_class_t, 'hessian')
def cusp_hessian(self, n_vectors: np.ndarray):
    """Cusp part of hessian
    https://sunlimingbit.wordpress.com/2018/09/23/hessian-of-radial-functions/
    d²f(r)/dxdy = ri ⊗ rj * f``(r) / r² + (δij - ri ⊗ rj / r²) * f`(r) / r
    """
    def impl(self, n_vectors: np.ndarray):
        hessian = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned, 3, 3))
        for i in range(self.orbitals_up):
            for j in range(self.neu):
                p = ao = 0
                for atom in range(n_vectors.shape[0]):
                    if not self.is_pseudoatom[atom]:
                        r = np.sqrt(n_vectors[atom, j] @ n_vectors[atom, j])
                        ri_rj = np.outer(n_vectors[atom, j], n_vectors[atom, j])
                        if r < self.rc[atom, i]:
                            hessian[i, j, :, :] = (
                                self.diff_2(atom, i, r) * ri_rj + self.diff_1(atom, i, r) * (np.eye(3) - ri_rj / r ** 2)
                            ) * self.exp(atom, i, r)
                        s_part = np.zeros(shape=(3, 3))
                        for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                                for primitive in range(self.primitives[nshell]):
                                    # look for Slater.hessian for s-radial gaussian function
                                    alpha = self.exponents[p + primitive]
                                    exponent = self.coefficients[p + primitive] * np.exp(-alpha * r * r)
                                    c = -2 * alpha
                                    s_part += (ri_rj * c + np.eye(3)) * c * exponent * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        # subtract uncusped s-part
                        hessian[i, j] -= s_part * self.norm

        for i in range(self.orbitals_up, self.orbitals_up + self.orbitals_down):
            for j in range(self.neu, self.neu + self.ned):
                p = ao = 0
                for atom in range(n_vectors.shape[0]):
                    if not self.is_pseudoatom[atom]:
                        r = np.sqrt(n_vectors[atom, j] @ n_vectors[atom, j])
                        ri_rj = np.outer(n_vectors[atom, j], n_vectors[atom, j])
                        if r < self.rc[atom, i]:
                            hessian[i, j, :, :] = (
                                self.diff_2(atom, i, r) * ri_rj + self.diff_1(atom, i, r) * (np.eye(3) - ri_rj / r ** 2)
                            ) * self.exp(atom, i, r)

                        s_part = np.zeros(shape=(3, 3))
                        for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                                for primitive in range(self.primitives[nshell]):
                                    # look for Slater.hessian for s-radial gaussian function
                                    alpha = self.exponents[p + primitive]
                                    exponent = self.coefficients[p + primitive] * np.exp(-alpha * r * r)
                                    c = -2 * alpha
                                    s_part += (ri_rj * c + np.eye(3)) * c * exponent * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        # subtract uncusped s-part
                        hessian[i, j] -= s_part * self.norm

        return hessian[:self.orbitals_up, :self.neu], hessian[self.orbitals_up:, self.neu:]
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Cusp_class_t, 'tressian')
def cusp_tressian(self, n_vectors: np.ndarray):
    """Cusp part of tressian
    d³f(r)/dxdydz = ri ⊗ rj ⊗ rk * f```(r) / r³ +
    (δik ⊗ rj + δjk ⊗ ri + δij ⊗ rk - 3 * ri ⊗ rj ⊗ rk / r²) * f``(r) / r² +
    (3 * ri ⊗ rj ⊗ rk / r**4 - δij ⊗ rk / r² - δik ⊗ rj / r² - δjk ⊗ ri / r²) * f`(r) / r
    """
    def impl(self, n_vectors: np.ndarray):
        tressian = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned, 3, 3, 3))

        for i in range(self.orbitals_up):
            for j in range(self.neu):
                p = ao = 0
                for atom in range(n_vectors.shape[0]):
                    if not self.is_pseudoatom[atom]:
                        r = np.sqrt(n_vectors[atom, j] @ n_vectors[atom, j])
                        ri_rj_rk = np.expand_dims(np.outer(n_vectors[atom, j], n_vectors[atom, j]), 2) * n_vectors[atom, j]
                        kronecker = (
                            np.expand_dims(np.eye(3), 2) * n_vectors[atom, j] +
                            np.expand_dims(np.eye(3), 1) * np.expand_dims(n_vectors[atom, j], 1) +
                            np.expand_dims(np.eye(3), 0) * np.expand_dims(np.expand_dims(n_vectors[atom, j], 1), 2)
                        )
                        if r < self.rc[atom, i]:
                            tressian[i, j, :, :, :] = (
                                self.diff_3(atom, i, r) * ri_rj_rk +
                                self.diff_2(atom, i, r) * (kronecker - 3 * ri_rj_rk / r ** 2) +
                                self.diff_1(atom, i, r) * (3 * ri_rj_rk / r ** 2 - kronecker) / r ** 2
                            ) * self.exp(atom, i, r)

                        s_part = np.zeros(shape=(3, 3, 3))
                        for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                                for primitive in range(self.primitives[nshell]):
                                    # look for Slater.tressian for s-radial gaussian function
                                    alpha = self.exponents[p + primitive]
                                    exponent = self.coefficients[p + primitive] * np.exp(-alpha * r * r)
                                    c = -2 * alpha
                                    s_part += (ri_rj_rk * c + kronecker) * c ** 2 * exponent * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        tressian[i, j] -= s_part * self.norm

        for i in range(self.orbitals_up, self.orbitals_up + self.orbitals_down):
            for j in range(self.neu, self.neu + self.ned):
                p = ao = 0
                for atom in range(n_vectors.shape[0]):
                    if not self.is_pseudoatom[atom]:
                        r = np.sqrt(n_vectors[atom, j] @ n_vectors[atom, j])
                        ri_rj_rk = np.expand_dims(np.outer(n_vectors[atom, j], n_vectors[atom, j]), 2) * n_vectors[atom, j]
                        kronecker = (
                                np.expand_dims(np.eye(3), 2) * n_vectors[atom, j] +
                                np.expand_dims(np.eye(3), 1) * np.expand_dims(n_vectors[atom, j], 1) +
                                np.expand_dims(np.eye(3), 0) * np.expand_dims(np.expand_dims(n_vectors[atom, j], 1), 2)
                        )
                        if r < self.rc[atom, i]:
                            tressian[i, j, :, :, :] = (
                                self.diff_3(atom, i, r) * ri_rj_rk +
                                self.diff_2(atom, i, r) * (kronecker - 3 * ri_rj_rk / r ** 2) +
                                self.diff_1(atom, i, r) * (3 * ri_rj_rk / r ** 2 - kronecker) / r ** 2
                            ) * self.exp(atom, i, r)

                        s_part = np.zeros(shape=(3, 3, 3))
                        for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                                for primitive in range(self.primitives[nshell]):
                                    # look for Slater.tressian for s-radial gaussian function
                                    alpha = self.exponents[p + primitive]
                                    exponent = self.coefficients[p + primitive] * np.exp(-alpha * r * r)
                                    c = -2 * alpha
                                    s_part += (ri_rj_rk * c + kronecker) * c ** 2 * exponent * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        # subtract uncusped s-part
                        tressian[i, j] -= s_part * self.norm

        return tressian[:self.orbitals_up, :self.neu], tressian[self.orbitals_up:, self.neu:]
    return impl


structref.define_boxing(Cusp_class_t, Cusp)


@nb.njit(nogil=True, parallel=False, cache=True)
def cusp_init(neu, ned, orbitals_up, orbitals_down, rc, shift, orbital_sign, alpha,
    mo, first_shells, shell_moments, primitives, coefficients, exponents, is_pseudoatom):
    self = structref.new(Cusp_t)
    self.neu = neu
    self.ned = ned
    self.norm = np.exp(-(math.lgamma(neu + 1) + math.lgamma(ned + 1)) / (neu + ned) / 2)
    self.orbitals_up = orbitals_up
    self.orbitals_down = orbitals_down
    self.rc = rc
    self.shift = shift
    self.orbital_sign = orbital_sign
    self.alpha = alpha
    self.mo = mo
    self.first_shells = first_shells
    self.shell_moments = shell_moments
    self.primitives = primitives
    self.coefficients = coefficients
    self.exponents = exponents
    self.is_pseudoatom = is_pseudoatom
    return self


class CuspFactory:

    def __init__(
            self, neu, ned, cusp_threshold, mo_up, mo_down, permutation_up, permutation_down,
            first_shells, shell_moments, primitives, coefficients, exponents, atom_positions,
            atom_charges, unrestricted, is_pseudoatom,
    ):
        self.neu = neu
        self.ned = ned
        self.orbitals_up = np.max(permutation_up) + 1 if neu else 0
        self.orbitals_down = np.max(permutation_down) + 1 if ned else 0
        self.norm = np.exp(-(math.lgamma(self.neu + 1) + math.lgamma(self.ned + 1)) / (self.neu + self.ned) / 2)
        self.casino_norm = np.exp(-(math.lgamma(self.neu + 1) + math.lgamma(self.neu + 1)) / (self.neu + self.neu) / 2)
        self.mo = np.concatenate((mo_up[:self.orbitals_up], mo_down[:self.orbitals_down]))
        self.first_shells = first_shells
        self.shell_moments = shell_moments
        self.primitives = primitives
        self.coefficients = coefficients
        self.exponents = exponents
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.cusp_threshold = cusp_threshold
        self.phi_0, _, _ = self.phi(np.zeros(shape=(self.atom_positions.shape[0], self.mo.shape[0])))
        self.orb_mask = np.abs(self.phi_0) > self.cusp_threshold
        self.beta = np.array([3.25819, -15.0126, 33.7308, -42.8705, 31.2276, -12.1316, 1.94692])
        # atoms, MO - Value of corrected orbital at nucleus
        self.phi_tilde_0 = self.phi_0
        # atoms, MO - cusp correction radius
        self.rc = self.rc_initial()
        # atoms, MO - sign of s-type Gaussian functions centered on the nucleus
        self.orbital_sign = self.phi_sign()
        # atoms, MO - shift chosen so that phi − shift is of one sign within rc
        self.shift = np.zeros((self.atom_positions.shape[0], self.mo.shape[0]))
        # atoms, MO - contribution from Gaussians on other nuclei
        self.eta = self.eta_data()
        self.unrestricted = unrestricted
        logger.info(
            ' Gaussian cusp correction\n'
            ' ========================\n'
            ' Activated.\n'
        )
        self.is_pseudoatom = is_pseudoatom

    def phi(self, rc):
        """Wfn of single electron of s-orbitals on each atom"""
        orbital = np.zeros((self.atom_positions.shape[0], self.mo.shape[0], self.mo.shape[1]))
        orbital_derivative = np.zeros((self.atom_positions.shape[0], self.mo.shape[0], self.mo.shape[1]))
        orbital_second_derivative = np.zeros((self.atom_positions.shape[0], self.mo.shape[0], self.mo.shape[1]))
        for orb in range(self.mo.shape[0]):
            p = ao = 0
            for atom in range(self.atom_positions.shape[0]):
                for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                    l = self.shell_moments[nshell]
                    s_part = s_derivative_part = s_second_derivative_part = 0.0
                    if self.shell_moments[nshell] == 0:
                        for primitive in range(self.primitives[nshell]):
                            r = rc[atom, orb]
                            alpha = self.exponents[p + primitive]
                            # FIXME: RuntimeWarning: underflow encountered in exp
                            exponent = self.coefficients[p + primitive] * np.exp(-alpha * r * r)
                            s_part += exponent
                            s_derivative_part -= 2 * alpha * r * exponent
                            s_second_derivative_part += 2 * alpha * (2 * alpha * r * r - 1) * exponent
                        orbital[atom, orb, ao] = s_part
                        orbital_derivative[atom, orb, ao] = s_derivative_part
                        orbital_second_derivative[atom, orb, ao] = s_second_derivative_part
                    ao += 2 * l + 1
                    p += self.primitives[nshell]
        return (
            np.sum(orbital * self.mo, axis=2) * self.norm,
            np.sum(orbital_derivative * self.mo, axis=2) * self.norm,
            np.sum(orbital_second_derivative * self.mo, axis=2) * self.norm
        )

    def eta_data(self):
        """Contribution from Gaussians on other nuclei"""
        orbital = np.zeros(shape=(self.atom_positions.shape[0], self.mo.shape[0], self.mo.shape[1]))
        for atom in range(self.atom_positions.shape[0]):
            for orb in range(self.mo.shape[0]):
                p = ao = 0
                for orb_atom in range(self.atom_positions.shape[0]):
                    x, y, z = self.atom_positions[atom] - self.atom_positions[orb_atom]
                    r2 = x * x + y * y + z * z
                    angular = value_angular_part(x, y, z)
                    for nshell in range(self.first_shells[orb_atom] - 1, self.first_shells[orb_atom + 1] - 1):
                        l = self.shell_moments[nshell]
                        radial = 0.0
                        if atom != orb_atom:
                            for primitive in range(self.primitives[nshell]):
                                # FIXME: RuntimeWarning: underflow encountered in exp
                                radial += self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r2)
                            for m in range(2 * l + 1):
                                orbital[atom, orb, ao+m] += angular[l*l+m] * radial
                        ao += 2 * l + 1
                        p += self.primitives[nshell]
        return np.sum(orbital * self.mo, axis=2) * self.norm

    def rc_initial(self):
        """Initial rc"""
        return np.where(self.orb_mask, 1 / self.atom_charges[:, np.newaxis], 0)

    def phi_sign(self):
        """Calculate phi sign."""
        return np.where(self.orb_mask, np.sign(self.phi_0), 0).astype(np.int_)

    def alpha_data(self, phi_tilde_0):
        """Calculate phi coefficients.
        shift variable chosen so that (phi−shift) is of one sign within rc.
        eta is a contribution from Gaussians on other nuclei.
        """
        rc = self.rc
        np.seterr(divide='ignore', invalid='ignore')
        alpha = np.zeros(shape=(5, self.atom_positions.shape[0], self.mo.shape[0]))
        phi_rc, phi_diff_rc, phi_diff_2_rc = self.phi(rc)
        R = phi_tilde_0 - self.shift
        X1 = np.log(np.abs(phi_rc - self.shift))                                  # (9)
        X2 = phi_diff_rc / (phi_rc - self.shift)                                  # (10)
        X3 = phi_diff_2_rc / (phi_rc - self.shift)                                # (11)
        X4 = -self.atom_charges[:, np.newaxis] * (self.shift + R + self.eta) / R  # (12)
        X5 = np.log(np.abs(R))                                                    # (13)
        # (14)
        alpha[0] = X5
        alpha[1] = X4
        alpha[2] = 6*X1/rc**2 - 3*X2/rc + X3/2 - 3*X4/rc - 6*X5/rc**2 - X2**2/2
        alpha[3] = -8*X1/rc**3 + 5*X2/rc**2 - X3/rc + 3*X4/rc**2 + 8*X5/rc**3 + X2**2/rc
        alpha[4] = 3*X1/rc**4 - 2*X2/rc**3 + X3/2/rc**2 - X4/rc**3 - 3*X5/rc**4 - X2**2/2/rc**2
        np.seterr(divide='warn', invalid='warn')
        # remove NaN from orbitals without s-part
        return np.nan_to_num(alpha, posinf=0, neginf=0)

    def phi_energy(self, r):
        """Effective one-electron local energy for gaussian s-part orbital.
        :param r:
        :return: energy
        """
        R = self.phi_0 - self.shift
        phi_rc, phi_diff_rc, phi_diff_2_rc = self.phi(r)
        z_eff = self.atom_charges[:, np.newaxis] * (1 + self.eta / (R + self.shift))  # (16)
        return - 0.5 * (2 * phi_diff_rc / r + phi_diff_2_rc) / phi_rc - z_eff / r     # (15)

    def phi_tilde_energy(self, r, alpha):
        """Effective one-electron local energy for corrected orbital.
        Equation (15)
        :param r:
        :return: energy
        """
        p = alpha[0] + alpha[1] * r + alpha[2] * r ** 2 + alpha[3] * r ** 3 + alpha[4] * r ** 4
        p_diff_1 = alpha[1] + 2 * alpha[2] * r + 3 * alpha[3] * r ** 2 + 4 * alpha[4] * r ** 3
        p_diff_2 = 2 * alpha[2] + 2 * 3 * alpha[3] * r + 3 * 4 * alpha[4] * r ** 2
        R = self.orbital_sign * np.exp(p)
        np.seterr(divide='ignore', invalid='ignore')
        z_eff = self.atom_charges[:, np.newaxis] * (1 + self.eta / (R + self.shift))  # (16)
        # np.where is not lazy
        # https://pretagteam.com/question/numpy-where-function-can-not-avoid-evaluate-sqrtnegative
        energy = np.where(
            r == 0,
            # apply L'Hôpital's rule to find energy limit at r=0 in (15)
            -0.5 * R / (R + self.shift) * (3 * p_diff_2 + p_diff_1 ** 2),
            -0.5 * R / (R + self.shift) * (2 * p_diff_1 / r + p_diff_2 + p_diff_1 ** 2) - z_eff / r
        )  # (15)
        np.seterr(divide='warn', invalid='warn')
        return energy

    def ideal_energy(self, r, beta0):
        """Ideal energy.
        :param r:
        :param beta0:
        :return:
        """
        return (beta0 + np.where(self.atom_charges[:, np.newaxis] == 1, 0, polyval(r, self.beta) * r**2)) * self.atom_charges[:, np.newaxis] ** 2  # (17)

    def get_energy_diff_max(self, alpha):
        """Maximum square deviation of phi_tilde energy from the ideal energy
        :return:
        """
        steps = 1000
        beta0 = (self.phi_tilde_energy(self.rc, alpha) - self.ideal_energy(self.rc, 0)) / self.atom_charges[:, np.newaxis] ** 2
        r = np.linspace(0, self.rc, steps + 1)
        energy = np.abs(self.phi_tilde_energy(r, alpha) - self.ideal_energy(r, beta0))
        return np.max(energy, axis=0)

    def optimize_rc(self):
        """Optimize rc"""
        rc = self.rc
        beta0 = (self.phi_energy(self.rc_initial()) - self.ideal_energy(self.rc_initial(), 0)) / self.atom_charges[:, np.newaxis] ** 2
        for atom in range(self.atom_positions.shape[0]):
            for orb in range(self.mo.shape[0]):
                r = rc[atom, orb]
                if r == 0.0:
                    rc[atom, orb] = 0.0
                    continue

                for r in np.linspace(rc[atom, orb], 0, int(rc[atom, orb] * 2000) + 1):
                    energy_delta = np.abs(self.phi_energy(rc) - self.ideal_energy(rc, beta0))
                    if (energy_delta > self.atom_charges[:, np.newaxis] ** 2 / 50)[atom, orb]:
                        rc[atom, orb] = r
                        break

        drc = rc * 0.05 * 2000

        for r in np.linspace(rc - 4 * drc, rc + 4 * drc, 9):
            self.optimize_phi_tilde_0(r, np.copy(self.phi_0))

        return rc

    def optimize_phi_tilde_0(self, phi_tilde_0):
        """Optimize phi_tilde at r=0
        :param phi_tilde_0: initial value
        :return:
        """
        nonzero_index = np.nonzero(self.orb_mask)
        nonzero_phi_tilde_0 = phi_tilde_0[nonzero_index]

        def f(x):
            phi_tilde_0[nonzero_index] = x
            alpha = self.alpha_data(phi_tilde_0)
            self.energy_diff_max = self.get_energy_diff_max(alpha)
            return np.sum(self.energy_diff_max[nonzero_index])

        options = dict(disp=False)
        res = minimize(f, nonzero_phi_tilde_0, method='Powell', options=options)
        phi_tilde_0[nonzero_index] = res.x
        return phi_tilde_0

    def create(self, casino_rc=False, casino_phi_tilde_0=False):
        """Create cusp class.
        :param casino_rc: get rc from CASINO
        :param casino_phi_tilde_0: get phi_tilde_0 from CASINO
        :return:
        """
        # He atom
        if self.orbitals_up == 1 and self.orbitals_down == 1:
            # atoms, MO - Value of uncorrected orbital at nucleus
            wfn_0_up = wfn_0_down = np.array([[1.307524154011]])
            # atoms, MO - cusp correction radius
            rc_up = rc_down = np.array([[0.4375]])
            # atoms, MO - Optimum corrected s orbital at nucleus
            phi_tilde_0_up = phi_tilde_0_down = np.array([[1.338322724162]])
        # Be atom
        elif self.orbitals_up == 2 and self.orbitals_down == 2:
            wfn_0_up = wfn_0_down = np.array([[-3.447246814709, -0.628316785317]])
            rc_up = rc_down = np.array([[0.1205, 0.1180]])
            phi_tilde_0_up = phi_tilde_0_down = np.array([[-3.481156233321, -0.634379297525]])
        # N atom
        elif self.orbitals_up == 5 and self.orbitals_down == 2:
            wfn_0_up = np.array([[6.069114031640, -1.397116693472, 0.0, 0.0, 0.0]])
            wfn_0_down = np.array([[6.095832387803, 1.268342737910]])
            rc_up = np.array([[0.0670, 0.0695, 0.0, 0.0, 0.0]])
            rc_down = np.array([[0.0675, 0.0680]])
            phi_tilde_0_up = np.array([[6.130043694767, -1.412040439372, 0.0, 0.0, 0.0]])
            phi_tilde_0_down = np.array([[6.155438260537, 1.280709246720]])
        # Ne atom
        elif self.orbitals_up == 5 and self.orbitals_down == 5:
            wfn_0_up = wfn_0_down = np.array([[10.523069754656, 2.470734575103, 0.0, 0.0, 0.0]])
            rc_up = rc_down = np.array([[0.0455, 0.0460, 0.0, 0.0, 0.0]])
            phi_tilde_0_up = phi_tilde_0_down = np.array([[10.624267229647, 2.494850990545, 0.0, 0.0, 0.0]])
        # Ar atom
        elif self.orbitals_up == 9 and self.orbitals_down == 9:
            wfn_0_up = wfn_0_down = np.array([[20.515046538335, 5.824658914949, 0.0, 0.0, 0.0, -1.820248905891, 0.0, 0.0, 0.0]])
            rc_up = rc_down = np.array([[0.0205, 0.0200, 0, 0, 0, 0.0205, 0, 0, 0]])
            phi_tilde_0_up = phi_tilde_0_down = np.array([[20.619199783780, 5.854393350981, 0.0, 0.0, 0.0, -1.829517070413, 0.0, 0.0, 0.0]])
        # Kr atom
        elif self.orbitals_up == 18 and self.orbitals_down == 18:
            wfn_0_up = wfn_0_down = np.array(([
                [43.608490133788, -13.720841107516, 0.0, 0.0, 0.0, -5.505781654931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.751185788791, 0.0, 0.0, 0.0],
            ]))
            rc_up = rc_down = np.array([[0.0045, 0.0045, 0, 0, 0, 0.0045, 0, 0, 0, 0, 0, 0, 0, 0, 0.0045, 0, 0, 0]])
            phi_tilde_0_up = phi_tilde_0_down = np.array(([
                [43.713171699758, -13.754783719428, 0.0, 0.0, 0.0, -5.518712340056, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.757882280257, 0.0, 0.0, 0.0],
            ]))
        # O3 molecule
        elif self.orbitals_up == 12 and self.orbitals_down == 12:
            wfn_0_up = np.array(([
                [-5.245016636407, -0.025034008898,  0.019182670511, -0.839192164211,  0.229570396176, -0.697628545957, 0.0, -0.140965538444, -0.015299796091, 0.0, -0.084998032927,  0.220208807573],
                [-0.024547538656,  5.241296804923, -0.002693454373, -0.611438043012, -0.806215116184,  0.550648084416, 0.0, -0.250758940038, -0.185619271170, 0.0,  0.007450966720, -0.023495021763],
                [-0.018654332490, -0.002776929419, -5.248498638985, -0.386055559344,  0.686627203383,  0.707083323432, 0.0, -0.029625096851,  0.443458560481, 0.0, -0.034753046153, -0.008117407260],
            ]))
            wfn_0_down = np.array(([
                [-5.245016636416, -0.025034009046,  0.019182670402,  0.839192164264, -0.229570396203, -0.697628545936, 0.0, -0.140965538413, -0.015299796160, 0.0,  0.084998032932, -0.220208807501],
                [-0.018654332375, -0.002776929447, -5.248498638992,  0.386055559309, -0.686627203339,  0.707083323455, 0.0, -0.029625097018,  0.443458560519, 0.0,  0.034753046180,  0.008117407241],
                [-0.024547538802,  5.241296804930, -0.002693454404,  0.611438042982,  0.806215116191,  0.550648084418, 0.0, -0.250758940010, -0.185619271253, 0.0, -0.007450966721,  0.023495021734],
            ]))
            rc_up = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
            ])
            rc_down = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
            ])
            phi_tilde_0_up = np.array(([
                [-5.296049272683, -0.025239447913,  0.019411088359, -0.861113290693,  0.232146160162, -0.696713729469, 0.0, -0.138314540320, -0.014622271569, 0.0, -0.081945526891,  0.210756161800],
                [-0.024767570243,  5.292572118624, -0.002698702824, -0.625758024602, -0.816314024031,  0.547346193861, 0.0, -0.243109138669, -0.182823180724, 0.0,  0.009584833192, -0.026173722205],
                [-0.018824846998, -0.002828414099, -5.299444354520, -0.398174414650,  0.701172068607,  0.709930839484, 0.0, -0.026979314507,  0.436599565939, 0.0, -0.032554800126, -0.012216984830],
            ]))
            phi_tilde_0_down = np.array(([
                [-5.296049272690, -0.025239448062,  0.019411088253,  0.861113290740, -0.232146160186, -0.696713729439, 0.0, -0.138314540293, -0.014622271629, 0.0,  0.081945526895, -0.210756161728],
                [-0.018824846884, -0.002828414129, -5.299444354528,  0.398174414615, -0.701172068569,  0.709930839504, 0.0, -0.026979314668,  0.436599565976, 0.0,  0.032554800149,  0.012216984810],
                [-0.024767570392,  5.292572118630, -0.002698702856,  0.625758024578,  0.816314024042,  0.547346193858, 0.0, -0.243109138643, -0.182823180811, 0.0, -0.009584833190,  0.026173722176],
            ]))
        if casino_rc:
            # atoms, MO - Value of uncorrected orbital at nucleus
            wfn_0 = np.concatenate((wfn_0_up, wfn_0_down), axis=1) * (self.norm / self.casino_norm)
            self.eta = wfn_0 - self.phi_0
            # atoms, MO - cusp correction radius
            self.rc = np.concatenate((rc_up, rc_down), axis=1)
        else:
            pass
            # rc = self.optimize_rc(rc)
        # atoms, MO - Value of corrected orbital at nucleus
        if casino_phi_tilde_0:
            self.phi_tilde_0 = np.concatenate((phi_tilde_0_up, phi_tilde_0_down), axis=1) * (self.norm / self.casino_norm)
        else:
            self.phi_tilde_0 = self.optimize_phi_tilde_0(np.copy(self.phi_0))

        alpha = self.alpha_data(self.phi_tilde_0)
        return Cusp(
            self.neu, self.ned, self.orbitals_up, self.orbitals_down, self.rc, self.shift, self.orbital_sign, alpha,
            self.mo, self.first_shells, self.shell_moments, self.primitives, self.coefficients, self.exponents, self.is_pseudoatom,
        )

    def cusp_info(self):
        """If cusp correction is set to T for an all-electron Gaussian basis set calculation,
        then casino will alter the orbitals inside a small radius around each nucleus in such a way
        that they obey the electron–nucleus cusp condition. If cusp info is set to T then information
        about precisely how this is done will be printed to the out file. Be aware that in large systems
        this may produce a lot of output.
        :return:
        """
        logger.info(
            ' Verbose print out flagged (turn off with cusp_info : F)\n'
        )
        for i in range(2) if self.unrestricted else range(1):
            if self.unrestricted:
                if i == 0:
                    logger.info(' UP SPIN\n')
                else:
                    logger.info(' DOWN SPIN\n')
            else:
                logger.info(' Spin restricted calculation.\n')
            for atom in range(self.atom_positions.shape[0]):
                for orb in range(self.orbitals_up) if i == 0 else range(self.orbitals_up, self.orbitals_up + self.orbitals_down):
                    logger.info(
                        f' Orbital {orb + 1 if i == 0 else orb + 1 - self.orbitals_up} at position of ion {atom + 1}'
                    )
                    if self.orb_mask[atom][orb]:
                        sign = 'positive' if self.orbital_sign[atom][orb] else 'negative'
                        z_eff = self.atom_charges[atom] * (1 + self.eta[atom][orb] / self.phi_0[atom][orb])
                        logger.info(
                            f' Sign of orbital at nucleus                : {sign}\n'
                            f' Cusp radius (au)                          : {self.rc[atom][orb]:16.12f}\n'
                            f' Value of uncorrected orbital at nucleus   : {(self.phi_0 + self.eta)[atom][orb]:16.12f}\n'
                            f' Value of s part of orbital at nucleus     : {self.phi_0[atom][orb]:16.12f}\n'
                            f' Optimum corrected s orbital at nucleus    : {self.phi_tilde_0[atom][orb]:16.12f}\n'
                            f' Maximum deviation from ideal local energy : {self.energy_diff_max[atom][orb]:16.12f}\n'
                            f' Effective nuclear charge                  : {z_eff:16.12f}\n'
                        )
                    else:
                        logger.info(' Orbital s component effectively zero at this nucleus.\n')
        nonzero_index = np.nonzero(self.orb_mask)
        logger.info(
            f' Maximum deviation from ideal (averaged over orbitals) : {np.mean(self.energy_diff_max[nonzero_index]):16.12f}.\n'
        )


class TestCuspFactory:

    def __init__(
            self, neu, ned, mo_up, mo_down, permutation_up, permutation_down,
            first_shells, shell_moments, primitives, coefficients, exponents
    ):
        self.neu = neu
        self.ned = ned
        self.orbitals_up = np.max(permutation_up) + 1
        self.orbitals_down = np.max(permutation_down) + 1
        self.norm = np.exp(-(math.lgamma(self.neu + 1) + math.lgamma(self.ned + 1)) / (self.neu + self.ned) / 2)
        self.casino_norm = np.exp(-(math.lgamma(self.neu + 1) + math.lgamma(self.neu + 1)) / (self.neu + self.neu) / 2)
        self.mo = np.concatenate((mo_up[:self.orbitals_up], mo_down[:self.orbitals_down]))
        self.first_shells = first_shells
        self.shell_moments = shell_moments
        self.primitives = primitives
        self.coefficients = coefficients
        self.exponents = exponents

    def create(self):
        if self.neu == 1 and self.ned == 1:
            is_pseudoatom = np.zeros(shape=(1, ), dtype=np.bool_)
            # atoms, MO - Value of uncorrected orbital at nucleus
            wfn_0_up = wfn_0_down = np.array([[1.307524154011]])
            # atoms, MO
            shift_up = shift_down = np.array([[0.0]])
            # atoms, MO - sign of s-type Gaussian functions centered on the nucleus
            orbital_sign_up = orbital_sign_down = np.array([[1]])
            # atoms, MO
            rc_up = rc_down = np.array([[0.4375]])
            # atoms, MO, alpha index
            alpha_up = alpha_down = np.array([[
                [0.29141713, -2.0, 0.25262478, -0.098352818, 0.11124336],
            ]])
        elif self.neu == 2 and self.ned == 2:
            is_pseudoatom = np.zeros(shape=(1, ), dtype=np.bool_)
            wfn_0_up = wfn_0_down = np.array([[-3.447246814709, -0.628316785317]])
            shift_up = shift_down = np.array([[0.0, 0.0]])
            orbital_sign_up = orbital_sign_down = np.array([[-1, -1]])
            rc_up = rc_down = np.array([[0.1205, 0.1180]])
            alpha_up = alpha_down = np.array([[
                [ 1.24736449, -4.0,  0.49675975, -0.30582868,  1.0897532],
                [-0.45510824, -4.0, -0.73882727, -0.89716308, -5.8491770]
            ]])
        elif self.neu == 5 and self.ned == 2:
            is_pseudoatom = np.zeros(shape=(1, ), dtype=np.bool_)
            wfn_0_up = np.array([[6.069114031640, -1.397116693472, 0.0, 0.0, 0.0]])
            wfn_0_down = np.array([[6.095832387803, 1.268342737910]])
            shift_up = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            shift_down = np.array([[0.0, 0.0]])
            orbital_sign_up = np.array([[1, -1, 0, 0, 0]])
            orbital_sign_down = np.array([[1, 1]])
            rc_up = np.array([[0.0670, 0.0695, 0.0, 0.0, 0.0]])
            rc_down = np.array([[0.0675, 0.0680]])
            alpha_up = np.array([[
                [1.81320188, -7.0,  0.66956651, 0.60574099E+01, -0.42786390E+02],
                [0.34503578, -7.0, -0.34059064E+01, -0.10410228E+02, -0.22372391E+02],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
            alpha_down = np.array([[
                [1.81733596, -7.0, 0.72913009, 0.19258618E+01, -0.12077748E+02],
                [0.24741402, -7.0, -0.36101513E+01, -0.11720244E+02, -0.17700238E+02],
            ]])
        elif self.neu == 5 and self.ned == 5:
            is_pseudoatom = np.zeros(shape=(1, ), dtype=np.bool_)
            wfn_0_up = wfn_0_down = np.array([[10.523069754656, 2.470734575103, 0.0, 0.0, 0.0]])
            shift_up = shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            orbital_sign_up = orbital_sign_down = np.array([[1, 1, 0, 0, 0]])
            rc_up = rc_down = np.array([[0.0455, 0.0460, 0.0, 0.0, 0.0]])
            alpha_up = alpha_down = np.array([[
                [2.36314075, -10.0,  0.81732253,  0.15573932E+02, -0.15756663E+03],
                [0.91422900, -10.0, -0.84570201E+01, -0.26889022E+02, -0.17583628E+03],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
        elif self.neu == 9 and self.ned == 9:
            is_pseudoatom = np.zeros(shape=(1, ), dtype=np.bool_)
            wfn_0_up = wfn_0_down = np.array([[20.515046538335, 5.824658914949, 0.0, 0.0, 0.0, -1.820248905891, 0.0, 0.0, 0.0]])
            shift_up = shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            orbital_sign_up = orbital_sign_down = np.array([[1, 1, 0, 0, 0, -1, 0, 0, 0]])
            rc_up = rc_down = np.array([[0.0205, 0.0200, 0, 0, 0, 0.0205, 0, 0, 0]])
            alpha_up = alpha_down = np.array([[
                [3.02622267, -18.0,  0.22734669E+01,  0.79076581E+02, -0.15595740E+04],
                [1.76719238, -18.0, -0.30835348E+02, -0.23112278E+03, -0.45351148E+03],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.60405204, -18.0, -0.35203155E+02, -0.13904842E+03, -0.35690426E+04],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
        elif self.neu == 18 and self.ned == 18:
            is_pseudoatom = np.zeros(shape=(1, ), dtype=np.bool_)
            wfn_0_up = wfn_0_down = np.array(([
                [43.608490133788, -13.720841107516, 0.0, 0.0, 0.0, -5.505781654931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.751185788791, 0.0, 0.0, 0.0],
            ]))
            shift_up = shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            orbital_sign_up = orbital_sign_down = np.array([[1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]])
            rc_up = rc_down = np.array([[0.0045, 0.0045, 0, 0, 0, 0.0045, 0, 0, 0, 0, 0, 0, 0, 0, 0.0045, 0, 0, 0]])
            alpha_up = alpha_down = np.array([[
                [3.77764947, -36.0,  0.22235586E+02, -0.56621947E+04, 0.62983424E+06],
                [2.62138667, -36.0, -0.12558804E+03, -0.72801257E+04, 0.58905979E+06],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.70814456, -36.0, -0.14280857E+03, -0.80481344E+04, 0.63438487E+06],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.56410983, -36.0, -0.14519895E+03, -0.85628812E+04, 0.69239963E+06],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
        elif self.neu == 12 and self.ned == 12:
            is_pseudoatom = np.zeros(shape=(3, ), dtype=np.bool_)
            wfn_0_up = np.array(([
                [-5.245016636407, -0.025034008898,  0.019182670511, -0.839192164211,  0.229570396176, -0.697628545957, 0.0, -0.140965538444, -0.015299796091, 0.0, -0.084998032927,  0.220208807573],
                [-0.024547538656,  5.241296804923, -0.002693454373, -0.611438043012, -0.806215116184,  0.550648084416, 0.0, -0.250758940038, -0.185619271170, 0.0,  0.007450966720, -0.023495021763],
                [-0.018654332490, -0.002776929419, -5.248498638985, -0.386055559344,  0.686627203383,  0.707083323432, 0.0, -0.029625096851,  0.443458560481, 0.0, -0.034753046153, -0.008117407260],
            ]))
            wfn_0_down = np.array(([
                [-5.245016636416, -0.025034009046,  0.019182670402,  0.839192164264, -0.229570396203, -0.697628545936, 0.0, -0.140965538413, -0.015299796160, 0.0,  0.084998032932, -0.220208807501],
                [-0.018654332375, -0.002776929447, -5.248498638992,  0.386055559309, -0.686627203339,  0.707083323455, 0.0, -0.029625097018,  0.443458560519, 0.0,  0.034753046180,  0.008117407241],
                [-0.024547538802,  5.241296804930, -0.002693454404,  0.611438042982,  0.806215116191,  0.550648084418, 0.0, -0.250758940010, -0.185619271253, 0.0, -0.007450966721,  0.023495021734],
            ]))
            shift_up = shift_down = np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])
            orbital_sign_up = np.array([
                [-1, -1,  1, -1,  1, -1, 0, -1, -1, 0, -1,  1],
                [-1,  1, -1, -1, -1,  1, 0, -1, -1, 0,  1, -1],
                [-1, -1, -1, -1,  1,  1, 0, -1,  1, 0, -1, -1],
            ])
            orbital_sign_down = np.array([
                [-1, -1,  1,  1, -1, -1, 0, -1, -1, 0,  1, -1],
                [-1, -1, -1,  1, -1,  1, 0, -1,  1, 0,  1,  1],
                [-1,  1, -1,  1,  1,  1, 0, -1, -1, 0, -1,  1],
            ])
            rc_up = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
            ])
            rc_down = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
            ])
            alpha_up = np.array([
                [
                    [ 1.66696112, -0.80000242E+01,  0.72538040E+00,  0.74822749E+01, -0.59832829E+02],
                    [-3.67934712, -0.80068146E+01,  0.52306712E+00,  0.74024477E+01, -0.66331792E+02],
                    [-3.94191081, -0.79787241E+01,  0.77594866E+00,  0.17932088E+01, -0.10979109E+02],
                    [-0.14952920, -0.78746605E+01, -0.43071992E+01, -0.96038217E+01, -0.73806352E+02],
                    [-1.46038810, -0.79908365E+01, -0.50007568E+01, -0.10260692E+02, -0.95143069E+02],
                    [-0.36138067, -0.80924033E+01, -0.55877946E+01, -0.12746613E+02, -0.98421943E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.97822491, -0.82393021E+01, -0.65029776E+01, -0.21458170E+02, -0.62457982E+02],
                    [-4.22520946, -0.84474893E+01, -0.73574279E+01, -0.66355424E+01, -0.22665330E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-2.50170056, -0.83886665E+01, -0.73121006E+01, -0.21788297E+02, -0.94723331E+02],
                    [-1.55705345, -0.84492012E+01, -0.78325729E+01, -0.13722665E+02, -0.18661446E+03],
                ],
                [
                    [-3.69822013, -0.80095431E+01,  0.11302149E+01, -0.43409054E+01,  0.46442078E+02],
                    [ 1.66630435, -0.80000141E+01,  0.72320378E+00,  0.81261796E+01, -0.65047801E+02],
                    [-5.91498406, -0.80763347E+01,  0.90077681E+00,  0.31742107E+00,  0.35324292E+01],
                    [-0.46879152, -0.78900614E+01, -0.41907688E+01, -0.88090450E+01, -0.77431698E+02],
                    [-0.20295616, -0.79750959E+01, -0.44477740E+01, -0.11660536E+02, -0.61243691E+02],
                    [-0.60267378, -0.81319664E+01, -0.57277864E+01, -0.14396561E+02, -0.91691179E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.41424481, -0.83393407E+01, -0.69933925E+01, -0.15868367E+02, -0.13371167E+03],
                    [-1.69923582, -0.82051284E+01, -0.64601471E+01, -0.99697087E+01, -0.15524536E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-4.64757331, -0.62408167E+01,  0.11841009E+01, -0.30996021E+01,  0.22886716E+02],
                    [-3.64299934, -0.72053551E+01, -0.11465606E+01, -0.73818077E+01,  0.16054618E+00],
                ],
                [
                    [-3.97257763, -0.80089446E+01,  0.11382835E+01, -0.33320223E+01,  0.38514125E+02],
                    [-5.86803911, -0.79164537E+01,  0.92088515E+00, -0.96065381E+00,  0.16259254E+02],
                    [1.66760198,  -0.79999530E+01,  0.72544044E+00,  0.70833021E+01, -0.56621874E+02],
                    [-0.92086514, -0.78245771E+01, -0.39180783E+01, -0.73789685E+01, -0.76429773E+02],
                    [-0.35500196, -0.79059822E+01, -0.43591899E+01, -0.11018127E+02, -0.64511456E+02],
                    [-0.34258772, -0.80478675E+01, -0.55020049E+01, -0.69743091E+01, -0.14002539E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.61268484, -0.88953684E+01, -0.10239355E+02, -0.14393680E+02, -0.29851351E+03],
                    [-0.82873883, -0.82083731E+01, -0.64401279E+01, -0.99864145E+01, -0.15474953E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.42483045, -0.86471163E+01, -0.90649452E+01, -0.17907642E+02, -0.21389596E+03],
                    [-4.40492810, -0.52578816E+01,  0.37442246E+01,  0.60259540E+01,  0.24340638E+01],
                ],
            ])
            alpha_down = np.array([
                [
                    [ 1.66696112, -0.80000242E+01,  0.72538040E+00,  0.74822749E+01, -0.59832829E+02],
                    [-3.67934711, -0.80068146E+01,  0.52306718E+00,  0.74024468E+01, -0.66331787E+02],
                    [-3.94191082, -0.79787241E+01,  0.77594861E+00,  0.17932097E+01, -0.10979114E+02],
                    [-0.14952920, -0.78746605E+01, -0.43071992E+01, -0.96038217E+01, -0.73806352E+02],
                    [-1.46038810, -0.79908365E+01, -0.50007568E+01, -0.10260692E+02, -0.95143069E+02],
                    [-0.36138067, -0.80924033E+01, -0.55877946E+01, -0.12746613E+02, -0.98421943E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.97822491, -0.82393021E+01, -0.65029776E+01, -0.21458170E+02, -0.62457982E+02],
                    [-4.22520946, -0.84474893E+01, -0.73574279E+01, -0.66355420E+01, -0.22665330E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-2.50170056, -0.83886665E+01, -0.73121006E+01, -0.21788297E+02, -0.94723331E+02],
                    [-1.55705345, -0.84492012E+01, -0.78325729E+01, -0.13722665E+02, -0.18661446E+03],
                ],
                [
                    [-3.97257764, -0.80089446E+01,  0.11382835E+01, -0.33320231E+01,  0.38514130E+02],
                    [-5.86803910, -0.79164537E+01,  0.92088495E+00, -0.96065034E+00,  0.16259234E+02],
                    [ 1.66760198, -0.79999530E+01,  0.72544044E+00,  0.70833021E+01, -0.56621874E+02],
                    [-0.92086514, -0.78245771E+01, -0.39180783E+01, -0.73789685E+01, -0.76429773E+02],
                    [-0.35500196, -0.79059822E+01, -0.43591899E+01, -0.11018127E+02, -0.64511456E+02],
                    [-0.34258772, -0.80478675E+01, -0.55020049E+01, -0.69743091E+01, -0.14002539E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.61268483, -0.88953684E+01, -0.10239355E+02, -0.14393679E+02, -0.29851351E+03],
                    [-0.82873883, -0.82083731E+01, -0.64401279E+01, -0.99864145E+01, -0.15474953E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.42483045, -0.86471163E+01, -0.90649452E+01, -0.17907642E+02, -0.21389596E+03],
                    [-4.40492810, -0.52578816E+01,  0.37442246E+01,  0.60259539E+01,  0.24340639E+01],
                ],
                [
                    [-3.69822013, -0.80095431E+01,  0.11302149E+01, -0.43409047E+01,  0.46442075E+02],
                    [ 1.66630435, -0.80000141E+01,  0.72320378E+00,  0.81261796E+01, -0.65047801E+02],
                    [-5.91498405, -0.80763347E+01,  0.90077692E+00,  0.31741986E+00,  0.35324359E+01],
                    [-0.46879152, -0.78900614E+01, -0.41907688E+01, -0.88090450E+01, -0.77431698E+02],
                    [-0.20295616, -0.79750959E+01, -0.44477740E+01, -0.11660536E+02, -0.61243691E+02],
                    [-0.60267378, -0.81319664E+01, -0.57277864E+01, -0.14396561E+02, -0.91691179E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.41424481, -0.83393407E+01, -0.69933925E+01, -0.15868367E+02, -0.13371167E+03],
                    [-1.69923582, -0.82051284E+01, -0.64601471E+01, -0.99697088E+01, -0.15524535E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-4.64757331, -0.62408167E+01,  0.11841009E+01, -0.30996020E+01,  0.22886715E+02],
                    [-3.64299934, -0.72053551E+01, -0.11465606E+01, -0.73818076E+01,  0.16054597E+00],
                ]
            ])
        # atoms, MO - cusp correction radius
        rc = np.concatenate((rc_up, rc_down), axis=1)
        # atoms, MO - shift chosen so that phi − shift is of one sign within rc
        shift = np.concatenate((shift_up, shift_down), axis=1)
        # atoms, MO - sign of s-type Gaussian functions centered on the nucleus
        orbital_sign = np.concatenate((orbital_sign_up, orbital_sign_down), axis=1)
        # atoms, MO, alpha index
        alpha = np.concatenate((alpha_up, alpha_down), axis=1)
        alpha = np.moveaxis(alpha, -1, 0)
        # because different normalization
        alpha[0] += np.where(alpha[0], np.log(self.norm / self.casino_norm), 0)
        alpha = np.ascontiguousarray(alpha)
        return Cusp(
            self.neu, self.ned, self.neu, self.ned, rc, shift, orbital_sign, alpha,
            self.mo, self.first_shells, self.shell_moments, self.primitives, self.coefficients, self.exponents,
            is_pseudoatom,
        )
        # atoms, MO - Optimum corrected s orbital at nucleus
        # phi_0 = np.concatenate((phi_0_up, phi_0_down), axis=1)
        np.concatenate((wfn_0_up, wfn_0_down), axis=1)


if __name__ == '__main__':
    """
    """

    for mol in ('He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3'):
        path = f'../tests/gwfn/{mol}/HF/cc-pVQZ/CBCS/Slater/'

        config = CasinoConfig(path)
        config.read()

        cusp = CuspFactory(
            config.input.neu, config.input.ned, config.input.cusp_threshold, config.wfn.mo_up, config.wfn.mo_down,
            config.mdet.permutation_up, config.mdet.permutation_down,
            config.wfn.first_shells, config.wfn.shell_moments, config.wfn.primitives,
            config.wfn.coefficients, config.wfn.exponents,
            config.wfn.atom_positions, config.wfn.atom_charges, config.wfn.unrestricted, config.wfn.is_pseudoatom,
        ).create(casino_rc=True, casino_phi_tilde_0=False)

        cusp_test = TestCuspFactory(
            config.input.neu, config.input.ned, config.wfn.mo_up, config.wfn.mo_down,
            config.mdet.permutation_up, config.mdet.permutation_down,
            config.wfn.first_shells, config.wfn.shell_moments, config.wfn.primitives,
            config.wfn.coefficients, config.wfn.exponents
        ).create()

        print(
            f'{mol}:',
            np.allclose(cusp.orbital_sign,  cusp_test.orbital_sign),
            np.allclose(cusp.shift, cusp_test.shift),
            np.allclose(cusp.rc, cusp_test.rc),
            np.allclose(cusp.alpha, cusp_test.alpha, atol=0.001),
        )
