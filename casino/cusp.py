import logging
import math

import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method
from numpy.polynomial.polynomial import polyder, polyval
from scipy.optimize import minimize_scalar

from casino.abstract import AbstractCusp
from casino.harmonics import Harmonics

logger = logging.getLogger(__name__)


@structref.register
class Cusp_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


Cusp_t = Cusp_class_t(
    [
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
    ]
)


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
@overload_method(Cusp_class_t, 'exp')
def cusp_exp(self, atom, orbital, r):
    """Exponent part"""

    def impl(self, atom, orbital, r) -> float:
        return self.orbital_sign[atom, orbital] * np.exp(
            # FIXME: use polyval(r, self.alpha[:, atom, i])
            self.alpha[0, atom, orbital] +
            self.alpha[1, atom, orbital] * r +
            self.alpha[2, atom, orbital] * r**2 +
            self.alpha[3, atom, orbital] * r**3 +
            self.alpha[4, atom, orbital] * r**4
        )  # fmt: skip

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
           ) / r  # fmt: skip

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
           ) / r ** 2  # fmt: skip

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
           ) / r ** 3  # fmt: skip

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

        return value[: self.orbitals_up, : self.neu], value[self.orbitals_up :, self.neu :]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Cusp_class_t, 'value_1e')
def cusp_value_1e(self, n_vector: np.ndarray, e: int):
    """Cusp correction for the orbitals of a single electron, the column of the slater
    matrix that an electron-by-electron move changes.
    :param n_vector: electron-nuclei vectors of that electron shape = (natom, 3)
    :param e: electron
    """

    def impl(self, n_vector: np.ndarray, e: int) -> np.ndarray:
        if e < self.neu:
            first, orbitals = 0, self.orbitals_up
        else:
            first, orbitals = self.orbitals_up, self.orbitals_down
        value = np.zeros(shape=orbitals)
        for i in range(first, first + orbitals):
            p = ao = 0
            for atom in range(n_vector.shape[0]):
                if not self.is_pseudoatom[atom]:
                    r = np.sqrt(n_vector[atom] @ n_vector[atom])
                    if r < self.rc[atom, i]:
                        value[i - first] = self.exp(atom, i, r) + self.shift[atom, i]
                    s_part = 0.0
                    for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                        l = self.shell_moments[nshell]
                        if r < self.rc[atom, i] and self.shell_moments[nshell] == 0:
                            for primitive in range(self.primitives[nshell]):
                                s_part += self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r * r) * self.mo[i, ao]
                        p += self.primitives[nshell]
                        ao += 2 * l + 1
                    # subtract uncusped s-part
                    value[i - first] -= s_part * self.norm
        return value

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

        return gradient[: self.orbitals_up, : self.neu], gradient[self.orbitals_up :, self.neu :]

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
                            laplacian[i, j] = (2 * self.diff_1(atom, i, r) + self.diff_2(atom, i, r) * r**2) * self.exp(atom, i, r)

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
                            laplacian[i, j] = (2 * self.diff_1(atom, i, r) + self.diff_2(atom, i, r) * r**2) * self.exp(atom, i, r)

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

        return laplacian[: self.orbitals_up, : self.neu], laplacian[self.orbitals_up :, self.neu :]

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
                            hessian[i, j, :, :] = (self.diff_2(atom, i, r) * ri_rj + self.diff_1(atom, i, r) * (np.eye(3) - ri_rj / r**2)) * self.exp(
                                atom, i, r
                            )
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
                            hessian[i, j, :, :] = (self.diff_2(atom, i, r) * ri_rj + self.diff_1(atom, i, r) * (np.eye(3) - ri_rj / r**2)) * self.exp(
                                atom, i, r
                            )

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

        return hessian[: self.orbitals_up, : self.neu], hessian[self.orbitals_up :, self.neu :]

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
                            np.expand_dims(np.eye(3), 2) * n_vectors[atom, j]
                            + np.expand_dims(np.eye(3), 1) * np.expand_dims(n_vectors[atom, j], 1)
                            + np.expand_dims(np.eye(3), 0) * np.expand_dims(np.expand_dims(n_vectors[atom, j], 1), 2)
                        )
                        if r < self.rc[atom, i]:
                            tressian[i, j, :, :, :] = (
                                self.diff_3(atom, i, r) * ri_rj_rk
                                + self.diff_2(atom, i, r) * (kronecker - 3 * ri_rj_rk / r**2)
                                + self.diff_1(atom, i, r) * (3 * ri_rj_rk / r**2 - kronecker) / r**2
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
                                    s_part += (ri_rj_rk * c + kronecker) * c**2 * exponent * self.mo[i, ao]
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
                            np.expand_dims(np.eye(3), 2) * n_vectors[atom, j]
                            + np.expand_dims(np.eye(3), 1) * np.expand_dims(n_vectors[atom, j], 1)
                            + np.expand_dims(np.eye(3), 0) * np.expand_dims(np.expand_dims(n_vectors[atom, j], 1), 2)
                        )
                        if r < self.rc[atom, i]:
                            tressian[i, j, :, :, :] = (
                                self.diff_3(atom, i, r) * ri_rj_rk
                                + self.diff_2(atom, i, r) * (kronecker - 3 * ri_rj_rk / r**2)
                                + self.diff_1(atom, i, r) * (3 * ri_rj_rk / r**2 - kronecker) / r**2
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
                                    s_part += (ri_rj_rk * c + kronecker) * c**2 * exponent * self.mo[i, ao]
                            p += self.primitives[nshell]
                            ao += 2 * l + 1
                        # subtract uncusped s-part
                        tressian[i, j] -= s_part * self.norm

        return tressian[: self.orbitals_up, : self.neu], tressian[self.orbitals_up :, self.neu :]

    return impl


structref.define_boxing(Cusp_class_t, Cusp)


@nb.njit(nogil=True, parallel=False, cache=True)
def cusp_init(
    neu,
    ned,
    orbitals_up,
    orbitals_down,
    rc,
    shift,
    orbital_sign,
    alpha,
    mo,
    first_shells,
    shell_moments,
    primitives,
    coefficients,
    exponents,
    is_pseudoatom,
):
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
    # radial grid on which nodes and the cusp radius are located
    orbgrid_spacing = 0.0005
    # radii closer than that to a node are ignored when fitting
    nodewidth = 0.02

    def __init__(self, config):
        self.neu = config.input.neu
        self.ned = config.input.ned
        self.orbitals_up = np.max(config.mdet.permutation_up) + 1 if self.neu else 0
        self.orbitals_down = np.max(config.mdet.permutation_down) + 1 if self.ned else 0
        self.norm = np.exp(-(math.lgamma(self.neu + 1) + math.lgamma(self.ned + 1)) / (self.neu + self.ned) / 2)
        self.mo = np.concatenate((config.wfn.mo_up[: self.orbitals_up], config.wfn.mo_down[: self.orbitals_down]))
        self.first_shells = config.wfn.first_shells
        self.shell_moments = config.wfn.shell_moments
        self.primitives = config.wfn.primitives
        self.coefficients = config.wfn.coefficients
        self.exponents = config.wfn.exponents
        self.atom_positions = config.wfn.atom_positions
        self.atom_charges = config.wfn.atom_charges
        self.cusp_threshold = config.input.cusp_threshold
        self.cusp_control = config.input.cusp_control
        self.harmonics = Harmonics(np.max(config.wfn.shell_moments))
        self.s_shells = self.s_shells_data()
        self.nearest_ion = self.nearest_ion_data()
        self.phi_0, _, _ = self.phi(np.zeros(shape=(self.atom_positions.shape[0], self.mo.shape[0])))
        self.orb_mask = np.abs(self.phi_0) > self.cusp_threshold
        self.beta = np.array([3.25819, -15.0126, 33.7308, -42.8705, 31.2276, -12.1316, 1.94692])
        # atoms, MO - Value of corrected orbital at nucleus
        self.phi_tilde_0 = np.copy(self.phi_0)
        # atoms, MO - cusp correction radius
        self.rc = np.zeros((self.atom_positions.shape[0], self.mo.shape[0]))
        # atoms, MO - sign of s-type Gaussian functions centered on the nucleus
        self.orbital_sign = self.phi_sign()
        # atoms, MO - shift chosen so that phi − shift is of one sign within rc
        self.shift = np.zeros((self.atom_positions.shape[0], self.mo.shape[0]))
        # atoms, MO - maximum deviation of the effective one-electron local energy from the ideal one
        self.energy_diff_max = np.zeros((self.atom_positions.shape[0], self.mo.shape[0]))
        # atoms, MO - contribution from Gaussians on other nuclei
        self.eta = self.eta_data()
        self.unrestricted = config.wfn.unrestricted
        logger.info(
            ' Gaussian cusp correction\n'
            ' ========================\n'
            ' Activated.\n'
        )  # fmt: skip
        self.is_pseudoatom = config.wfn.is_pseudoatom

    def phi(self, rc):
        """Wfn of single electron of s-orbitals on each atom"""
        orbital = np.zeros((self.atom_positions.shape[0], self.mo.shape[0], self.mo.shape[1]))
        orbital_derivative = np.zeros((self.atom_positions.shape[0], self.mo.shape[0], self.mo.shape[1]))
        orbital_second_derivative = np.zeros((self.atom_positions.shape[0], self.mo.shape[0], self.mo.shape[1]))
        # a tight gaussian evaluated away from its centre underflows to zero, which is the value
        # meant: the njit code drops those primitives outright on the gautol threshold
        with np.errstate(under='ignore'):
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
                np.sum(orbital_second_derivative * self.mo, axis=2) * self.norm,
            )

    def eta_data(self):
        """Contribution from Gaussians on other nuclei"""
        orbital = np.zeros(shape=(self.atom_positions.shape[0], self.mo.shape[0], self.mo.shape[1]))
        with np.errstate(under='ignore'):
            for atom in range(self.atom_positions.shape[0]):
                for orb in range(self.mo.shape[0]):
                    p = ao = 0
                    for orb_atom in range(self.atom_positions.shape[0]):
                        x, y, z = self.atom_positions[atom] - self.atom_positions[orb_atom]
                        r2 = x * x + y * y + z * z
                        angular = self.harmonics.get_value(x, y, z)
                        # angular = value_angular_part(x, y, z)
                        for nshell in range(self.first_shells[orb_atom] - 1, self.first_shells[orb_atom + 1] - 1):
                            l = self.shell_moments[nshell]
                            radial = 0.0
                            if atom != orb_atom:
                                for primitive in range(self.primitives[nshell]):
                                    radial += self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r2)
                                for m in range(2 * l + 1):
                                    orbital[atom, orb, ao + m] += angular[l * l + m] * radial
                            ao += 2 * l + 1
                            p += self.primitives[nshell]
            return np.sum(orbital * self.mo, axis=2) * self.norm

    def s_shells_data(self):
        """Exponents, coefficients and AO index of the s-type Gaussians centered on each atom"""
        result = []
        p = ao = 0
        for atom in range(self.atom_positions.shape[0]):
            exponents, coefficients, ao_index = [], [], []
            for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                l = self.shell_moments[nshell]
                if l == 0:
                    for primitive in range(self.primitives[nshell]):
                        exponents.append(self.exponents[p + primitive])
                        coefficients.append(self.coefficients[p + primitive])
                        ao_index.append(ao)
                ao += 2 * l + 1
                p += self.primitives[nshell]
            result.append((np.array(exponents), np.array(coefficients), np.array(ao_index, dtype=int)))
        return result

    def nearest_ion_data(self):
        """Nearest neighbour distance for each atom"""
        result = np.full(self.atom_positions.shape[0], np.inf)
        for atom in range(self.atom_positions.shape[0]):
            for other in range(self.atom_positions.shape[0]):
                if atom != other:
                    r = np.linalg.norm(self.atom_positions[atom] - self.atom_positions[other])
                    result[atom] = min(result[atom], r)
        return result

    def s_part(self, atom, r):
        """Wfn of single electron of s-orbitals on atom over a radial grid,
        its first and second radial derivatives, shape (grid, MO)
        """
        exponents, coefficients, ao = self.s_shells[atom]
        mo = self.mo[:, ao].T * self.norm
        with np.errstate(under='ignore'):
            exponent = coefficients * np.exp(-exponents * r[:, np.newaxis] ** 2)
            return (
                exponent @ mo,
                (-2 * exponents * r[:, np.newaxis] * exponent) @ mo,
                (2 * exponents * (2 * exponents * r[:, np.newaxis] ** 2 - 1) * exponent) @ mo,
            )

    def phi_sign(self):
        """Calculate phi sign."""
        return np.where(self.orb_mask, np.sign(self.phi_0), 0).astype(np.int_)

    def cusp_solve(self, rc, shift, phi_tilde_0, phi_rc, phi_diff_rc, phi_diff_2_rc, z_eff):
        """Solve for the polynomial coefficients satisfying the five constraints:
        continuity of p, p` and p`` at rc, the cusp condition at r=0 and a fixed phi_tilde(0).
        shift variable chosen so that (phi−shift) is of one sign within rc.
        """
        R = phi_tilde_0 - shift
        X1 = np.log(np.abs(phi_rc - shift))  # (9)
        X2 = phi_diff_rc / (phi_rc - shift)  # (10)
        X3 = phi_diff_2_rc / (phi_rc - shift)  # (11)
        X4 = -z_eff * phi_tilde_0 / R  # (12)
        X5 = np.log(np.abs(R))  # (13)
        return np.array([  # (14)
            X5,
            X4,
            6 * X1 / rc**2 - 3 * X2 / rc + X3 / 2 - 3 * X4 / rc - 6 * X5 / rc**2 - X2**2 / 2,
            -8 * X1 / rc**3 + 5 * X2 / rc**2 - X3 / rc + 3 * X4 / rc**2 + 8 * X5 / rc**3 + X2**2 / rc,
            3 * X1 / rc**4 - 2 * X2 / rc**3 + X3 / 2 / rc**2 - X4 / rc**3 - 3 * X5 / rc**4 - X2**2 / 2 / rc**2,
        ])  # fmt: skip

    def z_eff_data(self, phi_tilde_0):
        """Effective nuclear charge, equation (16)"""
        return self.atom_charges[:, np.newaxis] * (1 + self.eta / phi_tilde_0)

    def alpha_data(self, phi_tilde_0):
        """Calculate phi coefficients for every orbital and nucleus.
        eta is a contribution from Gaussians on other nuclei.
        """
        with np.errstate(divide='ignore', invalid='ignore', under='ignore'):
            phi_rc, phi_diff_rc, phi_diff_2_rc = self.phi(self.rc)
            alpha = self.cusp_solve(self.rc, self.shift, phi_tilde_0, phi_rc, phi_diff_rc, phi_diff_2_rc, self.z_eff_data(phi_tilde_0))
        # remove NaN from orbitals without s-part
        return np.nan_to_num(alpha, posinf=0, neginf=0)

    def phi_energy(self, phi, phi_diff_1, phi_diff_2, r, z_eff):
        """Effective one-electron local energy for gaussian s-part orbital, equation (15)"""
        return -(phi_diff_1 / r + phi_diff_2 / 2) / phi - z_eff / r

    def phi_tilde_energy(self, r, alpha, shift, orbital_sign, z_eff):
        """Effective one-electron local energy for corrected orbital, equation (15)"""
        p_diff_1 = polyval(r, polyder(alpha))
        p_diff_2 = polyval(r, polyder(alpha, 2))
        R = orbital_sign * np.exp(polyval(r, alpha))
        return -R / (R + shift) * (p_diff_1 / r + (p_diff_2 + p_diff_1**2) / 2) - z_eff / r

    def ideal_energy(self, r, rc, el_rc, z, z_eff):
        """Ideal energy, equation (17). It is fixed to el_rc at rc, hydrogen is a special case."""
        if z == 1:
            return np.full_like(r, el_rc)
        return (polyval(r, self.beta) * r**2 - polyval(rc, self.beta) * rc**2) * z_eff**2 + el_rc

    def find_nodes(self, atom, orb, r, phi):
        """Nodes of the s-part of an orbital, outermost first"""
        nodes = []
        for i in np.nonzero(phi[:-1] * phi[1:] < 0)[0][::-1]:
            lo, hi, phi_lo = r[i], r[i + 1], phi[i]
            while hi - lo > 1e-6:
                mid = (lo + hi) / 2
                phi_mid = self.s_part(atom, np.array([mid]))[0][0, orb]
                if phi_lo * phi_mid <= 0:
                    hi = mid
                else:
                    lo, phi_lo = mid, phi_mid
            nodes.append((lo + hi) / 2)
        return nodes

    def orb_solve(self, atom, orb, rc, r, good, z):
        """Optimum phi_tilde(0) for a given cusp radius and the corresponding maximum
        deviation of the effective one-electron local energy from the ideal one.
        The criterion for choosing phi_tilde(0) is that the local energy be as smooth
        as possible, so its maximum deviation within [0, rc] is minimized.
        """
        phi_rc, phi_diff_rc, phi_diff_2_rc = (x[0, orb] for x in self.s_part(atom, np.array([rc])))
        shift = self.shift[atom, orb]

        def energy_diff_max(phi_tilde_0):
            z_eff = z * (1 + self.eta[atom, orb] / phi_tilde_0)
            alpha = self.cusp_solve(rc, shift, phi_tilde_0, phi_rc, phi_diff_rc, phi_diff_2_rc, z_eff)
            el_rc = self.phi_energy(phi_rc, phi_diff_rc, phi_diff_2_rc, rc, z_eff)
            energy = self.phi_tilde_energy(r, alpha, shift, self.orbital_sign[atom, orb], z_eff)
            return np.max(np.abs(energy - self.ideal_energy(r, rc, el_rc, z, z_eff))[good])

        phi_0 = self.phi_0[atom, orb]
        res = minimize_scalar(energy_diff_max, bracket=(phi_0, phi_0 * 1.1), method='golden', options={'xtol': 1e-5})
        return res.x, res.fun

    def optimize_rc(self):
        """Cusp radius, shift and optimum phi_tilde(0) for every orbital and nucleus.
        The cusp radius is set to the largest radius below rcmax at which the deviation of the
        effective one-electron local energy of the uncorrected orbital from the ideal curve
        exceeds z²/cusp_control, then semi-optimized by varying it by ±20%.
        Radii closer than nodewidth to a node of the s-part are excluded from the fit, since
        the local energy diverges there.
        """
        spacing = self.orbgrid_spacing
        node_width = round(self.nodewidth / spacing)
        for atom in range(self.atom_positions.shape[0]):
            z = self.atom_charges[atom]
            el_check = z**2 / self.cusp_control
            rcmax_max = min(1.0, 0.9 * self.nearest_ion[atom])
            grid_size = int(rcmax_max / spacing)
            r = np.arange(1, grid_size + 1) * spacing
            phi, phi_diff_1, phi_diff_2 = self.s_part(atom, r)
            with np.errstate(divide='ignore', invalid='ignore', under='ignore'):
                # orbitals without s-part on this atom are skipped below
                el_gauss = self.phi_energy(phi, phi_diff_1, phi_diff_2, r[:, np.newaxis], z)
            for orb in range(self.mo.shape[0]):
                if not self.orb_mask[atom, orb]:
                    continue
                nodes = self.find_nodes(atom, orb, r, phi[:, orb])
                # a reasonable upper limit for the cusp radius
                n_rcmax = int(rcmax_max / z / spacing) - 1
                deviation = np.abs(el_gauss[:, orb] - self.ideal_energy(r, r[n_rcmax], el_gauss[n_rcmax, orb], z, z))
                exceeded = np.nonzero(deviation[: n_rcmax + 1] > el_check)[0]
                n_rcusp = exceeded[-1] if exceeded.size else 0
                # blank out grid points which lie close to a node
                good = np.ones(grid_size, dtype=bool)
                node_lo, node_hi = [], []
                for node in nodes:
                    lo = max(int(node / spacing) - 1 - node_width, 0)
                    hi = min(int(node / spacing) - 1 + node_width, grid_size - 1)
                    good[lo : hi + 1] = False
                    node_lo.append(lo)
                    node_hi.append(hi)
                # the innermost node is bounded by the nucleus
                node_hi.append(0)
                # if the cusp radius lies inside a nodal region, move it towards the nucleus
                cusp_inside_node = not good[n_rcusp]
                if cusp_inside_node:
                    n_rcusp = np.nonzero(good[: n_rcusp + 1])[0][-1]
                # the outermost node inside rcmax
                outer_node = next((j for j, lo in enumerate(node_lo) if r[lo] < r[n_rcmax]), None)
                # the algorithm cannot pass through a node since the local energy diverges there,
                # so if the outer node is a relatively long way out, look for a better rcusp inside it
                if outer_node is not None and (nodes[outer_node] > 0.25 or cusp_inside_node):
                    el_ideal = self.ideal_energy(r, r[n_rcmax], el_gauss[n_rcmax, orb], z, z)
                    deviation_old = abs(el_gauss[node_lo[outer_node], orb] - el_ideal[node_lo[outer_node]])
                    for j in range(outer_node, len(nodes)):
                        for i in range(node_lo[j] - 1, node_hi[j + 1] - 1, -1):
                            if abs(el_gauss[i, orb] - el_ideal[i]) > deviation_old:
                                n_rcmax = i
                                break
                            deviation_old = abs(el_gauss[i, orb] - el_ideal[i])
                        # go in as far as possible until the local energy difference exceeds the threshold
                        el_ideal = self.ideal_energy(r, r[n_rcmax], el_gauss[n_rcmax, orb], z, z)
                        for i in range(n_rcmax, node_hi[j + 1] - 1, -1):
                            if abs(el_gauss[i, orb] - el_ideal[i]) > el_check:
                                n_rcusp = i
                                break
                # if there is a node inside rc we cannot replace phi by a positive definite
                # exponential, so define a constant shift to temporarily add to phi_tilde
                if any(node < r[n_rcusp] for node in nodes):
                    if self.orbital_sign[atom, orb] > 0:
                        self.shift[atom, orb] = 2 * phi[:, orb].min()
                    else:
                        self.shift[atom, orb] = 2 * phi[:, orb].max()
                # semi-optimize rcusp by varying it by ±20% and picking the rcusp that
                # minimizes the maximum deviation from the ideal local energy
                step = int(r[n_rcusp] * 0.05 / spacing)
                energy_diff_max = np.inf
                for k in range(-4, 5) if step else [0]:
                    n = n_rcusp + k * step
                    if not 0 <= n < grid_size:
                        continue
                    if not good[n]:
                        if k > 0:
                            break
                        continue
                    phi_tilde_0, emax = self.orb_solve(atom, orb, r[n], r[: n + 1], good[: n + 1], z)
                    if emax < energy_diff_max:
                        energy_diff_max = emax
                        self.rc[atom, orb] = r[n]
                        self.phi_tilde_0[atom, orb] = phi_tilde_0
                        self.energy_diff_max[atom, orb] = emax

    def create(self):
        """Create cusp class."""
        self.optimize_rc()
        alpha = self.alpha_data(self.phi_tilde_0)
        return Cusp(
            self.neu,
            self.ned,
            self.orbitals_up,
            self.orbitals_down,
            self.rc,
            self.shift,
            self.orbital_sign,
            alpha,
            self.mo,
            self.first_shells,
            self.shell_moments,
            self.primitives,
            self.coefficients,
            self.exponents,
            self.is_pseudoatom,
        )

    def cusp_info(self):
        """If cusp correction is set to T for an all-electron Gaussian basis set calculation,
        then casino will alter the orbitals inside a small radius around each nucleus in such a way
        that they obey the electron–nucleus cusp condition. If cusp info is set to T then information
        about precisely how this is done will be printed to the out file. Be aware that in large systems
        this may produce a lot of output.
        :return:
        """
        logger.info(' Verbose print out flagged (turn off with cusp_info : F)\n')
        # zero phi_tilde_0 for orbitals without s-part, such elements are masked out below
        with np.errstate(divide='ignore', invalid='ignore'):
            z_eff_data = self.z_eff_data(self.phi_tilde_0)
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
                    logger.info(f' Orbital {orb + 1 if i == 0 else orb + 1 - self.orbitals_up} at position of ion {atom + 1}')
                    if self.orb_mask[atom][orb]:
                        sign = 'positive' if self.orbital_sign[atom][orb] > 0 else 'negative'
                        z_eff = z_eff_data[atom][orb]
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
        logger.info(f' Maximum deviation from ideal (averaged over orbitals) : {np.mean(self.energy_diff_max[nonzero_index]):16.12f}.\n')


class CasinoCuspFactory:
    def __init__(self, config):
        self.neu = config.input.neu
        self.ned = config.input.ned
        self.orbitals_up = np.max(config.mdet.permutation_up) + 1
        self.orbitals_down = np.max(config.mdet.permutation_down) + 1
        self.norm = np.exp(-(math.lgamma(self.neu + 1) + math.lgamma(self.ned + 1)) / (self.neu + self.ned) / 2)
        self.casino_norm = np.exp(-(math.lgamma(self.neu + 1) + math.lgamma(self.neu + 1)) / (self.neu + self.neu) / 2)
        self.mo = np.concatenate((config.wfn.mo_up[: self.orbitals_up], config.wfn.mo_down[: self.orbitals_down]))
        self.first_shells = config.wfn.first_shells
        self.shell_moments = config.wfn.shell_moments
        self.primitives = config.wfn.primitives
        self.coefficients = config.wfn.coefficients
        self.exponents = config.wfn.exponents

    def create(self):
        if self.neu == 1 and self.ned == 1:
            is_pseudoatom = np.zeros(shape=(1,), dtype=bool)
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
            ]])  # fmt: skip
        elif self.neu == 2 and self.ned == 2:
            is_pseudoatom = np.zeros(shape=(1,), dtype=bool)
            wfn_0_up = wfn_0_down = np.array([[-3.447246814709, -0.628316785317]])
            shift_up = shift_down = np.array([[0.0, 0.0]])
            orbital_sign_up = orbital_sign_down = np.array([[-1, -1]])
            rc_up = rc_down = np.array([[0.1205, 0.1180]])
            alpha_up = alpha_down = np.array([[
                [ 1.24736449, -4.0,  0.49675975, -0.30582868,  1.0897532],
                [-0.45510824, -4.0, -0.73882727, -0.89716308, -5.8491770]
            ]])  # fmt: skip
        elif self.neu == 5 and self.ned == 2:
            is_pseudoatom = np.zeros(shape=(1,), dtype=bool)
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
            ]])  # fmt: skip
            alpha_down = np.array([[
                [1.81733596, -7.0, 0.72913009, 0.19258618E+01, -0.12077748E+02],
                [0.24741402, -7.0, -0.36101513E+01, -0.11720244E+02, -0.17700238E+02],
            ]])  # fmt: skip
        elif self.neu == 5 and self.ned == 5:
            is_pseudoatom = np.zeros(shape=(1,), dtype=bool)
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
            ]])  # fmt: skip
        elif self.neu == 9 and self.ned == 9:
            is_pseudoatom = np.zeros(shape=(1,), dtype=bool)
            wfn_0_up = wfn_0_down = np.array([[20.515046538335, 5.824658914949, 0.0, 0.0, 0.0, -1.820248905891, 0.0, 0.0, 0.0]])  # fmt: skip
            shift_up = shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # fmt: skip
            orbital_sign_up = orbital_sign_down = np.array([[1, 1, 0, 0, 0, -1, 0, 0, 0]])  # fmt: skip
            rc_up = rc_down = np.array([[0.0205, 0.0200, 0, 0, 0, 0.0205, 0, 0, 0]])  # fmt: skip
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
            ]])  # fmt: skip
        elif self.neu == 18 and self.ned == 18:
            is_pseudoatom = np.zeros(shape=(1,), dtype=bool)
            wfn_0_up = wfn_0_down = np.array(([
                [43.608490133788, -13.720841107516, 0.0, 0.0, 0.0, -5.505781654931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.751185788791, 0.0, 0.0, 0.0],
            ]))  # fmt: skip
            shift_up = shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # fmt: skip
            orbital_sign_up = orbital_sign_down = np.array([[1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]])  # fmt: skip
            rc_up = rc_down = np.array([[0.0045, 0.0045, 0, 0, 0, 0.0045, 0, 0, 0, 0, 0, 0, 0, 0, 0.0045, 0, 0, 0]])  # fmt: skip
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
            ]])  # fmt: skip
        elif self.neu == 12 and self.ned == 12:
            is_pseudoatom = np.zeros(shape=(3,), dtype=bool)
            wfn_0_up = np.array(([
                [-5.245016636407, -0.025034008898,  0.019182670511, -0.839192164211,  0.229570396176, -0.697628545957, 0.0, -0.140965538444, -0.015299796091, 0.0, -0.084998032927,  0.220208807573],
                [-0.024547538656,  5.241296804923, -0.002693454373, -0.611438043012, -0.806215116184,  0.550648084416, 0.0, -0.250758940038, -0.185619271170, 0.0,  0.007450966720, -0.023495021763],
                [-0.018654332490, -0.002776929419, -5.248498638985, -0.386055559344,  0.686627203383,  0.707083323432, 0.0, -0.029625096851,  0.443458560481, 0.0, -0.034753046153, -0.008117407260],
            ]))  # fmt: skip
            wfn_0_down = np.array(([
                [-5.245016636416, -0.025034009046,  0.019182670402,  0.839192164264, -0.229570396203, -0.697628545936, 0.0, -0.140965538413, -0.015299796160, 0.0,  0.084998032932, -0.220208807501],
                [-0.018654332375, -0.002776929447, -5.248498638992,  0.386055559309, -0.686627203339,  0.707083323455, 0.0, -0.029625097018,  0.443458560519, 0.0,  0.034753046180,  0.008117407241],
                [-0.024547538802,  5.241296804930, -0.002693454404,  0.611438042982,  0.806215116191,  0.550648084418, 0.0, -0.250758940010, -0.185619271253, 0.0, -0.007450966721,  0.023495021734],
            ]))  # fmt: skip
            shift_up = shift_down = np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])  # fmt: skip
            orbital_sign_up = np.array([
                [-1, -1,  1, -1,  1, -1, 0, -1, -1, 0, -1,  1],
                [-1,  1, -1, -1, -1,  1, 0, -1, -1, 0,  1, -1],
                [-1, -1, -1, -1,  1,  1, 0, -1,  1, 0, -1, -1],
            ])  # fmt: skip
            orbital_sign_down = np.array([
                [-1, -1,  1,  1, -1, -1, 0, -1, -1, 0,  1, -1],
                [-1, -1, -1,  1, -1,  1, 0, -1,  1, 0,  1,  1],
                [-1,  1, -1,  1,  1,  1, 0, -1, -1, 0, -1,  1],
            ])  # fmt: skip
            rc_up = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
            ])  # fmt: skip
            rc_down = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
            ])  # fmt: skip
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
            ])  # fmt: skip
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
            ])  # fmt: skip
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
            self.neu,
            self.ned,
            self.neu,
            self.ned,
            rc,
            shift,
            orbital_sign,
            alpha,
            self.mo,
            self.first_shells,
            self.shell_moments,
            self.primitives,
            self.coefficients,
            self.exponents,
            is_pseudoatom,
        )
        # atoms, MO - Optimum corrected s orbital at nucleus
        # phi_0 = np.concatenate((phi_0_up, phi_0_down), axis=1)
        np.concatenate((wfn_0_up, wfn_0_down), axis=1)
