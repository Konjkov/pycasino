import math

import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino.overload import boys, comb, fact2


@structref.register
class HartreeFock_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


HartreeFock_t = HartreeFock_class_t(
    [
        ('neu', nb.int64),
        ('ned', nb.int64),
        ('atom_charges', nb.float64[::1]),
        ('atom_positions', nb.float64[:, ::1]),
        ('nbasis_functions', nb.int64),
        ('first_shells', nb.int64[::1]),
        ('orbital_types', nb.int64[::1]),
        ('shell_moments', nb.int64[::1]),
        ('slater_orders', nb.int64[::1]),
        ('primitives', nb.int64[::1]),
        ('coefficients', nb.float64[::1]),
        ('exponents', nb.float64[::1]),
        ('mo_up', nb.float64[:, ::1]),
        ('mo_down', nb.float64[:, ::1]),
        ('norm', nb.float64),
    ]
)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'binomial_prefactor')
def HartreeFock_binomial_prefactor(self, k, l1, l2, PA, PB):
    """The integral prefactor containing the binomial coefficients from Augspurger and Dykstra."""

    def impl(self, k, l1, l2, PA, PB) -> float:
        res = 0
        for q in range(-min(k, 2 * l2 - k), min(k, 2 * l1 - k) + 1, 2):
            i = (k + q) // 2
            j = (k - q) // 2
            res += comb(l1, i) * comb(l2, j) * PA ** (l1 - i) * PB ** (l2 - j)
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'overlap')
def HartreeFock_overlap(self, lmn1, lmn2, PA, PB, gamma):
    """Cartesian part of overlap integral."""

    def impl(self, lmn1, lmn2, PA, PB, gamma) -> float:
        overlap = np.zeros(shape=(3,))
        for l in range(0, lmn1[0] + lmn2[0] + 1, 2):
            overlap[0] += self.binomial_prefactor(l, lmn1[0], lmn2[0], PA[0], PB[0]) * fact2(l - 1) / (2 * gamma) ** (l / 2)
        for m in range(0, lmn1[1] + lmn2[1] + 1, 2):
            overlap[1] += self.binomial_prefactor(m, lmn1[1], lmn2[1], PA[1], PB[1]) * fact2(m - 1) / (2 * gamma) ** (m / 2)
        for n in range(0, lmn1[2] + lmn2[2] + 1, 2):
            overlap[2] += self.binomial_prefactor(n, lmn1[2], lmn2[2], PA[2], PB[2]) * fact2(n - 1) / (2 * gamma) ** (n / 2)
        return overlap

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'harmonics')
def HartreeFock_harmonics(self, l, m):
    """Angular harmonics."""

    def impl(self, l, m) -> list:
        if l == 0:
            return [(1.0, (0.0, 0.0, 0.0))]
        elif l == 1:
            if m == 0:
                return [(1.0, (1.0, 0.0, 0.0))]
            elif m == 1:
                return [(1.0, (0.0, 1.0, 0.0))]
            elif m == 2:
                return [(1.0, (0.0, 0.0, 1.0))]
        elif l == 2:
            if m == 0:
                # (3*z2 - r2) / 2
                return [(-0.5, (2.0, 0.0, 0.0)), (-0.5, (0.0, 2.0, 0.0)), (1.0, (0.0, 0.0, 2.0))]
            elif m == 1:
                # sqrt(3) * x*z
                return [(math.sqrt(3.0), (1.0, 0.0, 1.0))]
            elif m == 2:
                # sqrt(3) * y*z
                return [(math.sqrt(3.0), (0.0, 1.0, 1.0))]
            elif m == 3:
                # sqrt(3) * (x2 - y2) / 2
                return [(math.sqrt(3.0) / 2, (2.0, 0.0, 0.0)), (-math.sqrt(3.0) / 2, (0.0, 2.0, 0.0))]
            elif m == 4:
                # sqrt(3)*x*y
                return [(math.sqrt(3.0), (1.0, 1.0, 0.0))]
        elif l == 3:
            if m == 0:
                # z * (5*z2 - 3*r2) / 2
                return [(-1.5, (2.0, 0.0, 1.0)), (-1.5, (0.0, 2.0, 1.0)), (1.0, (0.0, 0.0, 3.0))]
            elif m == 1:
                # sqrt(6) * x * (5*z2 - r2) / 4
                return [(-math.sqrt(6.0) / 4, (3.0, 0.0, 0.0)), (-math.sqrt(6.0) / 4, (1.0, 2.0, 0.0)), (math.sqrt(6.0), (1.0, 0.0, 2.0))]
            elif m == 2:
                # sqrt(6) * y * (5*z2 - r2) / 4
                return [(-math.sqrt(6.0) / 4, (2.0, 1.0, 0.0)), (-math.sqrt(6.0) / 4, (0.0, 3.0, 0.0)), (math.sqrt(6.0), (0.0, 1.0, 2.0))]
            elif m == 3:
                # sqrt(15) * z * (x2 - y2) / 2
                return [(math.sqrt(15.0) / 2, (2.0, 0.0, 1.0)), (-math.sqrt(15.0) / 2, (0.0, 2.0, 1.0))]
            elif m == 4:
                # sqrt(15) * x * y * z
                return [(math.sqrt(15), (1.0, 1.0, 1.0))]
            elif m == 5:
                # sqrt(10) * x * (x2 - 3.0 * y2) / 4
                return [(math.sqrt(10) / 4, (3.0, 0.0, 0.0)), (-3 * math.sqrt(10) / 4, (1.0, 2.0, 0.0))]
            elif m == 6:
                # sqrt(10) * y * (3.0 * x2 - y2) / 4
                return [(3 * math.sqrt(10) / 4, (2.0, 1.0, 0.0)), (-math.sqrt(10) / 4, (0.0, 3.0, 0.0))]
        elif l == 4:
            if m == 0:
                # (35.0 * z**4 - 30.0 * z2 * r2 + 3.0 * r2**2) / 8.0 = (3x4 + 3y4 + 8z4 - 6x2y2 + 6x2z2 + 6y2z2 - 30x2z2 - 30y2z2) / 8
                return [
                    (3 / 8, (4.0, 0.0, 0.0)),
                    (3 / 8, (0.0, 4.0, 0.0)),
                    (1.0, (0.0, 0.0, 4.0)),
                    (3 / 4, (2.0, 2.0, 0.0)),
                    (-3.0, (2.0, 0.0, 2.0)),
                    (-3.0, (0.0, 2.0, 2.0)),
                ]
            elif m == 1:
                # sqrt(10) * x*z * (7.0 * z2 - 3.0 * r2) / 4
                return [(-3 * math.sqrt(10) / 4, (3.0, 0.0, 1.0)), (-3 * math.sqrt(10) / 4, (1.0, 2.0, 1.0)), (math.sqrt(10), (1.0, 0.0, 3.0))]
            elif m == 2:
                # sqrt(10) * y*z * (7.0 * z2 - 3.0 * r2) / 4
                return [(-3 * math.sqrt(10) / 4, (2.0, 1.0, 1.0)), (-3 * math.sqrt(10) / 4, (0.0, 3.0, 1.0)), (math.sqrt(10), (0.0, 1.0, 3.0))]
            elif m == 3:
                # sqrt(5) * (x2 - y2) * (7.0 * z2 - r2) / 4
                return [
                    (-math.sqrt(5) / 4, (4.0, 0.0, 0.0)),
                    (math.sqrt(5) / 4, (0.0, 4.0, 0.0)),
                    (3 * math.sqrt(5) / 2, (2.0, 0.0, 2.0)),
                    (-3 * math.sqrt(5) / 2, (0.0, 2.0, 2.0)),
                ]
            elif m == 4:
                # sqrt(5) * x*y * (7.0 * z2 - r2) / 2
                return [(-math.sqrt(5) / 2, (3.0, 1.0, 0.0)), (-math.sqrt(5) / 2, (1.0, 3.0, 0.0)), (3 * math.sqrt(5), (1.0, 1.0, 2.0))]
            elif m == 5:
                # sqrt(70) * x*z * (x2 - 3.0 * y2) / 4
                return [(math.sqrt(70) / 4, (3.0, 0.0, 1.0)), (-3 * math.sqrt(70) / 4, (1.0, 2.0, 1.0))]
            elif m == 6:
                # sqrt(70) * y*z * (3.0 * x2 - y2) / 4
                return [(3 * math.sqrt(70) / 4, (2.0, 1.0, 1.0)), (-math.sqrt(70) / 4, (0.0, 3.0, 1.0))]
            elif m == 7:
                # sqrt(35) * (x2**2 - 6.0 * x2 * y2 + y2**2) / 8
                return [(math.sqrt(35) / 8, (4.0, 0.0, 0.0)), (-3 * math.sqrt(35) / 4, (2.0, 2.0, 0.0)), (math.sqrt(35) / 8, (0.0, 4.0, 0.0))]
            elif m == 8:
                # sqrt(35) * x*y * (x2 - y2) / 2
                return [(math.sqrt(35) / 2, (3.0, 1.0, 0.0)), (-math.sqrt(35) / 2, (1.0, 3.0, 0.0))]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'S_angular')
def HartreeFock_S_angular(self, l1, l2, m1, m2, PA, PB, gamma):
    """Angular part of overlap integral."""

    def impl(self, l1, l2, m1, m2, PA, PB, gamma) -> float:
        res = 0
        for c1, lmn1 in self.harmonics(l1, m1):
            for c2, lmn2 in self.harmonics(l2, m2):
                res += c1 * c2 * np.prod(self.overlap(lmn1, lmn2, PA, PB, gamma))
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'T_angular')
def HartreeFock_T_angular(self, l1, l2, m1, m2, PA, PB, alpha1, gamma):
    """Angular part of electron kinetic energy."""

    def impl(self, l1, l2, m1, m2, PA, PB, alpha1, gamma) -> float:
        """Cartesian part of electron kinetic energy."""
        res = 0
        for c1, lmn1 in self.harmonics(l1, m1):
            lmn1_plus = [x + 2 for x in lmn1]
            lmn1_minus = [x - 2 for x in lmn1]
            for c2, lmn2 in self.harmonics(l2, m2):
                overlap_plus = self.overlap(lmn1_plus, lmn2, PA, PB, gamma)
                overlap = self.overlap(lmn1, lmn2, PA, PB, gamma)
                overlap_minus = self.overlap(lmn1_minus, lmn2, PA, PB, gamma)
                plus_term = (
                    4
                    * alpha1**2
                    * (
                        overlap_plus[0] * overlap[1] * overlap[2]
                        + overlap[0] * overlap_plus[1] * overlap[2]
                        + overlap[0] * overlap[1] * overlap_plus[2]
                    )
                )
                zero_term = 2 * alpha1 * (2 * sum(lmn1) + 3) * overlap[0] * overlap[1] * overlap[2]
                minus_term = (
                    lmn1[0] * (lmn1[0] - 1) * overlap_minus[0] * overlap[1] * overlap[2]
                    + lmn1[1] * (lmn1[1] - 1) * overlap[0] * overlap_minus[1] * overlap[2]
                    + lmn1[2] * (lmn1[2] - 1) * overlap[0] * overlap[1] * overlap_minus[2]
                )
                res += c1 * c2 * (plus_term - zero_term + minus_term)
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'ERI')
def HartreeFock_ERI(self):
    """Electrons repulsion integrals."""

    def impl(self) -> np.ndarray:
        res = np.zeros(shape=(self.nbasis_functions, self.nbasis_functions, self.nbasis_functions, self.nbasis_functions))
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'S')
def HartreeFock_S(self):
    """Electron overlap integrals."""

    def impl(self) -> np.ndarray:
        s_matrix = np.zeros(shape=(self.nbasis_functions, self.nbasis_functions))
        p1 = ao1 = 0
        for atom1 in range(self.atom_charges.size):
            A = self.atom_positions[atom1]
            for nshell1 in range(self.first_shells[atom1] - 1, self.first_shells[atom1 + 1] - 1):
                l1 = self.shell_moments[nshell1]
                p2 = ao2 = 0
                for atom2 in range(self.atom_charges.size):
                    B = self.atom_positions[atom2]
                    AB = np.linalg.norm(A - B)
                    for nshell2 in range(self.first_shells[atom2] - 1, self.first_shells[atom2 + 1] - 1):
                        l2 = self.shell_moments[nshell2]
                        for primitive1 in range(self.primitives[nshell1]):
                            alpha1 = self.exponents[p1 + primitive1]
                            coeff1 = self.coefficients[p1 + primitive1]
                            for primitive2 in range(self.primitives[nshell2]):
                                alpha2 = self.exponents[p2 + primitive2]
                                coeff2 = self.coefficients[p2 + primitive2]
                                gamma = alpha1 + alpha2
                                P = (alpha1 * A + alpha2 * B) / gamma
                                PA = P - A
                                PB = P - B
                                S_radial = coeff1 * coeff2 * math.exp(-alpha1 * alpha2 * AB / (alpha1 + alpha2)) * (np.pi / gamma) ** 1.5
                                for m1 in range(2 * l1 + 1):
                                    for m2 in range(2 * l2 + 1):
                                        s_matrix[ao1 + m1, ao2 + m2] += S_radial * self.S_angular(l1, l2, m1, m2, PA, PB, gamma)
                        ao2 += 2 * l2 + 1
                        p2 += self.primitives[nshell2]
                ao1 += 2 * l1 + 1
                p1 += self.primitives[nshell1]
        return s_matrix

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'T')
def HartreeFock_T(self):
    """Electrons kinetic energy."""

    def impl(self) -> np.ndarray:
        t_matrix = np.zeros(shape=(self.nbasis_functions, self.nbasis_functions))
        p1 = ao1 = 0
        for atom1 in range(self.atom_charges.size):
            A = self.atom_positions[atom1]
            for nshell1 in range(self.first_shells[atom1] - 1, self.first_shells[atom1 + 1] - 1):
                l1 = self.shell_moments[nshell1]
                p2 = ao2 = 0
                for atom2 in range(self.atom_charges.size):
                    B = self.atom_positions[atom2]
                    AB = np.linalg.norm(A - B)
                    for nshell2 in range(self.first_shells[atom2] - 1, self.first_shells[atom2 + 1] - 1):
                        l2 = self.shell_moments[nshell2]
                        for primitive1 in range(self.primitives[nshell1]):
                            alpha1 = self.exponents[p1 + primitive1]
                            coeff1 = self.coefficients[p1 + primitive1]
                            for primitive2 in range(self.primitives[nshell2]):
                                alpha2 = self.exponents[p2 + primitive2]
                                coeff2 = self.coefficients[p2 + primitive2]
                                gamma = alpha1 + alpha2
                                P = (alpha1 * A + alpha2 * B) / gamma
                                PA = P - A
                                PB = P - B
                                S_radial = coeff1 * coeff2 * math.exp(-alpha1 * alpha2 * AB / (alpha1 + alpha2)) * (np.pi / gamma) ** 1.5
                                for m1 in range(2 * l1 + 1):
                                    for m2 in range(2 * l2 + 1):
                                        t_matrix[ao1 + m1, ao2 + m2] += S_radial * self.T_angular(l1, l2, m1, m2, PA, PB, alpha1, gamma)
                        ao2 += 2 * l2 + 1
                        p2 += self.primitives[nshell2]
                ao1 += 2 * l1 + 1
                p1 += self.primitives[nshell1]
        return -t_matrix / 2

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'V')
def HartreeFock_V(self):
    """Electron-nuclear attraction."""

    def impl(self) -> np.ndarray:
        v_matrix = np.zeros(shape=(self.nbasis_functions, self.nbasis_functions))
        p1 = ao1 = 0
        for atom1 in range(self.atom_charges.size):
            A = self.atom_positions[atom1]
            for nshell1 in range(self.first_shells[atom1] - 1, self.first_shells[atom1 + 1] - 1):
                l1 = self.shell_moments[nshell1]
                p2 = ao2 = 0
                for atom2 in range(self.atom_charges.size):
                    B = self.atom_positions[atom2]
                    AB = np.linalg.norm(A - B)
                    for nshell2 in range(self.first_shells[atom2] - 1, self.first_shells[atom2 + 1] - 1):
                        l2 = self.shell_moments[nshell2]
                        for primitive1 in range(self.primitives[nshell1]):
                            alpha1 = self.exponents[p1 + primitive1]
                            coeff1 = self.coefficients[p1 + primitive1]
                            for primitive2 in range(self.primitives[nshell2]):
                                alpha2 = self.exponents[p2 + primitive2]
                                coeff2 = self.coefficients[p2 + primitive2]
                                gamma = alpha1 + alpha2
                                P = (alpha1 * A + alpha2 * B) / gamma
                                S_radial = coeff1 * coeff2 * math.exp(-alpha1 * alpha2 * AB / (alpha1 + alpha2)) * (math.pi / gamma) ** 1.5
                                for atom3 in range(self.atom_charges.size):
                                    C = self.atom_positions[atom3]
                                    PC = np.linalg.norm(P - C)
                                    for m1 in range(2 * l1 + 1):
                                        for m2 in range(2 * l2 + 1):
                                            v_matrix[ao1 + m1, ao2 + m2] += S_radial * boys(0, gamma * PC**2) / (math.pi / gamma) ** 0.5
                        ao2 += 2 * l2 + 1
                        p2 += self.primitives[nshell2]
                ao1 += 2 * l1 + 1
                p1 += self.primitives[nshell1]
        return v_matrix

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'H')
def HartreeFock_H(self):
    """Hamiltonian."""

    def impl(self) -> np.ndarray:
        res = np.zeros(shape=(self.nbasis_functions, self.nbasis_functions))
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'Fock')
def HartreeFock_Fock(self):
    """Fock matrix."""

    def impl(self) -> np.ndarray:
        res = np.zeros(shape=(self.nbasis_functions, self.nbasis_functions))
        return res

    return impl


class HartreeFock(structref.StructRefProxy):
    def __new__(cls, config):
        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(
            neu,
            ned,
            atom_charges,
            atom_positions,
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
        ):
            """Slater multideterminant wavefunction.
            :param neu: number of up electrons
            :param ned: number of down electrons
            :param atom_charges:
            :param atom_positions:
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
            """
            self = structref.new(HartreeFock_t)
            self.neu = neu
            self.ned = ned
            self.atom_charges = atom_charges
            self.atom_positions = atom_positions
            self.nbasis_functions = nbasis_functions
            self.first_shells = first_shells
            self.orbital_types = orbital_types
            self.shell_moments = shell_moments
            self.slater_orders = slater_orders
            self.primitives = primitives
            self.coefficients = coefficients
            self.exponents = exponents
            self.mo_up = mo_up
            self.mo_down = mo_down
            self.norm = np.exp(-(math.lgamma(neu + 1) + math.lgamma(ned + 1)) / (neu + ned) / 2)
            return self

        return init(
            config.input.neu,
            config.input.ned,
            config.wfn.atom_charges,
            config.wfn.atom_positions,
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
        )

    @nb.njit(nogil=True, parallel=False, cache=True)
    def S(self):
        return self.S()

    @nb.njit(nogil=True, parallel=False, cache=True)
    def T(self):
        return self.T()

    @nb.njit(nogil=True, parallel=False, cache=True)
    def V(self):
        return self.V()

    @nb.njit(nogil=True, parallel=False, cache=True)
    def H(self):
        return self.H()


structref.define_boxing(HartreeFock_class_t, HartreeFock)
