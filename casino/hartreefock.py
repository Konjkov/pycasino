import math

import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method


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
@overload_method(HartreeFock_class_t, 'fact2')
def HartreeFock_fact2(self, n):
    """n!!."""

    def impl(self, n) -> float:
        res = 1
        for i in range(n, 0, -2):
            res *= i
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'binomial_coefficient')
def HartreeFock_binomial_coefficient(self, n, k):
    """Binomial coefficient."""

    def impl(self, n, k) -> float:
        return math.gamma(n + 1) / math.gamma(k + 1) / math.gamma(n - k + 1)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'binomial_prefactor')
def HartreeFock_binomial_prefactor(self, k, l1, l2, PA, PB):
    """The integral prefactor containing the binomial coefficients from Augspurger and Dykstra."""

    def impl(self, k, l1, l2, PA, PB) -> float:
        res = 0
        for q in range(-min(k, 2 * l1 - k), min(k, 2 * l2 - k) + 1, 2):
            res += (
                self.binomial_coefficient(l1, (k + q) // 2)
                * self.binomial_coefficient(l2, (k - q) // 2)
                * PA ** (l1 - (k + q) // 2)
                * PB ** (l2 - (k - q) // 2)
            )
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'S_angular')
def HartreeFock_S_angular(self, l1, l2, PA, PB, gamma):
    """Angular part of overlap integral."""

    def impl(self, l1, l2, PA, PB, gamma) -> float:
        res = 0
        for i in range((l1 + l2) // 2 + 1):
            res += self.binomial_prefactor(2 * i, l1, l2, PA, PB) * self.fact2(2 * i - 1) / (2 * gamma) ** i
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
                                S_radial = coeff1 * coeff2 * np.exp(-alpha1 * alpha2 * AB / (alpha1 + alpha2)) * (np.pi / gamma) ** 1.5
                                for m1 in range(2 * l1 + 1):
                                    for m2 in range(2 * l2 + 1):
                                        if l1 + l2 <= 1:
                                            s_matrix[ao1 + m1, ao2 + m2] += (
                                                S_radial
                                                * self.S_angular(l1, l2, PA[0], PB[0], gamma)
                                                * self.S_angular(l1, l2, PA[1], PB[1], gamma)
                                                * self.S_angular(l1, l2, PA[2], PB[2], gamma)
                                            )
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
        res = np.zeros(shape=(self.nbasis_functions, self.nbasis_functions))
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(HartreeFock_class_t, 'V')
def HartreeFock_V(self):
    """Electron-nuclear attraction."""

    def impl(self) -> np.ndarray:
        res = np.zeros(shape=(self.nbasis_functions, self.nbasis_functions))
        return res

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
