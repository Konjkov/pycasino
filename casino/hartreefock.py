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
        ('atom_charges', nb.float64[:]),
        ('atom_positions', nb.float64[:, :]),
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
        p_i = ao_i = 0
        for atom_i in range(self.atom_charges.size):
            A = self.atom_positions[atom_i]
            for nshell_i in range(self.first_shells[atom_i] - 1, self.first_shells[atom_i + 1] - 1):
                l_i = self.shell_moments[nshell_i]
                p_j = ao_j = 0
                for atom_j in range(self.atom_charges.size):
                    B = self.atom_positions[atom_j]
                    AB = np.linalg.norm(A - B)
                    for nshell_j in range(self.first_shells[atom_j] - 1, self.first_shells[atom_j + 1] - 1):
                        l_j = self.shell_moments[nshell_j]
                        S = 0
                        for primitive_i in range(self.primitives[nshell_i]):
                            alpha_i = self.exponents[p_i + primitive_i]
                            coeff_i = self.coefficients[p_i + primitive_i]
                            for primitive_j in range(self.primitives[nshell_j]):
                                alpha_j = self.exponents[p_j + primitive_j]
                                coeff_j = self.coefficients[p_j + primitive_j]
                                # P = (alpha_i * A + alpha_j * B) / (alpha_i + alpha_j)
                                # PA = P - A
                                # PB = P - B
                                S += coeff_i * coeff_j * np.exp(-alpha_i * alpha_j * AB / (alpha_i + alpha_j)) * (np.pi / (alpha_i + alpha_j)) ** 1.5
                        if l_i == l_j == 0:
                            s_matrix[ao_i, ao_j] = S
                        # for m_i in range(2 * l_i + 1):
                        #     for m_j in range(2 * l_j + 1):
                        #         s_matrix[ao_i + m_i, ao_j + m_j] = S
                        ao_j += 2 * l_j + 1
                        p_j += self.primitives[nshell_j]
                ao_i += 2 * l_i + 1
                p_i += self.primitives[nshell_i]
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
