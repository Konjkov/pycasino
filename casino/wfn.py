import numpy as np
import numba as nb
from numba.core import types
from numba.experimental import structref
from numba.core.extending import overload_method

from casino import delta
from casino.slater import Slater, Slater_t
from casino.jastrow import Jastrow
from casino.backflow import Backflow
from casino.overload import block_diag
from casino.ppotential import PPotential


@structref.register
class Wfn_class_t(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


Wfn_t = Wfn_class_t([
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('atom_positions', nb.float64[:, :]),
    ('atom_charges', nb.float64[:]),
    ('nuclear_repulsion', nb.float64),
    ('slater', Slater_t),
    ('jastrow', nb.optional(Jastrow.class_type.instance_type)),
    ('backflow', nb.optional(Backflow.class_type.instance_type)),
    ('ppotential', nb.optional(PPotential.class_type.instance_type)),
])


class Wfn(structref.StructRefProxy):

    def __new__(cls, neu, ned, atom_positions, atom_charges, slater, jastrow, backflow, ppotential):
        """Wave function in general form.
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param atom_positions: atomic positions
        :param atom_charges: atomic charges
        :param slater: instance of Slater class
        :param jastrow: instance of Jastrow class
        :param backflow: instance of Backflow class
        :param ppotential: instance of Pseudopotential class
        """
        return wfn_new(neu, ned, atom_positions, atom_charges, slater, jastrow, backflow, ppotential)

    def energy(self, r_e) -> float:
        """Local energy.
        :param r_e: electron coordinates - array(nelec, 3)
        """
        return energy(self, r_e)

    @property
    def nuclear_repulsion(self) -> float:
        """Value of n-n repulsion."""
        return nuclear_repulsion(self)


@nb.njit(nogil=True, parallel=False, cache=True)
def nuclear_repulsion(self) -> float:
    """Value of n-n repulsion."""
    return self.nuclear_repulsion


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, '_relative_coordinates')
def wfn__relative_coordinates(self, r_e):
    """Get relative electron coordinates
    :param r_e: electron positions
    :return: e-e vectors - array(nelec, nelec, 3), e-n vectors - array(natom, nelec, 3)
    """
    def impl(self, r_e):
        e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
        n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(self.atom_positions, 1)
        return e_vectors, n_vectors
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, '_get_nuclear_repulsion')
def wfn__get_nuclear_repulsion(self) -> float:
    """Value of n-n repulsion."""
    def impl(self) -> float:
        res = 0.0
        for atom1 in range(self.atom_positions.shape[0] - 1):
            for atom2 in range(atom1 + 1, self.atom_positions.shape[0]):
                res += self.atom_charges[atom1] * self.atom_charges[atom2] / np.linalg.norm(self.atom_positions[atom1] - self.atom_positions[atom2])
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'coulomb')
def wfn_coulomb(self, r_e) -> float:
    """Value of e-e and e-n coulomb interaction."""
    def impl(self, r_e) -> float:
        res = 0.0
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        # e-e coulomb interaction
        for e1 in range(e_vectors.shape[0] - 1):
            for e2 in range(e1 + 1, e_vectors.shape[1]):
                res += 1 / np.linalg.norm(e_vectors[e1, e2])
        # e-n coulomb interaction
        for atom in range(n_vectors.shape[0]):
            for e1 in range(n_vectors.shape[1]):
                res -= self.atom_charges[atom] / np.linalg.norm(n_vectors[atom, e1])
        # local channel pseudopotential
        if self.ppotential is not None:
            potential = self.ppotential.get_ppotential(n_vectors)
            for atom in range(n_vectors.shape[0]):
                if self.ppotential.is_pseudoatom[atom]:
                    for e1 in range(self.neu + self.ned):
                        res += potential[atom][e1, 2]
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'value')
def wfn_value(self, r_e) -> float:
    """Value of wave function.
    :param r_e: electron positions
    """
    def impl(self, r_e) -> float:
        res = 1
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None:
            res *= np.exp(self.jastrow.value(e_vectors, n_vectors))
        if self.backflow is not None:
            n_vectors = self.backflow.value(e_vectors, n_vectors) + n_vectors
        res *= self.slater.value(n_vectors)
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'drift_velocity')
def wfn_drift_velocity(self, r_e):
    """Drift velocity
    drift velocity = 1/2 * 'drift or quantum force'
    where D is diffusion constant = 1/2
    """
    def impl(self, n_vectors: np.ndarray):
        e_vectors, n_vectors = self._relative_coordinates(r_e)

        if self.backflow is not None:
            b_g, b_v = self.backflow.gradient(e_vectors, n_vectors)
            s_g = self.slater.gradient(b_v + n_vectors)
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
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 't_move')
def wfn_t_move(self, r_e):
    """T-move
    :param r_e: electron positions - array(nelec, 3)
    :param step_size: DMC step size
    :return: next electrons positions
    """
    def impl(self, n_vectors: np.ndarray):
        if self.ppotential is not None:
            moved = False
            next_r_e = r_e.copy()
            for e1 in range(self.neu + self.ned):
                t_prob = [1.0]
                t_grid = [next_r_e]
                value = self.value(next_r_e)
                e_vectors, n_vectors = self._relative_coordinates(next_r_e)
                grid = self.ppotential.integration_grid(n_vectors)
                potential = self.ppotential.get_ppotential(n_vectors)
                for atom in range(n_vectors.shape[0]):
                    if self.ppotential.is_pseudoatom[atom]:
                        if potential[atom][e1, 0] or potential[atom][e1, 1]:
                            for q in range(grid.shape[2]):
                                cos_theta = (grid[atom, e1, q] @ n_vectors[atom, e1]) / (n_vectors[atom, e1] @ n_vectors[atom, e1])
                                r_e_q = next_r_e.copy()
                                r_e_q[e1] = grid[atom, e1, q] + self.atom_positions[atom]
                                value_ratio = self.value(r_e_q) / value
                                weight = self.ppotential.weight[atom][q]
                                v = 0
                                for l in range(2):
                                    v += potential[atom][e1, l] * self.ppotential.legendre(l, cos_theta) * weight * value_ratio
                                # negative probability is not possible
                                if v < 0:
                                    t_prob.append(-step_size * v)
                                    t_grid.append(r_e_q)
                t_prob = np.array(t_prob)
                i = np.searchsorted(np.cumsum(t_prob / np.sum(t_prob)), np.random.random())
                if i > 0:
                    moved = True
                    next_r_e = t_grid[i]
            return moved, next_r_e
        return False, r_e
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'local_potential')
def wfn_local_potential(self, r_e) -> float:
    """Local potential.
    :param r_e: electron positions - array(nelec, 3)
    """
    def impl(self, r_e) -> float:
        return self.coulomb(r_e) + self.nuclear_repulsion
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'nonlocal_potential')
def wfn_nonlocal_potential(self, r_e) -> float:
    """Nonlocal (pseudopotential) energy Wφ/φ.
    :param r_e: electron positions - array(nelec, 3)
    """
    def impl(self, r_e) -> float:
        res = 0.0
        if self.ppotential is not None:
            value = self.value(r_e)
            e_vectors, n_vectors = self._relative_coordinates(r_e)
            grid = self.ppotential.integration_grid(n_vectors)
            potential = self.ppotential.get_ppotential(n_vectors)
            for atom in range(n_vectors.shape[0]):
                if self.ppotential.is_pseudoatom[atom]:
                    for e1 in range(self.neu + self.ned):
                        if potential[atom][e1, 0] or potential[atom][e1, 1]:
                            for q in range(grid.shape[2]):
                                cos_theta = (grid[atom, e1, q] @ n_vectors[atom, e1]) / (n_vectors[atom, e1] @ n_vectors[atom, e1])
                                r_e_q = r_e.copy()
                                r_e_q[e1] = grid[atom, e1, q] + self.atom_positions[atom]
                                value_ratio = self.value(r_e_q) / value
                                weight = self.ppotential.weight[atom][q]
                                for l in range(2):
                                    res += potential[atom][e1, l] * self.ppotential.legendre(l, cos_theta) * weight * value_ratio
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'kinetic_energy')
def wfn_kinetic_energy(self, r_e) -> float:
    """Kinetic energy.
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
    def impl(self, r_e) -> float:
        with_F_and_T = True
        e_vectors, n_vectors = self._relative_coordinates(r_e)

        if self.backflow is not None:
            b_l, b_g, b_v = self.backflow.laplacian(e_vectors, n_vectors)
            s_h, s_g = self.slater.hessian(b_v + n_vectors)
            s_l = np.sum(s_h * (b_g @ b_g.T)) + s_g @ b_l
            if self.jastrow is not None:
                j_g = self.jastrow.gradient(e_vectors, n_vectors)
                j_l = self.jastrow.laplacian(e_vectors, n_vectors)
                s_g = s_g @ b_g
                F = np.sum((s_g + j_g)**2) / 2
                T = (np.sum(s_g**2) - s_l - j_l) / 4
                return 2 * T - F
            else:
                return - s_l / 2
        else:
            s_l = self.slater.laplacian(n_vectors)
            if self.jastrow is not None:
                j_g = self.jastrow.gradient(e_vectors, n_vectors)
                j_l = self.jastrow.laplacian(e_vectors, n_vectors)
                s_g = self.slater.gradient(n_vectors)
                F = np.sum((s_g + j_g)**2) / 2
                T = (np.sum(s_g**2) - s_l - j_l) / 4
                return 2 * T - F
            elif with_F_and_T:
                s_g = self.slater.gradient(n_vectors)
                F = np.sum(s_g**2) / 2
                T = (np.sum(s_g**2) - s_l) / 4
                return 2 * T - F
            else:
                return - s_l / 2
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
def energy(self, r_e) -> float:
    """Local energy.
    :param r_e: electron coordinates - array(nelec, 3)
    """
    return self.kinetic_energy(r_e) + self.local_potential(r_e) + self.nonlocal_potential(r_e)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'energy')
def wfn_energy(self, r_e) -> float:
    """Local energy.
    :param r_e: electron coordinates - array(nelec, 3)
    """
    def impl(self, r_e) -> float:
        return self.kinetic_energy(r_e) + self.local_potential(r_e) + self.nonlocal_potential(r_e)
    return impl


# This associates the proxy with MyStruct_t for the given set of fields.
# Notice how we are not constraining the type of each field.
# Field types remain generic.
structref.define_proxy(Wfn, Wfn_class_t, ['neu', 'ned',
    'atom_positions', 'atom_charges', 'slater', 'jastrow', 'backflow', 'ppotential'])


@nb.njit(nogil=True, parallel=False, cache=True)
def wfn_new(neu, ned, atom_positions, atom_charges, slater, jastrow, backflow, ppotential):
    self = structref.new(Wfn_t)
    self.neu = neu
    self.ned = ned
    self.atom_positions = atom_positions
    self.atom_charges = atom_charges
    self.slater = slater
    self.jastrow = jastrow
    self.backflow = backflow
    self.ppotential = ppotential
    return self
