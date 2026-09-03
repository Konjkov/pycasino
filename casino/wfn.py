import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino.abstract import AbstractWfn
from casino.backflow import Backflow_t
from casino.geminal import Geminal_t
from casino.jastrow import Jastrow_t
from casino.overload import block_diag
from casino.ppotential import PPotential_t
from casino.slater import Slater_t


@structref.register
class Wfn_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


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
def wfn__get_nuclear_repulsion(self):
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
def wfn_coulomb(self, r_e):
    """Value of e-e, e-n and n-n coulomb interaction."""

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
        return res + self.nuclear_repulsion

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'value')
def wfn_value(self, r_e):
    """Value of wave function.
    :param r_e: electron positions
    """

    def impl(self, r_e) -> float:
        res = 1
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None:
            res *= np.exp(self.jastrow.value(e_vectors, n_vectors))
        if self.backflow is not None:
            n_vectors += self.backflow.value(e_vectors, n_vectors)
        if self.geminal is not None:
            res *= self.geminal.value(n_vectors)
        else:
            res *= self.slater.value(n_vectors)
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'log_value')
def wfn_log_value(self, r_e):
    """Logarithm of the absolute value of wave function, and its sign.
    The jastrow is a logarithm already, so nothing of it is ever exponentiated here.
    :param r_e: electron positions
    :return: log(abs(psi)), sign(psi)
    """

    def impl(self, r_e) -> tuple[float, float]:
        res = 0.0
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None:
            res += self.jastrow.value(e_vectors, n_vectors)
        if self.backflow is not None:
            n_vectors += self.backflow.value(e_vectors, n_vectors)
        if self.geminal is not None:
            log_value, sign = self.geminal.log_value(n_vectors)
        else:
            log_value, sign = self.slater.log_value(n_vectors)
        return res + log_value, sign

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'caches')
def wfn_caches(self, r_e):
    """The two caches a walker carries between single-electron moves: the slater state and the
    powers of the e-n distances the jastrow reads. Built for a whole configuration, and from
    there on updated one electron at a time.
    :param r_e: electron positions - array(nelec, 3)
    :return: slater state, powers of the e-n distances
    """

    def impl(self, r_e):
        n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(self.atom_positions, 1)
        state = self.slater.state(n_vectors)
        if self.jastrow is not None:
            n_powers = self.jastrow.en_powers(n_vectors)
        else:
            n_powers = np.zeros(shape=(1, 1, 1))
        return state, n_powers

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'value_ratio_1e')
def wfn_value_ratio_1e(self, state, n_powers, r_e, e, next_r_e_1e):
    """Ratio of the wave function at the two ends of a single-electron move, out of the caches a
    walker carries rather than out of the whole configuration: the slater state gives the ratio of
    the determinants as one dot product per determinant, and the jastrow only the terms the moved
    electron takes part in. Neither cache is touched, accept_1e commits the move into them.
    Only meaningful without backflow, which spreads a single-electron move over the quasi-particle
    coordinates of every electron, and without a geminal, which is not a slater determinant.
    :param state: slater state of the current configuration
    :param n_powers: powers of the e-n distances of the current configuration
    :param r_e: electron positions - array(nelec, 3)
    :param e: electron being moved
    :param next_r_e_1e: its proposed position - array(3)
    :return: ln|psi'/psi|, sign(psi'/psi), orbitals of e, Q of each determinant
    """

    def impl(self, state, n_powers, r_e, e, next_r_e_1e) -> tuple[float, float, np.ndarray, np.ndarray]:
        # the determinant needs the nuclear distances of the moved electron and of nobody else,
        # and that column is cheaper to build than to look up in an array of all of them
        n_vector = next_r_e_1e - self.atom_positions
        log_value, sign, orbitals, q = state.ratio_1e(n_vector, e)
        log_ratio = log_value - state.log_value
        if self.jastrow is not None:
            # the nuclear distances of the electrons that stayed put are common to both ends of
            # the proposal, so only the row of the moved electron is replaced between the two
            # evaluations, and put back as the proposal may yet be rejected
            row = n_powers[:, e].copy()
            jastrow_value = self.jastrow.value_1e(r_e[e] - r_e, n_powers, e)
            self.jastrow.update_en_powers_1e(n_powers, n_vector, e)
            log_ratio += self.jastrow.value_1e(next_r_e_1e - r_e, n_powers, e) - jastrow_value
            n_powers[:, e] = row
        return log_ratio, sign * state.sign, orbitals, q

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'accept_1e')
def wfn_accept_1e(self, state, n_powers, e, next_r_e_1e, orbitals, q):
    """Move electron e into the caches of the walker.
    :param state: slater state of the configuration the electron leaves
    :param n_powers: powers of the e-n distances of that configuration
    :param e: electron being moved
    :param next_r_e_1e: its accepted position - array(3)
    :param orbitals: its orbitals, as returned by value_ratio_1e
    :param q: determinant ratios, as returned by value_ratio_1e
    :return: whether the slater state was updated, a False asking the caller to rebuild it
    """

    def impl(self, state, n_powers, e, next_r_e_1e, orbitals, q) -> bool:
        if self.jastrow is not None:
            self.jastrow.update_en_powers_1e(n_powers, next_r_e_1e - self.atom_positions, e)
        return state.accept_1e(e, orbitals, q)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'drift_velocity')
def wfn_drift_velocity(self, r_e):
    """Drift velocity
    drift velocity = 1/2 * 'drift or quantum force'
    where D is diffusion constant = 1/2
    """

    def impl(self, r_e):
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.backflow is not None:
            b_g, b_v = self.backflow.gradient(e_vectors, n_vectors)
            if self.geminal is not None:
                s_g = self.geminal.gradient(b_v + n_vectors) @ b_g
            else:
                s_g = self.slater.gradient(b_v + n_vectors) @ b_g
        elif self.geminal is not None:
            s_g = self.geminal.gradient(n_vectors)
        else:
            s_g = self.slater.gradient(n_vectors)
        if self.jastrow is not None:
            return s_g + self.jastrow.gradient(e_vectors, n_vectors)
        else:
            return s_g

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'drift_velocity_1e')
def wfn_drift_velocity_1e(self, state, n_powers, r_e, e, r_e_1e, q):
    """Drift velocity of a single electron, out of the caches a walker carries. An
    electron-by-electron step only ever needs the drift of the electron it is about to move, at
    the two ends of the proposal, and both come out of the caches without the drift of everybody
    else. Backflow and geminals have no such shortcut and pay the whole vector to read one row.
    :param state: slater state of the current configuration
    :param n_powers: powers of the e-n distances of the current configuration
    :param r_e: electron positions of that configuration - array(nelec, 3)
    :param e: electron
    :param r_e_1e: the position of that electron the drift is taken at - array(3)
    :param q: determinant ratios there, as returned by value_ratio_1e, ones at r_e[e]
    :return: array(3)
    """

    def impl(self, state, n_powers, r_e, e, r_e_1e, q) -> np.ndarray:
        if self.backflow is None and self.geminal is None:
            n_vector = r_e_1e - self.atom_positions
            res = state.gradient_1e(n_vector, e, q)
            if self.jastrow is not None:
                # as in value_ratio_1e the row of the electron is replaced for the duration of
                # the evaluation and put back, so that the caches are left as they were
                row = n_powers[:, e].copy()
                self.jastrow.update_en_powers_1e(n_powers, n_vector, e)
                res = res + self.jastrow.gradient_1e(r_e_1e - r_e, n_vector, n_powers, e)
                n_powers[:, e] = row
        else:
            next_r_e = r_e.copy()
            next_r_e[e] = r_e_1e
            res = self.drift_velocity(next_r_e).reshape(self.neu + self.ned, 3)[e].copy()
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 't_move')
def wfn_t_move(self, r_e, step_size):
    """T-move
    :param r_e: electron positions - array(nelec, 3)
    :param step_size: DMC step size
    :return: next electrons positions
    """

    def impl(self, r_e, step_size):
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
@overload_method(Wfn_class_t, 'nonlocal_potential')
def wfn_nonlocal_potential(self, r_e):
    """Nonlocal (pseudopotential) energy Wφ/φ.
    :param r_e: electron positions - array(nelec, 3)
    """

    def impl(self, r_e) -> float:
        res = 0.0
        if self.ppotential is not None:
            e_vectors, n_vectors = self._relative_coordinates(r_e)
            grid = self.ppotential.integration_grid(n_vectors)
            potential = self.ppotential.get_ppotential(n_vectors)
            # every point of the quadrature grid is one electron displaced, so the ratio it needs
            # is the single-electron one. backflow and geminals have no such ratio and pay the
            # whole wave function per point instead
            if self.backflow is None and self.geminal is None:
                state = self.slater.state(n_vectors)
                for atom in range(n_vectors.shape[0]):
                    if self.ppotential.is_pseudoatom[atom]:
                        for e1 in range(self.neu + self.ned):
                            if potential[atom][e1, 0] or potential[atom][e1, 1]:
                                n_vectors_q = n_vectors.copy()
                                if self.jastrow is not None:
                                    n_powers = self.jastrow.en_powers(n_vectors)
                                    jastrow_1e = self.jastrow.value_1e(e_vectors[e1], n_powers, e1)
                                for q in range(grid.shape[2]):
                                    cos_theta = (grid[atom, e1, q] @ n_vectors[atom, e1]) / (n_vectors[atom, e1] @ n_vectors[atom, e1])
                                    r_e_q1 = grid[atom, e1, q] + self.atom_positions[atom]
                                    n_vectors_q[:, e1] = r_e_q1 - self.atom_positions
                                    log_value, sign, _, _ = state.ratio_1e(n_vectors_q[:, e1], e1)
                                    value_ratio = sign * state.sign * np.exp(log_value - state.log_value)
                                    if self.jastrow is not None:
                                        self.jastrow.update_en_powers_1e(n_powers, n_vectors_q[:, e1], e1)
                                        value_ratio *= np.exp(self.jastrow.value_1e(r_e_q1 - r_e, n_powers, e1) - jastrow_1e)
                                    weight = self.ppotential.weight[atom][q]
                                    for l in range(2):
                                        res += potential[atom][e1, l] * self.ppotential.legendre(l, cos_theta) * weight * value_ratio
            else:
                value = self.value(r_e)
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
def wfn_kinetic_energy(self, r_e):
    """Kinetic energy.
    :param r_e: electron coordinates - array(nelec, 3)
    :return: local energy
    """

    def impl(self, r_e) -> float:
        with_F_and_T = True
        e_vectors, n_vectors = self._relative_coordinates(r_e)

        if self.backflow is not None:
            b_l, b_g, b_v = self.backflow.laplacian(e_vectors, n_vectors)
            if self.geminal is not None:
                s_h, s_g = self.geminal.hessian(b_v + n_vectors)
            else:
                s_h, s_g = self.slater.hessian(b_v + n_vectors)
            s_l = np.sum(b_g * (s_h @ b_g)) + s_g @ b_l
            s_g = s_g @ b_g
        elif self.geminal is not None:
            s_g = self.geminal.gradient(n_vectors)
            s_l = self.geminal.laplacian(n_vectors)
        else:
            s_l, s_g = self.slater.laplacian(n_vectors)

        if self.jastrow is not None:
            j_l, j_g = self.jastrow.laplacian(e_vectors, n_vectors)
            F = s_g + j_g
            T = s_g @ s_g - s_l - j_l
            return (T - F @ F) / 2
        elif with_F_and_T:
            F = s_g
            T = s_g @ s_g - s_l
            return (T - F @ F) / 2
        else:
            return -s_l / 2

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'energy')
def wfn_energy(self, r_e):
    """Local energy.
    :param r_e: electron coordinates - array(nelec, 3)
    """

    def impl(self, r_e) -> float:
        return self.kinetic_energy(r_e) + self.coulomb(r_e) + self.nonlocal_potential(r_e)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'second_fundamental_form')
def wfn_second_fundamental_form(self, r_e):
    """The second fundamental form of φ(r) = const surface at point r.

    Epsilon defines the thickness of the tubular neighborhood around the nodal surface
    used to approximate the surface integral via the co-area formula:

      ∫_S f dS ≈ 1/(2ε) · ∫_{|t_n|<ε} f(r_S(r)) · J(r) dr

    where t_n = |ψ|/|∇ψ| is the normal coordinate and J is the co-area Jacobian.

    The projection r → r_S(r) is a diffeomorphism provided t_n < R_cut, where
    R_cut = 1/max(κ_i) is determined by the principal curvatures on the CONCAVE
    side of the nodal surface (where geodesic normals converge). On the convex
    side there is no geometric restriction: normals diverge, J < 1 everywhere.

    Constraints on ε:
      - ε < R_cut  (concave side only): avoids the fold locus where J → ∞
      - ε not too small: insufficient VMC statistics near the node (ρ ∝ t_n²→ 0)
      - ε not too large: co-area approximation requires J ≈ 1, i.e. ε ≪ R_cut

    Optimal choice: ε ≈ R_cut / 3  (J deviates from 1 by ~10%)
    R_cut is estimated from kappas computed at the projected point r_S.
    When the concave side has no dangerous curvatures, R_cut = ∞ and ε is
    limited only by the requirement of sufficient sampling density.

    :param r_e: electron coordinates - array(nelec, 3)
    :return:
    """

    def impl(self, r_e) -> np.ndarray:
        ne = self.neu + self.ned
        epsilon = 0.4
        print(np.sqrt(np.sum(r_e**2, axis=1)))
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        value = self.slater.value(n_vectors)  # ψ
        value_orig = value
        hess, grad = self.slater.hessian(n_vectors)  # Hψ/ψ, ∇ψ/ψ
        norm_log_grad = np.linalg.norm(grad)  # |∇ψ/ψ| = |∇ψ|/|ψ|
        dist = 1 / norm_log_grad  # |ψ|/|∇ψ|
        dist_orig = dist
        i = 0
        if dist < epsilon:
            # Projection onto the nodal surface
            while (np.abs(value) > 1e-10 or dist > 1e-10) and i < 40:
                i += 1
                step = grad / norm_log_grad**2  # Δr = -ψ·∇ψ/|∇ψ|² = -g/|g|²
                r_e -= step
                e_vectors, n_vectors = self._relative_coordinates(r_e)
                value = self.slater.value(n_vectors)  # ψ
                hess, grad = self.slater.hessian(n_vectors)  # Hψ/ψ, ∇ψ/ψ
                norm_log_grad = np.linalg.norm(grad)  # |∇ψ/ψ| = |∇ψ|/|ψ|
                dist = 1 / norm_log_grad  # |ψ|/|∇ψ|
            P = np.eye(ne * 3) - np.outer(grad, grad) / norm_log_grad**2
            II_full = -(P @ hess @ P) / norm_log_grad  # the sign does not depend on ψ
            _, _, Vt = np.linalg.svd(P)
            E = Vt.T[:, :-1]  # tangent_basis
            II = E.T @ II_full @ E  # II in tangent basis (3N-1 × 3N-1)
            kappas = np.linalg.eigvalsh(II)  # Principal curvatures
            min_abs_kappa = np.min(np.abs(kappas))  # minimum curvature
            print(np.sqrt(np.sum(r_e**2, axis=1)), min_abs_kappa)
            # Jacobian = П 1/(1 - κ_i · t_n)
            J = np.prod(1 / (1 - kappas * dist_orig))
            # Invariants
            H = np.mean(kappas)  # average curvature
            K = np.prod(kappas)  # Gaussian curvature
            # Surface measure is always positive
            dS = J / (2.0 * epsilon * value_orig**2)
            # Fixed-node error measure
            HdF = H * norm_log_grad * abs(value)  # H · |∇ψ|
            return np.array([dS, H * dS, HdF * dS, K * dS, min_abs_kappa, J, dist, i])
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dist, 0])

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'value_parameters_d1')
def wfn_value_parameters_d1(self, r_e):
    """First-order derivatives of the wave function value w.r.t parameters.
    :param r_e: electron coordinates - array(nelec, 3)
    :return:
    """

    def impl(self, r_e):
        res = np.zeros(0)
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None and self.opt_jastrow:
            res = np.concatenate((res, self.jastrow.value_parameters_d1(e_vectors, n_vectors)))
        if self.backflow is not None and self.opt_backflow:
            b_v = self.backflow.value(e_vectors, n_vectors) + n_vectors
            if self.geminal is not None:
                s_g = self.geminal.gradient(b_v)
            else:
                s_g = self.slater.gradient(b_v)
            res = np.concatenate((res, self.backflow.value_parameters_d1(e_vectors, n_vectors) @ s_g))
        if self.geminal is not None and self.opt_geminal:
            if self.backflow is not None:
                gem_v = self.backflow.value(e_vectors, n_vectors) + n_vectors
            else:
                gem_v = n_vectors
            res = np.concatenate((res, self.geminal.value_parameters_d1(gem_v)))
        if self.slater.det_coeff.size > 1 and self.opt_det_coeff:
            if self.backflow is not None:
                n_vectors = self.backflow.value(e_vectors, n_vectors) + n_vectors
            res = np.concatenate((res, self.slater.value_parameters_d1(n_vectors)))
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'kinetic_energy_parameters_d1')
def wfn_kinetic_energy_parameters_d1(self, r_e):
    """First-order derivatives of kinetic energy w.r.t parameters.
    :param r_e: electron coordinates - array(nelec, 3)
    :return:
    """

    def impl(self, r_e):
        res = np.zeros(0)
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.backflow is not None:
            b_l, b_g, b_v = self.backflow.laplacian(e_vectors, n_vectors)
            if self.geminal is not None:
                s_g = self.geminal.gradient(b_v + n_vectors) @ b_g
            else:
                s_g = self.slater.gradient(b_v + n_vectors) @ b_g
        elif self.geminal is not None:
            s_g = self.geminal.gradient(n_vectors)
        else:
            s_g = self.slater.gradient(n_vectors)
        if self.jastrow is not None:
            j_g = self.jastrow.gradient(e_vectors, n_vectors)
        if self.jastrow is not None and self.opt_jastrow:
            # Jastrow parameters part
            j_g_d1 = self.jastrow.gradient_parameters_d1(e_vectors, n_vectors)
            j_l_d1 = self.jastrow.laplacian_parameters_d1(e_vectors, n_vectors)
            j_d1 = j_g_d1 @ (s_g + j_g) + j_l_d1 / 2
            res = np.concatenate((res, j_d1))
        if self.backflow is not None and self.opt_backflow:
            # backflow parameters part
            b_l_d1, b_g_d1, b_v_d1 = self.backflow.laplacian_parameters_d1(e_vectors, n_vectors)
            bb = b_g @ b_g.T
            # tressian is only ever contracted with bb over its last two axes, so compute that vector directly
            if self.geminal is not None:
                s_t_bb, s_h, s_g = self.geminal.tressian_dot(b_v + n_vectors, bb)
            else:
                s_t_bb, s_h, s_g = self.slater.tressian_dot(b_v + n_vectors, bb)
            s_g_d1 = b_v_d1 @ (s_h - np.outer(s_g, s_g))  # as hessian is d²ln(phi)/dxdy
            # Σ_bc (s_t - s_g ⊗ s_h)[a,b,c] · bb[b,c] = s_t_bb[a] - s_g[a]·Σ_bc s_h[b,c]·bb[b,c]
            s_h_d1_bb = b_v_d1 @ (s_t_bb - s_g * np.sum(s_h * bb))

            parameters = b_v_d1.shape[0]
            # d(b_g @ b_g.T) = b_g_d1 @ b_g.T + b_g @ b_g_d1.T = b_g_d1 @ b_g.T + (b_g_d1 @ b_g.T).T
            # and s_h is symmetric matrix, so sum(s_h * (b_g_d1[i] @ b_g.T)) = sum(b_g_d1[i] * (s_h @ b_g))
            bf_d1 = s_h_d1_bb / 2 + b_g_d1.reshape(parameters, -1) @ (s_h @ b_g).ravel() + (s_g_d1 @ b_l + b_l_d1 @ s_g) / 2
            if self.jastrow is not None:
                bf_d1 += s_g_d1 @ (b_g @ j_g) + b_g_d1.reshape(parameters, -1) @ np.outer(s_g, j_g).ravel()
            res = np.concatenate((res, bf_d1 @ self.backflow.parameters_projector))
        if self.geminal is not None and self.opt_geminal:
            # geminal parameters part. The laplacian of the geminal is its own second derivative
            # over its own value, so the square of its gradient is already inside it and the only
            # cross term left in the local energy is the one with the jastrow
            if self.backflow is not None:
                # under backflow the laplacian of the geminal over the electron coordinates is
                # its hessian over the quasi-particle ones contracted with the jacobian, plus its
                # gradient there against the laplacian of the transformation
                gem_g_d1 = self.geminal.gradient_parameters_d1(b_v + n_vectors)
                gem_d1 = (self.geminal.hessian_parameters_d1_dot(b_v + n_vectors, b_g @ b_g.T) + gem_g_d1 @ b_l) / 2
                gem_g_d1 = gem_g_d1 @ b_g
            else:
                gem_g_d1 = self.geminal.gradient_parameters_d1(n_vectors)
                gem_d1 = self.geminal.laplacian_parameters_d1(n_vectors) / 2
            if self.jastrow is not None:
                gem_d1 += gem_g_d1 @ j_g
            res = np.concatenate((res, gem_d1))
        if self.slater.det_coeff.size > 1 and self.opt_det_coeff:
            # determinants coefficients part
            if self.backflow is not None:
                s_g_d1 = self.slater.gradient_parameters_d1(b_v + n_vectors)
                s_h_d1 = self.slater.hessian_parameters_d1(b_v + n_vectors)
                # because slater gradient w.r.t parameters is already projected
                sl_d1 = np.zeros(shape=s_g_d1.shape[0])
                for i in range(s_g_d1.shape[0]):
                    # the laplacian of the transformation goes against the gradient over the
                    # quasi-particle coordinates, the one the jacobian has not yet been applied to
                    sl_d1[i] = (np.sum(s_h_d1[i] * (b_g @ b_g.T)) + s_g_d1[i] @ b_l) / 2
                s_g_d1 = s_g_d1 @ b_g
            else:
                s_g_d1 = self.slater.gradient_parameters_d1(n_vectors)
                sl_d1 = self.slater.laplacian_parameters_d1(n_vectors) / 2
            if self.jastrow is not None:
                # the laplacian of the determinant is taken in the d²Phi/Phi form, which carries
                # the square of its own gradient already, so the [nabla Phi / Phi] half of the
                # bracket of the log form cancels and the jastrow is the only cross term left
                sl_d1 += s_g_d1 @ j_g
            res = np.concatenate((res, sl_d1))
        return -res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'nonlocal_energy_parameters_d1')
def wfn_nonlocal_energy_parameters_d1(self, r_e):
    """First-order derivatives of pseudopotential energy w.r.t parameters.
    :param r_e: electron coordinates - array(nelec, 3)
    :return:
    """

    def impl(self, r_e):
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        grid = self.ppotential.integration_grid(n_vectors)
        potential = self.ppotential.get_ppotential(n_vectors)
        value_parameters_d1 = self.value_parameters_d1(r_e)
        res = np.zeros(shape=(value_parameters_d1.size,))
        for atom in range(n_vectors.shape[0]):
            if self.ppotential.is_pseudoatom[atom]:
                for e1 in range(self.neu + self.ned):
                    if potential[atom][e1, 0] or potential[atom][e1, 1]:
                        for q in range(grid.shape[2]):
                            cos_theta = (grid[atom, e1, q] @ n_vectors[atom, e1]) / (n_vectors[atom, e1] @ n_vectors[atom, e1])
                            r_e_q = r_e.copy()
                            r_e_q[e1] = grid[atom, e1, q] + self.atom_positions[atom]
                            value_q = self.value(r_e_q)
                            value_parameters_d1_q = self.value_parameters_d1(r_e_q)
                            weight = self.ppotential.weight[atom][q]
                            for l in range(2):
                                res += (
                                    potential[atom][e1, l]
                                    * self.ppotential.legendre(l, cos_theta)
                                    * weight
                                    * value_q
                                    * (value_parameters_d1_q - value_parameters_d1)
                                )
        return res / self.value(r_e)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'get_parameters')
def wfn_get_parameters(self, all_parameters=False):
    """Get WFN parameters to be optimized
    :param all_parameters: get all parameters or only independent
    """

    def impl(self, all_parameters=False):
        res = np.zeros(0)
        if self.jastrow is not None and self.opt_jastrow:
            res = np.concatenate((res, self.jastrow.get_parameters(all_parameters)))
        if self.backflow is not None and self.opt_backflow:
            res = np.concatenate((res, self.backflow.get_parameters(all_parameters)))
        if self.geminal is not None and self.opt_geminal:
            res = np.concatenate((res, self.geminal.get_parameters(all_parameters)))
        if self.slater.det_coeff.size > 1 and self.opt_det_coeff:
            res = np.concatenate((res, self.slater.get_parameters(all_parameters)))
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Wfn_class_t, 'set_parameters')
def wfn_set_parameters(self, parameters, all_parameters=False):
    """Update optimized parameters
    :param parameters: parameters to update
    :param all_parameters: set all parameters or only independent
    """

    def impl(self, parameters, all_parameters=False):
        if self.jastrow is not None and self.opt_jastrow:
            parameters = self.jastrow.set_parameters(parameters, all_parameters=all_parameters)
        if self.backflow is not None and self.opt_backflow:
            parameters = self.backflow.set_parameters(parameters, all_parameters=all_parameters)
        if self.geminal is not None and self.opt_geminal:
            parameters = self.geminal.set_parameters(parameters, all_parameters=all_parameters)
        if self.slater.det_coeff.size > 1 and self.opt_det_coeff:
            self.slater.set_parameters(parameters, all_parameters=all_parameters)

    return impl


Wfn_t = Wfn_class_t(
    [
        ('neu', nb.int64),
        ('ned', nb.int64),
        ('atom_positions', nb.float64[:, ::1]),
        ('atom_charges', nb.float64[::1]),
        ('nuclear_repulsion', nb.float64),
        ('slater', Slater_t),
        ('geminal', nb.optional(Geminal_t)),
        ('jastrow', nb.optional(Jastrow_t)),
        ('backflow', nb.optional(Backflow_t)),
        ('ppotential', nb.optional(PPotential_t)),
        ('opt_jastrow', nb.boolean),
        ('opt_backflow', nb.boolean),
        ('opt_geminal', nb.boolean),
        ('opt_orbitals', nb.boolean),
        ('opt_det_coeff', nb.boolean),
    ]
)


class Wfn(structref.StructRefProxy, AbstractWfn):
    def __new__(cls, config, slater, geminal=None, jastrow=None, backflow=None, ppotential=None):
        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(neu, ned, atom_positions, atom_charges, slater, geminal, jastrow, backflow, ppotential):
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
            self = structref.new(Wfn_t)
            self.neu = neu
            self.ned = ned
            self.atom_positions = atom_positions
            self.atom_charges = atom_charges
            self.nuclear_repulsion = self._get_nuclear_repulsion()
            self.slater = slater
            self.geminal = geminal
            self.jastrow = jastrow
            self.backflow = backflow
            self.ppotential = ppotential
            self.opt_jastrow = False
            self.opt_backflow = False
            self.opt_geminal = False
            self.opt_orbitals = False
            self.opt_det_coeff = False
            return self

        return init(
            config.input.neu, config.input.ned, config.wfn.atom_positions, config.wfn.atom_charges, slater, geminal, jastrow, backflow, ppotential
        )

    @nb.njit(nogil=True, parallel=False, cache=True)
    def _relative_coordinates(self, r_e):
        return self._relative_coordinates(r_e)

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def slater(self):
        return self.slater

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def geminal(self):
        return self.geminal

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def jastrow(self):
        return self.jastrow

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def backflow(self):
        return self.backflow

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def opt_jastrow(self):
        return self.opt_jastrow

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def opt_backflow(self):
        return self.opt_backflow

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def opt_geminal(self):
        return self.opt_geminal

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def opt_orbitals(self):
        return self.opt_orbitals

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def opt_det_coeff(self):
        return self.opt_det_coeff

    @opt_jastrow.setter
    @nb.njit(nogil=True, parallel=False, cache=True)
    def opt_jastrow(self, value):
        self.opt_jastrow = value

    @opt_backflow.setter
    @nb.njit(nogil=True, parallel=False, cache=True)
    def opt_backflow(self, value):
        self.opt_backflow = value

    @opt_geminal.setter
    @nb.njit(nogil=True, parallel=False, cache=True)
    def opt_geminal(self, value):
        self.opt_geminal = value

    @opt_orbitals.setter
    @nb.njit(nogil=True, parallel=False, cache=True)
    def opt_orbitals(self, value):
        self.opt_orbitals = value

    @opt_det_coeff.setter
    @nb.njit(nogil=True, parallel=False, cache=True)
    def opt_det_coeff(self, value):
        self.opt_det_coeff = value

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def nuclear_repulsion(self) -> float:
        """Value of n-n repulsion."""
        return self.nuclear_repulsion

    @nb.njit(nogil=True, parallel=False, cache=True)
    def kinetic_energy(self, r_e) -> float:
        """Kinetic energy.
        :param r_e: electron coordinates - array(nelec, 3)
        """
        return self.kinetic_energy(r_e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def drift_velocity(self, r_e):
        """Drift velocity.
        :param r_e: electron coordinates - array(nelec, 3)
        """
        return self.drift_velocity(r_e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def drift_kinetic_energy(self, r_e) -> float:
        """Kinetic energy in drift form, F@F/2. Has the same mean as kinetic_energy by parts,
        and a variance smaller by two orders of magnitude, being bounded where the laplacian
        form is not. This is the one the VMC step size sum rule contains.
        :param r_e: electron coordinates - array(nelec, 3)
        """
        F = self.drift_velocity(r_e)
        return F @ F / 2

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value(self, r_e) -> float:
        """Value of wave function.
        :param r_e: electron coordinates - array(nelec, 3)
        """
        return self.value(r_e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def log_value(self, r_e):
        """Logarithm of the absolute value of wave function, and its sign.
        :param r_e: electron coordinates - array(nelec, 3)
        """
        return self.log_value(r_e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def caches(self, r_e):
        """The caches a walker carries between single-electron moves.
        :param r_e: electron coordinates - array(nelec, 3)
        """
        return self.caches(r_e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value_ratio_1e(self, state, n_powers, r_e, e, next_r_e_1e):
        """Ratio of the wave function at the two ends of a single-electron move.
        :param r_e: electron coordinates - array(nelec, 3)
        """
        return self.value_ratio_1e(state, n_powers, r_e, e, next_r_e_1e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def accept_1e(self, state, n_powers, e, next_r_e_1e, orbitals, q):
        """Move electron e into the caches of the walker.
        :param next_r_e_1e: its accepted position - array(3)
        """
        return self.accept_1e(state, n_powers, e, next_r_e_1e, orbitals, q)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def drift_velocity_1e(self, state, n_powers, r_e, e, r_e_1e, q):
        """Drift velocity of a single electron.
        :param r_e_1e: the position of that electron the drift is taken at - array(3)
        """
        return self.drift_velocity_1e(state, n_powers, r_e, e, r_e_1e, q)

    @nb.njit(nogil=True, parallel=False, cache=True)
    # @nb.vectorize('float64(float64[:, :])', cache=True)
    def energy(self, r_e) -> float:
        """Local energy.
        :param r_e: electron coordinates - array(nelec, 3)
        """
        return self.energy(r_e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def second_fundamental_form(self, r_e):
        """The second fundamental form of φ(r) = const surface at point r.
        :param r_e: electron coordinates - array(nelec, 3)
        :return:
        """
        return self.second_fundamental_form(r_e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def get_parameters(self, all_parameters=False):
        """Get WFN parameters to be optimized
        :param all_parameters: get all parameters or only independent
        """
        return self.get_parameters(all_parameters)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def set_parameters(self, parameters, all_parameters=False):
        """Update optimized parameters
        :param parameters: parameters to update
        :param all_parameters: set all parameters or only independent
        """
        self.set_parameters(parameters, all_parameters)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def set_parameters_projector(self):
        """Update parameters projector"""
        if self.jastrow is not None and self.opt_jastrow:
            self.jastrow.set_parameters_projector()
        if self.backflow is not None and self.opt_backflow:
            self.backflow.set_parameters_projector()
        if self.geminal is not None and self.opt_geminal:
            self.geminal.set_parameters_projector()
        if self.slater.det_coeff.size > 1 and self.opt_det_coeff:
            self.slater.set_parameters_projector()

    @nb.njit(nogil=True, parallel=False, cache=True)
    def get_parameters_scale(self, all_parameters=False):
        """Characteristic scale of each optimized parameter.
        :param all_parameters: get all parameters scale or only independent
        """
        res = np.zeros(0)
        if self.jastrow is not None and self.opt_jastrow:
            res = np.concatenate((res, self.jastrow.get_parameters_scale(all_parameters)))
        if self.backflow is not None and self.opt_backflow:
            res = np.concatenate((res, self.backflow.get_parameters_scale(all_parameters)))
        if self.geminal is not None and self.opt_geminal:
            res = np.concatenate((res, self.geminal.get_parameters_scale(all_parameters)))
        if self.slater.det_coeff.size > 1 and self.opt_det_coeff:
            res = np.concatenate((res, self.slater.get_parameters_scale(all_parameters)))
        return res

    @nb.njit(nogil=True, parallel=False, cache=True)
    # @nb.vectorize('float64[:](float64[:, :])', cache=True)
    def value_parameters_d1(self, r_e):
        """First-order derivatives of the wave function value w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :return:
        """
        return self.value_parameters_d1(r_e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    # @nb.vectorize('float64[:, :](float64[:, :])', cache=True)
    def value_parameters_d2(self, r_e):
        """Second-order derivatives of the wave function value w.r.t parameters.
        1/wfn * d²wfn/dp² - 1/wfn * dwfn/dp * 1/wfn * dwfn/dp
        :param r_e: electron coordinates - array(nelec, 3)
        :return:
        """
        res = []
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        if self.jastrow is not None and self.opt_jastrow:
            res.append(self.jastrow.value_parameters_d2(e_vectors, n_vectors))
        if self.backflow is not None and self.opt_backflow:
            raise NotImplementedError
        if self.geminal is not None and self.opt_geminal:
            raise NotImplementedError
        if self.slater.det_coeff.size > 1 and self.opt_det_coeff:
            raise NotImplementedError
        return block_diag(res)

    @nb.njit(nogil=True, parallel=False, cache=True)
    # @nb.vectorize('float64[:](float64[:, :])', cache=True)
    def kinetic_energy_parameters_d1(self, r_e):
        """First-order derivatives of kinetic_ energy w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :return:
        """
        return self.kinetic_energy_parameters_d1(r_e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    # @nb.vectorize('float64[:](float64[:, :])', cache=True)
    def energy_parameters_d1(self, r_e):
        """First-order derivatives of local energy w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :return:
        """
        res = self.kinetic_energy_parameters_d1(r_e)
        if self.ppotential is not None:
            # pseudopotential part
            res += self.nonlocal_energy_parameters_d1(r_e)
        return res


structref.define_boxing(Wfn_class_t, Wfn)
