import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino import delta
from casino.abstract import AbstractBackflow
from casino.overload import block_diag, rref

eye3 = np.eye(3)


@nb.njit(nogil=True, parallel=False, cache=True)
def construct_c_matrix(trunc, phi_parameters, theta_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational):
    """C-matrix has the following rows:
    6 * (phi_en_order + phi_ee_order + 1) - 2 constraints imposed to satisfy phi-term conditions.
    ... constraints imposed to satisfy theta-term conditions.
    copy-paste from /CASINO/src/pbackflow.f90 SUBROUTINE construct_C
    """
    phi_ee_order = phi_parameters.shape[1] - 1
    phi_en_order = phi_parameters.shape[2] - 1

    ee_constrains = 2 * phi_en_order + 1
    en_constrains = phi_en_order + phi_ee_order + 1
    offset = 0
    if phi_cusp:
        # AE cusps: 6 * (ee + eN) - 8
        phi_constraints = 6 * en_constrains - 2
    else:
        # PP cusps: 2 * (ee + eN) - 2
        phi_constraints = 2 * en_constrains
    if spin_dep in (0, 2):
        # e-e cusps: 2 * eN - 1 from Theta, and same from some spin-deps of Phi
        phi_constraints += ee_constrains
        offset += ee_constrains
    if phi_cusp:
        # AE cusps: 5 * (ee + eN) - 7
        theta_constraints = 5 * en_constrains + ee_constrains - 2
    else:
        # PP cusps: ee + eN - 1
        theta_constraints = en_constrains + ee_constrains
    n_constraints = phi_constraints + theta_constraints
    if phi_irrotational:
        # ee * eN * eN + 2 * eN * (ee-1) + eN**2 constraints from Phi and Theta
        n_constraints += ((phi_en_order + 3) * (phi_ee_order + 2) - 4) * (phi_en_order + 1)
        if trunc == 0:
            n_constraints -= (phi_en_order + 1) * (phi_ee_order + 1)

    parameters_size = 2 * (phi_parameters.shape[1] * phi_parameters.shape[2] * phi_parameters.shape[3])
    c = np.zeros((n_constraints, parameters_size))
    cutoff_constraints = np.zeros(shape=(n_constraints,))
    p = 0
    # Do Phi bit of the constraint matrix.
    for m in range(phi_parameters.shape[1]):
        for l in range(phi_parameters.shape[2]):
            for k in range(phi_parameters.shape[3]):
                if spin_dep in (0, 2):  # e-e cusp
                    if m == 1:
                        c[k + l, p] = 1
                if phi_cusp:
                    if l == 0:  # 1b
                        c[k + m + offset + en_constrains, p] = 1
                        if m > 0:  # 3b
                            c[k + m - 1 + offset + 5 * en_constrains - 1, p] = m
                    elif l == 1:  # b2
                        c[k + m + offset + 3 * en_constrains, p] = 1
                    if k == 0:  # 1a
                        c[l + m + offset, p] = 1
                        if m > 0:  # 3a
                            c[l + m - 1 + offset + 4 * en_constrains, p] = m
                    elif k == 1:  # 2a
                        c[l + m + offset + 2 * en_constrains, p] = 1
                else:
                    if l == 0:  # 1b
                        c[k + m + offset + en_constrains, p] = -trunc / phi_cutoff
                        cutoff_constraints[k + m + offset + en_constrains] += trunc * phi_parameters[spin_dep, m, l, k] / phi_cutoff**2
                    elif l == 1:  # 1b
                        c[k + m + offset + en_constrains, p] = 1
                    if k == 0:  # 1a
                        c[l + m + offset, p] = -trunc / phi_cutoff
                        cutoff_constraints[l + m + offset] += trunc * phi_parameters[spin_dep, m, l, k] / phi_cutoff**2
                    elif k == 1:  # 1a
                        c[l + m + offset, p] = 1
                p += 1
    # Do Theta bit of the constraint matrix.
    offset = phi_constraints
    for m in range(phi_parameters.shape[1]):
        for l in range(phi_parameters.shape[2]):
            for k in range(phi_parameters.shape[3]):
                if m == 1:
                    c[k + l + offset, p] = 1
                if phi_cusp:
                    if l == 0:  # 2b
                        c[k + m + offset + ee_constrains + 2 * en_constrains, p] = -trunc / phi_cutoff
                        cutoff_constraints[k + m + offset + ee_constrains + 2 * en_constrains] += (
                            trunc * theta_parameters[spin_dep, m, l, k] / phi_cutoff**2
                        )
                        if m > 0:  # 3b
                            c[k + m - 1 + offset + ee_constrains + 4 * en_constrains - 1, p] = m
                    elif l == 1:  # 2b
                        c[k + m + offset + ee_constrains + 2 * en_constrains, p] = 1
                    if k == 0:  # 1a
                        c[l + m + offset + ee_constrains, p] = 1
                        if m > 0:  # 3a
                            c[l + m - 1 + offset + ee_constrains + 3 * en_constrains, p] = m
                    elif k == 1:  # 2a
                        c[l + m + offset + ee_constrains + en_constrains, p] = 1
                else:
                    if l == 0:  # 1a
                        c[k + m + offset + ee_constrains, p] = -trunc / phi_cutoff
                        cutoff_constraints[l + m + offset + ee_constrains] += trunc * theta_parameters[spin_dep, m, l, k] / phi_cutoff**2
                    elif l == 1:  # 1a
                        c[k + m + offset + ee_constrains, p] = 1
                p += 1
    # Do irrotational bit of the constraint matrix.
    n = phi_constraints + theta_constraints
    if phi_irrotational:
        p = 0
        inc_k = 1
        inc_l = inc_k * (phi_en_order + 1)
        inc_m = inc_l * (phi_en_order + 1)
        nphi = inc_m * (phi_ee_order + 1)
        for m in range(phi_parameters.shape[1]):
            for l in range(phi_parameters.shape[2]):
                for k in range(phi_parameters.shape[3]):
                    if trunc > 0:
                        if m > 0:
                            c[n, p - inc_m] = trunc + k
                            if k < phi_en_order:
                                c[n, p + inc_k - inc_m] = -phi_cutoff * (k + 1)
                        if m < phi_ee_order:
                            if k > 1:
                                c[n, p + nphi - 2 * inc_k + inc_m] = -(m + 1)
                            if k > 0:
                                c[n, p + nphi - inc_k + inc_m] = phi_cutoff * (m + 1)
                                cutoff_constraints[n] += (m + 1) * phi_parameters[spin_dep, m, l, k]
                    else:
                        if m > 0 and k < phi_en_order:
                            c[n, p + inc_k - inc_m] = k + 1
                        if k > 0 and m < phi_ee_order:
                            c[n, p + nphi - inc_k + inc_m] = -(m + 1)
                    p += 1
                    n += 1
        if trunc > 0:
            # Same as above, for m=N_ee+1...
            p = phi_ee_order * (phi_en_order + 1) ** 2
            for l in range(phi_parameters.shape[2]):
                for k in range(phi_parameters.shape[3]):
                    c[n, p] = trunc + k
                    if k < phi_en_order:
                        c[n, p + inc_k] = -phi_cutoff * (k + 1)
                        cutoff_constraints[n] -= (k + 1) * phi_parameters[spin_dep, :, l, k + 1].sum()
                    p += 1
                    n += 1
            # ...for k=N_eN+1...
            p = phi_en_order - 1
            for m in range(phi_parameters.shape[1] - 1):
                for l in range(phi_parameters.shape[2]):
                    c[n, p + nphi + inc_m] = -(m + 1)
                    c[n, p + nphi + inc_k + inc_m] = phi_cutoff * (m + 1)
                    cutoff_constraints[n] += (m + 1) * theta_parameters[spin_dep, m + 1, l, :].sum()
                    p += inc_l
                    n += 1
            # ...and for k=N_eN+2.
            p = phi_en_order
            for m in range(phi_parameters.shape[1] - 1):
                for l in range(phi_parameters.shape[2]):
                    c[n, p + nphi + inc_m] = -(m + 1)
                    p += inc_l
                    n += 1
        else:
            # Same as above, for m=N_ee+1...
            p = phi_ee_order * (phi_en_order + 1) ** 2
            for l in range(phi_parameters.shape[2]):
                for k in range(phi_parameters.shape[3] - 1):
                    c[n, p + inc_k] = 1  # just zeroes the corresponding param
                    p += 1
                    n += 1
            # ...and for k=N_eN+1.
            p = phi_en_order - 1
            for m in range(phi_parameters.shape[1] - 1):
                for l in range(phi_parameters.shape[2]):
                    c[n, p + nphi + inc_m] = 1  # just zeroes the corresponding param
                    p += inc_l
                    n += 1

    assert n == n_constraints
    return c, cutoff_constraints


@nb.njit(nogil=True, parallel=False, cache=True)
def construct_omega_matrix(trunc, omega_parameters, omega_cutoff, spin_dep):
    """Omega-matrix has the following rows:
    3 * (2 * omega_order + 1) no-cusp constraints, one set per pair distance,
    nd**3 (or 2 * nd**3 for a fully equal spin-triplet) particle-symmetry constraints.
    copy-paste from /CASINO/src/pbackflow.f90 SUBROUTINE construct_omega_constraints
    """
    nd = omega_parameters.shape[1]
    omega_order = nd - 1
    if omega_parameters.shape[0] == 1:
        omega_symm = 3
    else:
        omega_symm = (3, 1, 1, 3)[spin_dep]

    no_cusp_constraints = 3 * (2 * omega_order + 1)
    n_constraints = no_cusp_constraints
    if omega_symm == 1:
        n_constraints += nd**3
    elif omega_symm == 3:
        n_constraints += 2 * nd**3

    c = np.zeros((n_constraints, nd**3))
    cutoff_constraints = np.zeros(shape=(n_constraints,))
    # No-cusp conditions for each of the three pair distances, K_lmn: l -> rjk, m -> rki, n -> rij
    for a in range(2 * omega_order + 1):
        q1 = a
        q2 = q1 + 2 * omega_order + 1
        q3 = q2 + 2 * omega_order + 1
        for n in range(max(0, a - omega_order), min(omega_order, a) + 1):
            m = a - n
            # rjk -> 0
            c[q1, m * nd + n * nd**2] -= trunc / omega_cutoff
            c[q1, 1 + m * nd + n * nd**2] += 1
            cutoff_constraints[q1] += trunc * omega_parameters[spin_dep, 0, m, n] / omega_cutoff**2
            # rki -> 0
            c[q2, m + n * nd**2] -= trunc / omega_cutoff
            c[q2, m + nd + n * nd**2] += 1
            cutoff_constraints[q2] += trunc * omega_parameters[spin_dep, m, 0, n] / omega_cutoff**2
            # rij -> 0
            c[q3, m + n * nd] -= trunc / omega_cutoff
            c[q3, m + n * nd + nd**2] += 1
            cutoff_constraints[q3] += trunc * omega_parameters[spin_dep, m, n, 0] / omega_cutoff**2

    # Particle-symmetry constraints, as in construct_H in pjastrow.f90. NOTE: CASINO up to at
    # least v2.13.1313 sets both entries to +1 in construct_omega_constraints, which makes K_lmn
    # antisymmetric instead of symmetric, so omega changes sign under exchange of two identical
    # electrons and the backflow displacement is not equivariant. Omega parameters written by
    # those versions are therefore incompatible with this form.
    q = no_cusp_constraints
    if omega_symm > 0:
        for n in range(nd):
            for m in range(nd):
                for l in range(nd):
                    # first two indices interchangeable
                    c[q, l + m * nd + n * nd**2] += 1
                    c[q, m + l * nd + n * nd**2] -= 1
                    q += 1
                    if omega_symm == 3:
                        # last two indices interchangeable
                        c[q, l + m * nd + n * nd**2] += 1
                        c[q, l + n * nd + m * nd**2] -= 1
                        q += 1

    assert q == n_constraints
    return c, cutoff_constraints


@nb.njit(nogil=True, parallel=False, cache=True)
def construct_omega_folded_matrix(trunc, omega_parameters, omega_cutoff, spin_dep):
    """Same no-cusp constraints as construct_omega_matrix, but with the particle symmetry folded
    into the enumeration instead of being imposed by rows, as construct_a_matrix does for the
    Jastrow f-term. Only one representative per orbit of the index permutation is kept - the one
    with ascending indices, i.e. the one rref of the full matrix leaves free.
    Returns the (3 * (2 * omega_order + 1), n_representatives) matrix, the map of each K_lmn onto
    the column of its representative, and the indices of the representatives themselves.
    """
    nd = omega_parameters.shape[1]
    omega_order = nd - 1
    if omega_parameters.shape[0] == 1:
        omega_symm = 3
    else:
        omega_symm = (3, 1, 1, 3)[spin_dep]

    rep = np.zeros(shape=(nd, nd, nd), dtype=np.int64)
    rep_indices = np.zeros(shape=(nd**3, 3), dtype=np.int64)
    n_representatives = 0
    for n in range(nd):
        for m in range(nd):
            for l in range(nd):
                if omega_symm == 3:
                    is_representative = l <= m <= n
                elif omega_symm == 1:
                    is_representative = l <= m
                else:
                    is_representative = True
                if is_representative:
                    rep[l, m, n] = n_representatives
                    rep_indices[n_representatives] = l, m, n
                    n_representatives += 1
    for n in range(nd):
        for m in range(nd):
            for l in range(nd):
                if omega_symm == 3:
                    i1, i2, i3 = min(l, m, n), l + m + n - min(l, m, n) - max(l, m, n), max(l, m, n)
                elif omega_symm == 1:
                    i1, i2, i3 = min(l, m), max(l, m), n
                else:
                    i1, i2, i3 = l, m, n
                rep[l, m, n] = rep[i1, i2, i3]

    c = np.zeros(shape=(3 * (2 * omega_order + 1), n_representatives))
    # No-cusp conditions for each of the three pair distances, K_lmn: l -> rjk, m -> rki, n -> rij.
    # Coefficients are accumulated onto the representative's column, so the multiplicity of each
    # orbit is taken into account automatically.
    for a in range(2 * omega_order + 1):
        q1 = a
        q2 = q1 + 2 * omega_order + 1
        q3 = q2 + 2 * omega_order + 1
        for n in range(max(0, a - omega_order), min(omega_order, a) + 1):
            m = a - n
            # rjk -> 0
            c[q1, rep[0, m, n]] -= trunc / omega_cutoff
            c[q1, rep[1, m, n]] += 1
            # rki -> 0
            c[q2, rep[m, 0, n]] -= trunc / omega_cutoff
            c[q2, rep[m, 1, n]] += 1
            # rij -> 0
            c[q3, rep[m, n, 0]] -= trunc / omega_cutoff
            c[q3, rep[m, n, 1]] += 1

    return c, rep, rep_indices[:n_representatives]


@structref.register
class Backflow_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


labels_type = nb.int64[::1]
eta_parameters_type = nb.float64[:, ::1]
mu_parameters_type = nb.float64[:, ::1]
phi_parameters_type = nb.float64[:, :, :, ::1]
theta_parameters_type = nb.float64[:, :, :, ::1]
omega_parameters_type = nb.float64[:, :, :, ::1]
eta_parameters_mask_type = nb.boolean[:, ::1]
mu_parameters_mask_type = nb.boolean[:, ::1]
phi_parameters_mask_type = nb.boolean[:, :, :, ::1]
theta_parameters_mask_type = nb.boolean[:, :, :, ::1]
omega_parameters_mask_type = nb.boolean[:, :, :, ::1]

Backflow_t = Backflow_class_t(
    [
        ('neu', nb.int64),
        ('ned', nb.int64),
        ('trunc', nb.int64),
        ('eta_parameters', eta_parameters_type),
        ('mu_parameters', nb.types.ListType(mu_parameters_type)),
        ('phi_parameters', nb.types.ListType(phi_parameters_type)),
        ('theta_parameters', nb.types.ListType(theta_parameters_type)),
        ('omega_parameters', omega_parameters_type),
        ('eta_parameters_optimizable', eta_parameters_mask_type),
        ('mu_parameters_optimizable', nb.types.ListType(mu_parameters_mask_type)),
        ('phi_parameters_optimizable', nb.types.ListType(phi_parameters_mask_type)),
        ('theta_parameters_optimizable', nb.types.ListType(theta_parameters_mask_type)),
        ('omega_parameters_optimizable', omega_parameters_mask_type),
        ('eta_parameters_available', eta_parameters_mask_type),
        ('mu_parameters_available', nb.types.ListType(mu_parameters_mask_type)),
        ('phi_parameters_available', nb.types.ListType(phi_parameters_mask_type)),
        ('theta_parameters_available', nb.types.ListType(theta_parameters_mask_type)),
        ('omega_parameters_available', omega_parameters_mask_type),
        ('eta_cutoff', nb.float64[:]),
        ('eta_cutoff_optimizable', nb.boolean[:]),
        ('mu_cutoff', nb.float64[:]),
        ('mu_cutoff_optimizable', nb.boolean[:]),
        ('phi_cutoff', nb.float64[:]),
        ('phi_cutoff_optimizable', nb.boolean[:]),
        ('omega_cutoff', nb.float64[:]),
        ('omega_cutoff_optimizable', nb.boolean[:]),
        ('mu_labels', nb.types.ListType(labels_type)),
        ('phi_labels', nb.types.ListType(labels_type)),
        ('max_ee_order', nb.int64),
        ('max_en_order', nb.int64),
        ('mu_cusp', nb.boolean[::1]),
        ('phi_cusp', nb.boolean[::1]),
        ('phi_irrotational', nb.boolean[::1]),
        ('ae_cutoff', nb.float64[:]),
        ('ae_cutoff_optimizable', nb.boolean[:]),
        ('parameters_projector', nb.float64[:, ::1]),
        ('cutoffs_optimizable', nb.boolean),
    ]
)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'fix_optimizable')
def backflow_fix_optimizable(self):
    """Set parameter fixed if there is no corresponded spin-pairs"""

    def impl(self):
        if self.neu + self.ned == 1:
            # H-atom
            for i in range(len(self.eta_cutoff_optimizable)):
                self.eta_cutoff_optimizable[i] = False
            for i in range(len(self.phi_cutoff_optimizable)):
                self.phi_cutoff_optimizable[i] = False

        ee_order = 2
        if self.eta_parameters.shape[0] == 2:
            if self.neu < ee_order and self.ned < ee_order:
                self.eta_parameters_available[0] = False
            if self.neu + self.ned < ee_order:
                self.eta_parameters_available[1] = False
        elif self.eta_parameters.shape[0] == 3:
            if self.neu < ee_order:
                self.eta_parameters_available[0] = False
            if self.neu + self.ned < ee_order:
                self.eta_parameters_available[1] = False
            if self.ned < ee_order:
                self.eta_parameters_available[2] = False

        ee_order = 1
        for mu_parameters_optimizable in self.mu_parameters_optimizable:
            mu_parameters_available = np.ones_like(mu_parameters_optimizable)
            if mu_parameters_optimizable.shape[0] == 2:
                if self.neu < ee_order:
                    mu_parameters_available[0] = False
                if self.ned < ee_order:
                    mu_parameters_available[1] = False
            self.mu_parameters_available.append(mu_parameters_available)

        ee_order = 2
        for phi_parameters_optimizable in self.phi_parameters_optimizable:
            phi_parameters_available = np.ones_like(phi_parameters_optimizable)
            if phi_parameters_optimizable.shape[0] == 2:
                if self.neu < ee_order and self.ned < ee_order:
                    phi_parameters_available[0] = False
                if self.neu + self.ned < ee_order:
                    phi_parameters_available[1] = False
            elif phi_parameters_optimizable.shape[0] == 3:
                if self.neu < ee_order:
                    phi_parameters_available[0] = False
                if self.neu + self.ned < ee_order:
                    phi_parameters_available[1] = False
                if self.ned < ee_order:
                    phi_parameters_available[2] = False
            self.phi_parameters_available.append(phi_parameters_available)

        for theta_parameters_optimizable in self.theta_parameters_optimizable:
            theta_parameters_available = np.ones_like(theta_parameters_optimizable)
            if theta_parameters_optimizable.shape[0] == 2:
                if self.neu < ee_order and self.ned < ee_order:
                    theta_parameters_available[0] = False
                if self.neu + self.ned < ee_order:
                    theta_parameters_available[1] = False
            elif theta_parameters_optimizable.shape[0] == 3:
                if self.neu < ee_order:
                    theta_parameters_available[0] = False
                if self.neu + self.ned < ee_order:
                    theta_parameters_available[1] = False
                if self.ned < ee_order:
                    theta_parameters_available[2] = False
            self.theta_parameters_available.append(theta_parameters_available)

        # striplet_exists rule (monte_carlo.f90 assign_spin_deps): the doubled spin
        # is always the lower index, so udd is never registered for a 2-spin system
        if self.omega_parameters.shape[0] == 1:
            if not (self.neu > 2 or self.ned > 2 or (self.neu > 1 and self.ned > 0)):
                self.omega_parameters_available[0] = False
        elif self.omega_parameters.shape[0] == 4:
            if self.neu < 3:
                self.omega_parameters_available[0] = False
            if self.neu < 2 or self.ned < 1:
                self.omega_parameters_available[1] = False
            self.omega_parameters_available[2] = False
            if self.ned < 3:
                self.omega_parameters_available[3] = False

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'ee_powers')
def backflow_ee_powers(self, e_vectors: np.ndarray):
    """Powers of e-e distances
    :param e_vectors: e-e vectors - array(nelec, nelec, 3)
    :return:
    """

    def impl(self, e_vectors: np.ndarray) -> np.ndarray:
        res = np.ones(shape=(e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(1, e_vectors.shape[0]):
            for j in range(i):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(1, self.max_ee_order):
                    res[i, j, k] = res[j, i, k] = r_ee**k
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'en_powers')
def backflow_en_powers(self, n_vectors: np.ndarray):
    """Powers of e-n distances
    :param n_vectors: e-n vectors - array(natom, nelec, 3)
    :return:
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        res = np.ones(shape=(n_vectors.shape[0], n_vectors.shape[1], self.max_en_order))
        for i in range(n_vectors.shape[0]):
            for j in range(n_vectors.shape[1]):
                r_eI = np.linalg.norm(n_vectors[i, j])
                for k in range(1, self.max_en_order):
                    res[i, j, k] = r_eI**k
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'ae_multiplier')
def backflow_ae_multiplier(self, n_vectors, n_powers):
    """Zeroing the backflow displacement at AE atoms."""

    def impl(self, n_vectors, n_powers):
        res = np.ones(shape=(2, self.neu + self.ned, 3))
        for i in range(n_vectors.shape[0]):
            Lg = self.ae_cutoff[i]
            for j in range(self.neu + self.ned):
                r = n_powers[i, j, 1]
                if r < Lg:
                    res[1, j] = (r / Lg) ** 2 * (6 - 8 * (r / Lg) + 3 * (r / Lg) ** 2)
        return res.reshape(2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'ae_multiplier_gradient')
def backflow_ae_multiplier_gradient(self, n_vectors, n_powers):
    """Zeroing the backflow displacement at AE atoms.
    Gradient of spherically symmetric function (in 3-D space) is:
        ∇(f) = df/dr * r_vec/r
    """

    def impl(self, n_vectors, n_powers):
        res = np.zeros(shape=(2, self.neu + self.ned, 3, self.neu + self.ned, 3))
        for i in range(n_vectors.shape[0]):
            Lg = self.ae_cutoff[i]
            for j in range(self.neu + self.ned):
                r_vec = n_vectors[i, j]
                r = n_powers[i, j, 1]
                if r < Lg:
                    res[1, j, :, j, :] = 12 * r_vec / Lg**2 * (1 - r / Lg) ** 2
        return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'ae_multiplier_laplacian')
def backflow_ae_multiplier_laplacian(self, n_vectors, n_powers):
    """Zeroing the backflow displacement at AE atoms.
    Laplace operator of spherically symmetric function (in 3-D space) is:
        ∇²(f) = d²f/dr² + 2/r * df/dr
    """

    def impl(self, n_vectors, n_powers):
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        for i in range(n_vectors.shape[0]):
            Lg = self.ae_cutoff[i]
            for j in range(self.neu + self.ned):
                r = n_powers[i, j, 1]
                if r < Lg:
                    res[1, j] = 12 / Lg**2 * (3 - 8 * (r / Lg) + 5 * (r / Lg) ** 2)
        return res.reshape(2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'eta_term')
def backflow_eta_term(self, e_powers, e_vectors):
    """
    :param e_vectors: e-e vectors
    :param e_powers: powers of e-e distances
    :return: displacements of electrons - array(nelec, 3)
    """

    def impl(self, e_powers, e_vectors):
        ae_cutoff_condition = 1
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res.reshape(2, (self.neu + self.ned) * 3)

        C = self.trunc
        parameters = self.eta_parameters
        k_max = parameters.shape[1]
        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                r_vec = e_vectors[e1, e2]
                r = e_powers[e1, e2, 1]
                eta_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % parameters.shape[0]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    poly = parameters[eta_set] @ e_powers[e1, e2, :k_max]
                    bf = (1 - r / L) ** C * poly * r_vec
                    res[ae_cutoff_condition, e1] += bf
                    res[ae_cutoff_condition, e2] -= bf
        return res.reshape(2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'mu_term')
def backflow_mu_term(self, n_powers, n_vectors):
    """
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    :return: displacements of electrons - array(2, nelec, 3)
    """

    def impl(self, n_powers, n_vectors):
        C = self.trunc
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        for parameters, L, mu_labels in zip(self.mu_parameters, self.mu_cutoff, self.mu_labels):
            k_max = parameters.shape[1]
            for label in mu_labels:
                for e1 in range(self.neu + self.ned):
                    r_vec = n_vectors[label, e1]
                    r = n_powers[label, e1, 1]
                    if r < L:
                        mu_set = int(e1 >= self.neu) % parameters.shape[0]
                        # cutoff_condition
                        # 0: AE cutoff exactly not applied
                        # 1: AE cutoff maybe applied
                        ae_cutoff_condition = int(r > self.ae_cutoff[label])
                        poly = parameters[mu_set] @ n_powers[label, e1, :k_max]
                        res[ae_cutoff_condition, e1] += (1 - r / L) ** C * poly * r_vec
        return res.reshape(2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'phi_term')
def backflow_phi_term(self, e_powers, n_powers, e_vectors, n_vectors):
    """
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :return: displacements of electrons - array(2, nelec, 3)
    """

    def impl(self, e_powers, n_powers, e_vectors, n_vectors):
        C = self.trunc
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        for phi_parameters, theta_parameters, L, phi_labels in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_labels):
            for label in phi_labels:
                for e1 in range(self.neu + self.ned):
                    for e2 in range(self.neu + self.ned):
                        if e1 == e2:
                            continue
                        r_e1I_vec = n_vectors[label, e1]
                        r_ee_vec = e_vectors[e1, e2]
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            phi_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % phi_parameters.shape[0]
                            phi_poly = theta_poly = 0.0
                            for m in range(phi_parameters.shape[1]):
                                for l in range(phi_parameters.shape[2]):
                                    for k in range(phi_parameters.shape[3]):
                                        poly = e_powers[e1, e2, m] * n_powers[label, e2, l] * n_powers[label, e1, k]
                                        phi_poly += phi_parameters[phi_set, m, l, k] * poly
                                        theta_poly += theta_parameters[phi_set, m, l, k] * poly
                            # cutoff_condition
                            # 0: AE cutoff exactly not applied
                            # 1: AE cutoff maybe applied
                            ae_cutoff_condition = int(r_e1I > self.ae_cutoff[label])
                            res[ae_cutoff_condition, e1] += (
                                (1 - r_e1I / L) ** C * (1 - r_e2I / L) ** C * (phi_poly * r_ee_vec + theta_poly * r_e1I_vec)
                            )
        return res.reshape(2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'omega_term')
def backflow_omega_term(self, e_powers, e_vectors):
    """
    :param e_powers: powers of e-e distances
    :param e_vectors: e-e vectors
    :return: displacements of electrons - array(2, nelec * 3)
    """

    def impl(self, e_powers, e_vectors):
        ae_cutoff_condition = 1
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        if not self.omega_cutoff.any():
            return res.reshape(2, (self.neu + self.ned) * 3)

        C = self.trunc
        parameters = self.omega_parameters
        for e1 in range(2, self.neu + self.ned):
            for e2 in range(1, e1):
                for e3 in range(e2):
                    # canonical triplet order (i, j, k) = (e3, e2, e1), i < j < k
                    r_ij = e_powers[e3, e2, 1]
                    r_jk = e_powers[e2, e1, 1]
                    r_ki = e_powers[e1, e3, 1]
                    omega_set = (int(e1 >= self.neu) + int(e2 >= self.neu) + int(e3 >= self.neu)) % parameters.shape[0]
                    L = self.omega_cutoff[omega_set % self.omega_cutoff.shape[0]]
                    if r_ij < L and r_jk < L and r_ki < L:
                        poly = 0.0
                        for l in range(parameters.shape[1]):
                            for m in range(parameters.shape[2]):
                                for n in range(parameters.shape[3]):
                                    poly += parameters[omega_set, l, m, n] * e_powers[e2, e1, l] * e_powers[e1, e3, m] * e_powers[e3, e2, n]
                        w = (1 - r_jk / L) ** C * (1 - r_ki / L) ** C * (1 - r_ij / L) ** C * poly
                        res[ae_cutoff_condition, e3] += w * (e_vectors[e3, e2] + e_vectors[e3, e1])
                        res[ae_cutoff_condition, e2] += w * (e_vectors[e2, e3] + e_vectors[e2, e1])
                        res[ae_cutoff_condition, e1] += w * (e_vectors[e1, e3] + e_vectors[e1, e2])
        return res.reshape(2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'eta_term_gradient')
def backflow_eta_term_gradient(self, e_powers, e_vectors):
    """
    :param e_vectors: e-e vectors
    :param e_powers: powers of e-e distances
    Gradient of spherically symmetric function (in 3-D space) is df/dr * (x, y, z)
    :return: partial derivatives of displacements of electrons - array(nelec * 3, nelec * 3)
    """

    def impl(self, e_powers, e_vectors):
        ae_cutoff_condition = 1
        res = np.zeros(shape=(2, self.neu + self.ned, 3, self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

        C = self.trunc
        parameters = self.eta_parameters
        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                r_vec = e_vectors[e1, e2]
                r = e_powers[e1, e2, 1]
                eta_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % parameters.shape[0]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    poly = poly_diff = 0
                    for k in range(parameters.shape[1]):
                        p = parameters[eta_set, k] * e_powers[e1, e2, k]
                        poly += p
                        poly_diff += p * k

                    bf = (1 - r / L) ** C * ((poly_diff - C * r / (L - r) * poly) * np.outer(r_vec, r_vec) / r**2 + poly * eye3)
                    res[ae_cutoff_condition, e1, :, e1, :] += bf
                    res[ae_cutoff_condition, e1, :, e2, :] -= bf
                    res[ae_cutoff_condition, e2, :, e1, :] -= bf
                    res[ae_cutoff_condition, e2, :, e2, :] += bf

        return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'mu_term_gradient')
def backflow_mu_term_gradient(self, n_powers, n_vectors):
    """
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    :return: partial derivatives of displacements of electrons - array(2, nelec * 3, nelec * 3)
    """

    def impl(self, n_powers, n_vectors):
        C = self.trunc
        res = np.zeros(shape=(2, self.neu + self.ned, 3, self.neu + self.ned, 3))
        for parameters, L, mu_labels in zip(self.mu_parameters, self.mu_cutoff, self.mu_labels):
            for label in mu_labels:
                for e1 in range(self.neu + self.ned):
                    r_vec = n_vectors[label, e1]
                    r = n_powers[label, e1, 1]
                    if r < L:
                        mu_set = int(e1 >= self.neu) % parameters.shape[0]
                        poly = poly_diff = 0.0
                        for k in range(parameters.shape[1]):
                            p = parameters[mu_set, k] * n_powers[label, e1, k]
                            poly += p
                            poly_diff += k * p
                        # cutoff_condition
                        # 0: AE cutoff exactly not applied
                        # 1: AE cutoff maybe applied
                        ae_cutoff_condition = int(r > self.ae_cutoff[label])
                        res[ae_cutoff_condition, e1, :, e1, :] += (1 - r / L) ** C * (
                            (poly_diff - C * r / (L - r) * poly) * np.outer(r_vec, r_vec) / r**2 + poly * eye3
                        )

        return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'phi_term_gradient')
def backflow_phi_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors):
    """
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    """

    def impl(self, e_powers, n_powers, e_vectors, n_vectors):
        C = self.trunc
        res = np.zeros(shape=(2, self.neu + self.ned, 3, self.neu + self.ned, 3))
        for phi_parameters, theta_parameters, L, phi_labels in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_labels):
            for label in phi_labels:
                for e1 in range(self.neu + self.ned):
                    for e2 in range(self.neu + self.ned):
                        if e1 == e2:
                            continue
                        r_e1I_vec = n_vectors[label, e1]
                        r_e2I_vec = n_vectors[label, e2]
                        r_ee_vec = r_e1I_vec - r_e2I_vec
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        r_ee = e_powers[e1, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            phi_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % phi_parameters.shape[0]
                            phi_poly = phi_poly_diff_e1I = phi_poly_diff_e2I = phi_poly_diff_ee = 0.0
                            theta_poly = theta_poly_diff_e1I = theta_poly_diff_e2I = theta_poly_diff_ee = 0.0
                            for m in range(phi_parameters.shape[1]):
                                for l in range(phi_parameters.shape[2]):
                                    for k in range(phi_parameters.shape[3]):
                                        phi_p = phi_parameters[phi_set, m, l, k]
                                        theta_p = theta_parameters[phi_set, m, l, k]
                                        poly = e_powers[e1, e2, m] * n_powers[label, e2, l] * n_powers[label, e1, k]
                                        phi_poly += poly * phi_p
                                        theta_poly += poly * theta_p
                                        poly_diff_e1I = k * poly
                                        phi_poly_diff_e1I += poly_diff_e1I * phi_p
                                        theta_poly_diff_e1I += poly_diff_e1I * theta_p
                                        poly_diff_e2I = l * poly
                                        phi_poly_diff_e2I += poly_diff_e2I * phi_p
                                        theta_poly_diff_e2I += poly_diff_e2I * theta_p
                                        poly_diff_ee = m * poly
                                        phi_poly_diff_ee += poly_diff_ee * phi_p
                                        theta_poly_diff_ee += poly_diff_ee * theta_p
                            # cutoff_condition
                            # 0: AE cutoff exactly not applied
                            # 1: AE cutoff maybe applied
                            ae_cutoff_condition = int(r_e1I > self.ae_cutoff[label])
                            cutoff = (1 - r_e1I / L) ** C * (1 - r_e2I / L) ** C
                            r_ee_r_ee = np.outer(r_ee_vec, r_ee_vec)
                            r_ee_r_e1I = np.outer(r_ee_vec, r_e1I_vec)
                            res[ae_cutoff_condition, e1, :, e1, :] += cutoff * (
                                (phi_poly_diff_e1I - C * r_e1I / (L - r_e1I) * phi_poly) * r_ee_r_e1I / r_e1I**2
                                + phi_poly_diff_ee * r_ee_r_ee / r_ee**2
                                + phi_poly * eye3
                                + (theta_poly_diff_e1I - C * r_e1I / (L - r_e1I) * theta_poly) * np.outer(r_e1I_vec, r_e1I_vec) / r_e1I**2
                                + theta_poly_diff_ee * r_ee_r_e1I.T / r_ee**2
                                + theta_poly * eye3
                            )
                            res[ae_cutoff_condition, e1, :, e2, :] += cutoff * (
                                (phi_poly_diff_e2I - C * r_e2I / (L - r_e2I) * phi_poly) * np.outer(r_ee_vec, r_e2I_vec) / r_e2I**2
                                - phi_poly_diff_ee * r_ee_r_ee / r_ee**2
                                - phi_poly * eye3
                                + (theta_poly_diff_e2I - C * r_e2I / (L - r_e2I) * theta_poly) * np.outer(r_e1I_vec, r_e2I_vec) / r_e2I**2
                                - theta_poly_diff_ee * r_ee_r_e1I.T / r_ee**2
                            )

        return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'omega_term_gradient')
def backflow_omega_term_gradient(self, e_powers, e_vectors):
    """
    :param e_powers: powers of e-e distances
    :param e_vectors: e-e vectors
    omega depends on the positions only through the three pair distances, so with
        A = r_j - r_k, B = r_k - r_i, C = r_i - r_j and a, b, c their lengths
        grad_i(omega) = -omega_b * B/b + omega_c * C/c   (cyclically for j, k)
    and the displacement u_p = 2*r_p - (the other two) has d(u_p)/d(r_q) = (3*delta_pq - 1) * I
    :return: partial derivatives of displacements of electrons - array(2, nelec * 3, nelec * 3)
    """

    def impl(self, e_powers, e_vectors):
        ae_cutoff_condition = 1
        res = np.zeros(shape=(2, self.neu + self.ned, 3, self.neu + self.ned, 3))
        if not self.omega_cutoff.any():
            return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

        C = self.trunc
        parameters = self.omega_parameters
        u = np.zeros(shape=(3, 3))
        d = np.zeros(shape=(3, 3))
        for e1 in range(2, self.neu + self.ned):
            for e2 in range(1, e1):
                for e3 in range(e2):
                    # canonical triplet order (i, j, k) = (e3, e2, e1), i < j < k
                    r_ij = e_powers[e3, e2, 1]
                    r_jk = e_powers[e2, e1, 1]
                    r_ki = e_powers[e1, e3, 1]
                    omega_set = (int(e1 >= self.neu) + int(e2 >= self.neu) + int(e3 >= self.neu)) % parameters.shape[0]
                    L = self.omega_cutoff[omega_set % self.omega_cutoff.shape[0]]
                    if r_ij < L and r_jk < L and r_ki < L:
                        poly = poly_jk = poly_ki = poly_ij = 0.0
                        for l in range(parameters.shape[1]):
                            for m in range(parameters.shape[2]):
                                for n in range(parameters.shape[3]):
                                    p = parameters[omega_set, l, m, n] * e_powers[e2, e1, l] * e_powers[e1, e3, m] * e_powers[e3, e2, n]
                                    poly += p
                                    poly_jk += p * l
                                    poly_ki += p * m
                                    poly_ij += p * n

                        f = (1 - r_jk / L) ** C * (1 - r_ki / L) ** C * (1 - r_ij / L) ** C
                        w = f * poly
                        w_jk = f * (poly_jk / r_jk - C / (L - r_jk) * poly)
                        w_ki = f * (poly_ki / r_ki - C / (L - r_ki) * poly)
                        w_ij = f * (poly_ij / r_ij - C / (L - r_ij) * poly)

                        jk_vec = e_vectors[e2, e1]
                        ki_vec = e_vectors[e1, e3]
                        ij_vec = e_vectors[e3, e2]
                        u[0] = ij_vec - ki_vec
                        u[1] = jk_vec - ij_vec
                        u[2] = ki_vec - jk_vec
                        d[0] = w_ij * ij_vec / r_ij - w_ki * ki_vec / r_ki
                        d[1] = w_jk * jk_vec / r_jk - w_ij * ij_vec / r_ij
                        d[2] = w_ki * ki_vec / r_ki - w_jk * jk_vec / r_jk

                        triplet = (e3, e2, e1)
                        for p1 in range(3):
                            for p2 in range(3):
                                bf = np.outer(u[p1], d[p2])
                                if p1 == p2:
                                    bf += 2 * w * eye3
                                else:
                                    bf -= w * eye3
                                res[ae_cutoff_condition, triplet[p1], :, triplet[p2], :] += bf

        return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'eta_term_laplacian')
def backflow_eta_term_laplacian(self, e_powers, e_vectors):
    """
    :param e_vectors: e-e vectors
    :param e_powers: powers of e-e distances
    Laplace operator of spherically symmetric function (in 3-D space) is
        ∇²(f) = d²f/dr² + 2/r * df/dr
    :return: vector laplacian - array(nelec * 3)
    """

    def impl(self, e_powers, e_vectors):
        ae_cutoff_condition = 1
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res.reshape(2, (self.neu + self.ned) * 3)

        C = self.trunc
        parameters = self.eta_parameters
        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                r_vec = e_vectors[e1, e2]
                r = e_powers[e1, e2, 1]
                eta_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % parameters.shape[0]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    poly = 0.0
                    for k in range(parameters.shape[1]):
                        poly += (
                            (C * (C - 1) / (L - r) ** 2 - 2 * C * (k + 2) / r / (L - r) + k * (k + 3) / r**2)
                            * parameters[eta_set, k]
                            * e_powers[e1, e2, k]
                        )
                    # double because we add half of the matrix
                    bf = 2 * (1 - r / L) ** C * poly * r_vec
                    res[ae_cutoff_condition, e1] += bf
                    res[ae_cutoff_condition, e2] -= bf

        return res.reshape(2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'mu_term_laplacian')
def backflow_mu_term_laplacian(self, n_powers, n_vectors):
    """
    :param n_powers: powers of e-n distances
    :param n_vectors: e-n vectors
    """

    def impl(self, n_powers, n_vectors):
        C = self.trunc
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        for parameters, L, mu_labels in zip(self.mu_parameters, self.mu_cutoff, self.mu_labels):
            for label in mu_labels:
                for e1 in range(self.neu + self.ned):
                    r_vec = n_vectors[label, e1]
                    r = n_powers[label, e1, 1]
                    if r < L:
                        mu_set = int(e1 >= self.neu) % parameters.shape[0]
                        poly = 0.0
                        for k in range(parameters.shape[1]):
                            poly += (
                                (C * (C - 1) / (L - r) ** 2 - 2 * C * (k + 2) / r / (L - r) + k * (k + 3) / r**2)
                                * parameters[mu_set, k]
                                * n_powers[label, e1, k]
                            )
                        # cutoff_condition
                        # 0: AE cutoff exactly not applied
                        # 1: AE cutoff maybe applied
                        ae_cutoff_condition = int(r > self.ae_cutoff[label])
                        res[ae_cutoff_condition, e1] += (1 - r / L) ** C * poly * r_vec

        return res.reshape(2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'phi_term_laplacian')
def backflow_phi_term_laplacian(self, e_powers, n_powers, e_vectors, n_vectors):
    """
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    phi-term is a product of two spherically symmetric functions f(r_eI) and g(r_ee) so using
        ∇²(f*g) = ∇²(f)*g + 2*∇(f)*∇(g) + f*∇²(g)
    Laplace operator of spherically symmetric function (in 3-D space) is
        ∇²(f) = d²f/dr² + 2/r * df/dr
    :return: vector laplacian - array(2, nelec * 3)
    """

    def impl(self, e_powers, n_powers, e_vectors, n_vectors):
        C = self.trunc
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        for phi_parameters, theta_parameters, L, phi_labels in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_labels):
            for label in phi_labels:
                for e1 in range(self.neu + self.ned):
                    for e2 in range(self.neu + self.ned):
                        if e1 == e2:
                            continue
                        r_e1I_vec = n_vectors[label, e1]
                        r_e2I_vec = n_vectors[label, e2]
                        r_ee_vec = e_vectors[e1, e2]
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        r_ee = e_powers[e1, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            phi_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % phi_parameters.shape[0]
                            phi_poly = phi_poly_diff_e1I = phi_poly_diff_e2I = phi_poly_diff_ee = 0.0
                            phi_poly_diff_e1I_2 = phi_poly_diff_e2I_2 = phi_poly_diff_ee_2 = 0.0
                            phi_poly_diff_e1I_ee = phi_poly_diff_e2I_ee = 0.0
                            theta_poly = theta_poly_diff_e1I = theta_poly_diff_e2I = theta_poly_diff_ee = 0.0
                            theta_poly_diff_e1I_2 = theta_poly_diff_e2I_2 = theta_poly_diff_ee_2 = 0.0
                            theta_poly_diff_e1I_ee = theta_poly_diff_e2I_ee = 0.0
                            cutoff_diff_e1I = C * r_e1I / (L - r_e1I)
                            cutoff_diff_e2I = C * r_e2I / (L - r_e2I)
                            cutoff_diff_e1I_2 = C * (C - 1) * r_e1I**2 / (L - r_e1I) ** 2
                            cutoff_diff_e2I_2 = C * (C - 1) * r_e2I**2 / (L - r_e2I) ** 2
                            for m in range(phi_parameters.shape[1]):
                                for l in range(phi_parameters.shape[2]):
                                    for k in range(phi_parameters.shape[3]):
                                        phi_p = phi_parameters[phi_set, m, l, k]
                                        theta_p = theta_parameters[phi_set, m, l, k]
                                        poly = e_powers[e1, e2, m] * n_powers[label, e2, l] * n_powers[label, e1, k]
                                        phi_poly += poly * phi_p
                                        theta_poly += poly * theta_p
                                        poly_diff_e1I = k * poly
                                        phi_poly_diff_e1I += poly_diff_e1I * phi_p
                                        theta_poly_diff_e1I += poly_diff_e1I * theta_p
                                        poly_diff_e2I = l * poly
                                        phi_poly_diff_e2I += poly_diff_e2I * phi_p
                                        theta_poly_diff_e2I += poly_diff_e2I * theta_p
                                        poly_diff_ee = m * poly
                                        phi_poly_diff_ee += poly_diff_ee * phi_p
                                        theta_poly_diff_ee += poly_diff_ee * theta_p
                                        poly_diff_e1I_2 = k * (k + 1) * poly
                                        phi_poly_diff_e1I_2 += poly_diff_e1I_2 * phi_p
                                        theta_poly_diff_e1I_2 += poly_diff_e1I_2 * theta_p
                                        poly_diff_e2I_2 = l * (l + 1) * poly
                                        phi_poly_diff_e2I_2 += poly_diff_e2I_2 * phi_p
                                        theta_poly_diff_e2I_2 += poly_diff_e2I_2 * theta_p
                                        poly_diff_ee_2 = m * (m + 1) * poly
                                        phi_poly_diff_ee_2 += poly_diff_ee_2 * phi_p
                                        theta_poly_diff_ee_2 += poly_diff_ee_2 * theta_p
                                        poly_diff_e1I_ee = k * m * poly
                                        phi_poly_diff_e1I_ee += poly_diff_e1I_ee * phi_p
                                        theta_poly_diff_e1I_ee += poly_diff_e1I_ee * theta_p
                                        poly_diff_e2I_ee = l * m * poly
                                        phi_poly_diff_e2I_ee += poly_diff_e2I_ee * phi_p
                                        theta_poly_diff_e2I_ee += poly_diff_e2I_ee * theta_p

                            phi_ee = (
                                phi_poly * cutoff_diff_e1I_2 / r_e1I**2
                                - 2 * (phi_poly_diff_e1I + phi_poly) * cutoff_diff_e1I / r_e1I**2
                                + phi_poly_diff_e1I_2 / r_e1I**2
                                + phi_poly * cutoff_diff_e2I_2 / r_e2I**2
                                - 2 * (phi_poly_diff_e2I + phi_poly) * cutoff_diff_e2I / r_e2I**2
                                + phi_poly_diff_e2I_2 / r_e2I**2
                                + 2 * phi_poly_diff_ee_2 / r_ee**2
                                + 4 * phi_poly_diff_ee / r_ee**2
                            )
                            phi_dot_product = 2 * (
                                (phi_poly_diff_e1I - phi_poly * cutoff_diff_e1I) * r_e1I_vec / r_e1I**2
                                - (phi_poly_diff_e2I - phi_poly * cutoff_diff_e2I) * r_e2I_vec / r_e2I**2
                                + (phi_poly_diff_e1I_ee - phi_poly_diff_ee * cutoff_diff_e1I) * r_ee_vec * (r_ee_vec @ r_e1I_vec) / r_e1I**2 / r_ee**2
                                - (phi_poly_diff_e2I_ee - phi_poly_diff_ee * cutoff_diff_e2I) * r_ee_vec * (r_ee_vec @ r_e2I_vec) / r_e2I**2 / r_ee**2
                            )
                            theta_e1I = (
                                theta_poly * cutoff_diff_e1I_2 / r_e1I**2
                                - 2 * (theta_poly_diff_e1I + theta_poly) * cutoff_diff_e1I / r_e1I**2
                                + theta_poly_diff_e1I_2 / r_e1I**2
                                + theta_poly * cutoff_diff_e2I_2 / r_e2I**2
                                - 2 * (theta_poly_diff_e2I + theta_poly) * cutoff_diff_e2I / r_e2I**2
                                + theta_poly_diff_e2I_2 / r_e2I**2
                                + 2 * theta_poly_diff_ee_2 / r_ee**2
                                + 2 * (theta_poly_diff_e1I - theta_poly * cutoff_diff_e1I) / r_e1I**2
                            )
                            theta_dot_product = (
                                2
                                * (
                                    (theta_poly_diff_e1I_ee - theta_poly_diff_ee * cutoff_diff_e1I) * r_e1I_vec * (r_e1I_vec @ r_ee_vec) / r_e1I**2
                                    - (theta_poly_diff_e2I_ee - theta_poly_diff_ee * cutoff_diff_e2I) * r_e1I_vec * (r_e2I_vec @ r_ee_vec) / r_e2I**2
                                    + theta_poly_diff_ee * r_ee_vec
                                )
                                / r_ee**2
                            )
                            # cutoff_condition
                            # 0: AE cutoff exactly not applied
                            # 1: AE cutoff maybe applied
                            ae_cutoff_condition = int(r_e1I > self.ae_cutoff[label])
                            cutoff = (1 - r_e1I / L) ** C * (1 - r_e2I / L) ** C
                            res[ae_cutoff_condition, e1] += cutoff * (phi_ee * r_ee_vec + phi_dot_product + theta_e1I * r_e1I_vec + theta_dot_product)

        return res.reshape(2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'omega_term_laplacian')
def backflow_omega_term_laplacian(self, e_powers, e_vectors):
    """
    :param e_powers: powers of e-e distances
    :param e_vectors: e-e vectors
    Each electron of a triplet moves only two of the three pair distances, so e.g. for i
        lap_i(omega) = omega_bb + omega_cc - 2 * omega_bc * (B.C)/(b*c) + 2*omega_b/b + 2*omega_c/c
    Since omega is translationally invariant the first-derivative terms collapse:
        sum_q lap_q(u_p * omega) = lap(omega) * u_p + 6 * grad_p(omega)
    :return: vector laplacian - array(2, nelec * 3)
    """

    def impl(self, e_powers, e_vectors):
        ae_cutoff_condition = 1
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        if not self.omega_cutoff.any():
            return res.reshape(2, (self.neu + self.ned) * 3)

        C = self.trunc
        parameters = self.omega_parameters
        for e1 in range(2, self.neu + self.ned):
            for e2 in range(1, e1):
                for e3 in range(e2):
                    # canonical triplet order (i, j, k) = (e3, e2, e1), i < j < k
                    r_ij = e_powers[e3, e2, 1]
                    r_jk = e_powers[e2, e1, 1]
                    r_ki = e_powers[e1, e3, 1]
                    omega_set = (int(e1 >= self.neu) + int(e2 >= self.neu) + int(e3 >= self.neu)) % parameters.shape[0]
                    L = self.omega_cutoff[omega_set % self.omega_cutoff.shape[0]]
                    if r_ij < L and r_jk < L and r_ki < L:
                        poly = poly_jk = poly_ki = poly_ij = 0.0
                        poly_jk_2 = poly_ki_2 = poly_ij_2 = 0.0
                        poly_jk_ki = poly_jk_ij = poly_ki_ij = 0.0
                        for l in range(parameters.shape[1]):
                            for m in range(parameters.shape[2]):
                                for n in range(parameters.shape[3]):
                                    p = parameters[omega_set, l, m, n] * e_powers[e2, e1, l] * e_powers[e1, e3, m] * e_powers[e3, e2, n]
                                    poly += p
                                    poly_jk += p * l
                                    poly_ki += p * m
                                    poly_ij += p * n
                                    poly_jk_2 += p * l * (l - 1)
                                    poly_ki_2 += p * m * (m - 1)
                                    poly_ij_2 += p * n * (n - 1)
                                    poly_jk_ki += p * l * m
                                    poly_jk_ij += p * l * n
                                    poly_ki_ij += p * m * n

                        f = (1 - r_jk / L) ** C * (1 - r_ki / L) ** C * (1 - r_ij / L) ** C
                        g_jk = -C / (L - r_jk)
                        g_ki = -C / (L - r_ki)
                        g_ij = -C / (L - r_ij)
                        h = C * (C - 1)
                        w_jk = f * (poly_jk / r_jk + g_jk * poly)
                        w_ki = f * (poly_ki / r_ki + g_ki * poly)
                        w_ij = f * (poly_ij / r_ij + g_ij * poly)
                        w_jk_2 = f * (poly_jk_2 / r_jk**2 + 2 * g_jk * poly_jk / r_jk + h / (L - r_jk) ** 2 * poly)
                        w_ki_2 = f * (poly_ki_2 / r_ki**2 + 2 * g_ki * poly_ki / r_ki + h / (L - r_ki) ** 2 * poly)
                        w_ij_2 = f * (poly_ij_2 / r_ij**2 + 2 * g_ij * poly_ij / r_ij + h / (L - r_ij) ** 2 * poly)
                        w_jk_ki = f * (poly_jk_ki / r_jk / r_ki + g_jk * poly_ki / r_ki + g_ki * poly_jk / r_jk + g_jk * g_ki * poly)
                        w_jk_ij = f * (poly_jk_ij / r_jk / r_ij + g_jk * poly_ij / r_ij + g_ij * poly_jk / r_jk + g_jk * g_ij * poly)
                        w_ki_ij = f * (poly_ki_ij / r_ki / r_ij + g_ki * poly_ij / r_ij + g_ij * poly_ki / r_ki + g_ki * g_ij * poly)

                        jk_vec = e_vectors[e2, e1]
                        ki_vec = e_vectors[e1, e3]
                        ij_vec = e_vectors[e3, e2]
                        # every pair distance is moved by exactly two of the three electrons
                        lap = (
                            2 * (w_jk_2 + w_ki_2 + w_ij_2)
                            - 2 * w_ki_ij * (ki_vec @ ij_vec) / r_ki / r_ij
                            - 2 * w_jk_ij * (jk_vec @ ij_vec) / r_jk / r_ij
                            - 2 * w_jk_ki * (jk_vec @ ki_vec) / r_jk / r_ki
                            + 4 * (w_jk / r_jk + w_ki / r_ki + w_ij / r_ij)
                        )
                        d_i = w_ij * ij_vec / r_ij - w_ki * ki_vec / r_ki
                        d_j = w_jk * jk_vec / r_jk - w_ij * ij_vec / r_ij
                        d_k = w_ki * ki_vec / r_ki - w_jk * jk_vec / r_jk
                        res[ae_cutoff_condition, e3] += lap * (ij_vec - ki_vec) + 6 * d_i
                        res[ae_cutoff_condition, e2] += lap * (jk_vec - ij_vec) + 6 * d_j
                        res[ae_cutoff_condition, e1] += lap * (ki_vec - jk_vec) + 6 * d_k

        return res.reshape(2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'value')
def backflow_value(self, e_vectors, n_vectors):
    """Backflow displacements
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return: backflow displacement array(nelec * 3)
    """

    def impl(self, e_vectors, n_vectors):
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        eta_term = self.eta_term(e_powers, e_vectors)
        mu_term = self.mu_term(n_powers, n_vectors)
        phi_term = self.phi_term(e_powers, n_powers, e_vectors, n_vectors)
        omega_term = self.omega_term(e_powers, e_vectors)

        ae_value = eta_term + mu_term + phi_term + omega_term
        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)

        return np.sum(ae_value * ae_multiplier, axis=0).reshape((self.neu + self.ned), 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'gradient')
def backflow_gradient(self, e_vectors, n_vectors):
    """Gradient with respect to e-coordinates
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors, n_vectors):
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        eta_term = self.eta_term(e_powers, e_vectors)
        mu_term = self.mu_term(n_powers, n_vectors)
        phi_term = self.phi_term(e_powers, n_powers, e_vectors, n_vectors)
        omega_term = self.omega_term(e_powers, e_vectors)

        eta_term_gradient = self.eta_term_gradient(e_powers, e_vectors)
        mu_term_gradient = self.mu_term_gradient(n_powers, n_vectors)
        phi_term_gradient = self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_gradient = self.omega_term_gradient(e_powers, e_vectors)

        ae_value = eta_term + mu_term + phi_term + omega_term
        ae_gradient = eta_term_gradient + mu_term_gradient + phi_term_gradient + omega_term_gradient

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)
        ae_multiplier_gradient = self.ae_multiplier_gradient(n_vectors, n_powers)

        value = np.sum(ae_value * ae_multiplier, axis=0).reshape((self.neu + self.ned), 3)

        gradient = np.sum(ae_multiplier_gradient * np.expand_dims(ae_value, 2) + ae_gradient * np.expand_dims(ae_multiplier, 2), axis=0) + np.eye(
            (self.neu + self.ned) * 3
        )

        return gradient, value

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'laplacian')
def backflow_laplacian(self, e_vectors, n_vectors):
    """Backflow laplacian, gradient, value
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors, n_vectors):
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        eta_term = self.eta_term(e_powers, e_vectors)
        mu_term = self.mu_term(n_powers, n_vectors)
        phi_term = self.phi_term(e_powers, n_powers, e_vectors, n_vectors)
        omega_term = self.omega_term(e_powers, e_vectors)

        eta_term_gradient = self.eta_term_gradient(e_powers, e_vectors)
        mu_term_gradient = self.mu_term_gradient(n_powers, n_vectors)
        phi_term_gradient = self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_gradient = self.omega_term_gradient(e_powers, e_vectors)

        eta_term_laplacian = self.eta_term_laplacian(e_powers, e_vectors)
        mu_term_laplacian = self.mu_term_laplacian(n_powers, n_vectors)
        phi_term_laplacian = self.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_laplacian = self.omega_term_laplacian(e_powers, e_vectors)

        ae_value = eta_term + mu_term + phi_term + omega_term
        ae_gradient = eta_term_gradient + mu_term_gradient + phi_term_gradient + omega_term_gradient
        ae_laplacian = eta_term_laplacian + mu_term_laplacian + phi_term_laplacian + omega_term_laplacian

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)
        ae_multiplier_gradient = self.ae_multiplier_gradient(n_vectors, n_powers)
        ae_multiplier_laplacian = self.ae_multiplier_laplacian(n_vectors, n_powers)

        value = np.sum(ae_value * ae_multiplier, axis=0).reshape((self.neu + self.ned), 3)

        gradient = np.sum(ae_multiplier_gradient * np.expand_dims(ae_value, 2) + ae_gradient * np.expand_dims(ae_multiplier, 2), axis=0) + np.eye(
            (self.neu + self.ned) * 3
        )

        laplacian = np.sum(
            ae_multiplier_laplacian * ae_value + 2 * (ae_gradient * ae_multiplier_gradient).sum(axis=-1) + ae_laplacian * ae_multiplier, axis=0
        )

        return laplacian, gradient, value

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'fix_eta_parameters')
def backflow_fix_eta_parameters(self):
    """Fix eta-term dependent parameters"""

    def impl(self):
        C = self.trunc
        L = self.eta_cutoff[0]
        self.eta_parameters[0, 1] = C * self.eta_parameters[0, 0] / L
        if self.eta_parameters.shape[0] == 3:
            L = self.eta_cutoff[2] or self.eta_cutoff[0]
            self.eta_parameters[2, 1] = C * self.eta_parameters[2, 0] / L

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'fix_mu_parameters')
def backflow_fix_mu_parameters(self):
    """Fix mu-term dependent parameters"""

    def impl(self):
        C = self.trunc
        for mu_parameters, L, mu_cusp in zip(self.mu_parameters, self.mu_cutoff, self.mu_cusp):
            if mu_cusp:
                # AE atoms (d0,I = 0; Lμ,I * d1,I = C * d0,I)
                mu_parameters[:, 0:2] = 0
            else:
                # PP atoms (Lμ,I * d1,I = C * d0,I)
                mu_parameters[:, 1] = C * mu_parameters[:, 0] / L

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'fix_phi_parameters')
def backflow_fix_phi_parameters(self):
    """Fix phi-term dependent parameters"""

    def impl(self):
        for phi_parameters, theta_parameters, phi_cutoff, phi_cusp, phi_irrotational in zip(
            self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_cusp, self.phi_irrotational
        ):
            for spin_dep in range(phi_parameters.shape[0]):
                c, _ = construct_c_matrix(self.trunc, phi_parameters, theta_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational)
                c, pivot_positions = rref(c)
                c = c[: pivot_positions.size, :]

                b = np.zeros((c.shape[0],))
                p = 0
                for m in range(phi_parameters.shape[1]):
                    for l in range(phi_parameters.shape[2]):
                        for k in range(phi_parameters.shape[3]):
                            if p not in pivot_positions:
                                for temp in range(c.shape[0]):
                                    b[temp] -= c[temp, p] * phi_parameters[spin_dep, m, l, k]
                            p += 1

                for m in range(phi_parameters.shape[1]):
                    for l in range(phi_parameters.shape[2]):
                        for k in range(phi_parameters.shape[3]):
                            if p not in pivot_positions:
                                for temp in range(c.shape[0]):
                                    b[temp] -= c[temp, p] * theta_parameters[spin_dep, m, l, k]
                            p += 1

                x = np.linalg.solve(c[:, pivot_positions], b)

                p = 0
                temp = 0
                for m in range(phi_parameters.shape[1]):
                    for l in range(phi_parameters.shape[2]):
                        for k in range(phi_parameters.shape[3]):
                            if temp in pivot_positions:
                                phi_parameters[spin_dep, m, l, k] = x[p]
                                p += 1
                            temp += 1

                for m in range(phi_parameters.shape[1]):
                    for l in range(phi_parameters.shape[2]):
                        for k in range(phi_parameters.shape[3]):
                            if temp in pivot_positions:
                                theta_parameters[spin_dep, m, l, k] = x[p]
                                p += 1
                            temp += 1

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'fix_omega_parameters')
def backflow_fix_omega_parameters(self):
    """Fix omega-term dependent parameters"""

    def impl(self):
        for spin_dep in range(self.omega_parameters.shape[0]):
            if not self.omega_parameters_available[spin_dep].any():
                continue
            omega_cutoff = self.omega_cutoff[spin_dep % self.omega_cutoff.shape[0]]
            c, rep, rep_indices = construct_omega_folded_matrix(self.trunc, self.omega_parameters, omega_cutoff, spin_dep)
            c, pivot_positions = rref(c)

            # the matrix is in reduced form, so the pivot submatrix is the identity and no
            # linear system has to be solved
            x = np.zeros(shape=(rep_indices.shape[0],))
            for q in range(rep_indices.shape[0]):
                x[q] = self.omega_parameters[spin_dep, rep_indices[q, 0], rep_indices[q, 1], rep_indices[q, 2]]
            for p in pivot_positions:
                x[p] = 0
            for temp in range(pivot_positions.size):
                p = pivot_positions[temp]
                res = 0.0
                for q in range(c.shape[1]):
                    if q != p:
                        res -= c[temp, q] * x[q]
                x[p] = res

            for n in range(self.omega_parameters.shape[3]):
                for m in range(self.omega_parameters.shape[2]):
                    for l in range(self.omega_parameters.shape[1]):
                        self.omega_parameters[spin_dep, l, m, n] = x[rep[l, m, n]]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'get_parameters_mask')
def backflow_get_parameters_mask(self):
    """Optimizable mask of each parameter."""

    def impl(self) -> np.ndarray:
        res = []
        if self.eta_cutoff.any():
            for eta_cutoff, eta_cutoff_optimizable in zip(self.eta_cutoff, self.eta_cutoff_optimizable):
                if eta_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(1)
            for j1 in range(self.eta_parameters.shape[0]):
                for j2 in range(self.eta_parameters.shape[1]):
                    if self.eta_parameters_available[j1, j2]:
                        res.append(self.eta_parameters_optimizable[j1, j2])

        if self.mu_cutoff.any():
            for i, (mu_parameters, mu_parameters_optimizable, mu_cutoff, mu_cutoff_optimizable, mu_parameters_available) in enumerate(
                zip(self.mu_parameters, self.mu_parameters_optimizable, self.mu_cutoff, self.mu_cutoff_optimizable, self.mu_parameters_available)
            ):
                if mu_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(1)
                for j1 in range(mu_parameters.shape[0]):
                    for j2 in range(mu_parameters.shape[1]):
                        if mu_parameters_available[j1, j2]:
                            res.append(mu_parameters_optimizable[j1, j2])

        if self.phi_cutoff.any():
            for i, (
                phi_parameters,
                phi_parameters_optimizable,
                phi_parameters_available,
                theta_parameters_optimizable,
                theta_parameters_available,
                phi_cutoff,
                phi_cutoff_optimizable,
            ) in enumerate(
                zip(
                    self.phi_parameters,
                    self.phi_parameters_optimizable,
                    self.phi_parameters_available,
                    self.theta_parameters_optimizable,
                    self.theta_parameters_available,
                    self.phi_cutoff,
                    self.phi_cutoff_optimizable,
                )
            ):
                if phi_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(1)
                for j1 in range(phi_parameters.shape[0]):
                    for j2 in range(phi_parameters.shape[1]):
                        for j3 in range(phi_parameters.shape[2]):
                            for j4 in range(phi_parameters.shape[3]):
                                if phi_parameters_available[j1, j2, j3, j4]:
                                    res.append(phi_parameters_optimizable[j1, j2, j3, j4])

                    for j2 in range(phi_parameters.shape[1]):
                        for j3 in range(phi_parameters.shape[2]):
                            for j4 in range(phi_parameters.shape[3]):
                                if theta_parameters_available[j1, j2, j3, j4]:
                                    res.append(theta_parameters_optimizable[j1, j2, j3, j4])

        if self.omega_cutoff.any():
            for omega_cutoff_optimizable in self.omega_cutoff_optimizable:
                if omega_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(1)
            for j1 in range(self.omega_parameters.shape[0]):
                for j2 in range(self.omega_parameters.shape[1]):
                    for j3 in range(self.omega_parameters.shape[2]):
                        for j4 in range(self.omega_parameters.shape[3]):
                            if self.omega_parameters_available[j1, j2, j3, j4]:
                                res.append(self.omega_parameters_optimizable[j1, j2, j3, j4])

        for ae_cutoff_optimizable in self.ae_cutoff_optimizable:
            if ae_cutoff_optimizable and self.cutoffs_optimizable:
                res.append(1)

        return np.array(res)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'get_parameters_scale')
def backflow_get_parameters_scale(self, all_parameters):
    """Characteristic scale of each variable. Setting x_scale is equivalent
    to reformulating the problem in scaled variables xs = x / x_scale.
    An alternative view is that the size of a trust region along j-th
    dimension is proportional to x_scale[j].
    The purpose of this method is to reformulate the optimization problem
    with dimensionless variables having only one dimensional parameter - cutoff length.
    """

    def impl(self, all_parameters):
        res = []
        ne = self.neu + self.ned
        if self.eta_cutoff.any():
            for eta_cutoff, eta_cutoff_optimizable in zip(self.eta_cutoff, self.eta_cutoff_optimizable):
                if eta_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(1)
            for j1 in range(self.eta_parameters.shape[0]):
                for j2 in range(self.eta_parameters.shape[1]):
                    if (self.eta_parameters_optimizable[j1, j2] or all_parameters) and self.eta_parameters_available[j1, j2]:
                        res.append(2 / self.eta_cutoff[0] ** j2 / ne**2)

        if self.mu_cutoff.any():
            for i, (mu_parameters, mu_parameters_optimizable, mu_cutoff, mu_cutoff_optimizable, mu_parameters_available) in enumerate(
                zip(self.mu_parameters, self.mu_parameters_optimizable, self.mu_cutoff, self.mu_cutoff_optimizable, self.mu_parameters_available)
            ):
                if mu_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(1)
                for j1 in range(mu_parameters.shape[0]):
                    for j2 in range(mu_parameters.shape[1]):
                        if (mu_parameters_optimizable[j1, j2] or all_parameters) and mu_parameters_available[j1, j2]:
                            res.append(1 / mu_cutoff**j2 / ne)

        if self.phi_cutoff.any():
            for i, (
                phi_parameters,
                phi_parameters_optimizable,
                phi_parameters_available,
                theta_parameters_optimizable,
                theta_parameters_available,
                phi_cutoff,
                phi_cutoff_optimizable,
            ) in enumerate(
                zip(
                    self.phi_parameters,
                    self.phi_parameters_optimizable,
                    self.phi_parameters_available,
                    self.theta_parameters_optimizable,
                    self.theta_parameters_available,
                    self.phi_cutoff,
                    self.phi_cutoff_optimizable,
                )
            ):
                if phi_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(1)
                for j1 in range(phi_parameters.shape[0]):
                    for j2 in range(phi_parameters.shape[1]):
                        for j3 in range(phi_parameters.shape[2]):
                            for j4 in range(phi_parameters.shape[3]):
                                if (phi_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and phi_parameters_available[j1, j2, j3, j4]:
                                    res.append(2 / phi_cutoff ** (j2 + j3 + j4) / ne**3)

                    for j2 in range(phi_parameters.shape[1]):
                        for j3 in range(phi_parameters.shape[2]):
                            for j4 in range(phi_parameters.shape[3]):
                                if (theta_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and theta_parameters_available[j1, j2, j3, j4]:
                                    res.append(2 / phi_cutoff ** (j2 + j3 + j4) / ne**3)

        if self.omega_cutoff.any():
            for omega_cutoff_optimizable in self.omega_cutoff_optimizable:
                if omega_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(1)
            for j1 in range(self.omega_parameters.shape[0]):
                omega_cutoff = self.omega_cutoff[j1 % self.omega_cutoff.shape[0]]
                for j2 in range(self.omega_parameters.shape[1]):
                    for j3 in range(self.omega_parameters.shape[2]):
                        for j4 in range(self.omega_parameters.shape[3]):
                            if (self.omega_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and self.omega_parameters_available[
                                j1, j2, j3, j4
                            ]:
                                res.append(6 / omega_cutoff ** (j2 + j3 + j4) / ne**3)

        for ae_cutoff_optimizable in self.ae_cutoff_optimizable:
            if ae_cutoff_optimizable and self.cutoffs_optimizable:
                res.append(1)

        return np.array(res)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'get_parameters_constraints')
def backflow_get_parameters_constraints(self):
    """Returns parameters in the following order
    eta-cutoff, eta-linear parameters,
    for every mu-set: mu-cutoff, mu-linear parameters,
    for every phi/theta-set: phi-cutoff, phi-linear parameters, theta-linear parameters.
    :return:
    """

    def impl(self):
        a_list = []
        b_list = []

        if self.eta_cutoff.any():
            # c0*C - c1*L = 0 only for like-spin electrons
            eta_spin_deps = [0]
            if self.eta_parameters.shape[0] == 2:
                eta_spin_deps = [0, 1]
                if self.neu < 2 and self.ned < 2:
                    eta_spin_deps = [x for x in eta_spin_deps if x != 0]
                if self.neu + self.ned < 2:
                    eta_spin_deps = [x for x in eta_spin_deps if x != 1]
            elif self.eta_parameters.shape[0] == 3:
                eta_spin_deps = [0, 1, 2]
                if self.neu < 2:
                    eta_spin_deps = [x for x in eta_spin_deps if x != 0]
                if self.neu + self.ned < 2:
                    eta_spin_deps = [x for x in eta_spin_deps if x != 1]
                if self.ned < 2:
                    eta_spin_deps = [x for x in eta_spin_deps if x != 2]

            eta_list = []
            eta_cutoff_matrix = []
            for spin_dep in eta_spin_deps:
                # e-e term is affected by constraints only for like-spin electrons
                if spin_dep in (0, 2):
                    eta_matrix = np.zeros(shape=(1, self.eta_parameters.shape[1]))
                    eta_matrix[0, 0] = self.trunc
                    eta_matrix[0, 1] = -self.eta_cutoff[spin_dep]
                    eta_list.append(eta_matrix)
                    b_list.append(0)
                    for _ in range(self.eta_cutoff_optimizable.sum()):
                        eta_cutoff_matrix.append(self.eta_parameters[spin_dep, 1])
                else:
                    # no constrains
                    eta_matrix = np.zeros(shape=(0, self.eta_parameters.shape[1]))
                    eta_list.append(eta_matrix)
            eta_block = block_diag(eta_list)
            if self.eta_cutoff_optimizable.any() and self.cutoffs_optimizable:
                eta_block = np.hstack(
                    (
                        # FIXME: check if two Cut-off radii
                        -np.array(eta_cutoff_matrix).reshape(-1, self.eta_cutoff_optimizable.sum()),
                        eta_block,
                    )
                )
            a_list.append(eta_block)

        for mu_parameters, mu_cutoff, mu_cutoff_optimizable, mu_cusp in zip(
            self.mu_parameters, self.mu_cutoff, self.mu_cutoff_optimizable, self.mu_cusp
        ):
            if mu_cusp:
                # AE atoms (d0,I = 0; Lμ,I * d1,I = C * d0,I) after differentiation on variables: d0, d1, L
                # -d1 * dL + С * d(d0) - L * d(d1) = 0
                mu_matrix = np.zeros(shape=(2, mu_parameters.shape[1]))
                mu_matrix[0, 0] = 1
                mu_matrix[1, 0] = self.trunc
                mu_matrix[1, 1] = -mu_cutoff
            else:
                # PP atoms (Lμ,I * d1,I = C * d0,I) after differentiation on variables: d0, d1, L
                # -d1 * dL + С * d(d0) - L * d(d1) = 0
                mu_matrix = np.zeros(shape=(1, mu_parameters.shape[1]))
                mu_matrix[0, 0] = self.trunc
                mu_matrix[0, 1] = -mu_cutoff

            if mu_parameters.shape[0] == 2:
                mu_spin_deps = [0, 1]
                if mu_cusp:
                    mu_cutoff_matrix = [0, mu_parameters[0, 1], 0, mu_parameters[1, 1]]
                else:
                    mu_cutoff_matrix = [mu_parameters[0, 1], mu_parameters[1, 1]]
                if self.neu < 1:
                    mu_spin_deps = [1]
                    if mu_cusp:
                        mu_cutoff_matrix = [0, mu_parameters[1, 1]]
                    else:
                        mu_cutoff_matrix = [mu_parameters[1, 1]]
                if self.ned < 1:
                    mu_spin_deps = [0]
                    if mu_cusp:
                        mu_cutoff_matrix = [0, mu_parameters[0, 1]]
                    else:
                        mu_cutoff_matrix = [mu_parameters[0, 1]]
            else:
                mu_spin_deps = [0]
                if mu_cusp:
                    mu_cutoff_matrix = [0, mu_parameters[0, 1]]
                else:
                    mu_cutoff_matrix = [mu_parameters[0, 1]]

            mu_block = block_diag([mu_matrix] * len(mu_spin_deps))
            if mu_cutoff_optimizable and self.cutoffs_optimizable:
                # does not matter for AE atoms
                mu_block = np.hstack((-np.array(mu_cutoff_matrix).reshape(-1, 1), mu_block))
            a_list.append(mu_block)
            if mu_cusp:
                b_list += [0] * 2 * len(mu_spin_deps)
            else:
                b_list += [0] * len(mu_spin_deps)

        for phi_parameters, theta_parameters, phi_cutoff, phi_cutoff_optimizable, phi_cusp, phi_irrotational in zip(
            self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_cutoff_optimizable, self.phi_cusp, self.phi_irrotational
        ):
            phi_spin_deps = [0]
            if phi_parameters.shape[0] == 2:
                phi_spin_deps = [0, 1]
                if self.neu < 2 and self.ned < 2:
                    phi_spin_deps = [x for x in phi_spin_deps if x != 0]
                if self.neu + self.ned < 2:
                    phi_spin_deps = [x for x in phi_spin_deps if x != 1]
            elif phi_parameters.shape[0] == 3:
                phi_spin_deps = [0, 1, 2]
                if self.neu < 2:
                    phi_spin_deps = [x for x in phi_spin_deps if x != 0]
                if self.neu + self.ned < 2:
                    phi_spin_deps = [x for x in phi_spin_deps if x != 1]
                if self.ned < 2:
                    phi_spin_deps = [x for x in phi_spin_deps if x != 2]

            phi_list = []
            phi_cutoff_matrix = np.zeros(0)
            for spin_dep in phi_spin_deps:
                phi_matrix, cutoff_constraints = construct_c_matrix(
                    self.trunc, phi_parameters, theta_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational
                )
                phi_constrains_size, phi_parameters_size = phi_matrix.shape
                phi_list.append(phi_matrix)
                phi_cutoff_matrix = np.concatenate((phi_cutoff_matrix, cutoff_constraints))
                b_list += [0] * phi_constrains_size

            phi_block = block_diag(phi_list)
            if phi_cutoff_optimizable and self.cutoffs_optimizable:
                phi_block = np.hstack((phi_cutoff_matrix.reshape(-1, 1), phi_block))
            a_list.append(phi_block)

        if self.omega_cutoff.any():
            omega_spin_deps = []
            omega_list = []
            omega_cutoff_list = []
            for spin_dep in range(self.omega_parameters.shape[0]):
                if not self.omega_parameters_available[spin_dep].any():
                    continue
                omega_matrix, cutoff_constraints = construct_omega_matrix(
                    self.trunc, self.omega_parameters, self.omega_cutoff[spin_dep % self.omega_cutoff.shape[0]], spin_dep
                )
                omega_spin_deps.append(spin_dep)
                omega_list.append(omega_matrix)
                omega_cutoff_list.append(cutoff_constraints)
                b_list += [0] * omega_matrix.shape[0]

            omega_block = block_diag(omega_list)
            if self.omega_cutoff_optimizable.any() and self.cutoffs_optimizable:
                # each set's rows only depend on that set's cutoff
                cutoff_block = np.zeros(shape=(omega_block.shape[0], self.omega_cutoff_optimizable.sum()))
                row = 0
                for k in range(len(omega_list)):
                    col = 0
                    for i in range(self.omega_cutoff.shape[0]):
                        if self.omega_cutoff_optimizable[i]:
                            if i == omega_spin_deps[k] % self.omega_cutoff.shape[0]:
                                cutoff_block[row : row + omega_list[k].shape[0], col] = omega_cutoff_list[k]
                            col += 1
                    row += omega_list[k].shape[0]
                omega_block = np.hstack((cutoff_block, omega_block))
            a_list.append(omega_block)

        if self.ae_cutoff_optimizable.any() and self.cutoffs_optimizable:
            a_list.append(np.zeros(shape=(0, self.ae_cutoff_optimizable.sum())))

        return block_diag(a_list), np.array(b_list)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'set_parameters_projector')
def backflow_set_parameters_projector(self):
    """Set Projector matrix"""

    def impl(self):
        a, b = self.get_parameters_constraints()
        p = np.eye(a.shape[1]) - a.T @ np.linalg.pinv(a.T)
        mask_idx = np.argwhere(self.get_parameters_mask()).ravel()
        inv_p = np.linalg.inv(p[:, mask_idx][mask_idx, :])
        self.parameters_projector = p[:, mask_idx] @ inv_p

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'get_parameters')
def backflow_get_parameters(self, all_parameters):
    """Returns parameters in the following order:
    eta-cutoff(s), eta-linear parameters,
    for every mu-set: mu-cutoff, mu-linear parameters,
    for every phi/theta-set: phi-cutoff, phi-linear parameters, theta-linear parameters.
    :param all_parameters:
    :return:
    """

    def impl(self, all_parameters):
        res = []
        if self.eta_cutoff.any():
            for eta_cutoff, eta_cutoff_optimizable in zip(self.eta_cutoff, self.eta_cutoff_optimizable):
                if eta_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(eta_cutoff)
            for j1 in range(self.eta_parameters.shape[0]):
                for j2 in range(self.eta_parameters.shape[1]):
                    if (self.eta_parameters_optimizable[j1, j2] or all_parameters) and self.eta_parameters_available[j1, j2]:
                        res.append(self.eta_parameters[j1, j2])

        if self.mu_cutoff.any():
            for mu_parameters, mu_parameters_optimizable, mu_cutoff, mu_cutoff_optimizable, mu_parameters_available in zip(
                self.mu_parameters, self.mu_parameters_optimizable, self.mu_cutoff, self.mu_cutoff_optimizable, self.mu_parameters_available
            ):
                if mu_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(mu_cutoff)
                for j1 in range(mu_parameters.shape[0]):
                    for j2 in range(mu_parameters.shape[1]):
                        if (mu_parameters_optimizable[j1, j2] or all_parameters) and mu_parameters_available[j1, j2]:
                            res.append(mu_parameters[j1, j2])

        if self.phi_cutoff.any():
            for (
                phi_parameters,
                phi_parameters_optimizable,
                phi_parameters_available,
                theta_parameters,
                theta_parameters_optimizable,
                theta_parameters_available,
                phi_cutoff,
                phi_cutoff_optimizable,
            ) in zip(
                self.phi_parameters,
                self.phi_parameters_optimizable,
                self.phi_parameters_available,
                self.theta_parameters,
                self.theta_parameters_optimizable,
                self.theta_parameters_available,
                self.phi_cutoff,
                self.phi_cutoff_optimizable,
            ):
                if phi_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(phi_cutoff)
                for j1 in range(phi_parameters.shape[0]):
                    for j2 in range(phi_parameters.shape[1]):
                        for j3 in range(phi_parameters.shape[2]):
                            for j4 in range(phi_parameters.shape[3]):
                                if (phi_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and phi_parameters_available[j1, j2, j3, j4]:
                                    res.append(phi_parameters[j1, j2, j3, j4])

                    for j2 in range(theta_parameters.shape[1]):
                        for j3 in range(theta_parameters.shape[2]):
                            for j4 in range(theta_parameters.shape[3]):
                                if (theta_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and theta_parameters_available[j1, j2, j3, j4]:
                                    res.append(theta_parameters[j1, j2, j3, j4])

        if self.omega_cutoff.any():
            for omega_cutoff, omega_cutoff_optimizable in zip(self.omega_cutoff, self.omega_cutoff_optimizable):
                if omega_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(omega_cutoff)
            for j1 in range(self.omega_parameters.shape[0]):
                for j2 in range(self.omega_parameters.shape[1]):
                    for j3 in range(self.omega_parameters.shape[2]):
                        for j4 in range(self.omega_parameters.shape[3]):
                            if (self.omega_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and self.omega_parameters_available[
                                j1, j2, j3, j4
                            ]:
                                res.append(self.omega_parameters[j1, j2, j3, j4])

        for i, ae_cutoff_optimizable in enumerate(self.ae_cutoff_optimizable):
            if ae_cutoff_optimizable and self.cutoffs_optimizable:
                res.append(self.ae_cutoff[i])

        return np.array(res)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'set_parameters')
def backflow_set_parameters(self, parameters, all_parameters):
    """Set parameters in the following order:
    eta-cutoff(s), eta-linear parameters,
    for every mu-set: mu-cutoff, mu-linear parameters,
    for every phi/theta-set: phi-cutoff, phi-linear parameters, theta-linear parameters.
    :param parameters:
    :param all_parameters:
    :return:
    """

    def impl(self, parameters, all_parameters):
        n = 0
        if self.eta_cutoff.any():
            for j1 in range(self.eta_cutoff.shape[0]):
                if self.eta_cutoff_optimizable[j1] and self.cutoffs_optimizable:
                    self.eta_cutoff[j1] = parameters[n]
                    n += 1
            for j1 in range(self.eta_parameters.shape[0]):
                for j2 in range(self.eta_parameters.shape[1]):
                    if (self.eta_parameters_optimizable[j1, j2] or all_parameters) and self.eta_parameters_available[j1, j2]:
                        self.eta_parameters[j1, j2] = parameters[n]
                        n += 1
            if not all_parameters:
                self.fix_eta_parameters()

        if self.mu_cutoff.any():
            for i, (mu_parameters, mu_parameters_optimizable, mu_cutoff_optimizable, mu_parameters_available) in enumerate(
                zip(self.mu_parameters, self.mu_parameters_optimizable, self.mu_cutoff_optimizable, self.mu_parameters_available)
            ):
                if mu_cutoff_optimizable and self.cutoffs_optimizable:
                    self.mu_cutoff[i] = parameters[n]
                    n += 1
                for j1 in range(mu_parameters.shape[0]):
                    for j2 in range(mu_parameters.shape[1]):
                        if (mu_parameters_optimizable[j1, j2] or all_parameters) and mu_parameters_available[j1, j2]:
                            mu_parameters[j1, j2] = parameters[n]
                            n += 1
            if not all_parameters:
                self.fix_mu_parameters()

        if self.phi_cutoff.any():
            for i, (
                phi_parameters,
                phi_parameters_optimizable,
                phi_parameters_available,
                theta_parameters,
                theta_parameters_optimizable,
                phi_cutoff_optimizable,
                theta_parameters_available,
            ) in enumerate(
                zip(
                    self.phi_parameters,
                    self.phi_parameters_optimizable,
                    self.phi_parameters_available,
                    self.theta_parameters,
                    self.theta_parameters_optimizable,
                    self.phi_cutoff_optimizable,
                    self.theta_parameters_available,
                )
            ):
                if phi_cutoff_optimizable and self.cutoffs_optimizable:
                    self.phi_cutoff[i] = parameters[n]
                    n += 1
                for j1 in range(phi_parameters.shape[0]):
                    for j2 in range(phi_parameters.shape[1]):
                        for j3 in range(phi_parameters.shape[2]):
                            for j4 in range(phi_parameters.shape[3]):
                                if (phi_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and phi_parameters_available[j1, j2, j3, j4]:
                                    phi_parameters[j1, j2, j3, j4] = parameters[n]
                                    n += 1

                    for j2 in range(theta_parameters.shape[1]):
                        for j3 in range(theta_parameters.shape[2]):
                            for j4 in range(theta_parameters.shape[3]):
                                if (theta_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and theta_parameters_available[j1, j2, j3, j4]:
                                    theta_parameters[j1, j2, j3, j4] = parameters[n]
                                    n += 1
            if not all_parameters:
                self.fix_phi_parameters()

        if self.omega_cutoff.any():
            for j1 in range(self.omega_cutoff.shape[0]):
                if self.omega_cutoff_optimizable[j1] and self.cutoffs_optimizable:
                    self.omega_cutoff[j1] = parameters[n]
                    n += 1
            for j1 in range(self.omega_parameters.shape[0]):
                for j2 in range(self.omega_parameters.shape[1]):
                    for j3 in range(self.omega_parameters.shape[2]):
                        for j4 in range(self.omega_parameters.shape[3]):
                            if (self.omega_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and self.omega_parameters_available[
                                j1, j2, j3, j4
                            ]:
                                self.omega_parameters[j1, j2, j3, j4] = parameters[n]
                                n += 1
            if not all_parameters:
                self.fix_omega_parameters()

        for i, cutoff_optimizable in enumerate(self.ae_cutoff_optimizable):
            if cutoff_optimizable and self.cutoffs_optimizable:
                self.ae_cutoff[i] = parameters[n]
                n += 1

        return parameters[n:]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'eta_term_d1')
def backflow_eta_term_d1(self, e_powers, e_vectors):
    """First derivatives of log wfn w.r.t eta-term parameters
    :param e_vectors: e-e vectors
    :param e_powers: powers of e-e distances
    """

    def impl(self, e_powers, e_vectors):
        C = self.trunc
        ae_cutoff_condition = 1
        if not self.eta_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3))

        size = self.eta_parameters_available.sum() + (self.cutoffs_optimizable and self.eta_cutoff_optimizable.sum())
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3))

        n = -1
        for i in range(self.eta_cutoff.shape[0]):
            if self.eta_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.eta_cutoff[i] -= delta
                res[n] -= self.eta_term(e_powers, e_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.eta_cutoff[i] += 2 * delta
                res[n] += self.eta_term(e_powers, e_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.eta_cutoff[i] -= delta

        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                n = (self.cutoffs_optimizable and self.eta_cutoff_optimizable.sum()) - 1
                r = e_powers[e1, e2, 1]
                eta_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % self.eta_parameters.shape[0]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    r_vec = e_vectors[e1, e2]
                    for j1 in range(self.eta_parameters.shape[0]):
                        for j2 in range(self.eta_parameters.shape[1]):
                            if self.eta_parameters_available[j1, j2]:
                                n += 1
                                if eta_set == j1:
                                    bf = (1 - r / L) ** C * e_powers[e1, e2, j2] * r_vec
                                    res[n, ae_cutoff_condition, e1] += bf
                                    res[n, ae_cutoff_condition, e2] -= bf

        return res.reshape(size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'mu_term_d1')
def backflow_mu_term_d1(self, n_powers, n_vectors):
    """First derivatives of log wfn w.r.t mu-term parameters
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    """

    def impl(self, n_powers, n_vectors):
        C = self.trunc
        if not self.mu_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3))

        size = sum(
            [
                mu_parameters_available.sum() + (mu_cutoff_optimizable and self.cutoffs_optimizable)
                for mu_parameters_available, mu_cutoff_optimizable in zip(self.mu_parameters_available, self.mu_cutoff_optimizable)
            ]
        )
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3))

        n = -1
        for i, (mu_parameters, mu_parameters_available, mu_labels) in enumerate(
            zip(self.mu_parameters, self.mu_parameters_available, self.mu_labels)
        ):
            if self.mu_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.mu_cutoff[i] -= delta
                res[n] -= self.mu_term(n_powers, n_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.mu_cutoff[i] += 2 * delta
                res[n] += self.mu_term(n_powers, n_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.mu_cutoff[i] -= delta

            n_start = n
            L = self.mu_cutoff[i]
            for label in mu_labels:
                for e1 in range(self.neu + self.ned):
                    n = n_start
                    r = n_powers[label, e1, 1]
                    if r < L:
                        r_vec = n_vectors[label, e1]
                        # cutoff_condition
                        # 0: AE cutoff exactly not applied
                        # 1: AE cutoff maybe applied
                        ae_cutoff_condition = int(r > self.ae_cutoff[label])
                        mu_set = int(e1 >= self.neu) % mu_parameters.shape[0]
                        for j1 in range(mu_parameters.shape[0]):
                            for j2 in range(mu_parameters.shape[1]):
                                if mu_parameters_available[j1, j2]:
                                    n += 1
                                    if mu_set == j1:
                                        poly = n_powers[label, e1, j2]
                                        res[n, ae_cutoff_condition, e1] += poly * (1 - r / L) ** C * r_vec

        return res.reshape(size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'phi_term_d1')
def backflow_phi_term_d1(self, e_powers, n_powers, e_vectors, n_vectors):
    """First derivatives of log wfn w.r.t phi-term parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    """

    def impl(self, e_powers, n_powers, e_vectors, n_vectors):
        C = self.trunc
        if not self.phi_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3))

        size = sum(
            [
                phi_parameters_available.sum() + theta_parameters_available.sum() + (phi_cutoff_optimizable and self.cutoffs_optimizable)
                for phi_parameters_available, theta_parameters_available, phi_cutoff_optimizable in zip(
                    self.phi_parameters_available, self.theta_parameters_available, self.phi_cutoff_optimizable
                )
            ]
        )
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3))

        n = -1
        for i, (phi_parameters, phi_parameters_available, theta_parameters, theta_parameters_available, phi_labels) in enumerate(
            zip(self.phi_parameters, self.phi_parameters_available, self.theta_parameters, self.theta_parameters_available, self.phi_labels)
        ):
            if self.phi_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.phi_cutoff[i] -= delta
                res[n] -= self.phi_term(e_powers, n_powers, e_vectors, n_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.phi_cutoff[i] += 2 * delta
                res[n] += self.phi_term(e_powers, n_powers, e_vectors, n_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.phi_cutoff[i] -= delta

            n_start = n
            L = self.phi_cutoff[i]
            for label in phi_labels:
                for e1 in range(self.neu + self.ned):
                    for e2 in range(self.neu + self.ned):
                        if e1 == e2:
                            continue
                        n = n_start
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            r_e1I_vec = n_vectors[label, e1]
                            r_ee_vec = e_vectors[e1, e2]
                            # cutoff_condition
                            # 0: AE cutoff exactly not applied
                            # 1: AE cutoff maybe applied
                            ae_cutoff_condition = int(r_e1I > self.ae_cutoff[label])
                            cutoff = (1 - r_e1I / L) ** C * (1 - r_e2I / L) ** C
                            phi_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % phi_parameters.shape[0]
                            for j1 in range(phi_parameters.shape[0]):
                                dn = np.sum(phi_parameters_available[j1])
                                for j2 in range(phi_parameters.shape[1]):
                                    for j3 in range(phi_parameters.shape[2]):
                                        for j4 in range(phi_parameters.shape[3]):
                                            if phi_parameters_available[j1, j2, j3, j4]:
                                                n += 1
                                                if phi_set == j1:
                                                    poly = e_powers[e1, e2, j2] * n_powers[label, e1, j4] * n_powers[label, e2, j3]
                                                    res[n, ae_cutoff_condition, e1] += cutoff * poly * r_ee_vec
                                                    res[n + dn, ae_cutoff_condition, e1] += cutoff * poly * r_e1I_vec
                                n += dn

        return res.reshape(size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'ae_multiplier_d1')
def backflow_ae_multiplier_d1(self, n_vectors, n_powers):
    """First derivatives of log wfn w.r.t ae_cutoff
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    """

    def impl(self, n_vectors, n_powers):
        size = self.cutoffs_optimizable and self.ae_cutoff_optimizable.sum()
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3))
        n = -1
        for i in range(n_vectors.shape[0]):
            if self.ae_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                Lg = self.ae_cutoff[i]
                for j in range(self.neu + self.ned):
                    r = n_powers[i, j, 1]
                    if r < Lg:
                        res[n, 1, j] = -12 / Lg * (r / Lg) ** 2 * (1 - r / Lg) ** 2

        return res.reshape(size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'omega_term_d1')
def backflow_omega_term_d1(self, e_powers, e_vectors):
    """First derivatives of log wfn w.r.t omega-term parameters
    :param e_powers: powers of e-e distances
    :param e_vectors: e-e vectors
    """

    def impl(self, e_powers, e_vectors):
        ae_cutoff_condition = 1
        if not self.omega_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3))

        C = self.trunc
        parameters = self.omega_parameters
        size = self.omega_parameters_available.sum() + (self.cutoffs_optimizable and self.omega_cutoff_optimizable.sum())
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3))

        n = -1
        for i in range(self.omega_cutoff.shape[0]):
            if self.omega_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.omega_cutoff[i] -= delta
                res[n] -= self.omega_term(e_powers, e_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.omega_cutoff[i] += 2 * delta
                res[n] += self.omega_term(e_powers, e_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.omega_cutoff[i] -= delta

        n_start = n
        for e1 in range(2, self.neu + self.ned):
            for e2 in range(1, e1):
                for e3 in range(e2):
                    n = n_start
                    # canonical triplet order (i, j, k) = (e3, e2, e1), i < j < k
                    r_ij = e_powers[e3, e2, 1]
                    r_jk = e_powers[e2, e1, 1]
                    r_ki = e_powers[e1, e3, 1]
                    omega_set = (int(e1 >= self.neu) + int(e2 >= self.neu) + int(e3 >= self.neu)) % parameters.shape[0]
                    L = self.omega_cutoff[omega_set % self.omega_cutoff.shape[0]]
                    if r_ij < L and r_jk < L and r_ki < L:
                        cutoff = (1 - r_jk / L) ** C * (1 - r_ki / L) ** C * (1 - r_ij / L) ** C
                        for j1 in range(parameters.shape[0]):
                            for j2 in range(parameters.shape[1]):
                                for j3 in range(parameters.shape[2]):
                                    for j4 in range(parameters.shape[3]):
                                        if self.omega_parameters_available[j1, j2, j3, j4]:
                                            n += 1
                                            if omega_set == j1:
                                                w = cutoff * e_powers[e2, e1, j2] * e_powers[e1, e3, j3] * e_powers[e3, e2, j4]
                                                res[n, ae_cutoff_condition, e3] += w * (e_vectors[e3, e2] + e_vectors[e3, e1])
                                                res[n, ae_cutoff_condition, e2] += w * (e_vectors[e2, e3] + e_vectors[e2, e1])
                                                res[n, ae_cutoff_condition, e1] += w * (e_vectors[e1, e3] + e_vectors[e1, e2])

        return res.reshape(size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'eta_term_gradient_d1')
def backflow_eta_term_gradient_d1(self, e_powers, e_vectors):
    """First derivatives of log wfn w.r.t eta-term parameters
    :param e_vectors: e-e vectors
    :param e_powers: powers of e-e distances
    """

    def impl(self, e_powers, e_vectors):
        C = self.trunc
        ae_cutoff_condition = 1
        if not self.eta_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3))

        size = self.eta_parameters_available.sum() + (self.cutoffs_optimizable and self.eta_cutoff_optimizable.sum())
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3, (self.neu + self.ned), 3))

        n = -1
        for i in range(self.eta_cutoff.shape[0]):
            if self.eta_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.eta_cutoff[i] -= delta
                res[n] -= self.eta_term_gradient(e_powers, e_vectors).reshape(2, (self.neu + self.ned), 3, (self.neu + self.ned), 3) / delta / 2
                self.eta_cutoff[i] += 2 * delta
                res[n] += self.eta_term_gradient(e_powers, e_vectors).reshape(2, (self.neu + self.ned), 3, (self.neu + self.ned), 3) / delta / 2
                self.eta_cutoff[i] -= delta

        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                n = (self.cutoffs_optimizable and self.eta_cutoff_optimizable.sum()) - 1
                r = e_powers[e1, e2, 1]
                eta_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % self.eta_parameters.shape[0]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    r_vec = e_vectors[e1, e2]
                    cutoff = (1 - r / L) ** C
                    outer_vec = np.outer(r_vec, r_vec)
                    for j1 in range(self.eta_parameters.shape[0]):
                        for j2 in range(self.eta_parameters.shape[1]):
                            if self.eta_parameters_available[j1, j2]:
                                n += 1
                                if eta_set == j1:
                                    poly = cutoff * e_powers[e1, e2, j2]
                                    bf = ((j2 / r - C / (L - r)) * outer_vec / r + eye3) * poly
                                    res[n, ae_cutoff_condition, e1, :, e1, :] += bf
                                    res[n, ae_cutoff_condition, e1, :, e2, :] -= bf
                                    res[n, ae_cutoff_condition, e2, :, e1, :] -= bf
                                    res[n, ae_cutoff_condition, e2, :, e2, :] += bf

        return res.reshape(size, 2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'mu_term_gradient_d1')
def backflow_mu_term_gradient_d1(self, n_powers, n_vectors):
    """First derivatives of log wfn w.r.t mu-term parameters
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    """

    def impl(self, n_powers, n_vectors):
        C = self.trunc
        if not self.mu_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3))

        size = sum(
            [
                mu_parameters_available.sum() + (mu_cutoff_optimizable and self.cutoffs_optimizable)
                for mu_parameters_available, mu_cutoff_optimizable in zip(self.mu_parameters_available, self.mu_cutoff_optimizable)
            ]
        )
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3, (self.neu + self.ned), 3))

        n = -1
        for i, (mu_parameters, mu_parameters_available, mu_labels) in enumerate(
            zip(self.mu_parameters, self.mu_parameters_available, self.mu_labels)
        ):
            if self.mu_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.mu_cutoff[i] -= delta
                res[n] -= self.mu_term_gradient(n_powers, n_vectors).reshape(2, (self.neu + self.ned), 3, (self.neu + self.ned), 3) / delta / 2
                self.mu_cutoff[i] += 2 * delta
                res[n] += self.mu_term_gradient(n_powers, n_vectors).reshape(2, (self.neu + self.ned), 3, (self.neu + self.ned), 3) / delta / 2
                self.mu_cutoff[i] -= delta

            n_start = n
            L = self.mu_cutoff[i]
            for label in mu_labels:
                for e1 in range(self.neu + self.ned):
                    n = n_start
                    r = n_powers[label, e1, 1]
                    if r < L:
                        r_vec = n_vectors[label, e1]
                        # cutoff_condition
                        # 0: AE cutoff exactly not applied
                        # 1: AE cutoff maybe applied
                        ae_cutoff_condition = int(r > self.ae_cutoff[label])
                        mu_set = int(e1 >= self.neu) % mu_parameters.shape[0]
                        cutoff = (1 - r / L) ** C
                        outer_vec = np.outer(r_vec, r_vec)
                        for j1 in range(mu_parameters.shape[0]):
                            for j2 in range(mu_parameters.shape[1]):
                                if mu_parameters_available[j1, j2]:
                                    n += 1
                                    if mu_set == j1:
                                        poly = cutoff * n_powers[label, e1, j2]
                                        res[n, ae_cutoff_condition, e1, :, e1, :] += ((j2 / r - C / (L - r)) * outer_vec / r + eye3) * poly

        return res.reshape(size, 2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'phi_term_gradient_d1')
def backflow_phi_term_gradient_d1(self, e_powers, n_powers, e_vectors, n_vectors):
    """First derivatives of log wfn w.r.t phi-term parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    """

    def impl(self, e_powers, n_powers, e_vectors, n_vectors):
        C = self.trunc
        if not self.phi_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3))

        size = sum(
            [
                phi_parameters_available.sum() + theta_parameters_available.sum() + (phi_cutoff_optimizable and self.cutoffs_optimizable)
                for phi_parameters_available, theta_parameters_available, phi_cutoff_optimizable in zip(
                    self.phi_parameters_available, self.theta_parameters_available, self.phi_cutoff_optimizable
                )
            ]
        )
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3, (self.neu + self.ned), 3))

        n = -1
        for i, (phi_parameters, phi_parameters_available, theta_parameters, theta_parameters_available, phi_labels) in enumerate(
            zip(self.phi_parameters, self.phi_parameters_available, self.theta_parameters, self.theta_parameters_available, self.phi_labels)
        ):
            if self.phi_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.phi_cutoff[i] -= delta
                res[n] -= (
                    self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors).reshape(2, (self.neu + self.ned), 3, (self.neu + self.ned), 3)
                    / delta
                    / 2
                )
                self.phi_cutoff[i] += 2 * delta
                res[n] += (
                    self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors).reshape(2, (self.neu + self.ned), 3, (self.neu + self.ned), 3)
                    / delta
                    / 2
                )
                self.phi_cutoff[i] -= delta

            n_start = n
            L = self.phi_cutoff[i]
            for label in phi_labels:
                for e1 in range(self.neu + self.ned):
                    for e2 in range(self.neu + self.ned):
                        if e1 == e2:
                            continue
                        n = n_start
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            r_ee = e_powers[e1, e2, 1]
                            r_e1I_vec = n_vectors[label, e1]
                            r_e2I_vec = n_vectors[label, e2]
                            r_ee_vec = e_vectors[e1, e2]
                            cutoff = (1 - r_e1I / L) ** C * (1 - r_e2I / L) ** C
                            cutoff_diff_e1I = C * r_e1I / (L - r_e1I)
                            cutoff_diff_e2I = C * r_e2I / (L - r_e2I)
                            # cutoff_condition
                            # 0: AE cutoff exactly not applied
                            # 1: AE cutoff maybe applied
                            ae_cutoff_condition = int(r_e1I > self.ae_cutoff[label])
                            phi_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % phi_parameters.shape[0]
                            for j1 in range(phi_parameters.shape[0]):
                                dn = np.sum(phi_parameters_available[j1])
                                for j2 in range(phi_parameters.shape[1]):
                                    for j3 in range(phi_parameters.shape[2]):
                                        for j4 in range(phi_parameters.shape[3]):
                                            if phi_parameters_available[j1, j2, j3, j4]:
                                                n += 1
                                                if phi_set == j1:
                                                    poly = cutoff * e_powers[e1, e2, j2] * n_powers[label, e2, j3] * n_powers[label, e1, j4]
                                                    for t1 in range(3):
                                                        for t2 in range(3):
                                                            res[n, ae_cutoff_condition, e1, t1, e1, t2] += (
                                                                (j4 - cutoff_diff_e1I) * r_ee_vec[t1] * r_e1I_vec[t2] / r_e1I**2
                                                                + j2 * r_ee_vec[t1] * r_ee_vec[t2] / r_ee**2
                                                                + eye3[t1, t2]
                                                            ) * poly
                                                            res[n + dn, ae_cutoff_condition, e1, t1, e1, t2] += (
                                                                (j4 - cutoff_diff_e1I) * r_e1I_vec[t1] * r_e1I_vec[t2] / r_e1I**2
                                                                + j2 * r_ee_vec[t2] * r_e1I_vec[t1] / r_ee**2
                                                                + eye3[t1, t2]
                                                            ) * poly
                                                            res[n, ae_cutoff_condition, e1, t1, e2, t2] += (
                                                                (j3 - cutoff_diff_e2I) * r_ee_vec[t1] * r_e2I_vec[t2] / r_e2I**2
                                                                - j2 * r_ee_vec[t1] * r_ee_vec[t2] / r_ee**2
                                                                - eye3[t1, t2]
                                                            ) * poly
                                                            res[n + dn, ae_cutoff_condition, e1, t1, e2, t2] += (
                                                                (j3 - cutoff_diff_e2I) * r_e1I_vec[t1] * r_e2I_vec[t2] / r_e2I**2
                                                                - j2 * r_ee_vec[t2] * r_e1I_vec[t1] / r_ee**2
                                                            ) * poly
                                n += dn

        return res.reshape(size, 2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'ae_multiplier_gradient_d1')
def backflow_ae_multiplier_gradient_d1(self, n_vectors, n_powers):
    """First derivatives of gradient w.r.t ae_cutoff
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    """

    def impl(self, n_vectors, n_powers):
        size = self.cutoffs_optimizable and self.ae_cutoff_optimizable.sum()
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3, self.neu + self.ned, 3))
        n = -1
        for i in range(n_vectors.shape[0]):
            if self.ae_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                Lg = self.ae_cutoff[i]
                for j in range(self.neu + self.ned):
                    r_vec = n_vectors[i, j]
                    r = n_powers[i, j, 1]
                    if r < Lg:
                        res[n, 1, j, :, j, :] = -24 * r_vec / Lg**3 * (1 - 3 * (r / Lg) + 2 * (r / Lg) ** 2)

        return res.reshape(size, 2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'eta_term_laplacian_d1')
def backflow_eta_term_laplacian_d1(self, e_powers, e_vectors):
    """First derivatives of laplacian w.r.t eta-term parameters
    :param e_vectors: e-e vectors
    :param e_powers: powers of e-e distances
    """

    def impl(self, e_powers, e_vectors):
        C = self.trunc
        ae_cutoff_condition = 1
        if not self.eta_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3))

        size = self.eta_parameters_available.sum() + (self.cutoffs_optimizable and self.eta_cutoff_optimizable.sum())
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3))

        n = -1
        for i in range(self.eta_cutoff.shape[0]):
            if self.eta_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.eta_cutoff[i] -= delta
                res[n] -= self.eta_term_laplacian(e_powers, e_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.eta_cutoff[i] += 2 * delta
                res[n] += self.eta_term_laplacian(e_powers, e_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.eta_cutoff[i] -= delta

        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                n = (self.cutoffs_optimizable and self.eta_cutoff_optimizable.sum()) - 1
                r = e_powers[e1, e2, 1]
                eta_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % self.eta_parameters.shape[0]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    r_vec = e_vectors[e1, e2]
                    for j1 in range(self.eta_parameters.shape[0]):
                        for j2 in range(self.eta_parameters.shape[1]):
                            if self.eta_parameters_available[j1, j2]:
                                n += 1
                                if eta_set == j1:
                                    poly = C * (C - 1) / (L - r) ** 2 - 2 * C * (j2 + 2) / r / (L - r) + j2 * (j2 + 3) / r**2
                                    bf = 2 * (1 - r / L) ** C * poly * r_vec * e_powers[e1, e2, j2]
                                    res[n, ae_cutoff_condition, e1] += bf
                                    res[n, ae_cutoff_condition, e2] -= bf

        return res.reshape(size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'mu_term_laplacian_d1')
def backflow_mu_term_laplacian_d1(self, n_powers, n_vectors):
    """First derivatives of log wfn w.r.t mu-term parameters
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    """

    def impl(self, n_powers, n_vectors):
        C = self.trunc
        if not self.mu_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3))

        size = sum(
            [
                mu_parameters_available.sum() + (mu_cutoff_optimizable and self.cutoffs_optimizable)
                for mu_parameters_available, mu_cutoff_optimizable in zip(self.mu_parameters_available, self.mu_cutoff_optimizable)
            ]
        )
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3))

        n = -1
        for i, (mu_parameters, mu_parameters_available, mu_labels) in enumerate(
            zip(self.mu_parameters, self.mu_parameters_available, self.mu_labels)
        ):
            if self.mu_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.mu_cutoff[i] -= delta
                res[n] -= self.mu_term_laplacian(n_powers, n_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.mu_cutoff[i] += 2 * delta
                res[n] += self.mu_term_laplacian(n_powers, n_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.mu_cutoff[i] -= delta

            n_start = n
            L = self.mu_cutoff[i]
            for label in mu_labels:
                for e1 in range(self.neu + self.ned):
                    n = n_start
                    r = n_powers[label, e1, 1]
                    if r < L:
                        r_vec = n_vectors[label, e1]
                        # cutoff_condition
                        # 0: AE cutoff exactly not applied
                        # 1: AE cutoff maybe applied
                        ae_cutoff_condition = int(r > self.ae_cutoff[label])
                        mu_set = int(e1 >= self.neu) % mu_parameters.shape[0]
                        for j1 in range(mu_parameters.shape[0]):
                            for j2 in range(mu_parameters.shape[1]):
                                if mu_parameters_available[j1, j2]:
                                    n += 1
                                    if mu_set == j1:
                                        poly = C * (C - 1) / (L - r) ** 2 - 2 * C * (j2 + 2) / r / (L - r) + j2 * (j2 + 3) / r**2
                                        bf = (1 - r / L) ** C * poly * r_vec * n_powers[label, e1, j2]
                                        res[n, ae_cutoff_condition, e1] += bf

        return res.reshape(size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'phi_term_laplacian_d1')
def backflow_phi_term_laplacian_d1(self, e_powers, n_powers, e_vectors, n_vectors):
    """First derivatives of laplacian w.r.t phi-term parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    """

    def impl(self, e_powers, n_powers, e_vectors, n_vectors):
        C = self.trunc
        if not self.phi_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3))

        size = sum(
            [
                phi_parameters_available.sum() + theta_parameters_available.sum() + (phi_cutoff_optimizable and self.cutoffs_optimizable)
                for phi_parameters_available, theta_parameters_available, phi_cutoff_optimizable in zip(
                    self.phi_parameters_available, self.theta_parameters_available, self.phi_cutoff_optimizable
                )
            ]
        )
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3))

        n = -1
        for i, (phi_parameters, phi_parameters_available, theta_parameters, theta_parameters_available, phi_labels) in enumerate(
            zip(self.phi_parameters, self.phi_parameters_available, self.theta_parameters, self.theta_parameters_available, self.phi_labels)
        ):
            if self.phi_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.phi_cutoff[i] -= delta
                res[n] -= self.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.phi_cutoff[i] += 2 * delta
                res[n] += self.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors).reshape(2, (self.neu + self.ned), 3) / delta / 2
                self.phi_cutoff[i] -= delta

            n_start = n
            L = self.phi_cutoff[i]
            for label in phi_labels:
                for e1 in range(self.neu + self.ned):
                    for e2 in range(self.neu + self.ned):
                        if e1 == e2:
                            continue
                        n = n_start
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            r_e1I_vec = n_vectors[label, e1]
                            r_e2I_vec = n_vectors[label, e2]
                            r_ee_vec = e_vectors[e1, e2]
                            r_ee = e_powers[e1, e2, 1]
                            cutoff = (1 - r_e1I / L) ** C * (1 - r_e2I / L) ** C
                            cutoff_diff_e1I = C * r_e1I / (L - r_e1I)
                            cutoff_diff_e2I = C * r_e2I / (L - r_e2I)
                            cutoff_diff_e1I_2 = C * (C - 1) * r_e1I**2 / (L - r_e1I) ** 2
                            cutoff_diff_e2I_2 = C * (C - 1) * r_e2I**2 / (L - r_e2I) ** 2
                            # cutoff_condition
                            # 0: AE cutoff exactly not applied
                            # 1: AE cutoff maybe applied
                            ae_cutoff_condition = int(r_e1I > self.ae_cutoff[label])
                            phi_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % phi_parameters.shape[0]
                            vec_1 = r_ee_vec * (r_ee_vec @ r_e1I_vec)
                            vec_2 = r_ee_vec * (r_ee_vec @ r_e2I_vec)
                            vec_3 = r_e1I_vec * (r_ee_vec @ r_e1I_vec)
                            vec_4 = r_e1I_vec * (r_ee_vec @ r_e2I_vec)
                            for j1 in range(phi_parameters.shape[0]):
                                dn = np.sum(phi_parameters_available[j1])
                                for j2 in range(phi_parameters.shape[1]):
                                    for j3 in range(phi_parameters.shape[2]):
                                        for j4 in range(phi_parameters.shape[3]):
                                            if phi_parameters_available[j1, j2, j3, j4]:
                                                n += 1
                                                if phi_set == j1:
                                                    phi_diff_1 = (
                                                        (j4 - cutoff_diff_e1I) / r_e1I**2 + (j3 - cutoff_diff_e2I) / r_e2I**2 + 4 * j2 / r_ee**2
                                                    )
                                                    phi_diff_2 = (
                                                        (cutoff_diff_e1I_2 - 2 * j4 * cutoff_diff_e1I + j4 * (j4 - 1)) / r_e1I**2
                                                        + (cutoff_diff_e2I_2 - 2 * j3 * cutoff_diff_e2I + j3 * (j3 - 1)) / r_e2I**2
                                                        + 2 * j2 * (j2 - 1) / r_ee**2
                                                    )
                                                    phi_dot_product = (
                                                        (j4 - cutoff_diff_e1I) * r_e1I_vec / r_e1I**2
                                                        - (j3 - cutoff_diff_e2I) * r_e2I_vec / r_e2I**2
                                                        + (j4 - cutoff_diff_e1I) * j2 * vec_1 / r_e1I**2 / r_ee**2
                                                        - (j3 - cutoff_diff_e2I) * j2 * vec_2 / r_e2I**2 / r_ee**2
                                                    )
                                                    theta_diff_1 = (
                                                        2 * (j4 - cutoff_diff_e1I) / r_e1I**2 + (j3 - cutoff_diff_e2I) / r_e2I**2 + 2 * j2 / r_ee**2
                                                    )
                                                    theta_diff_2 = (
                                                        (cutoff_diff_e1I_2 - 2 * j4 * cutoff_diff_e1I + j4 * (j4 - 1)) / r_e1I**2
                                                        + (cutoff_diff_e2I_2 - 2 * j3 * cutoff_diff_e2I + j3 * (j3 - 1)) / r_e2I**2
                                                        + 2 * j2 * (j2 - 1) / r_ee**2
                                                    )
                                                    theta_dot_product = (
                                                        (
                                                            (j4 - cutoff_diff_e1I) * vec_3 / r_e1I**2
                                                            - (j3 - cutoff_diff_e2I) * vec_4 / r_e2I**2
                                                            + r_ee_vec
                                                        )
                                                        * j2
                                                        / r_ee**2
                                                    )
                                                    poly = cutoff * e_powers[e1, e2, j2] * n_powers[label, e2, j3] * n_powers[label, e1, j4]
                                                    res[n, ae_cutoff_condition, e1] += (
                                                        (phi_diff_2 + 2 * phi_diff_1) * r_ee_vec + 2 * phi_dot_product
                                                    ) * poly
                                                    res[n + dn, ae_cutoff_condition, e1] += (
                                                        (theta_diff_2 + 2 * theta_diff_1) * r_e1I_vec + 2 * theta_dot_product
                                                    ) * poly
                                n += dn

        return res.reshape(size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'omega_term_gradient_d1')
def backflow_omega_term_gradient_d1(self, e_powers, e_vectors):
    """First derivatives of gradient w.r.t omega-term parameters.
    The parameters enter the displacement linearly, so this is omega_term_gradient with the
    polynomial replaced by the single monomial belonging to each parameter.
    :param e_powers: powers of e-e distances
    :param e_vectors: e-e vectors
    """

    def impl(self, e_powers, e_vectors):
        ae_cutoff_condition = 1
        if not self.omega_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3))

        C = self.trunc
        parameters = self.omega_parameters
        size = self.omega_parameters_available.sum() + (self.cutoffs_optimizable and self.omega_cutoff_optimizable.sum())
        res = np.zeros(shape=(size, 2, self.neu + self.ned, 3, self.neu + self.ned, 3))

        n = -1
        for i in range(self.omega_cutoff.shape[0]):
            if self.omega_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                shape = (2, self.neu + self.ned, 3, self.neu + self.ned, 3)
                self.omega_cutoff[i] -= delta
                res[n] -= self.omega_term_gradient(e_powers, e_vectors).reshape(shape) / delta / 2
                self.omega_cutoff[i] += 2 * delta
                res[n] += self.omega_term_gradient(e_powers, e_vectors).reshape(shape) / delta / 2
                self.omega_cutoff[i] -= delta

        n_start = n
        u = np.zeros(shape=(3, 3))
        d = np.zeros(shape=(3, 3))
        for e1 in range(2, self.neu + self.ned):
            for e2 in range(1, e1):
                for e3 in range(e2):
                    n = n_start
                    # canonical triplet order (i, j, k) = (e3, e2, e1), i < j < k
                    r_ij = e_powers[e3, e2, 1]
                    r_jk = e_powers[e2, e1, 1]
                    r_ki = e_powers[e1, e3, 1]
                    omega_set = (int(e1 >= self.neu) + int(e2 >= self.neu) + int(e3 >= self.neu)) % parameters.shape[0]
                    L = self.omega_cutoff[omega_set % self.omega_cutoff.shape[0]]
                    if r_ij < L and r_jk < L and r_ki < L:
                        f = (1 - r_jk / L) ** C * (1 - r_ki / L) ** C * (1 - r_ij / L) ** C
                        g_jk = -C / (L - r_jk)
                        g_ki = -C / (L - r_ki)
                        g_ij = -C / (L - r_ij)
                        jk_vec = e_vectors[e2, e1]
                        ki_vec = e_vectors[e1, e3]
                        ij_vec = e_vectors[e3, e2]
                        u[0] = ij_vec - ki_vec
                        u[1] = jk_vec - ij_vec
                        u[2] = ki_vec - jk_vec
                        triplet = (e3, e2, e1)
                        for j1 in range(parameters.shape[0]):
                            for j2 in range(parameters.shape[1]):
                                for j3 in range(parameters.shape[2]):
                                    for j4 in range(parameters.shape[3]):
                                        if self.omega_parameters_available[j1, j2, j3, j4]:
                                            n += 1
                                            if omega_set == j1:
                                                poly = e_powers[e2, e1, j2] * e_powers[e1, e3, j3] * e_powers[e3, e2, j4]
                                                w = f * poly
                                                w_jk = f * poly * (j2 / r_jk + g_jk)
                                                w_ki = f * poly * (j3 / r_ki + g_ki)
                                                w_ij = f * poly * (j4 / r_ij + g_ij)
                                                d[0] = w_ij * ij_vec / r_ij - w_ki * ki_vec / r_ki
                                                d[1] = w_jk * jk_vec / r_jk - w_ij * ij_vec / r_ij
                                                d[2] = w_ki * ki_vec / r_ki - w_jk * jk_vec / r_jk
                                                for p1 in range(3):
                                                    for p2 in range(3):
                                                        bf = np.outer(u[p1], d[p2])
                                                        if p1 == p2:
                                                            bf += 2 * w * eye3
                                                        else:
                                                            bf -= w * eye3
                                                        res[n, ae_cutoff_condition, triplet[p1], :, triplet[p2], :] += bf

        return res.reshape(size, 2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'omega_term_laplacian_d1')
def backflow_omega_term_laplacian_d1(self, e_powers, e_vectors):
    """First derivatives of laplacian w.r.t omega-term parameters, as omega_term_laplacian
    with the polynomial replaced by the single monomial belonging to each parameter.
    :param e_powers: powers of e-e distances
    :param e_vectors: e-e vectors
    """

    def impl(self, e_powers, e_vectors):
        ae_cutoff_condition = 1
        if not self.omega_cutoff.any():
            return np.zeros(shape=(0, 2, (self.neu + self.ned) * 3))

        C = self.trunc
        parameters = self.omega_parameters
        size = self.omega_parameters_available.sum() + (self.cutoffs_optimizable and self.omega_cutoff_optimizable.sum())
        res = np.zeros(shape=(size, 2, self.neu + self.ned, 3))

        n = -1
        for i in range(self.omega_cutoff.shape[0]):
            if self.omega_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                shape = (2, self.neu + self.ned, 3)
                self.omega_cutoff[i] -= delta
                res[n] -= self.omega_term_laplacian(e_powers, e_vectors).reshape(shape) / delta / 2
                self.omega_cutoff[i] += 2 * delta
                res[n] += self.omega_term_laplacian(e_powers, e_vectors).reshape(shape) / delta / 2
                self.omega_cutoff[i] -= delta

        n_start = n
        for e1 in range(2, self.neu + self.ned):
            for e2 in range(1, e1):
                for e3 in range(e2):
                    n = n_start
                    # canonical triplet order (i, j, k) = (e3, e2, e1), i < j < k
                    r_ij = e_powers[e3, e2, 1]
                    r_jk = e_powers[e2, e1, 1]
                    r_ki = e_powers[e1, e3, 1]
                    omega_set = (int(e1 >= self.neu) + int(e2 >= self.neu) + int(e3 >= self.neu)) % parameters.shape[0]
                    L = self.omega_cutoff[omega_set % self.omega_cutoff.shape[0]]
                    if r_ij < L and r_jk < L and r_ki < L:
                        f = (1 - r_jk / L) ** C * (1 - r_ki / L) ** C * (1 - r_ij / L) ** C
                        g_jk = -C / (L - r_jk)
                        g_ki = -C / (L - r_ki)
                        g_ij = -C / (L - r_ij)
                        h_jk = C * (C - 1) / (L - r_jk) ** 2
                        h_ki = C * (C - 1) / (L - r_ki) ** 2
                        h_ij = C * (C - 1) / (L - r_ij) ** 2
                        jk_vec = e_vectors[e2, e1]
                        ki_vec = e_vectors[e1, e3]
                        ij_vec = e_vectors[e3, e2]
                        cos_ki_ij = (ki_vec @ ij_vec) / r_ki / r_ij
                        cos_jk_ij = (jk_vec @ ij_vec) / r_jk / r_ij
                        cos_jk_ki = (jk_vec @ ki_vec) / r_jk / r_ki
                        for j1 in range(parameters.shape[0]):
                            for j2 in range(parameters.shape[1]):
                                for j3 in range(parameters.shape[2]):
                                    for j4 in range(parameters.shape[3]):
                                        if self.omega_parameters_available[j1, j2, j3, j4]:
                                            n += 1
                                            if omega_set == j1:
                                                poly = f * e_powers[e2, e1, j2] * e_powers[e1, e3, j3] * e_powers[e3, e2, j4]
                                                w_jk = poly * (j2 / r_jk + g_jk)
                                                w_ki = poly * (j3 / r_ki + g_ki)
                                                w_ij = poly * (j4 / r_ij + g_ij)
                                                w_jk_2 = poly * (j2 * (j2 - 1) / r_jk**2 + 2 * g_jk * j2 / r_jk + h_jk)
                                                w_ki_2 = poly * (j3 * (j3 - 1) / r_ki**2 + 2 * g_ki * j3 / r_ki + h_ki)
                                                w_ij_2 = poly * (j4 * (j4 - 1) / r_ij**2 + 2 * g_ij * j4 / r_ij + h_ij)
                                                w_jk_ki = poly * (j2 * j3 / r_jk / r_ki + g_jk * j3 / r_ki + g_ki * j2 / r_jk + g_jk * g_ki)
                                                w_jk_ij = poly * (j2 * j4 / r_jk / r_ij + g_jk * j4 / r_ij + g_ij * j2 / r_jk + g_jk * g_ij)
                                                w_ki_ij = poly * (j3 * j4 / r_ki / r_ij + g_ki * j4 / r_ij + g_ij * j3 / r_ki + g_ki * g_ij)
                                                lap = (
                                                    2 * (w_jk_2 + w_ki_2 + w_ij_2)
                                                    - 2 * w_ki_ij * cos_ki_ij
                                                    - 2 * w_jk_ij * cos_jk_ij
                                                    - 2 * w_jk_ki * cos_jk_ki
                                                    + 4 * (w_jk / r_jk + w_ki / r_ki + w_ij / r_ij)
                                                )
                                                d_i = w_ij * ij_vec / r_ij - w_ki * ki_vec / r_ki
                                                d_j = w_jk * jk_vec / r_jk - w_ij * ij_vec / r_ij
                                                d_k = w_ki * ki_vec / r_ki - w_jk * jk_vec / r_jk
                                                res[n, ae_cutoff_condition, e3] += lap * (ij_vec - ki_vec) + 6 * d_i
                                                res[n, ae_cutoff_condition, e2] += lap * (jk_vec - ij_vec) + 6 * d_j
                                                res[n, ae_cutoff_condition, e1] += lap * (ki_vec - jk_vec) + 6 * d_k

        return res.reshape(size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'ae_multiplier_laplacian_d1')
def backflow_ae_multiplier_laplacian_d1(self, n_vectors, n_powers):
    """First derivatives of laplacian w.r.t ae_cutoff
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    """

    def impl(self, n_vectors, n_powers):
        size = self.cutoffs_optimizable and self.ae_cutoff_optimizable.sum()
        res = np.zeros(shape=(size, 2, (self.neu + self.ned), 3))
        n = -1
        for i in range(n_vectors.shape[0]):
            if self.ae_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                Lg = self.ae_cutoff[i]
                for j in range(self.neu + self.ned):
                    r = n_powers[i, j, 1]
                    if r < Lg:
                        res[n, 1, j] = -24 / Lg**3 * (3 - 12 * (r / Lg) + 10 * (r / Lg) ** 2)

        return res.reshape(size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
def backflow_reduce_parameters_d1(value, gradient, laplacian, offset, f, f_gradient, f_laplacian, g, g_gradient, g_laplacian):
    """Parameter derivative of the product g*f (Leibniz rule), accumulated into value/gradient/laplacian
    at the given row offset:
        value[p,a]     = Σ_s f*g
        gradient[p,a,b]= Σ_s (∇g*f + ∇f*g)
        laplacian[p,a] = Σ_s (∇²g*f + 2 Σ_b ∇f*∇g + ∇²f*g)
    Indices:
        p       optimizable parameter of the term
        a, b    collective-coordinate components, m = 3N
        s       AE-cutoff branch (not spin): 0 -> multiplier 1 (no zeroing),
                1 -> g(r_iI) when r_iI < Lg,I (zeroing the displacement at AE atoms), else 1
    Explicit loops avoid the large (size, 2, m, m) temporaries numba materialises for the broadcast form.
    """
    m = g.shape[1]
    for p in range(f.shape[0]):
        row = offset + p
        for s in range(2):
            for a in range(m):
                fv = f[p, s, a]
                gv = g[s, a]
                value[row, a] += fv * gv
                dot = 0.0
                for b in range(m):
                    fg = f_gradient[p, s, a, b]
                    gg = g_gradient[s, a, b]
                    gradient[row, a, b] += gg * fv + fg * gv
                    dot += fg * gg
                laplacian[row, a] += g_laplacian[s, a] * fv + 2 * dot + f_laplacian[p, s, a] * gv


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'value_parameters_d1')
def backflow_value_parameters_d1(self, e_vectors, n_vectors):
    """First derivatives of backflow w.r.t the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    """

    def impl(self, e_vectors, n_vectors):
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        eta_term = self.eta_term(e_powers, e_vectors)
        mu_term = self.mu_term(n_powers, n_vectors)
        phi_term = self.phi_term(e_powers, n_powers, e_vectors, n_vectors)
        omega_term = self.omega_term(e_powers, e_vectors)
        ae_value = eta_term + mu_term + phi_term + omega_term

        eta_term_d1 = self.eta_term_d1(e_powers, e_vectors)
        mu_term_d1 = self.mu_term_d1(n_powers, n_vectors)
        phi_term_d1 = self.phi_term_d1(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_d1 = self.omega_term_d1(e_powers, e_vectors)

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)

        ae_multiplier_d1 = self.ae_multiplier_d1(n_vectors, n_powers)

        return self.parameters_projector.T @ np.concatenate(
            (
                np.sum(eta_term_d1 * ae_multiplier, axis=1),
                np.sum(mu_term_d1 * ae_multiplier, axis=1),
                np.sum(phi_term_d1 * ae_multiplier, axis=1),
                np.sum(omega_term_d1 * ae_multiplier, axis=1),
                np.sum(ae_value * ae_multiplier_d1, axis=1),
            )
        )

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'gradient_parameters_d1')
def backflow_gradient_parameters_d1(self, e_vectors, n_vectors):
    """First derivatives of backflow gradient w.r.t. the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors, n_vectors):
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        eta_term = self.eta_term(e_powers, e_vectors)
        mu_term = self.mu_term(n_powers, n_vectors)
        phi_term = self.phi_term(e_powers, n_powers, e_vectors, n_vectors)
        omega_term = self.omega_term(e_powers, e_vectors)
        ae_value = eta_term + mu_term + phi_term + omega_term

        eta_term_gradient = self.eta_term_gradient(e_powers, e_vectors)
        mu_term_gradient = self.mu_term_gradient(n_powers, n_vectors)
        phi_term_gradient = self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_gradient = self.omega_term_gradient(e_powers, e_vectors)
        ae_gradient = eta_term_gradient + mu_term_gradient + phi_term_gradient + omega_term_gradient

        eta_term_d1 = self.eta_term_d1(e_powers, e_vectors)
        mu_term_d1 = self.mu_term_d1(n_powers, n_vectors)
        phi_term_d1 = self.phi_term_d1(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_d1 = self.omega_term_d1(e_powers, e_vectors)

        eta_term_gradient_d1 = self.eta_term_gradient_d1(e_powers, e_vectors)
        mu_term_gradient_d1 = self.mu_term_gradient_d1(n_powers, n_vectors)
        phi_term_gradient_d1 = self.phi_term_gradient_d1(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_gradient_d1 = self.omega_term_gradient_d1(e_powers, e_vectors)

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)
        ae_multiplier_gradient = self.ae_multiplier_gradient(n_vectors, n_powers)

        ae_multiplier_d1 = self.ae_multiplier_d1(n_vectors, n_powers)
        ae_multiplier_gradient_d1 = self.ae_multiplier_gradient_d1(n_vectors, n_powers)

        value = np.concatenate(
            (
                np.sum(eta_term_d1 * ae_multiplier, axis=1),
                np.sum(mu_term_d1 * ae_multiplier, axis=1),
                np.sum(phi_term_d1 * ae_multiplier, axis=1),
                np.sum(omega_term_d1 * ae_multiplier, axis=1),
                np.sum(ae_value * ae_multiplier_d1, axis=1),
            )
        )

        gradient = np.concatenate(
            (
                np.sum(ae_multiplier_gradient * np.expand_dims(eta_term_d1, 3) + eta_term_gradient_d1 * np.expand_dims(ae_multiplier, 2), axis=1),
                np.sum(ae_multiplier_gradient * np.expand_dims(mu_term_d1, 3) + mu_term_gradient_d1 * np.expand_dims(ae_multiplier, 2), axis=1),
                np.sum(ae_multiplier_gradient * np.expand_dims(phi_term_d1, 3) + phi_term_gradient_d1 * np.expand_dims(ae_multiplier, 2), axis=1),
                np.sum(ae_multiplier_gradient * np.expand_dims(omega_term_d1, 3) + omega_term_gradient_d1 * np.expand_dims(ae_multiplier, 2), axis=1),
                np.sum(ae_multiplier_gradient_d1 * np.expand_dims(ae_value, 2) + ae_gradient * np.expand_dims(ae_multiplier_d1, 3), axis=1),
            )
        )

        return gradient, value

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'laplacian_parameters_d1')
def backflow_laplacian_parameters_d1(self, e_vectors, n_vectors):
    """First derivatives of backflow laplacian w.r.t the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors, n_vectors):
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        eta_term = self.eta_term(e_powers, e_vectors)
        mu_term = self.mu_term(n_powers, n_vectors)
        phi_term = self.phi_term(e_powers, n_powers, e_vectors, n_vectors)
        omega_term = self.omega_term(e_powers, e_vectors)
        ae_value = eta_term + mu_term + phi_term + omega_term

        eta_term_gradient = self.eta_term_gradient(e_powers, e_vectors)
        mu_term_gradient = self.mu_term_gradient(n_powers, n_vectors)
        phi_term_gradient = self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_gradient = self.omega_term_gradient(e_powers, e_vectors)
        ae_gradient = eta_term_gradient + mu_term_gradient + phi_term_gradient + omega_term_gradient

        eta_term_laplacian = self.eta_term_laplacian(e_powers, e_vectors)
        mu_term_laplacian = self.mu_term_laplacian(n_powers, n_vectors)
        phi_term_laplacian = self.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_laplacian = self.omega_term_laplacian(e_powers, e_vectors)
        ae_laplacian = eta_term_laplacian + mu_term_laplacian + phi_term_laplacian + omega_term_laplacian

        eta_term_d1 = self.eta_term_d1(e_powers, e_vectors)
        mu_term_d1 = self.mu_term_d1(n_powers, n_vectors)
        phi_term_d1 = self.phi_term_d1(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_d1 = self.omega_term_d1(e_powers, e_vectors)

        eta_term_gradient_d1 = self.eta_term_gradient_d1(e_powers, e_vectors)
        mu_term_gradient_d1 = self.mu_term_gradient_d1(n_powers, n_vectors)
        phi_term_gradient_d1 = self.phi_term_gradient_d1(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_gradient_d1 = self.omega_term_gradient_d1(e_powers, e_vectors)

        eta_term_laplacian_d1 = self.eta_term_laplacian_d1(e_powers, e_vectors)
        mu_term_laplacian_d1 = self.mu_term_laplacian_d1(n_powers, n_vectors)
        phi_term_laplacian_d1 = self.phi_term_laplacian_d1(e_powers, n_powers, e_vectors, n_vectors)
        omega_term_laplacian_d1 = self.omega_term_laplacian_d1(e_powers, e_vectors)

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)
        ae_multiplier_gradient = self.ae_multiplier_gradient(n_vectors, n_powers)
        ae_multiplier_laplacian = self.ae_multiplier_laplacian(n_vectors, n_powers)

        ae_multiplier_d1 = self.ae_multiplier_d1(n_vectors, n_powers)
        ae_multiplier_gradient_d1 = self.ae_multiplier_gradient_d1(n_vectors, n_powers)
        ae_multiplier_laplacian_d1 = self.ae_multiplier_laplacian_d1(n_vectors, n_powers)

        ne = ae_multiplier.shape[1]
        eta_size = eta_term_d1.shape[0]
        mu_size = mu_term_d1.shape[0]
        phi_size = phi_term_d1.shape[0]
        omega_size = omega_term_d1.shape[0]
        ae_size = ae_multiplier_d1.shape[0]
        total = eta_size + mu_size + phi_size + omega_size + ae_size
        value = np.zeros(shape=(total, ne))
        gradient = np.zeros(shape=(total, ne, ne))
        laplacian = np.zeros(shape=(total, ne))

        backflow_reduce_parameters_d1(
            value,
            gradient,
            laplacian,
            0,
            eta_term_d1,
            eta_term_gradient_d1,
            eta_term_laplacian_d1,
            ae_multiplier,
            ae_multiplier_gradient,
            ae_multiplier_laplacian,
        )
        backflow_reduce_parameters_d1(
            value,
            gradient,
            laplacian,
            eta_size,
            mu_term_d1,
            mu_term_gradient_d1,
            mu_term_laplacian_d1,
            ae_multiplier,
            ae_multiplier_gradient,
            ae_multiplier_laplacian,
        )
        backflow_reduce_parameters_d1(
            value,
            gradient,
            laplacian,
            eta_size + mu_size,
            phi_term_d1,
            phi_term_gradient_d1,
            phi_term_laplacian_d1,
            ae_multiplier,
            ae_multiplier_gradient,
            ae_multiplier_laplacian,
        )
        backflow_reduce_parameters_d1(
            value,
            gradient,
            laplacian,
            eta_size + mu_size + phi_size,
            omega_term_d1,
            omega_term_gradient_d1,
            omega_term_laplacian_d1,
            ae_multiplier,
            ae_multiplier_gradient,
            ae_multiplier_laplacian,
        )
        backflow_reduce_parameters_d1(
            value,
            gradient,
            laplacian,
            eta_size + mu_size + phi_size + omega_size,
            ae_multiplier_d1,
            ae_multiplier_gradient_d1,
            ae_multiplier_laplacian_d1,
            ae_value,
            ae_gradient,
            ae_laplacian,
        )

        return laplacian, gradient, value

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'eta_term_d2')
def backflow_eta_term_d2(self, e_powers, e_vectors):
    """Second derivatives of logarithm wfn w.r.t eta-term parameters
    :param e_vectors: e-e vectors
    :param e_powers: powers of e-e distances
    """

    def impl(self, e_powers, e_vectors):
        if not self.eta_cutoff.any():
            return np.zeros(shape=(0, 0, 2, (self.neu + self.ned) * 3))

        size = self.eta_parameters_available.sum() + (self.cutoffs_optimizable and self.eta_cutoff_optimizable.sum())
        res = np.zeros(shape=(size, size, 2, (self.neu + self.ned), 3))
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'mu_term_d2')
def backflow_mu_term_d2(self, n_powers, n_vectors):
    """Second derivatives of logarithm wfn w.r.t mu-term parameters
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    """

    def impl(self, n_powers, n_vectors):
        if not self.mu_cutoff.any():
            return np.zeros(shape=(0, 0, 2, (self.neu + self.ned) * 3))

        size = sum(
            [
                mu_parameters_available.sum() + (mu_cutoff_optimizable and self.cutoffs_optimizable)
                for mu_parameters_available, mu_cutoff_optimizable in zip(self.mu_parameters_available, self.mu_cutoff_optimizable)
            ]
        )
        res = np.zeros(shape=(size, size, 2, (self.neu + self.ned), 3))
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'phi_term_d2')
def backflow_phi_term_d2(self, e_powers, n_powers, e_vectors, n_vectors):
    """Second derivatives of logarithm wfn w.r.t phi-term parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    """

    def impl(self, e_powers, n_powers, e_vectors, n_vectors):
        if not self.phi_cutoff.any():
            return np.zeros(shape=(0, 0, 2, (self.neu + self.ned) * 3))

        size = sum(
            [
                phi_parameters_available.sum() + theta_parameters_available.sum() + (phi_cutoff_optimizable and self.cutoffs_optimizable)
                for phi_parameters_available, theta_parameters_available, phi_cutoff_optimizable in zip(
                    self.phi_parameters_available, self.theta_parameters_available, self.phi_cutoff_optimizable
                )
            ]
        )
        res = np.zeros(shape=(size, size, 2, (self.neu + self.ned), 3))
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'ae_multiplier_d2')
def backflow_ae_multiplier_d2(self, n_vectors, n_powers):
    """Second derivatives of logarithm wfn w.r.t ae_cutoff
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    """

    def impl(self, n_vectors, n_powers):
        size = self.cutoffs_optimizable and self.ae_cutoff_optimizable.sum()
        res = np.zeros(shape=(size, size, 2, (self.neu + self.ned), 3))

        return res.reshape(size, size, 2, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'value_parameters_d2')
def backflow_value_parameters_d2(self, e_vectors, n_vectors):
    """Second derivatives backflow w.r.t the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    """

    def impl(self, e_vectors, n_vectors):
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)

        return (
            self.parameters_projector.T
            @ block_diag(
                (
                    np.sum(self.eta_term_parameters_d2(e_powers, e_vectors) * ae_multiplier, axis=2),
                    np.sum(self.mu_term_parameters_d2(n_powers, n_vectors) * ae_multiplier, axis=2),
                    np.sum(self.phi_term_parameters_d2(e_powers, n_powers, e_vectors, n_vectors) * ae_multiplier, axis=2),
                )
            )
            @ self.parameters_projector
        )

    return impl


class Backflow(structref.StructRefProxy, AbstractBackflow):
    def __new__(cls, config):
        @nb.njit(nogil=True, parallel=False, cache=True)
        def backflow_init(
            neu,
            ned,
            trunc,
            eta_parameters,
            eta_parameters_optimizable,
            eta_cutoff,
            mu_parameters,
            mu_parameters_optimizable,
            mu_cutoff,
            mu_cusp,
            mu_labels,
            phi_parameters,
            phi_parameters_optimizable,
            theta_parameters,
            theta_parameters_optimizable,
            phi_cutoff,
            phi_cusp,
            phi_labels,
            phi_irrotational,
            omega_parameters,
            omega_parameters_optimizable,
            omega_cutoff,
            ae_cutoff,
            ae_cutoff_optimizable,
        ):
            self = structref.new(Backflow_t)
            self.neu = neu
            self.ned = ned
            self.trunc = trunc
            # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
            self.eta_cutoff = eta_cutoff['value']
            self.eta_cutoff_optimizable = eta_cutoff['optimizable']
            self.eta_parameters = eta_parameters
            self.eta_parameters_optimizable = eta_parameters_optimizable
            self.eta_parameters_available = np.ones_like(eta_parameters_optimizable)
            # spin dep (0->u=d; 1->u/=d)
            self.mu_cusp = mu_cusp
            self.mu_labels = mu_labels
            self.mu_cutoff = mu_cutoff['value']
            self.mu_cutoff_optimizable = mu_cutoff['optimizable']
            self.mu_parameters = mu_parameters
            self.mu_parameters_optimizable = mu_parameters_optimizable
            self.mu_parameters_available = nb.typed.List.empty_list(mu_parameters_mask_type)
            # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
            self.phi_irrotational = phi_irrotational
            self.phi_cusp = phi_cusp
            self.phi_labels = phi_labels
            self.phi_cutoff = phi_cutoff['value']
            self.phi_cutoff_optimizable = phi_cutoff['optimizable']
            self.phi_parameters = phi_parameters
            self.theta_parameters = theta_parameters
            self.phi_parameters_optimizable = phi_parameters_optimizable
            self.theta_parameters_optimizable = theta_parameters_optimizable
            self.phi_parameters_available = nb.typed.List.empty_list(phi_parameters_mask_type)
            self.theta_parameters_available = nb.typed.List.empty_list(theta_parameters_mask_type)
            # spin dep (0->uuu=uud=udd=ddd; 1->uuu/=uud/=udd/=ddd)
            self.omega_cutoff = omega_cutoff['value']
            self.omega_cutoff_optimizable = omega_cutoff['optimizable']
            self.omega_parameters = omega_parameters
            self.omega_parameters_optimizable = omega_parameters_optimizable
            self.omega_parameters_available = np.ones_like(omega_parameters_optimizable)

            self.max_ee_order = max(
                (
                    self.eta_parameters.shape[1],
                    max([p.shape[1] for p in self.phi_parameters]) if self.phi_parameters else 0,
                    self.omega_parameters.shape[1],
                )
            )
            self.max_en_order = max(
                (
                    max([p.shape[1] for p in self.mu_parameters]) if self.mu_parameters else 0,
                    max([p.shape[2] for p in self.phi_parameters]) if self.phi_parameters else 0,
                    2,
                )
            )
            self.ae_cutoff = ae_cutoff
            self.ae_cutoff_optimizable = ae_cutoff_optimizable
            self.cutoffs_optimizable = True
            self.fix_optimizable()
            return self

        return backflow_init(
            config.input.neu,
            config.input.ned,
            config.backflow.trunc,
            config.backflow.eta_parameters,
            config.backflow.eta_parameters_optimizable,
            config.backflow.eta_cutoff,
            config.backflow.mu_parameters,
            config.backflow.mu_parameters_optimizable,
            config.backflow.mu_cutoff,
            config.backflow.mu_cusp,
            config.backflow.mu_labels,
            config.backflow.phi_parameters,
            config.backflow.phi_parameters_optimizable,
            config.backflow.theta_parameters,
            config.backflow.theta_parameters_optimizable,
            config.backflow.phi_cutoff,
            config.backflow.phi_cusp,
            config.backflow.phi_labels,
            config.backflow.phi_irrotational,
            config.backflow.omega_parameters,
            config.backflow.omega_parameters_optimizable,
            config.backflow.omega_cutoff,
            config.backflow.ae_cutoff,
            config.backflow.ae_cutoff_optimizable,
        )

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def trunc(self):
        return self.trunc

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def eta_cutoff(self):
        return self.eta_cutoff

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def mu_cutoff(self):
        return self.mu_cutoff

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def phi_cutoff(self):
        return self.phi_cutoff

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def cutoffs_optimizable(self):
        return self.cutoffs_optimizable

    @cutoffs_optimizable.setter
    @nb.njit(nogil=True, parallel=False, cache=True)
    def cutoffs_optimizable(self, value):
        self.cutoffs_optimizable = value

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def parameters_projector(self):
        return self.parameters_projector

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def eta_parameters(self):
        return self.eta_parameters

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def mu_parameters(self):
        return self.mu_parameters

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def phi_parameters(self):
        return self.phi_parameters

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def theta_parameters(self):
        return self.theta_parameters

    @nb.njit(nogil=True, parallel=False, cache=True)
    def ee_powers(self, e_vectors):
        return self.ee_powers(e_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def en_powers(self, n_vectors):
        return self.en_powers(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def eta_term(self, e_powers, e_vectors):
        return self.eta_term(e_powers, e_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def mu_term(self, n_powers, n_vectors):
        return self.mu_term(n_powers, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def phi_term(self, e_powers, n_powers, e_vectors, n_vectors):
        return self.phi_term(e_powers, n_powers, e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def eta_term_gradient(self, e_powers, e_vectors):
        return self.eta_term_gradient(e_powers, e_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def mu_term_gradient(self, n_powers, n_vectors):
        return self.mu_term_gradient(n_powers, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def phi_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors):
        return self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def eta_term_laplacian(self, e_powers, e_vectors):
        return self.eta_term_laplacian(e_powers, e_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def mu_term_laplacian(self, n_powers, n_vectors):
        return self.mu_term_laplacian(n_powers, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def phi_term_laplacian(self, e_powers, n_powers, e_vectors, n_vectors):
        return self.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value(self, e_vectors, n_vectors):
        return self.value(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def gradient(self, e_vectors, n_vectors):
        return self.gradient(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def laplacian(self, e_vectors, n_vectors):
        return self.laplacian(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value_parameters_d1(self, e_vectors, n_vectors):
        return self.value_parameters_d1(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def gradient_parameters_d1(self, e_vectors, n_vectors):
        return self.gradient_parameters_d1(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def laplacian_parameters_d1(self, e_vectors, n_vectors):
        return self.laplacian_parameters_d1(e_vectors, n_vectors)


structref.define_boxing(Backflow_class_t, Backflow)
