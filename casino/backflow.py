import numpy as np
import numba as nb
from numpy.polynomial.polynomial import polyval
from numba.core import types
from numba.experimental import structref
from numba.core.extending import overload_method


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
    cutoff_constraints = np.zeros(shape=(n_constraints, ))
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
                        cutoff_constraints[k + m + offset + en_constrains] += trunc * phi_parameters[spin_dep, m, l, k] / phi_cutoff ** 2
                    elif l == 1:  # 1b
                        c[k + m + offset + en_constrains, p] = 1
                    if k == 0:  # 1a
                        c[l + m + offset, p] = -trunc / phi_cutoff
                        cutoff_constraints[l + m + offset] += trunc * phi_parameters[spin_dep, m, l, k] / phi_cutoff ** 2
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
                        cutoff_constraints[k + m + offset + ee_constrains + 2 * en_constrains] += trunc * theta_parameters[spin_dep, m, l, k] / phi_cutoff ** 2
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
                        cutoff_constraints[l + m + offset + ee_constrains] += trunc * theta_parameters[spin_dep, m, l, k] / phi_cutoff ** 2
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
                        cutoff_constraints[n] -= (k + 1) * phi_parameters[spin_dep, :, l, k+1].sum()
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


@structref.register
class Backflow_class_t(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


labels_type = nb.int64[:]
eta_parameters_type = nb.float64[:, :]
mu_parameters_type = nb.float64[:, :]
phi_parameters_type = nb.float64[:, :, :, :]
theta_parameters_type = nb.float64[:, :, :, :]
eta_parameters_mask_type = nb.boolean[:, :]
mu_parameters_mask_type = nb.boolean[:, :]
phi_parameters_mask_type = nb.boolean[:, :, :, :]
theta_parameters_mask_type = nb.boolean[:, :, :, :]

Backflow_t = Backflow_class_t([
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('trunc', nb.int64),
    ('eta_parameters', eta_parameters_type),
    ('mu_parameters', nb.types.ListType(mu_parameters_type)),
    ('phi_parameters', nb.types.ListType(phi_parameters_type)),
    ('theta_parameters', nb.types.ListType(theta_parameters_type)),
    ('eta_parameters_optimizable', eta_parameters_mask_type),
    ('mu_parameters_optimizable', nb.types.ListType(mu_parameters_mask_type)),
    ('mu_parameters_available', nb.types.ListType(mu_parameters_mask_type)),
    ('phi_parameters_optimizable', nb.types.ListType(phi_parameters_mask_type)),
    ('theta_parameters_optimizable', nb.types.ListType(theta_parameters_mask_type)),
    ('eta_parameters_available', eta_parameters_mask_type),
    ('phi_parameters_available', nb.types.ListType(phi_parameters_mask_type)),
    ('theta_parameters_available', nb.types.ListType(theta_parameters_mask_type)),
    ('eta_cutoff', nb.float64[:]),
    ('eta_cutoff_optimizable', nb.boolean[:]),
    ('mu_cutoff', nb.float64[:]),
    ('mu_cutoff_optimizable', nb.boolean[:]),
    ('phi_cutoff', nb.float64[:]),
    ('phi_cutoff_optimizable', nb.boolean[:]),
    ('mu_labels', nb.types.ListType(labels_type)),
    ('phi_labels', nb.types.ListType(labels_type)),
    ('max_ee_order', nb.int64),
    ('max_en_order', nb.int64),
    ('mu_cusp', nb.boolean[:]),
    ('phi_cusp', nb.boolean[:]),
    ('phi_irrotational', nb.boolean[:]),
    ('ae_cutoff', nb.float64[:]),
    ('ae_cutoff_optimizable', nb.boolean[:]),
    ('parameters_projector', nb.float64[:, :]),
    ('cutoffs_optimizable', nb.boolean),
])


class Backflow(structref.StructRefProxy):

    def __new__(self, neu, trunc, eta_parameters, eta_parameters_optimizable, eta_cutoff,
        mu_parameters, mu_parameters_optimizable, mu_cutoff, mu_cusp, mu_labels,
        phi_parameters, phi_parameters_optimizable, theta_parameters, theta_parameters_optimizable,
        phi_cutoff, phi_cusp, phi_labels, phi_irrotational, ae_cutoff, ae_cutoff_optimizable):
        return backflow_new(neu, ned, trunc, eta_parameters, eta_parameters_optimizable, eta_cutoff,
        mu_parameters, mu_parameters_optimizable, mu_cutoff, mu_cusp, mu_labels,
        phi_parameters, phi_parameters_optimizable, theta_parameters, theta_parameters_optimizable,
        phi_cutoff, phi_cusp, phi_labels, phi_irrotational, ae_cutoff, ae_cutoff_optimizable)


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
                    res[i, j, k] = res[j, i, k] = r_ee ** k
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
                    res[i, j, k] = r_eI ** k
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
                    res[1, j] = (r/Lg)**2 * (6 - 8 * (r/Lg) + 3 * (r/Lg)**2)
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
                    res[1, j, :, j, :] = 12*r_vec/Lg**2 * (1 - r/Lg)**2
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
                    res[1, j] = 12/Lg**2 * (3 - 8 * (r/Lg) + 5 * (r/Lg)**2)
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
        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                r_vec = e_vectors[e1, e2]
                r = e_powers[e1, e2, 1]
                eta_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % parameters.shape[0]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    bf = (1 - r/L) ** C * r_vec * polyval(r, parameters[eta_set])
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
                        res[ae_cutoff_condition, e1] += (1 - r/L) ** C * r_vec * polyval(r, parameters[mu_set])
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
                            res[ae_cutoff_condition, e1] += (1-r_e1I/L) ** C * (1-r_e2I/L) ** C * (phi_poly * r_ee_vec + theta_poly * r_e1I_vec)
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

                    bf = (1 - r/L)**C * (
                        (poly_diff - C*r/(L - r)*poly) * np.outer(r_vec, r_vec)/r**2 + poly * eye3
                    )
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
                        res[ae_cutoff_condition, e1, :, e1, :] += (1 - r/L)**C * (
                            (poly_diff - C*r/(L - r)*poly) * np.outer(r_vec, r_vec)/r**2 + poly * eye3
                        )

        return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'phi_term_gradient')
def backflow_phi_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors):
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
                            cutoff = (1-r_e1I/L) ** C * (1-r_e2I/L) ** C
                            r_ee_r_ee = np.outer(r_ee_vec, r_ee_vec)
                            r_ee_r_e1I = np.outer(r_ee_vec, r_e1I_vec)
                            res[ae_cutoff_condition, e1, :, e1, :] += cutoff * (
                                (phi_poly_diff_e1I - C*r_e1I/(L - r_e1I)*phi_poly) * r_ee_r_e1I/r_e1I**2 +
                                phi_poly_diff_ee * r_ee_r_ee / r_ee**2 + phi_poly * eye3 +
                                (theta_poly_diff_e1I - C*r_e1I/(L - r_e1I) * theta_poly) * np.outer(r_e1I_vec, r_e1I_vec)/r_e1I**2 +
                                theta_poly_diff_ee * r_ee_r_e1I.T / r_ee**2 + theta_poly * eye3
                            )
                            res[ae_cutoff_condition, e1, :, e2, :] += cutoff * (
                                (phi_poly_diff_e2I - C*r_e2I/(L - r_e2I)*phi_poly) * np.outer(r_ee_vec, r_e2I_vec)/r_e2I**2 -
                                phi_poly_diff_ee * r_ee_r_ee / r_ee**2 - phi_poly * eye3 +
                                (theta_poly_diff_e2I - C*r_e2I/(L - r_e2I) * theta_poly) * np.outer(r_e1I_vec, r_e2I_vec) / r_e2I**2 -
                                theta_poly_diff_ee * r_ee_r_e1I.T / r_ee**2
                            )

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
                    poly = poly_diff = poly_diff_2 = 0
                    for k in range(parameters.shape[1]):
                        p = parameters[eta_set, k] * e_powers[e1, e2, k]
                        poly += p
                        poly_diff += k * p
                        poly_diff_2 += k * (k - 1) * p

                    bf = 2 * (1 - r/L)**C * (
                        4*(poly_diff - C*r/(L - r) * poly) +
                        (C*(C - 1)*r**2/(L - r)**2*poly - 2*C*r/(L - r)*poly_diff + poly_diff_2)
                    ) * r_vec / r**2
                    res[ae_cutoff_condition, e1] += bf
                    res[ae_cutoff_condition, e2] -= bf

        return res.reshape(2, (self.neu + self.ned) * 3)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'mu_term_laplacian')
def backflow_mu_term_laplacian(self, n_powers, n_vectors):
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
                        poly = poly_diff = poly_diff_2 = 0.0
                        for k in range(parameters.shape[1]):
                            p = parameters[mu_set, k] * n_powers[label, e1, k]
                            poly += p
                            poly_diff += k * p
                            poly_diff_2 += k * (k - 1) * p
                        # cutoff_condition
                        # 0: AE cutoff exactly not applied
                        # 1: AE cutoff maybe applied
                        ae_cutoff_condition = int(r > self.ae_cutoff[label])
                        res[ae_cutoff_condition, e1] += (1 - r / L) ** C * (
                                4 * (poly_diff - C * r / (L - r) * poly) +
                                (C * (C - 1) * r ** 2 / (L - r) ** 2 * poly - 2 * C * r / (L - r) * poly_diff + poly_diff_2)
                        ) * r_vec / r ** 2

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
                            cutoff_diff_e1I_2 = C * (C - 1) * r_e1I ** 2 / (L - r_e1I) ** 2
                            cutoff_diff_e2I_2 = C * (C - 1) * r_e2I ** 2 / (L - r_e2I) ** 2
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
                                        poly_diff_e1I_2 = k * (k-1) * poly
                                        phi_poly_diff_e1I_2 += poly_diff_e1I_2 * phi_p
                                        theta_poly_diff_e1I_2 += poly_diff_e1I_2 * theta_p
                                        poly_diff_e2I_2 = l * (l-1) * poly
                                        phi_poly_diff_e2I_2 += poly_diff_e2I_2 * phi_p
                                        theta_poly_diff_e2I_2 += poly_diff_e2I_2 * theta_p
                                        poly_diff_ee_2 = m * (m-1) * poly
                                        phi_poly_diff_ee_2 += poly_diff_ee_2 * phi_p
                                        theta_poly_diff_ee_2 += poly_diff_ee_2 * theta_p
                                        poly_diff_e1I_ee = k * m * poly
                                        phi_poly_diff_e1I_ee += poly_diff_e1I_ee * phi_p
                                        theta_poly_diff_e1I_ee += poly_diff_e1I_ee * theta_p
                                        poly_diff_e2I_ee = l * m * poly
                                        phi_poly_diff_e2I_ee += poly_diff_e2I_ee * phi_p
                                        theta_poly_diff_e2I_ee += poly_diff_e2I_ee * theta_p

                            phi_diff_1 = (
                                (phi_poly_diff_e1I - phi_poly*cutoff_diff_e1I)/r_e1I**2 +
                                (phi_poly_diff_e2I - phi_poly*cutoff_diff_e2I)/r_e2I**2 +
                                4 * phi_poly_diff_ee/r_ee**2
                            )
                            phi_diff_2 = (
                                phi_poly*cutoff_diff_e1I_2/r_e1I**2 - 2*phi_poly_diff_e1I*cutoff_diff_e1I/r_e1I**2 + phi_poly_diff_e1I_2/r_e1I**2 +
                                phi_poly*cutoff_diff_e2I_2/r_e2I**2 - 2*phi_poly_diff_e2I*cutoff_diff_e2I/r_e2I**2 + phi_poly_diff_e2I_2/r_e2I**2 +
                                2 * phi_poly_diff_ee_2/r_ee**2
                            )
                            phi_dot_product = (
                                (phi_poly_diff_e1I - phi_poly*cutoff_diff_e1I) * r_e1I_vec/r_e1I**2 -
                                (phi_poly_diff_e2I - phi_poly*cutoff_diff_e2I) * r_e2I_vec/r_e2I**2 +
                                (phi_poly_diff_e1I_ee - phi_poly_diff_ee*cutoff_diff_e1I) * r_ee_vec * (r_ee_vec @ r_e1I_vec)/r_e1I**2/r_ee**2 -
                                (phi_poly_diff_e2I_ee - phi_poly_diff_ee*cutoff_diff_e2I) * r_ee_vec * (r_ee_vec @ r_e2I_vec)/r_e2I**2/r_ee**2
                            )
                            theta_diff_1 = (
                                2 * (theta_poly_diff_e1I - theta_poly*cutoff_diff_e1I)/r_e1I**2 +
                                (theta_poly_diff_e2I - theta_poly*cutoff_diff_e2I)/r_e2I**2 +
                                2 * theta_poly_diff_ee/r_ee**2
                            )
                            theta_diff_2 = (
                                theta_poly*cutoff_diff_e1I_2/r_e1I**2 - 2*theta_poly_diff_e1I*cutoff_diff_e1I/r_e1I**2 + theta_poly_diff_e1I_2/r_e1I**2 +
                                theta_poly*cutoff_diff_e2I_2/r_e2I**2 - 2*theta_poly_diff_e2I*cutoff_diff_e2I/r_e2I**2 + theta_poly_diff_e2I_2/r_e2I**2 +
                                2 * theta_poly_diff_ee_2/r_ee**2
                            )
                            theta_dot_product = (
                                (theta_poly_diff_e1I_ee - theta_poly_diff_ee*cutoff_diff_e1I) * r_e1I_vec * (r_e1I_vec @ r_ee_vec)/r_e1I**2 -
                                (theta_poly_diff_e2I_ee - theta_poly_diff_ee*cutoff_diff_e2I) * r_e1I_vec * (r_e2I_vec @ r_ee_vec)/r_e2I**2 +
                                theta_poly_diff_ee * r_ee_vec
                            ) / r_ee**2
                            # cutoff_condition
                            # 0: AE cutoff exactly not applied
                            # 1: AE cutoff maybe applied
                            ae_cutoff_condition = int(r_e1I > self.ae_cutoff[label])
                            res[ae_cutoff_condition, e1] += (1-r_e1I/L)**C * (1-r_e2I/L)**C * (
                                (phi_diff_2 + 2 * phi_diff_1) * r_ee_vec + 2 * phi_dot_product +
                                (theta_diff_2 + 2 * theta_diff_1) * r_e1I_vec + 2 * theta_dot_product
                            )

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

        ae_value = eta_term + mu_term + phi_term
        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)

        return np.sum(
            ae_value * ae_multiplier,
            axis=0
        ).reshape((self.neu + self.ned), 3)
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

        eta_term_gradient = self.eta_term_gradient(e_powers, e_vectors)
        mu_term_gradient = self.mu_term_gradient(n_powers, n_vectors)
        phi_term_gradient = self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)

        ae_value = eta_term + mu_term + phi_term
        ae_gradient = eta_term_gradient + mu_term_gradient + phi_term_gradient

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)
        ae_multiplier_gradient = self.ae_multiplier_gradient(n_vectors, n_powers)

        value = np.sum(
            ae_value * ae_multiplier,
            axis=0
        ).reshape((self.neu + self.ned), 3)

        gradient = np.sum(
            ae_multiplier_gradient * np.expand_dims(ae_value, 2) +
            ae_gradient * np.expand_dims(ae_multiplier, 2),
            axis=0
        ) + np.eye((self.neu + self.ned) * 3)

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

        eta_term_gradient = self.eta_term_gradient(e_powers, e_vectors)
        mu_term_gradient = self.mu_term_gradient(n_powers, n_vectors)
        phi_term_gradient = self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)

        eta_term_laplacian = self.eta_term_laplacian(e_powers, e_vectors)
        mu_term_laplacian = self.mu_term_laplacian(n_powers, n_vectors)
        phi_term_laplacian = self.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)

        ae_value = eta_term + mu_term + phi_term
        ae_gradient = eta_term_gradient + mu_term_gradient + phi_term_gradient
        ae_laplacian = eta_term_laplacian + mu_term_laplacian + phi_term_laplacian

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)
        ae_multiplier_gradient = self.ae_multiplier_gradient(n_vectors, n_powers)
        ae_multiplier_laplacian = self.ae_multiplier_laplacian(n_vectors, n_powers)

        value = np.sum(
            ae_value * ae_multiplier,
            axis=0
        ).reshape((self.neu + self.ned), 3)

        gradient = np.sum(
            ae_multiplier_gradient * np.expand_dims(ae_value, 2) +
            ae_gradient * np.expand_dims(ae_multiplier, 2),
            axis=0
        ) + np.eye((self.neu + self.ned) * 3)

        laplacian = np.sum(
            ae_multiplier_laplacian * ae_value +
            2 * (ae_gradient * ae_multiplier_gradient).sum(axis=-1) +
            ae_laplacian * ae_multiplier,
            axis=0
        )

        return laplacian, gradient, value
    return impl


# This associates the proxy with MyStruct_t for the given set of fields.
# Notice how we are not constraining the type of each field.
# Field types remain generic.
structref.define_proxy(Backflow, Backflow_class_t, ['neu', 'ned',
    'trunc', 'eta_parameters', 'mu_parameters', 'phi_parameters', 'theta_parameters',
    'eta_parameters_optimizable', 'mu_parameters_optimizable', 'mu_parameters_available',
    'phi_parameters_optimizable', 'theta_parameters_optimizable', 'eta_parameters_available',
    'phi_parameters_available', 'theta_parameters_available', 'eta_cutoff',
    'eta_cutoff_optimizable', 'mu_cutoff', 'mu_cutoff_optimizable', 'phi_cutoff',
    'phi_cutoff_optimizable', 'mu_labels', 'phi_labels', 'max_ee_order', 'max_en_order',
    'mu_cusp', 'phi_cusp', 'phi_irrotational', 'ae_cutoff', 'ae_cutoff_optimizable',
    'parameters_projector', 'cutoffs_optimizable'])

@nb.njit(nogil=True, parallel=False, cache=True)
def backflow_new(neu, ned, trunc, eta_parameters, eta_parameters_optimizable, eta_cutoff,
        mu_parameters, mu_parameters_optimizable, mu_cutoff, mu_cusp, mu_labels,
        phi_parameters, phi_parameters_optimizable, theta_parameters, theta_parameters_optimizable,
        phi_cutoff, phi_cusp, phi_labels, phi_irrotational, ae_cutoff, ae_cutoff_optimizable
    ):
    self = structref.new(Jastrow_t)
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

    self.max_ee_order = max((
        self.eta_parameters.shape[1],
        max([p.shape[1] for p in self.phi_parameters]) if self.phi_parameters else 0,
    ))
    self.max_en_order = max((
        max([p.shape[1] for p in self.mu_parameters]) if self.mu_parameters else 0,
        max([p.shape[2] for p in self.phi_parameters]) if self.phi_parameters else 0,
        2
    ))
    self.ae_cutoff = ae_cutoff
    self.ae_cutoff_optimizable = ae_cutoff_optimizable
    self.cutoffs_optimizable = True
    self.fix_optimizable()
    return self
