import numpy as np
import numba as nb
from numba.core import types
from numba.experimental import structref
from numba.core.extending import overload_method

from casino.overload import block_diag, rref


@nb.njit(nogil=True, parallel=False, cache=True)
def construct_a_matrix(trunc, f_parameters, f_cutoff, spin_dep, no_dup_u_term, no_dup_chi_term):
    """A-matrix has the following rows:
    (2 * f_en_order + 1) constraints imposed to satisfy electron–electron no-cusp condition.
    (f_en_order + f_ee_order + 1) constraints imposed to satisfy electron–nucleus no-cusp condition.
    (f_ee_order + 1) constraints imposed to prevent duplication of u-term
    (f_en_order + 1) constraints imposed to prevent duplication of chi-term
    copy-paste from /CASINO/src/pjastrow.f90 SUBROUTINE construct_A
    """
    f_ee_order = f_parameters.shape[1] - 1
    f_en_order = f_parameters.shape[2] - 1
    ee_constrains = 2 * f_en_order + 1
    en_constrains = f_en_order + f_ee_order + 1
    no_dup_u_constrains = f_ee_order + 1
    no_dup_chi_constrains = f_en_order + 1

    n_constraints = ee_constrains + en_constrains
    if no_dup_u_term:
        n_constraints += no_dup_u_constrains
    if no_dup_chi_term:
        n_constraints += no_dup_chi_constrains

    parameters_size = (f_en_order + 1) * (f_en_order + 2) * (f_ee_order + 1) // 2
    a = np.zeros(shape=(n_constraints, parameters_size))
    # cutoff constraints column for projector matrix
    cutoff_constraints = np.zeros(shape=(n_constraints, ))
    p = 0
    for n in range(f_ee_order + 1):
        for m in range(f_en_order + 1):
            for l in range(m, f_en_order + 1):
                if n == 1:
                    if l == m:
                        a[l + m, p] = 1
                    else:
                        a[l + m, p] = 2
                if m == 1:
                    a[l + n + ee_constrains, p] = -f_cutoff
                    cutoff_constraints[l + n + ee_constrains] -= f_parameters[spin_dep, n, m, l]
                elif m == 0:
                    a[l + n + ee_constrains, p] = trunc
                    if l == 1:
                        a[n + ee_constrains, p] = -f_cutoff
                        cutoff_constraints[n + ee_constrains] -= f_parameters[spin_dep, n, m, l]
                    elif l == 0:
                        a[n + ee_constrains, p] = trunc
                    if no_dup_u_term:
                        if l == 0:
                            a[n + ee_constrains + en_constrains, p] = 1
                        if no_dup_chi_term and n == 0:
                            a[l + ee_constrains + en_constrains + no_dup_u_constrains, p] = 1
                    else:
                        if no_dup_chi_term and n == 0:
                            a[l + ee_constrains + en_constrains, p] = 1
                p += 1
    return a, cutoff_constraints

@structref.register
class Jastrow_class_t(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


Jastrow_t = Jastrow_class_t([
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('trunc', nb.int64),
    ('u_parameters', u_parameters_type),
    ('chi_parameters', nb.types.ListType(chi_parameters_type)),
    ('f_parameters', nb.types.ListType(f_parameters_type)),
    ('u_parameters_optimizable', u_parameters_mask_type),
    ('chi_parameters_optimizable', nb.types.ListType(chi_parameters_mask_type)),
    ('f_parameters_optimizable', nb.types.ListType(f_parameters_mask_type)),
    ('u_parameters_available', u_parameters_mask_type),
    ('chi_parameters_available', nb.types.ListType(chi_parameters_mask_type)),
    ('f_parameters_available', nb.types.ListType(f_parameters_mask_type)),
    ('u_cutoff', nb.float64),
    ('u_cutoff_optimizable', nb.boolean),
    ('chi_cutoff', nb.float64[:]),
    ('chi_cutoff_optimizable', nb.boolean[:]),
    ('f_cutoff', nb.float64[:]),
    ('f_cutoff_optimizable', nb.boolean[:]),
    ('chi_labels', nb.types.ListType(labels_type)),
    ('f_labels', nb.types.ListType(labels_type)),
    ('max_ee_order', nb.int64),
    ('max_en_order', nb.int64),
    ('chi_cusp', nb.boolean[:]),
    ('no_dup_u_term', nb.boolean[:]),
    ('no_dup_chi_term', nb.boolean[:]),
    ('parameters_projector', nb.float64[:, :]),
    ('cutoffs_optimizable', nb.boolean),
])


class Jastrow(structref.StructRefProxy):

    def __new__(self, neu, ned, trunc, u_parameters, u_parameters_optimizable, u_cutoff,
        chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_labels, chi_cusp,
        f_parameters, f_parameters_optimizable, f_cutoff, f_labels, no_dup_u_term, no_dup_chi_term):
        return jastrow_new(neu, ned, trunc, u_parameters, u_parameters_optimizable, u_cutoff,
        chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_labels, chi_cusp,
        f_parameters, f_parameters_optimizable, f_cutoff, f_labels, no_dup_u_term, no_dup_chi_term)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'fix_optimizable')
def jastrow_fix_optimizable(self):
    """Set parameter optimisation to "fixed" if there is no corresponded spin-pairs"""
    def impl(self):
        if self.neu + self.ned == 1:
            # H-atom
            self.u_cutoff_optimizable = False
            for i in range(len(self.f_cutoff_optimizable)):
                self.f_cutoff_optimizable[i] = False

        ee_order = 2
        if self.u_parameters.shape[0] == 2:
            if self.neu < ee_order and self.ned < ee_order:
                self.u_parameters_available[0] = False
            if self.neu + self.ned < ee_order:
                self.u_parameters_available[1] = False
        elif self.u_parameters.shape[1] == 3:
            if self.neu < ee_order:
                self.u_parameters_available[0] = False
            if self.neu + self.ned < ee_order:
                self.u_parameters_available[1] = False
            if self.ned < ee_order:
                self.u_parameters_available[2] = False

        ee_order = 1
        for chi_parameters_optimizable in self.chi_parameters_optimizable:
            chi_parameters_available = np.ones_like(chi_parameters_optimizable)
            if chi_parameters_optimizable.shape[0] == 2:
                if self.neu < ee_order:
                    chi_parameters_available[0] = False
                if self.ned < ee_order:
                    chi_parameters_available[1] = False
            self.chi_parameters_available.append(chi_parameters_available)

        ee_order = 2
        for f_parameters_optimizable in self.f_parameters_optimizable:
            f_parameters_available = np.ones_like(f_parameters_optimizable)
            for j1 in range(f_parameters_optimizable.shape[2]):
                for j2 in range(f_parameters_optimizable.shape[3]):
                    if j1 > j2:
                        f_parameters_available[:, :, j1, j2] = False
            if f_parameters_optimizable.shape[0] == 2:
                if self.neu < ee_order and self.ned < ee_order:
                    f_parameters_available[0] = False
                if self.neu + self.ned < ee_order:
                    f_parameters_available[1] = False
            elif f_parameters_optimizable.shape[0] == 3:
                if self.neu < ee_order:
                    f_parameters_available[0] = False
                if self.neu + self.ned < ee_order:
                    f_parameters_available[1] = False
                if self.ned < ee_order:
                    f_parameters_available[2] = False
            self.f_parameters_available.append(f_parameters_available)
        return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'ee_powers')
def jastrow_ee_powers(self, e_vectors: np.ndarray):
    """Powers of e-e distances
    :param e_vectors: e-e vectors - array(nelec, nelec, 3)
    :return: powers of e-e distances - array(nelec, nelec, max_ee_order)
    """
    def impl(self, e_vectors: np.ndarray) -> np.ndarray:
        res = np.ones(shape=(e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(1, self.max_ee_order):
                    res[i, j, k] = res[j, i, k] = r_ee ** k
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'en_powers')
def jastrow_en_powers(self, n_vectors: np.ndarray):
    """Powers of e-n distances
    :param n_vectors: e-n vectors - array(natom, nelec, 3)
    :return: powers of e-n distances - array(natom, nelec, max_en_order)
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
@overload_method(Jastrow_class_t, 'u_term')
def jastrow_u_term(self, e_powers: np.ndarray):
    """Jastrow u-term
    :param e_powers: powers of e-e distances
    :return:
    """
    def impl(self, e_powers: np.ndarray) -> float:
        res = 0.0
        if not self.u_cutoff:
            return res

        C = self.trunc
        parameters = self.u_parameters
        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                r = e_powers[e1, e2, 1]
                if r < self.u_cutoff:
                    cusp_set = int(e1 >= self.neu) + int(e2 >= self.neu)
                    u_set = cusp_set % parameters.shape[0]
                    poly = 0.0
                    for k in range(parameters.shape[1]):
                        if parameters.shape[0] == 1 and cusp_set == 1 and k == 1:
                            p = parameters[u_set, k] * 2
                        else:
                            p = parameters[u_set, k]
                        poly += p * e_powers[e1, e2, k]
                    res += poly * (r - self.u_cutoff) ** C
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'chi_term')
def jastrow_chi_term(self, n_powers: np.ndarray):
    """Jastrow chi-term
    :param n_powers: powers of e-n distances
    :return:
    """
    def impl(self, n_powers: np.ndarray) -> float:
        res = 0.0
        C = self.trunc
        for parameters, L, chi_labels in zip(self.chi_parameters, self.chi_cutoff, self.chi_labels):
            for label in chi_labels:
                for e1 in range(self.neu + self.ned):
                    r = n_powers[label, e1, 1]
                    if r < L:
                        chi_set = int(e1 >= self.neu) % parameters.shape[0]
                        res += polyval(r, parameters[chi_set]) * (r - L) ** C
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'f_term')
def jastrow_f_term(self, e_powers, n_powers):
    """Jastrow f-term
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :return:
    """
    def impl(self, e_powers, n_powers) -> float:
        res = 0.0
        C = self.trunc
        for parameters, L, f_labels in zip(self.f_parameters, self.f_cutoff, self.f_labels):
            for label in f_labels:
                for e1 in range(1, self.neu + self.ned):
                    for e2 in range(e1):
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % parameters.shape[0]
                            # FIXME: polyval3d not supported
                            # r_ee = e_powers[e1, e2, 1]
                            # res += (r_e1I - L)**C * (r_e2I - L)**C * polyval3d(r_ee, r_e1I, r_e2I, parameters[f_set])
                            poly = 0.0
                            for l in range(parameters.shape[3]):
                                for m in range(l, parameters.shape[2]):
                                    en_part = n_powers[label, e1, l] * n_powers[label, e2, m]
                                    if l != m:
                                        en_part += n_powers[label, e1, m] * n_powers[label, e2, l]
                                    for n in range(parameters.shape[1]):
                                        poly += parameters[f_set, n, m, l] * en_part * e_powers[e1, e2, n]
                            res += poly * (r_e1I - L) ** C * (r_e2I - L) ** C
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'u_term_gradient')
def jastrow_u_term_gradient(self):
    """Jastrow u-term gradient w.r.t e-coordinates
    :param e_powers: powers of e-e distances
    :param e_vectors: e-e vectors
    :return:
    """
    def impl(self) -> np.ndarray:
        res = np.zeros(shape=(self.neu + self.ned, 3))

        if not self.u_cutoff:
            return res.ravel()

        C = self.trunc
        L = self.u_cutoff
        parameters = self.u_parameters
        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                r = e_powers[e1, e2, 1]
                if r < L:
                    r_vec = e_vectors[e1, e2] / r
                    cusp_set = (int(e1 >= self.neu) + int(e2 >= self.neu))
                    u_set = cusp_set % parameters.shape[0]
                    poly = 0.0
                    for k in range(parameters.shape[1]):
                        if parameters.shape[0] == 1 and cusp_set == 1 and k == 1:
                            p = parameters[u_set, k] * e_powers[e1, e2, k] * 2
                        else:
                            p = parameters[u_set, k] * e_powers[e1, e2, k]
                        poly += (C/(r-L) + k/r) * p

                    gradient = r_vec * (r-L) ** C * poly
                    res[e1, :] += gradient
                    res[e2, :] -= gradient
        return res.ravel()
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'chi_term_gradient')
def jastrow_chi_term_gradient(self):
    """Jastrow chi-term gradient w.r.t e-coordinates
    :param n_powers: powers of e-n distances
    :param n_vectors: e-n vectors
    :return:
    """
    def impl(self) -> np.ndarray:
        C = self.trunc
        res = np.zeros(shape=(self.neu + self.ned, 3))
        for parameters, L, chi_labels in zip(self.chi_parameters, self.chi_cutoff, self.chi_labels):
            k = np.arange(parameters.shape[1])
            for label in chi_labels:
                for e1 in range(self.neu + self.ned):
                    r = n_powers[label, e1, 1]
                    if r < L:
                        r_vec = n_vectors[label, e1] / r
                        chi_set = int(e1 >= self.neu) % parameters.shape[0]
                        res[e1, :] += r_vec * (r-L) ** C * polyval(r, (C*r/(r-L) + k) * parameters[chi_set]) / r
        return res.ravel()
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'f_term_gradient')
def jastrow_f_term_gradient(self):
    """Jastrow f-term gradient w.r.t e-coordinates
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """
    def impl(self) -> np.ndarray:
        C = self.trunc
        res = np.zeros(shape=(self.neu + self.ned, 3))
        for parameters, L, f_labels in zip(self.f_parameters, self.f_cutoff, self.f_labels):
            for label in f_labels:
                for e1 in range(1, self.neu + self.ned):
                    for e2 in range(e1):
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        r_ee = e_powers[e1, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            r_e1I_vec = n_vectors[label, e1] / r_e1I
                            r_e2I_vec = n_vectors[label, e2] / r_e2I
                            r_ee_vec = e_vectors[e1, e2] / r_ee
                            f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % parameters.shape[0]
                            # FIXME: polyval3d not supported
                            # r_ee = e_powers[e1, e2, 1]
                            # k = np.arange(parameters.shape[0])
                            # cutoff = (r_e1I - L) ** C * (r_e2I - L) ** C
                            # e1_gradient = r_e1I_vec * (C/(r_e1I - L) * poly + poly_diff_e1I/r_e1I)
                            # e2_gradient = r_e2I_vec * (C/(r_e2I - L) * poly + poly_diff_e2I/r_e2I)
                            # ee_gradient = r_ee_vec * polyval3d(r_ee, r_e1I, r_e2I, k[np.newaxis, np.newaxis, :] * parameters[f_set])/r_ee
                            # res[e1] += cutoff * (e1_gradient + ee_gradient)
                            # res[e2] += cutoff * (e2_gradient - ee_gradient)
                            poly = poly_diff_e1I = poly_diff_e2I = poly_diff_ee = 0.0
                            for l in range(parameters.shape[3]):
                                for m in range(parameters.shape[2]):
                                    en_part = n_powers[label, e1, l] * n_powers[label, e2, m]
                                    for n in range(parameters.shape[1]):
                                        p = parameters[f_set, n, m, l] * en_part * e_powers[e1, e2, n]
                                        poly += p
                                        poly_diff_e1I += l * p
                                        poly_diff_e2I += m * p
                                        poly_diff_ee += n * p
                            cutoff = (r_e1I - L) ** C * (r_e2I - L) ** C
                            # workaround do not create temporary 1-d numpy array
                            for t1 in range(3):
                                e1_gradient = r_e1I_vec[t1] * (C/(r_e1I - L) * poly + poly_diff_e1I/r_e1I)
                                e2_gradient = r_e2I_vec[t1] * (C/(r_e2I - L) * poly + poly_diff_e2I/r_e2I)
                                ee_gradient = r_ee_vec[t1] * poly_diff_ee/r_ee
                                res[e1, t1] += cutoff * (e1_gradient + ee_gradient)
                                res[e2, t1] += cutoff * (e2_gradient - ee_gradient)
        return res.ravel()
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'u_term_laplacian')
def jastrow_u_term_laplacian(self):
    """Jastrow u-term laplacian w.r.t e-coordinates
    :param e_powers: powers of e-e distances
    :return:
    """
    def impl(self) -> float:
        res = 0.0
        if not self.u_cutoff:
            return res

        C = self.trunc
        L = self.u_cutoff
        parameters = self.u_parameters
        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                r = e_powers[e1, e2, 1]
                if r < L:
                    cusp_set = (int(e1 >= self.neu) + int(e2 >= self.neu))
                    u_set = cusp_set % parameters.shape[0]
                    poly = poly_diff = poly_diff_2 = 0.0
                    for k in range(parameters.shape[1]):
                        if parameters.shape[1] == 1 and cusp_set == 1 and k == 1:
                            p = parameters[u_set, k] * e_powers[e1, e2, k] * 2
                        else:
                            p = parameters[u_set, k] * e_powers[e1, e2, k]
                        poly += p
                        poly_diff += k * p
                        poly_diff_2 += k * (k-1) * p

                    res += (r-L)**C * (
                        C*(C-1)*r**2/(r-L)**2 * poly +
                        2 * C*r/(r-L) * (poly + poly_diff) +
                        poly_diff_2 + 2 * poly_diff
                    ) / r**2
        return 2 * res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'chi_term_laplacian')
def jastrow_chi_term_laplacian(self):
    """Jastrow chi-term laplacian w.r.t e-coordinates
    :param n_powers: powers of e-n distances
    :return:
    """
    def impl(self) -> float:
        res = 0.0
        C = self.trunc
        for parameters, L, chi_labels in zip(self.chi_parameters, self.chi_cutoff, self.chi_labels):
            k = np.arange(parameters.shape[1])
            k_1 = np.arange(-1, parameters.shape[1] - 1)
            for label in chi_labels:
                for e1 in range(self.neu + self.ned):
                    r = n_powers[label, e1, 1]
                    if r < L:
                        chi_set = int(e1 >= self.neu) % parameters.shape[0]
                        res += (r-L) ** C * polyval(r, (C*(C-1)*r**2/(r-L)**2 + 2*C*r/(r-L)*(1 + k) + 2*k + k * k_1) * parameters[chi_set]) / r**2
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'f_term_laplacian')
def jastrow_f_term_laplacian(self):
    """Jastrow f-term laplacian w.r.t e-coordinates
    f-term is a product of two spherically symmetric functions f(r_eI) and g(r_ee) so using
        ∇²(f*g) = ∇²(f)*g + 2*∇(f)*∇(g) + f*∇²(g)
    then Laplace operator of spherically symmetric function (in 3-D space) is
        ∇²(f) = d²f/dr² + 2/r * df/dr
    also using: r_ee_vec = r_e1I_vec - r_e2I_vec
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """
    def impl(self) -> float:
        res = 0.0
        C = self.trunc
        for parameters, L, f_labels in zip(self.f_parameters, self.f_cutoff, self.f_labels):
            for label in f_labels:
                r_e1I_vec_dot_r_e2I_vec = n_vectors[label] @ n_vectors[label].T
                for e1 in range(1, self.neu + self.ned):
                    for e2 in range(e1):
                        # r_e1I_vec = n_vectors[label, e1]
                        # r_e2I_vec = n_vectors[label, e2]
                        # r_ee_vec = e_vectors[e1, e2]
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        r_ee = e_powers[e1, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % parameters.shape[0]
                            # FIXME: polyval3d not supported
                            cutoff_diff_e1I = C*r_e1I/(r_e1I - L)
                            cutoff_diff_e2I = C*r_e2I/(r_e2I - L)
                            poly = poly_diff_e1I = poly_diff_e2I = 0.0
                            poly_diff_ee = poly_diff_e1I_2 = poly_diff_e2I_2 = 0.0
                            poly_diff_ee_2 = poly_diff_e1I_ee = poly_diff_e2I_ee = 0.0
                            for l in range(parameters.shape[3]):
                                for m in range(parameters.shape[2]):
                                    en_part = n_powers[label, e1, l] * n_powers[label, e2, m]
                                    for n in range(parameters.shape[1]):
                                        p = parameters[f_set, n, m, l] * en_part * e_powers[e1, e2, n]
                                        poly += p
                                        poly_diff_e1I += l * p
                                        poly_diff_e2I += m * p
                                        poly_diff_ee += n * p
                                        poly_diff_e1I_2 += l * (l-1) * p
                                        poly_diff_e2I_2 += m * (m-1) * p
                                        poly_diff_ee_2 += n * (n-1) * p
                                        poly_diff_e1I_ee += l * n * p
                                        poly_diff_e2I_ee += m * n * p

                            diff_1 = (
                                (cutoff_diff_e1I * poly + poly_diff_e1I) / r_e1I**2 +
                                (cutoff_diff_e2I * poly + poly_diff_e2I) / r_e2I**2 +
                                2 * poly_diff_ee / r_ee**2
                            )
                            diff_2 = (
                                C*(C-1)/(r_e1I - L)**2 * poly +
                                C*(C-1)/(r_e2I - L)**2 * poly +
                                poly_diff_e1I_2 / r_e1I**2 + poly_diff_e2I_2 / r_e2I**2 + 2 * poly_diff_ee_2 / r_ee**2 +
                                2 * cutoff_diff_e1I * poly_diff_e1I / r_e1I**2 +
                                2 * cutoff_diff_e2I * poly_diff_e2I / r_e2I**2
                            )
                            # dot_product = (
                            #     (r_e1I_vec @ r_ee_vec) * (cutoff_diff_e1I * poly_diff_ee + poly_diff_e1I_ee) / r_e1I**2 -
                            #     (r_e2I_vec @ r_ee_vec) * (cutoff_diff_e2I * poly_diff_ee + poly_diff_e2I_ee) / r_e2I**2
                            # ) / r_ee**2
                            dot_product = (
                                (1 - r_e1I_vec_dot_r_e2I_vec[e1, e2] / r_e1I**2) * (cutoff_diff_e1I * poly_diff_ee + poly_diff_e1I_ee) +
                                (1 - r_e1I_vec_dot_r_e2I_vec[e1, e2] / r_e2I**2) * (cutoff_diff_e2I * poly_diff_ee + poly_diff_e2I_ee)
                            ) / r_ee**2
                            res += (r_e1I - L) ** C * (r_e2I - L) ** C * (diff_2 + 2 * diff_1 + 2 * dot_product)
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'value')
def jastrow_value(self, e_vectors, n_vectors):
    """Jastrow value
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :param neu: number of up electrons
    :return:
    """
    def impl(self, e_vectors, n_vectors) -> float:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term(e_powers) +
            self.chi_term(n_powers) +
            self.f_term(e_powers, n_powers)
        )
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'gradient')
def jastrow_gradient(self, e_vectors, n_vectors):
    """Jastrow gradient w.r.t. e-coordinates
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """
    def impl(self, e_vectors, n_vectors) -> np.ndarray:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term_gradient(e_powers, e_vectors) +
            self.chi_term_gradient(n_powers, n_vectors) +
            self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
        )
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'laplacian')
def jastrow_laplacian(self, e_vectors, n_vectors):
    """Jastrow laplacian w.r.t e-coordinates
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """
    def impl(self, e_vectors, n_vectors) -> float:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term_laplacian(e_powers) +
            self.chi_term_laplacian(n_powers) +
            self.f_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)
        )
    return impl


# This associates the proxy with MyStruct_t for the given set of fields.
# Notice how we are not constraining the type of each field.
# Field types remain generic.
structref.define_proxy(Jastrow, Jastrow_class_t, ['neu', 'ned',
    'trunc', 'u_parameters', 'chi_parameters', 'f_parameters',
    'u_parameters_optimizable', 'chi_parameters_optimizable', 'f_parameters_optimizable',
    'u_parameters_available', 'chi_parameters_available', 'f_parameters_available',
    'u_cutoff', 'u_cutoff_optimizable', 'chi_cutoff', 'chi_cutoff_optimizable', 'f_cutoff', 'f_cutoff_optimizable',
    'chi_labels', 'f_labels', 'max_ee_order', 'max_en_order', 'chi_cusp', 'no_dup_u_term', 'no_dup_chi_term',
    'parameters_projector', 'cutoffs_optimizable'])


@nb.njit(nogil=True, parallel=False, cache=True)
def jastrow_new(neu, ned, trunc, u_parameters, u_parameters_optimizable, u_cutoff,
        chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_labels, chi_cusp,
        f_parameters, f_parameters_optimizable, f_cutoff, f_labels, no_dup_u_term, no_dup_chi_term
    ):
    self = structref.new(Jastrow_t)
    self.neu = neu
    self.ned = ned
    self.trunc = trunc
    # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
    self.u_cutoff = u_cutoff[0]['value']
    self.u_cutoff_optimizable = u_cutoff[0]['optimizable']
    self.u_parameters = u_parameters
    self.u_parameters_optimizable = u_parameters_optimizable
    self.u_parameters_available = np.ones_like(u_parameters_optimizable)
    # spin dep (0->u=d; 1->u/=d)
    self.chi_labels = chi_labels
    self.chi_cutoff = chi_cutoff['value']
    self.chi_cutoff_optimizable = chi_cutoff['optimizable']
    self.chi_parameters = chi_parameters
    self.chi_parameters_optimizable = chi_parameters_optimizable
    self.chi_parameters_available = nb.typed.List.empty_list(chi_parameters_mask_type)
    # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
    self.f_labels = f_labels
    self.f_cutoff = f_cutoff['value']
    self.f_cutoff_optimizable = f_cutoff['optimizable']
    self.f_parameters = f_parameters
    self.f_parameters_optimizable = f_parameters_optimizable
    self.f_parameters_available = nb.typed.List.empty_list(f_parameters_mask_type)

    self.max_ee_order = max((
        self.u_parameters.shape[1],
        max([p.shape[1] for p in self.f_parameters]) if self.f_parameters else 0,
    ))
    self.max_en_order = max((
        max([p.shape[1] for p in self.chi_parameters]) if self.chi_parameters else 0,
        max([p.shape[2] for p in self.f_parameters]) if self.f_parameters else 0,
    ))
    self.chi_cusp = chi_cusp
    self.no_dup_u_term = no_dup_u_term
    self.no_dup_chi_term = no_dup_chi_term
    self.cutoffs_optimizable = True
    self.fix_optimizable()
    return self