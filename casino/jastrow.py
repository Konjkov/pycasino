import numpy as np
import numba as nb
from numpy.polynomial.polynomial import polyval
from numba.core import types
from numba.experimental import structref
from numba.core.extending import overload_method

from casino import delta
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


labels_type = nb.int64[::1]
u_parameters_type = nb.float64[:, ::1]
chi_parameters_type = nb.float64[:, ::1]
f_parameters_type = nb.float64[:, :, :, ::1]
u_parameters_mask_type = nb.boolean[:, ::1]
chi_parameters_mask_type = nb.boolean[:, ::1]
f_parameters_mask_type = nb.boolean[:, :, :, ::1]

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
    ('chi_cusp', nb.boolean[::1]),
    ('no_dup_u_term', nb.boolean[::1]),
    ('no_dup_chi_term', nb.boolean[::1]),
    ('parameters_projector', nb.float64[:, ::1]),
    ('cutoffs_optimizable', nb.boolean),
])


class Jastrow(structref.StructRefProxy):

    def __new__(cls, neu, ned, trunc, u_parameters, u_parameters_optimizable, u_cutoff,
        chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_labels, chi_cusp,
        f_parameters, f_parameters_optimizable, f_cutoff, f_labels, no_dup_u_term, no_dup_chi_term):
        return jastrow_new(neu, ned, trunc, u_parameters, u_parameters_optimizable, u_cutoff,
        chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_labels, chi_cusp,
        f_parameters, f_parameters_optimizable, f_cutoff, f_labels, no_dup_u_term, no_dup_chi_term)

    @property
    def u_cutoff(self):
        return jastrow_u_cutoff_get(self)

    @property
    def cutoffs_optimizable(self):
        jastrow_cutoffs_optimizable_get(self)

    @cutoffs_optimizable.setter
    def cutoffs_optimizable(self, value):
        jastrow_cutoffs_optimizable_set(self, value)


@nb.njit(nogil=True, parallel=False, cache=True)
def jastrow_u_cutoff_get(self) -> float:
    """u_cuoff."""
    return self.u_cutoff


@nb.njit(nogil=True, parallel=False, cache=True)
def jastrow_cutoffs_optimizable_get(self):
    """cutoffs_optimizable getter."""
    return self.cutoffs_optimizable


@nb.njit(nogil=True, parallel=False, cache=True)
def jastrow_cutoffs_optimizable_set(self, value):
    """cutoffs_optimizable setter."""
    self.cutoffs_optimizable = value


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
def jastrow_u_term_gradient(self, e_powers, e_vectors):
    """Jastrow u-term gradient w.r.t e-coordinates
    :param e_powers: powers of e-e distances
    :param e_vectors: e-e vectors
    :return:
    """
    def impl(self, e_powers, e_vectors) -> np.ndarray:
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
def jastrow_chi_term_gradient(self, n_powers, n_vectors):
    """Jastrow chi-term gradient w.r.t e-coordinates
    :param n_powers: powers of e-n distances
    :param n_vectors: e-n vectors
    :return:
    """
    def impl(self, n_powers, n_vectors) -> np.ndarray:
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
def jastrow_f_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors):
    """Jastrow f-term gradient w.r.t e-coordinates
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """
    def impl(self, e_powers, n_powers, e_vectors, n_vectors) -> np.ndarray:
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
def jastrow_u_term_laplacian(self, e_powers):
    """Jastrow u-term laplacian w.r.t e-coordinates
    :param e_powers: powers of e-e distances
    :return:
    """
    def impl(self, e_powers) -> float:
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
def jastrow_chi_term_laplacian(self, n_powers):
    """Jastrow chi-term laplacian w.r.t e-coordinates
    :param n_powers: powers of e-n distances
    :return:
    """
    def impl(self, n_powers) -> float:
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
def jastrow_f_term_laplacian(self, e_powers, n_powers, e_vectors, n_vectors):
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
    def impl(self, e_powers, n_powers, e_vectors, n_vectors) -> float:
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


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'set_u_parameters_for_emin')
def jastrow_set_u_parameters_for_emin(self):
    """Set u-term dependent parameters for CASINO emin."""
    def impl(self):
        C = self.trunc
        L = self.u_cutoff
        Gamma = 1 / np.array([4, 2, 4][:self.u_parameters.shape[0]])
        self.u_parameters[:, 0] = -L * Gamma / (-L) ** C / C
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'fix_u_parameters')
def jastrow_fix_u_parameters(self):
    """Fix u-term dependent parameters."""
    def impl(self):
        C = self.trunc
        L = self.u_cutoff
        Gamma = 1 / np.array([4, 2, 4][:self.u_parameters.shape[0]])
        self.u_parameters[:, 1] = Gamma / (-L) ** C + self.u_parameters[:, 0] * C / L
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'fix_chi_parameters')
def jastrow_fix_chi_parameters(self):
    """Fix chi-term dependent parameters."""
    def impl(self):
        C = self.trunc
        for chi_parameters, L, chi_cusp in zip(self.chi_parameters, self.chi_cutoff, self.chi_cusp):
            chi_parameters[:, 1] = chi_parameters[:, 0] * C / L
            if chi_cusp:
                pass
                # FIXME: chi cusp not implemented
                # chi_parameters[:, 1] -= charge / (-L) ** C
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'fix_f_parameters')
def jastrow_fix_f_parameters(self):
    """Fix f-term dependent parameters.
    To find the dependent coefficients of f-term it is necessary to solve
    the system of linear equations:  A*x=b
    A-matrix has the following rows:
    (2 * f_en_order + 1) constraints imposed to satisfy electron–electron no-cusp condition.
    (f_en_order + f_ee_order + 1) constraints imposed to satisfy electron–nucleus no-cusp condition.
    (f_ee_order + 1) constraints imposed to prevent duplication of u-term
    (f_en_order + 1) constraints imposed to prevent duplication of chi-term
    b-column has the sum of independent coefficients for each condition.
    """
    def impl(self):
        for f_parameters, L, no_dup_u_term, no_dup_chi_term in zip(self.f_parameters, self.f_cutoff, self.no_dup_u_term, self.no_dup_chi_term):
            f_spin_dep = f_parameters.shape[0] - 1
            f_ee_order = f_parameters.shape[1] - 1
            f_en_order = f_parameters.shape[2] - 1

            a, _ = construct_a_matrix(self.trunc, f_parameters, L, 0, no_dup_u_term, no_dup_chi_term)
            a, pivot_positions = rref(a)
            # remove zero-rows
            a = a[:pivot_positions.size, :]
            b = np.zeros(shape=(f_spin_dep+1, a.shape[0]))
            p = 0
            for n in range(f_ee_order + 1):
                for m in range(f_en_order + 1):
                    for l in range(m, f_en_order + 1):
                        if p not in pivot_positions:
                            for temp in range(pivot_positions.size):
                                b[:, temp] -= a[temp, p] * f_parameters[:, n, m, l]
                        p += 1

            x = np.empty(shape=(f_spin_dep + 1, pivot_positions.size))
            for i in range(f_spin_dep + 1):
                x[i, :] = np.linalg.solve(a[:, pivot_positions], b[i])

            p = 0
            temp = 0
            for n in range(f_ee_order + 1):
                for m in range(f_en_order + 1):
                    for l in range(m, f_en_order + 1):
                        if temp in pivot_positions:
                            f_parameters[:, n, m, l] = f_parameters[:, n, l, m] = x[:, p]
                            p += 1
                        temp += 1
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'get_parameters_mask')
def jastrow_get_parameters_mask(self):
    """Mask optimizable parameters."""
    def impl(self) -> np.ndarray:
        res = []
        if self.u_cutoff:
            if self.u_cutoff_optimizable and self.cutoffs_optimizable:
                res.append(1)
            for j1 in range(self.u_parameters.shape[0]):
                for j2 in range(self.u_parameters.shape[1]):
                    if self.u_parameters_available[j1, j2]:
                        res.append(self.u_parameters_optimizable[j1, j2])

        if self.chi_cutoff.any():
            for chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_cutoff_optimizable, chi_parameters_available in zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_cutoff, self.chi_cutoff_optimizable, self.chi_parameters_available):
                if chi_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(1)
                for j1 in range(chi_parameters.shape[0]):
                    for j2 in range(chi_parameters.shape[1]):
                        if chi_parameters_available[j1, j2]:
                            res.append(chi_parameters_optimizable[j1, j2])

        if self.f_cutoff.any():
            for f_parameters, f_parameters_optimizable, f_cutoff, f_cutoff_optimizable, f_parameters_available in zip(self.f_parameters, self.f_parameters_optimizable, self.f_cutoff, self.f_cutoff_optimizable, self.f_parameters_available):
                if f_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(1)
                for j1 in range(f_parameters.shape[0]):
                    for j2 in range(f_parameters.shape[1]):
                        for j3 in range(f_parameters.shape[2]):
                            for j4 in range(j3, f_parameters.shape[3]):
                                if f_parameters_available[j1, j2, j3, j4]:
                                    res.append(f_parameters_optimizable[j1, j2, j3, j4])

        return np.array(res)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'get_parameters_scale')
def jastrow_get_parameters_scale(self, all_parameters):
    """Characteristic scale of each variable. Setting x_scale is equivalent
    to reformulating the problem in scaled variables xs = x / x_scale.
    An alternative view is that the size of a trust region along j-th
    dimension is proportional to x_scale[j].
    The purpose of this method is to reformulate the optimization problem
    with dimensionless variables having only one dimensional parameter - scale.
    """
    def impl(self, all_parameters) -> np.ndarray:
        scale = []
        ne = self.neu + self.ned
        if self.u_cutoff:
            if self.u_cutoff_optimizable and self.cutoffs_optimizable:
                scale.append(1)
            for j1 in range(self.u_parameters.shape[0]):
                for j2 in range(self.u_parameters.shape[1]):
                    if (self.u_parameters_optimizable[j1, j2] or all_parameters) and self.u_parameters_available[j1, j2]:
                        scale.append(2 / self.u_cutoff ** j2 / ne ** 2)

        if self.chi_cutoff.any():
            for chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_cutoff_optimizable, chi_parameters_available in zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_cutoff, self.chi_cutoff_optimizable, self.chi_parameters_available):
                if chi_cutoff_optimizable and self.cutoffs_optimizable:
                    scale.append(1)
                for j1 in range(chi_parameters.shape[0]):
                    for j2 in range(chi_parameters.shape[1]):
                        if (chi_parameters_optimizable[j1, j2] or all_parameters) and chi_parameters_available[j1, j2]:
                            scale.append(1 / chi_cutoff ** j2 / ne)

        if self.f_cutoff.any():
            for f_parameters, f_parameters_optimizable, f_cutoff, f_cutoff_optimizable, f_parameters_available in zip(self.f_parameters, self.f_parameters_optimizable, self.f_cutoff, self.f_cutoff_optimizable, self.f_parameters_available):
                if f_cutoff_optimizable and self.cutoffs_optimizable:
                    scale.append(1)
                for j1 in range(f_parameters.shape[0]):
                    for j2 in range(f_parameters.shape[1]):
                        for j3 in range(f_parameters.shape[2]):
                            for j4 in range(j3, f_parameters.shape[3]):
                                if (f_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and f_parameters_available[j1, j2, j3, j4]:
                                    scale.append(2 / f_cutoff ** (j2 + j3 + j4) / ne ** 3)

        return np.array(scale)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'get_parameters_constraints')
def jastrow_get_parameters_constraints(self):
    """Returns parameters constraints in the following order
    u-cutoff, u-linear parameters,
    for every chi-set: chi-cutoff, chi-linear parameters,
    for every f-set: f-cutoff, f-linear parameters.
    :return:
    """
    def impl(self):
        a_list = []
        b_list = [0.0]

        if self.u_cutoff:
            # a0*C - a1*L = gamma/(-L)**(C-1) after differentiation on variables: a0, a1, L
            # (-(C-1)*gamma/(-L)**C - a1) * dL + С * da0 - L * da1 = 0
            u_matrix = np.zeros(shape=(1, self.u_parameters.shape[1]))
            u_matrix[0, 0] = self.trunc
            u_matrix[0, 1] = -self.u_cutoff

            u_spin_deps = self.u_parameters.shape[0]
            c = 1 / (-self.u_cutoff) ** (self.trunc - 1)
            if u_spin_deps == 3:
                u_b = [c/4, c/2, c/4]
                u_spin_deps = [0, 1, 2]
                if self.neu < 2:
                    u_b = [c/2, c/4]
                    u_spin_deps = [x for x in u_spin_deps if x != 0]
                if self.neu + self.ned < 2:
                    u_b = [c/4, c/4]
                    u_spin_deps = [x for x in u_spin_deps if x != 1]
                if self.ned < 2:
                    u_b = [c/4, c/2]
                    u_spin_deps = [x for x in u_spin_deps if x != 2]
            elif u_spin_deps == 2:
                u_b = [c/4, c/2]
                u_spin_deps = [0, 1]
                if self.neu < 2 and self.ned < 2:
                    u_b = [c/2]
                    u_spin_deps = [x for x in u_spin_deps if x != 0]
                if self.neu + self.ned < 2:
                    u_b = [c/4]
                    u_spin_deps = [x for x in u_spin_deps if x != 1]
            else:
                # FIXME: u_spin_deps == 1
                u_b = [c/4]
                u_spin_deps = [0]

            u_block = block_diag([u_matrix] * len(u_spin_deps))
            if self.u_cutoff_optimizable and self.cutoffs_optimizable:
                u_block = np.hstack((
                    -((1 - self.trunc) * np.array(u_b) / self.u_cutoff + self.u_parameters[np.array(u_spin_deps), 1]).reshape(-1, 1),
                    u_block
                ))
            a_list.append(u_block)
            b_list += u_b

        for chi_parameters, chi_cutoff, chi_cutoff_optimizable in zip(self.chi_parameters, self.chi_cutoff, self.chi_cutoff_optimizable):
            # a0*C - a1*L = -Z/(-L)**(C-1) or 0 if cusp imposed by WFN, after differentiation on variables: a0, a1, L
            # -a1 * dL + С * da0 - L * da1 = 0
            chi_matrix = np.zeros(shape=(1, chi_parameters.shape[1]))
            chi_matrix[0, 0] = self.trunc
            chi_matrix[0, 1] = -chi_cutoff

            if chi_parameters.shape[0] == 2:
                chi_spin_deps = [0, 1]
                if self.neu < 1:
                    chi_spin_deps = [1]
                if self.ned < 1:
                    chi_spin_deps = [0]
            else:
                chi_spin_deps = [0]

            chi_block = block_diag([chi_matrix] * len(chi_spin_deps))
            if chi_cutoff_optimizable and self.cutoffs_optimizable:
                chi_block = np.hstack((
                    -chi_parameters[np.array(chi_spin_deps), 1].reshape(-1, 1),
                    chi_block
                ))
            a_list.append(chi_block)
            b_list += [0] * len(chi_spin_deps)

        for f_parameters, f_cutoff, f_cutoff_optimizable, no_dup_u_term, no_dup_chi_term in zip(self.f_parameters, self.f_cutoff, self.f_cutoff_optimizable, self.no_dup_u_term, self.no_dup_chi_term):
            if f_parameters.shape[0] == 3:
                f_spin_deps = [0, 1, 2]
                if self.neu < 2:
                    f_spin_deps = [x for x in f_spin_deps if x != 0]
                if self.neu + self.ned < 2:
                    f_spin_deps = [x for x in f_spin_deps if x != 1]
                if self.ned < 2:
                    f_spin_deps = [x for x in f_spin_deps if x != 2]
            elif f_parameters.shape[0] == 2:
                f_spin_deps = [0, 1]
                if self.neu < 2 and self.ned < 2:
                    f_spin_deps = [x for x in f_spin_deps if x != 0]
                if self.neu + self.ned < 2:
                    f_spin_deps = [x for x in f_spin_deps if x != 1]
            else:
                f_spin_deps = [0]

            f_list = []
            f_constrains_size = 0
            f_cutoff_matrix = np.zeros(0)
            for spin_dep in f_spin_deps:
                f_matrix, cutoff_constraints = construct_a_matrix(self.trunc, f_parameters, f_cutoff, spin_dep, no_dup_u_term, no_dup_chi_term)
                f_constrains_size, f_parameters_size = f_matrix.shape
                f_list.append(f_matrix)
                f_cutoff_matrix = np.concatenate((f_cutoff_matrix, cutoff_constraints))
            f_block = block_diag(f_list)
            if f_cutoff_optimizable and self.cutoffs_optimizable:
                f_block = np.hstack((f_cutoff_matrix.reshape(-1, 1), f_block))
            a_list.append(f_block)
            b_list += [0] * f_constrains_size * len(f_spin_deps)

        return block_diag(a_list), np.array(b_list[1:])
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'set_parameters_projector')
def jastrow_set_parameters_projector(self):
    """Set Projector matrix"""
    def impl(self):
        a, b = self.get_parameters_constraints()
        p = np.eye(a.shape[1]) - a.T @ np.linalg.pinv(a.T)
        mask_idx = np.argwhere(self.get_parameters_mask()).ravel()
        inv_p = np.linalg.inv(p[:, mask_idx][mask_idx, :])
        self.parameters_projector = p[:, mask_idx] @ inv_p
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'get_parameters')
def jastrow_get_parameters(self, all_parameters):
    """Returns parameters in the following order:
    u-cutoff, u-linear parameters,
    for every chi-set: chi-cutoff, chi-linear parameters,
    for every f-set: f-cutoff, f-linear parameters.
    :return:
    """
    def impl(self, all_parameters):
        res = []
        if self.u_cutoff:
            if self.u_cutoff_optimizable and self.cutoffs_optimizable:
                res.append(self.u_cutoff)
            for j1 in range(self.u_parameters.shape[0]):
                for j2 in range(self.u_parameters.shape[1]):
                    if (self.u_parameters_optimizable[j1, j2] or all_parameters) and self.u_parameters_available[j1, j2]:
                        res.append(self.u_parameters[j1, j2])

        if self.chi_cutoff.any():
            for chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_cutoff_optimizable, chi_parameters_available in zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_cutoff, self.chi_cutoff_optimizable, self.chi_parameters_available):
                if chi_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(chi_cutoff)
                for j1 in range(chi_parameters.shape[0]):
                    for j2 in range(chi_parameters.shape[1]):
                        if (chi_parameters_optimizable[j1, j2] or all_parameters) and chi_parameters_available[j1, j2]:
                            res.append(chi_parameters[j1, j2])

        if self.f_cutoff.any():
            for f_parameters, f_parameters_optimizable, f_cutoff, f_cutoff_optimizable, f_parameters_available in zip(self.f_parameters, self.f_parameters_optimizable, self.f_cutoff, self.f_cutoff_optimizable, self.f_parameters_available):
                if f_cutoff_optimizable and self.cutoffs_optimizable:
                    res.append(f_cutoff)
                for j1 in range(f_parameters.shape[0]):
                    for j2 in range(f_parameters.shape[1]):
                        for j3 in range(f_parameters.shape[2]):
                            for j4 in range(j3, f_parameters.shape[3]):
                                if (f_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and f_parameters_available[j1, j2, j3, j4]:
                                    res.append(f_parameters[j1, j2, j3, j4])

        return np.array(res)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'set_parameters')
def jastrow_set_parameters(self, parameters, all_parameters):
    """Set parameters in the following order:
    u-cutoff, u-linear parameters,
    for every chi-set: chi-cutoff, chi-linear parameters,
    for every f-set: f-cutoff, f-linear parameters.
    :param parameters:
    :param all_parameters:
    :return:
    """
    def impl(self, parameters, all_parameters):
        n = 0
        if self.u_cutoff:
            if self.u_cutoff_optimizable and self.cutoffs_optimizable:
                self.u_cutoff = parameters[n]
                n += 1
            for j1 in range(self.u_parameters.shape[0]):
                for j2 in range(self.u_parameters.shape[1]):
                    if (self.u_parameters_optimizable[j1, j2] or all_parameters) and self.u_parameters_available[j1, j2]:
                        self.u_parameters[j1, j2] = parameters[n]
                        n += 1
            if not all_parameters:
                self.fix_u_parameters()

        if self.chi_cutoff.any():
            for i, (chi_parameters, chi_parameters_optimizable, chi_cutoff_optimizable, chi_parameters_available) in enumerate(zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_cutoff_optimizable, self.chi_parameters_available)):
                if chi_cutoff_optimizable and self.cutoffs_optimizable:
                    # Sequence type is a pointer, but numeric type is not.
                    self.chi_cutoff[i] = parameters[n]
                    n += 1
                for j1 in range(chi_parameters.shape[0]):
                    for j2 in range(chi_parameters.shape[1]):
                        if (chi_parameters_optimizable[j1, j2] or all_parameters) and chi_parameters_available[j1, j2]:
                            chi_parameters[j1, j2] = parameters[n]
                            n += 1
            if not all_parameters:
                self.fix_chi_parameters()

        if self.f_cutoff.any():
            for i, (f_parameters, f_parameters_optimizable, f_cutoff_optimizable, f_parameters_available) in enumerate(zip(self.f_parameters, self.f_parameters_optimizable, self.f_cutoff_optimizable, self.f_parameters_available)):
                if f_cutoff_optimizable and self.cutoffs_optimizable:
                    # Sequence types is a pointer, but numeric types is not.
                    self.f_cutoff[i] = parameters[n]
                    n += 1
                for j1 in range(f_parameters.shape[0]):
                    for j2 in range(f_parameters.shape[1]):
                        for j3 in range(f_parameters.shape[2]):
                            for j4 in range(j3, f_parameters.shape[3]):
                                if (f_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and f_parameters_available[j1, j2, j3, j4]:
                                    f_parameters[j1, j2, j3, j4] = f_parameters[j1, j2, j4, j3] = parameters[n]
                                    n += 1
            if not all_parameters:
                self.fix_f_parameters()

        return parameters[n:]
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'u_term_parameters_d1')
def jastrow_u_term_parameters_d1(self, e_powers):
    """First derivatives of log wfn w.r.t u-term parameters
    :param e_powers: powers of e-e distances
    """
    def impl(self, e_powers) -> np.ndarray:
        if not self.u_cutoff:
            return np.zeros((0,))

        C = self.trunc
        L = self.u_cutoff
        size = self.u_parameters_available.sum() + (self.u_cutoff_optimizable and self.cutoffs_optimizable)
        res = np.zeros(shape=(size,))

        n = -1
        if self.u_cutoff_optimizable and self.cutoffs_optimizable:
            n += 1
            self.u_cutoff -= delta
            res[n] -= self.u_term(e_powers) / delta / 2
            self.u_cutoff += 2 * delta
            res[n] += self.u_term(e_powers) / delta / 2
            self.u_cutoff -= delta

        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                n = int(self.u_cutoff_optimizable and self.cutoffs_optimizable) - 1
                r = e_powers[e1, e2, 1]
                cutoff = (r - L) ** C
                if r < self.u_cutoff:
                    u_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % self.u_parameters.shape[0]
                    for j1 in range(self.u_parameters.shape[0]):
                        for j2 in range(self.u_parameters.shape[1]):
                            if self.u_parameters_available[j1, j2]:
                                n += 1
                                if u_set == j1:
                                    res[n] += e_powers[e1, e2, j2] * cutoff
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'chi_term_parameters_d1')
def jastrow_chi_term_parameters_d1(self, n_powers):
    """First derivatives of log wfn w.r.t chi-term parameters
    :param n_powers: powers of e-n distances
    """
    def impl(self, n_powers) -> np.ndarray:
        if not self.chi_cutoff.any():
            return np.zeros((0,))

        C = self.trunc
        size = sum([
            chi_parameters_available.sum() + (chi_cutoff_optimizable and self.cutoffs_optimizable)
            for chi_parameters_available, chi_cutoff_optimizable
            in zip(self.chi_parameters_available, self.chi_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size,))

        n = -1
        for i, (chi_parameters, chi_parameters_available, chi_labels) in enumerate(zip(self.chi_parameters, self.chi_parameters_available, self.chi_labels)):
            if self.chi_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.chi_cutoff[i] -= delta
                res[n] -= self.chi_term(n_powers) / delta / 2
                self.chi_cutoff[i] += 2 * delta
                res[n] += self.chi_term(n_powers) / delta / 2
                self.chi_cutoff[i] -= delta

            n_start = n
            L = self.chi_cutoff[i]
            for label in chi_labels:
                for e1 in range(self.neu + self.ned):
                    n = n_start
                    r = n_powers[label, e1, 1]
                    cutoff = (r - L) ** C
                    if r < L:
                        chi_set = int(e1 >= self.neu) % chi_parameters.shape[0]
                        for j1 in range(chi_parameters.shape[0]):
                            for j2 in range(chi_parameters.shape[1]):
                                if chi_parameters_available[j1, j2]:
                                    n += 1
                                    if chi_set == j1:
                                        res[n] += n_powers[label, e1, j2] * cutoff
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'f_term_parameters_d1')
def jastrow_f_term_parameters_d1(self, e_powers, n_powers):
    """First derivatives of log wfn w.r.t f-term parameters
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    """
    def impl(self, e_powers, n_powers) -> np.ndarray:
        if not self.f_cutoff.any():
            return np.zeros((0,))

        C = self.trunc
        size = sum([
            f_parameters_available.sum() + (f_cutoff_optimizable and self.cutoffs_optimizable)
            for f_parameters_available, f_cutoff_optimizable
            in zip(self.f_parameters_available, self.f_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size,))

        n = -1

        for i, (f_parameters, f_parameters_available, f_labels) in enumerate(zip(self.f_parameters, self.f_parameters_available, self.f_labels)):
            if self.f_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.f_cutoff[i] -= delta
                res[n] -= self.f_term(e_powers, n_powers) / delta / 2
                self.f_cutoff[i] += 2 * delta
                res[n] += self.f_term(e_powers, n_powers) / delta / 2
                self.f_cutoff[i] -= delta

            n_start = n
            L = self.f_cutoff[i]
            for label in f_labels:
                for e1 in range(1, self.neu + self.ned):
                    for e2 in range(e1):
                        n = n_start
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % f_parameters.shape[0]
                            cutoff = (r_e1I - L) ** C * (r_e2I - L) ** C
                            for j1 in range(f_parameters.shape[0]):
                                for j2 in range(f_parameters.shape[1]):
                                    for j3 in range(f_parameters.shape[2]):
                                        for j4 in range(j3, f_parameters.shape[3]):
                                            if f_parameters_available[j1, j2, j3, j4]:
                                                n += 1
                                                if f_set == j1:
                                                    en_part = n_powers[label, e1, j3] * n_powers[label, e2, j4]
                                                    if j3 != j4:
                                                        en_part += n_powers[label, e1, j4] * n_powers[label, e2, j3]
                                                    res[n] += en_part * e_powers[e1, e2, j2] * cutoff
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'u_term_gradient_parameters_d1')
def jastrow_u_term_gradient_parameters_d1(self, e_powers, e_vectors):
    """Gradient w.r.t the parameters
    :param e_vectors: e-e vectors
    :param e_powers: powers of e-e distances
    :return:
    """
    def impl(self, e_powers, e_vectors) -> np.ndarray:
        if not self.u_cutoff:
            return np.zeros((0, (self.neu + self.ned) * 3))

        C = self.trunc
        L = self.u_cutoff
        size = self.u_parameters_available.sum() + (self.u_cutoff_optimizable and self.cutoffs_optimizable)
        res = np.zeros(shape=(size, (self.neu + self.ned), 3))

        n = -1
        if self.u_cutoff_optimizable and self.cutoffs_optimizable:
            n += 1
            self.u_cutoff -= delta
            res[n] -= self.u_term_gradient(e_powers, e_vectors).reshape((self.neu + self.ned), 3) / delta / 2
            self.u_cutoff += 2 * delta
            res[n] += self.u_term_gradient(e_powers, e_vectors).reshape((self.neu + self.ned), 3) / delta / 2
            self.u_cutoff -= delta

        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                n = int(self.u_cutoff_optimizable and self.cutoffs_optimizable) - 1
                r = e_powers[e1, e2, 1]
                if r < self.u_cutoff:
                    r_vec = e_vectors[e1, e2] / r
                    u_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % self.u_parameters.shape[0]
                    cutoff = (r - L) ** C
                    for j1 in range(self.u_parameters.shape[0]):
                        for j2 in range(self.u_parameters.shape[1]):
                            if self.u_parameters_available[j1, j2]:
                                n += 1
                                if u_set == j1:
                                    poly = e_powers[e1, e2, j2]
                                    gradient = r_vec * cutoff * (C / (r - L) + j2 / r) * poly
                                    res[n, e1, :] += gradient
                                    res[n, e2, :] -= gradient
        return res.reshape(size, (self.neu + self.ned) * 3)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'chi_term_gradient_parameters_d1')
def jastrow_chi_term_gradient_parameters_d1(self, n_powers, n_vectors):
    """Gradient w.r.t the parameters
    :param n_vectors: e-n vectors
    :param n_powers: powers of e-n distances
    :return:
    """
    def impl(self, n_powers, n_vectors) -> np.ndarray:
        if not self.chi_cutoff.any():
            return np.zeros((0, (self.neu + self.ned) * 3))

        C = self.trunc
        size = sum([
            chi_parameters_available.sum() + (chi_cutoff_optimizable and self.cutoffs_optimizable)
            for chi_parameters_available, chi_cutoff_optimizable
            in zip(self.chi_parameters_available, self.chi_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size, (self.neu + self.ned), 3))

        n = -1
        for i, (chi_parameters, chi_parameters_available, chi_labels) in enumerate(zip(self.chi_parameters, self.chi_parameters_available, self.chi_labels)):
            if self.chi_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.chi_cutoff[i] -= delta
                res[n] -= self.chi_term_gradient(n_powers, n_vectors).reshape((self.neu + self.ned), 3) / delta / 2
                self.chi_cutoff[i] += 2 * delta
                res[n] += self.chi_term_gradient(n_powers, n_vectors).reshape((self.neu + self.ned), 3) / delta / 2
                self.chi_cutoff[i] -= delta

            n_start = n
            L = self.chi_cutoff[i]
            for label in chi_labels:
                for e1 in range(self.neu + self.ned):
                    n = n_start
                    r = n_powers[label, e1, 1]
                    if r < L:
                        r_vec = n_vectors[label, e1] / r
                        chi_set = int(e1 >= self.neu) % chi_parameters.shape[0]
                        cutoff = (r - L) ** C
                        for j1 in range(chi_parameters.shape[0]):
                            for j2 in range(chi_parameters.shape[1]):
                                if chi_parameters_available[j1, j2]:
                                    n += 1
                                    if chi_set == j1:
                                        poly = n_powers[label, e1, j2]
                                        res[n, e1, :] += r_vec * cutoff * (C / (r - L) + j2 / r) * poly
        return res.reshape(size, (self.neu + self.ned) * 3)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'f_term_gradient_parameters_d1')
def jastrow_f_term_gradient_parameters_d1(self, e_powers, n_powers, e_vectors, n_vectors):
    """Gradient w.r.t the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :return:
    """
    def impl(self, e_powers, n_powers, e_vectors, n_vectors) -> np.ndarray:
        if not self.f_cutoff.any():
            return np.zeros((0, (self.neu + self.ned) * 3))

        C = self.trunc
        size = sum([
            f_parameters_available.sum() + (f_cutoff_optimizable and self.cutoffs_optimizable)
            for f_parameters_available, f_cutoff_optimizable
            in zip(self.f_parameters_available, self.f_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size, (self.neu + self.ned), 3))

        n = -1
        for i, (f_parameters, f_parameters_available, f_labels) in enumerate(zip(self.f_parameters, self.f_parameters_available, self.f_labels)):
            if self.f_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.f_cutoff[i] -= delta
                res[n] -= self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors).reshape((self.neu + self.ned), 3) / delta / 2
                self.f_cutoff[i] += 2 * delta
                res[n] += self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors).reshape((self.neu + self.ned), 3) / delta / 2
                self.f_cutoff[i] -= delta

            n_start = n
            L = self.f_cutoff[i]
            for label in f_labels:
                for e1 in range(1, self.neu + self.ned):
                    for e2 in range(e1):
                        n = n_start
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        r_ee = e_powers[e1, e2, 1]
                        r_e1I_vec = n_vectors[label, e1] / r_e1I
                        r_e2I_vec = n_vectors[label, e2] / r_e2I
                        r_ee_vec = e_vectors[e1, e2] / r_ee
                        cutoff = (r_e1I - L) ** C * (r_e2I - L) ** C
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % f_parameters.shape[0]
                            for j1 in range(f_parameters.shape[0]):
                                for j2 in range(f_parameters.shape[1]):
                                    for j3 in range(f_parameters.shape[2]):
                                        for j4 in range(j3, f_parameters.shape[3]):
                                            if f_parameters_available[j1, j2, j3, j4]:
                                                n += 1
                                                if f_set == j1:
                                                    poly_1 = cutoff * e_powers[e1, e2, j2] * n_powers[label, e1, j3] * n_powers[label, e2, j4]
                                                    poly_2 = cutoff * e_powers[e1, e2, j2] * n_powers[label, e1, j4] * n_powers[label, e2, j3]
                                                    # workaround to not create temporary 1-d numpy array
                                                    for t1 in range(3):
                                                        e1_gradient = r_e1I_vec[t1] * (C / (r_e1I - L) + j3 / r_e1I)
                                                        e2_gradient = r_e2I_vec[t1] * (C / (r_e2I - L) + j4 / r_e2I)
                                                        ee_gradient = r_ee_vec[t1] * j2 / r_ee
                                                        res[n, e1, t1] += (e1_gradient + ee_gradient) * poly_1
                                                        res[n, e2, t1] += (e2_gradient - ee_gradient) * poly_1

                                                        if j3 != j4:
                                                            e1_gradient = r_e1I_vec[t1] * (C / (r_e1I - L) + j4 / r_e1I)
                                                            e2_gradient = r_e2I_vec[t1] * (C / (r_e2I - L) + j3 / r_e2I)
                                                            res[n, e1, t1] += (e1_gradient + ee_gradient) * poly_2
                                                            res[n, e2, t1] += (e2_gradient - ee_gradient) * poly_2
        return res.reshape(size, (self.neu + self.ned) * 3)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'u_term_laplacian_parameters_d1')
def jastrow_u_term_laplacian_parameters_d1(self, e_powers):
    """Laplacian w.r.t the parameters
    :param e_powers: powers of e-e distances
    :return:
    """
    def impl(self, e_powers) -> np.ndarray:
        if not self.u_cutoff:
            return np.zeros((0, ))

        C = self.trunc
        L = self.u_cutoff
        size = self.u_parameters_available.sum() + (self.u_cutoff_optimizable and self.cutoffs_optimizable)
        res = np.zeros(shape=(size, ))

        n = -1
        if self.u_cutoff_optimizable and self.cutoffs_optimizable:
            n += 1
            self.u_cutoff -= delta
            res[n] -= self.u_term_laplacian(e_powers) / delta / 2
            self.u_cutoff += 2 * delta
            res[n] += self.u_term_laplacian(e_powers) / delta / 2
            self.u_cutoff -= delta

        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                n = int(self.u_cutoff_optimizable and self.cutoffs_optimizable) - 1
                r = e_powers[e1, e2, 1]
                if r < self.u_cutoff:
                    u_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % self.u_parameters.shape[0]
                    cutoff = (r - L) ** C
                    for j1 in range(self.u_parameters.shape[0]):
                        for j2 in range(self.u_parameters.shape[1]):
                            if self.u_parameters_available[j1, j2]:
                                n += 1
                                if u_set == j1:
                                    poly = e_powers[e1, e2, j2]
                                    res[n] += 2 * cutoff * (
                                        C*(C - 1)/(r-L)**2 +
                                        2 * C/(r-L) * (j2 + 1) / r +
                                        j2 * (j2 + 1) / r**2
                                    ) * poly
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'chi_term_laplacian_parameters_d1')
def jastrow_chi_term_laplacian_parameters_d1(self, n_powers):
    """Laplacian w.r.t the parameters
    :param n_powers: powers of e-n distances
    :return:
    """
    def impl(self, n_powers) -> np.ndarray:
        if not self.chi_cutoff.any():
            return np.zeros((0, ))

        C = self.trunc
        size = sum([
            chi_parameters_available.sum() + (chi_cutoff_optimizable and self.cutoffs_optimizable)
            for chi_parameters_available, chi_cutoff_optimizable
            in zip(self.chi_parameters_available, self.chi_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size, ))

        n = -1
        for i, (chi_parameters, chi_parameters_available, chi_labels) in enumerate(zip(self.chi_parameters, self.chi_parameters_available, self.chi_labels)):
            if self.chi_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.chi_cutoff[i] -= delta
                res[n] -= self.chi_term_laplacian(n_powers) / delta / 2
                self.chi_cutoff[i] += 2 * delta
                res[n] += self.chi_term_laplacian(n_powers) / delta / 2
                self.chi_cutoff[i] -= delta

            n_start = n
            L = self.chi_cutoff[i]
            for label in chi_labels:
                for e1 in range(self.neu + self.ned):
                    n = n_start
                    r = n_powers[label, e1, 1]
                    if r < L:
                        chi_set = int(e1 >= self.neu) % chi_parameters.shape[0]
                        cutoff = (r - L) ** C
                        for j1 in range(chi_parameters.shape[0]):
                            for j2 in range(chi_parameters.shape[1]):
                                if chi_parameters_available[j1, j2]:
                                    n += 1
                                    if chi_set == j1:
                                        poly = n_powers[label, e1, j2]
                                        res[n] += cutoff * (
                                            C*(C - 1)/(r-L)**2 +
                                            2 * C/(r-L) * (j2 + 1) / r +
                                            j2 * (j2 + 1) / r**2
                                        ) * poly
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'f_term_laplacian_parameters_d1')
def jastrow_f_term_laplacian_parameters_d1(self, e_powers, n_powers, e_vectors, n_vectors):
    """Laplacian w.r.t the parameters
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :return:
    """
    def impl(self, e_powers, n_powers, e_vectors, n_vectors) -> np.ndarray:
        if not self.f_cutoff.any():
            return np.zeros((0, ))

        size = sum([
            f_parameters_available.sum() + (f_cutoff_optimizable and self.cutoffs_optimizable)
            for f_parameters_available, f_cutoff_optimizable
            in zip(self.f_parameters_available, self.f_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size, ))

        n = -1
        C = self.trunc
        for i, (f_parameters, f_parameters_available, f_labels) in enumerate(zip(self.f_parameters, self.f_parameters_available, self.f_labels)):
            if self.f_cutoff_optimizable[i] and self.cutoffs_optimizable:
                n += 1
                self.f_cutoff[i] -= delta
                res[n] -= self.f_term_laplacian(e_powers, n_powers, e_vectors, n_vectors) / delta / 2
                self.f_cutoff[i] += 2 * delta
                res[n] += self.f_term_laplacian(e_powers, n_powers, e_vectors, n_vectors) / delta / 2
                self.f_cutoff[i] -= delta

            n_start = n
            L = self.f_cutoff[i]
            for label in f_labels:
                r_e1I_vec_dot_r_e2I_vec = n_vectors[label] @ n_vectors[label].T
                for e1 in range(1, self.neu + self.ned):
                    for e2 in range(e1):
                        n = n_start
                        # r_e1I_vec = n_vectors[label, e1]
                        # r_e2I_vec = n_vectors[label, e2]
                        # r_ee_vec = e_vectors[e1, e2]
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        r_ee = e_powers[e1, e2, 1]
                        cutoff = (r_e1I - L) ** C * (r_e2I - L) ** C
                        vec_1 = 1 - r_e1I_vec_dot_r_e2I_vec[e1, e2] / r_e1I**2
                        vec_2 = 1 - r_e1I_vec_dot_r_e2I_vec[e1, e2] / r_e2I**2
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % f_parameters.shape[0]
                            for j1 in range(f_parameters.shape[0]):
                                for j2 in range(f_parameters.shape[1]):
                                    for j3 in range(f_parameters.shape[2]):
                                        for j4 in range(j3, f_parameters.shape[3]):
                                            if f_parameters_available[j1, j2, j3, j4]:
                                                n += 1
                                                if f_set == j1:
                                                    poly = cutoff * e_powers[e1, e2, j2] * n_powers[label, e1, j3] * n_powers[label, e2, j4]
                                                    diff_1 = (
                                                        (C * r_e1I / (r_e1I - L) + j3) / r_e1I**2 +
                                                        (C * r_e2I / (r_e2I - L) + j4) / r_e2I**2 +
                                                        2 * j2 / r_ee**2
                                                    )
                                                    diff_2 = (
                                                        C * (C - 1) / (r_e1I - L) ** 2 +
                                                        C * (C - 1) / (r_e2I - L) ** 2 +
                                                        (j3 * (j3-1) / r_e1I**2 + j4 * (j4-1) / r_e2I**2 + 2 * j2 * (j2-1) / r_ee**2) +
                                                        2 * C / (r_e1I - L) * j3 / r_e1I +
                                                        2 * C / (r_e2I - L) * j4 / r_e2I
                                                    )
                                                    dot_product = (
                                                        vec_1 * (C * r_e1I / (r_e1I - L) + j3) +
                                                        vec_2 * (C * r_e2I / (r_e2I - L) + j4)
                                                    ) * j2 / r_ee**2
                                                    res[n] += (diff_2 + 2 * diff_1 + 2 * dot_product) * poly
                                                    if j3 != j4:
                                                        poly = cutoff  * e_powers[e1, e2, j2] * n_powers[label, e1, j4] * n_powers[label, e2, j3]
                                                        diff_1 = (
                                                            (C * r_e1I / (r_e1I - L) + j4) / r_e1I**2 +
                                                            (C * r_e2I / (r_e2I - L) + j3) / r_e2I**2 +
                                                            2 * j2 / r_ee**2
                                                        )
                                                        diff_2 = (
                                                            C * (C - 1) / (r_e1I - L) ** 2 +
                                                            C * (C - 1) / (r_e2I - L) ** 2 +
                                                            (j4 * (j4 - 1) / r_e1I**2 + j3 * (j3 - 1) / r_e2I**2 + 2 * j2 * (j2 - 1) / r_ee**2) +
                                                            2 * C / (r_e1I - L) * j4 / r_e1I +
                                                            2 * C / (r_e2I - L) * j3 / r_e2I
                                                        )
                                                        dot_product = (
                                                            vec_1 * (C * r_e1I / (r_e1I - L) + j4) +
                                                            vec_2 * (C * r_e2I / (r_e2I - L) + j3)
                                                        ) * j2 / r_ee**2
                                                        res[n] += (diff_2 + 2 * diff_1 + 2 * dot_product) * poly
        return res
    return impl



@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'value_parameters_d1')
def jastrow_value_parameters_d1(self, e_vectors, n_vectors):
    """First derivatives logarithm Jastrow w.r.t the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    """
    def impl(self, e_vectors, n_vectors) -> np.ndarray:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.parameters_projector.T @ np.concatenate((
            self.u_term_parameters_d1(e_powers),
            self.chi_term_parameters_d1(n_powers),
            self.f_term_parameters_d1(e_powers, n_powers),
        ))
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'gradient_parameters_d1')
def jastrow_gradient_parameters_d1(self, e_vectors, n_vectors):
    """First derivatives of Jastrow gradient w.r.t the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """
    def impl(self, e_vectors, n_vectors) -> np.ndarray:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.parameters_projector.T @ np.concatenate((
            self.u_term_gradient_parameters_d1(e_powers, e_vectors),
            self.chi_term_gradient_parameters_d1(n_powers, n_vectors),
            self.f_term_gradient_parameters_d1(e_powers, n_powers, e_vectors, n_vectors),
        ))
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'laplacian_parameters_d1')
def jastrow_laplacian_parameters_d1(self, e_vectors, n_vectors):
    """First derivatives of Jastrow laplacian w.r.t the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """
    def impl(self, e_vectors, n_vectors) -> np.ndarray:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.parameters_projector.T @ np.concatenate((
            self.u_term_laplacian_parameters_d1(e_powers),
            self.chi_term_laplacian_parameters_d1(n_powers),
            self.f_term_laplacian_parameters_d1(e_powers, n_powers, e_vectors, n_vectors),
        ))
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'u_term_parameters_d2')
def jastrow_u_term_parameters_d2(self, e_powers):
    def impl(self, e_powers) -> np.ndarray:
        pass
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'chi_term_parameters_d2')
def jastrow_chi_term_parameters_d2(self, e_powers):
    """Second derivatives of wfn w.r.t. u-term parameters
    :param e_powers: powers of e-e distances
    """
    def impl(self, e_powers) -> np.ndarray:
        pass
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'f_term_parameters_d2')
def jastrow_f_term_parameters_d2(self, e_powers):
    """Second derivatives of wfn w.r.t. chi-term parameters
    :param n_powers: powers of e-n distances
    """
    def impl(self, e_powers) -> np.ndarray:
        pass
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Jastrow_class_t, 'value_parameters_d2')
def jastrow_value_parameters_d2(self, e_vectors, n_vectors):
    """Second derivatives Jastrow w.r.t the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    """
    def impl(self, e_vectors, n_vectors) -> np.ndarray:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.parameters_projector.T @ block_diag((
            self.u_term_parameters_d2(e_powers),
            self.chi_term_parameters_d2(n_powers),
            self.f_term_parameters_d2(e_powers, n_powers),
        )) @ self.parameters_projector
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