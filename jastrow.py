#!/usr/bin/env python3

import os
from timeit import default_timer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import numba as nb

from numpy.polynomial.polynomial import polyval, polyval3d

from readers.casino import CasinoConfig
from overload import subtract_outer
from logger import logging

logger = logging.getLogger('vmc')

labels_type = nb.int64[:]
u_mask_type = nb.boolean[:, :]
chi_mask_type = nb.boolean[:, :]
f_mask_type = nb.boolean[:, :, :, :]
u_parameters_type = nb.float64[:, :]
chi_parameters_type = nb.float64[:, :]
f_parameters_type = nb.float64[:, :, :, :]

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('trunc', nb.int64),
    ('u_mask', u_mask_type),
    ('chi_mask', nb.types.ListType(chi_mask_type)),
    ('f_mask', nb.types.ListType(f_mask_type)),
    ('u_parameters', u_parameters_type),
    ('chi_parameters', nb.types.ListType(chi_parameters_type)),
    ('f_parameters', nb.types.ListType(f_parameters_type)),
    ('u_cutoff', nb.float64),
    ('chi_cutoff', nb.float64[:]),
    ('f_cutoff', nb.float64[:]),
    ('chi_labels', nb.types.ListType(labels_type)),
    ('f_labels', nb.types.ListType(labels_type)),
    ('max_ee_order', nb.int64),
    ('max_en_order', nb.int64),
    ('chi_cusp', nb.boolean[:]),
    ('no_dup_u_term', nb.boolean[:]),
    ('no_dup_chi_term', nb.boolean[:]),
    ('u_cusp_const', nb.float64[:]),
]


@nb.experimental.jitclass(spec)
class Jastrow:

    def __init__(
        self, neu, ned, trunc, u_parameters, u_mask, u_cutoff, u_cusp_const, chi_parameters, chi_mask, chi_cutoff, chi_labels,
        f_parameters, f_mask, f_cutoff, f_labels, no_dup_u_term, no_dup_chi_term, chi_cusp
    ):
        self.neu = neu
        self.ned = ned
        self.trunc = trunc
        self.u_mask = u_mask
        self.chi_mask = nb.typed.List.empty_list(chi_mask_type)
        [self.chi_mask.append(m) for m in chi_mask]
        self.f_mask = nb.typed.List.empty_list(f_mask_type)
        [self.f_mask.append(m) for m in f_mask]
        # last index (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        self.u_parameters = u_parameters
        # last index (0->u=d; 1->u/=d)
        self.chi_parameters = nb.typed.List.empty_list(chi_parameters_type)
        [self.chi_parameters.append(p) for p in chi_parameters]
        # last index (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        self.f_parameters = nb.typed.List.empty_list(f_parameters_type)
        [self.f_parameters.append(p) for p in f_parameters]
        self.u_cutoff = u_cutoff
        self.chi_cutoff = chi_cutoff
        self.f_cutoff = f_cutoff
        self.chi_labels = nb.typed.List.empty_list(labels_type)
        [self.chi_labels.append(p) for p in chi_labels]
        self.f_labels = nb.typed.List.empty_list(labels_type)
        [self.f_labels.append(p) for p in f_labels]
        self.max_ee_order = max((
            self.u_parameters.shape[0],
            max([p.shape[2] for p in self.f_parameters]) if self.f_parameters else 0,
        ))
        self.max_en_order = max((
            max([p.shape[0] for p in self.chi_parameters]) if self.chi_parameters else 0,
            max([p.shape[0] for p in self.f_parameters]) if self.f_parameters else 0,
        ))
        self.chi_cusp = chi_cusp
        self.no_dup_u_term = no_dup_u_term
        self.no_dup_chi_term = no_dup_chi_term
        self.u_cusp_const = u_cusp_const

    def ee_powers(self, e_vectors):
        """Powers of e-e distances
        :param e_vectors: e-e vectors - array(nelec, nelec, 3)
        :return: powers of e-e distances - array(nelec, nelec, max_ee_order)
        """
        res = np.ones(shape=(e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(1, self.max_ee_order):
                    res[i, j, k] = res[j, i, k] = r_ee ** k
        return res

    def en_powers(self, n_vectors):
        """Powers of e-n distances
        :param n_vectors: e-n vectors - array(natom, nelec, 3)
        :return: powers of e-n distances - array(natom, nelec, max_en_order)
        """
        res = np.ones(shape=(n_vectors.shape[0], n_vectors.shape[1], self.max_en_order))
        for i in range(n_vectors.shape[0]):
            for j in range(n_vectors.shape[1]):
                r_eI = np.linalg.norm(n_vectors[i, j])
                for k in range(1, self.max_en_order):
                    res[i, j, k] = r_eI ** k
        return res

    def u_term(self, e_powers) -> float:
        """Jastrow u-term
        :param e_powers: powers of e-e distances
        :return:
        """
        res = 0.0
        if not self.u_cutoff:
            return res

        C = self.trunc
        parameters = self.u_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r = e_powers[i, j, 1]
                if r < self.u_cutoff:
                    cusp_set = int(i >= self.neu) + int(j >= self.neu)
                    u_set = cusp_set % parameters.shape[1]
                    poly = 0.0
                    for k in range(parameters.shape[0]):
                        if k == 1:
                            p = self.u_cusp_const[cusp_set]
                        else:
                            p = parameters[k, u_set]
                        poly += p * e_powers[i, j, k]
                    res += poly * (r - self.u_cutoff) ** C
        return res

    def chi_term(self, n_powers) -> float:
        """Jastrow chi-term
        :param n_powers: powers of e-e distances
        :return:
        """
        res = 0.0
        if not self.chi_cutoff.any():
            return res

        C = self.trunc
        for parameters, L, chi_labels in zip(self.chi_parameters, self.chi_cutoff, self.chi_labels):
            for i in chi_labels:
                for j in range(self.neu + self.ned):
                    r = n_powers[i, j, 1]
                    if r < L:
                        chi_set = int(j >= self.neu) % parameters.shape[1]
                        # FIXME: maybe in next numba
                        # res += polyval(r, parameters[:, chi_set]) * (r - L) ** C
                        poly = 0.0
                        for k in range(parameters.shape[0]):
                            poly += parameters[k, chi_set] * n_powers[i, j, k]
                        res += poly * (r - L) ** C
        return res

    def f_term(self, e_powers, n_powers) -> float:
        """Jastrow f-term
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :return:
        """
        res = 0.0
        if not self.f_cutoff.any():
            return res

        C = self.trunc
        for parameters, L, f_labels in zip(self.f_parameters, self.f_cutoff, self.f_labels):
            for i in f_labels:
                for j in range(1, self.neu + self.ned):
                    for k in range(j):
                        r_e1I = n_powers[i, j, 1]
                        r_e2I = n_powers[i, k, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(j >= self.neu) + int(k >= self.neu)) % parameters.shape[3]
                            poly = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        poly += parameters[l, m, n, f_set] * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n]
                            res += poly * (r_e1I - L) ** C * (r_e2I - L) ** C
        return res

    def u_term_gradient(self, e_powers, e_vectors):
        """Jastrow u-term gradient with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :param e_vectors: e-e vectors
        :return:
        """
        res = np.zeros(shape=(self.neu + self.ned, 3))

        if not self.u_cutoff:
            return res.ravel()

        C = self.trunc
        L = self.u_cutoff
        parameters = self.u_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_vec = e_vectors[i, j]
                r = e_powers[i, j, 1]
                if r < L:
                    cusp_set = (int(i >= self.neu) + int(j >= self.neu))
                    u_set = cusp_set % parameters.shape[1]
                    poly = poly_diff = 0.0
                    for k in range(parameters.shape[0]):
                        if k == 1:
                            p = self.u_cusp_const[cusp_set]
                        else:
                            p = parameters[k, u_set]
                        poly += p * e_powers[i, j, k]
                        if k > 0:
                            poly_diff += p * k * e_powers[i, j, k-1]

                    gradient = r_vec/r * (r-L) ** C * (C/(r-L) * poly + poly_diff)
                    res[i, :] += gradient
                    res[j, :] -= gradient
        return res.ravel()

    def chi_term_gradient(self, n_powers, n_vectors):
        """Jastrow chi-term gradient with respect to a e-coordinates
        :param n_powers: powers of e-n distances
        :param n_vectors: e-n vectors
        :return:
        """
        res = np.zeros(shape=(self.neu + self.ned, 3))

        if not self.chi_cutoff.any():
            return res.ravel()

        C = self.trunc
        for parameters, L, chi_labels in zip(self.chi_parameters, self.chi_cutoff, self.chi_labels):
            for i in chi_labels:
                for j in range(self.neu + self.ned):
                    r_vec = n_vectors[i, j]
                    r = n_powers[i, j, 1]
                    if r < L:
                        chi_set = int(j >= self.neu) % parameters.shape[1]
                        poly = poly_diff = 0.0
                        for k in range(parameters.shape[0]):
                            p = parameters[k, chi_set]
                            poly += p * n_powers[i, j, k]
                            if k > 0:
                                poly_diff += p * k * n_powers[i, j, k-1]

                        res[j, :] += r_vec/r * (r-L) ** C * (C/(r-L) * poly + poly_diff)
        return res.ravel()

    def f_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors):
        """Jastrow f-term gradient with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        res = np.zeros(shape=(self.neu + self.ned, 3))

        if not self.f_cutoff.any():
            return res.ravel()

        C = self.trunc
        for parameters, L, f_labels in zip(self.f_parameters, self.f_cutoff, self.f_labels):
            for i in f_labels:
                for j in range(1, self.neu + self.ned):
                    for k in range(j):
                        r_e1I_vec = n_vectors[i, j]
                        r_e2I_vec = n_vectors[i, k]
                        r_ee_vec = e_vectors[j, k]
                        r_e1I = n_powers[i, j, 1]
                        r_e2I = n_powers[i, k, 1]
                        r_ee = e_powers[j, k, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(j >= self.neu) + int(k >= self.neu)) % parameters.shape[3]
                            poly = poly_diff_e1I = poly_diff_e2I = poly_diff_ee = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        p = parameters[l, m, n, f_set]
                                        poly += n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n] * p
                                        if l > 0:
                                            poly_diff_e1I += l * n_powers[i, j, l-1] * n_powers[i, k, m] * e_powers[j, k, n] * p
                                        if m > 0:
                                            poly_diff_e2I += m * n_powers[i, j, l] * n_powers[i, k, m-1] * e_powers[j, k, n] * p
                                        if n > 0:
                                            poly_diff_ee += n * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n-1] * p

                            e1_gradient = r_e1I_vec/r_e1I * (C/(r_e1I - L) * poly + poly_diff_e1I)
                            e2_gradient = r_e2I_vec/r_e2I * (C/(r_e2I - L) * poly + poly_diff_e2I)
                            ee_gradient = r_ee_vec/r_ee * poly_diff_ee
                            res[j, :] += (r_e1I - L) ** C * (r_e2I - L) ** C * (e1_gradient + ee_gradient)
                            res[k, :] += (r_e1I - L) ** C * (r_e2I - L) ** C * (e2_gradient - ee_gradient)
        return res.ravel()

    def u_term_laplacian(self, e_powers) -> float:
        """Jastrow u-term laplacian with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :return:
        """
        res = 0.0
        if not self.u_cutoff:
            return res

        C = self.trunc
        L = self.u_cutoff
        parameters = self.u_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r = e_powers[i, j, 1]
                if r < L:
                    cusp_set = (int(i >= self.neu) + int(j >= self.neu))
                    u_set = cusp_set % parameters.shape[1]
                    poly = poly_diff = poly_diff_2 = 0.0
                    for k in range(parameters.shape[0]):
                        if k == 1:
                            p = self.u_cusp_const[cusp_set]
                        else:
                            p = parameters[k, u_set]
                        poly += p * e_powers[i, j, k]
                        if k > 0:
                            poly_diff += k * p * e_powers[i, j, k-1]
                        if k > 1:
                            poly_diff_2 += k * (k-1) * p * e_powers[i, j, k-2]

                    res += (r-L)**C * (
                        C*(C - 1)/(r-L)**2 * poly + 2 * C/(r-L) * poly_diff + poly_diff_2 +
                        2 * (C/(r-L) * poly + poly_diff) / r
                    )
        return 2 * res

    def chi_term_laplacian(self, n_powers) -> float:
        """Jastrow chi-term laplacian with respect to a e-coordinates
        :param n_powers: powers of e-n distances
        :return:
        """
        res = 0.0
        if not self.chi_cutoff.any():
            return res

        C = self.trunc
        for parameters, L, chi_labels in zip(self.chi_parameters, self.chi_cutoff, self.chi_labels):
            for i in chi_labels:
                for j in range(self.neu + self.ned):
                    r = n_powers[i, j, 1]
                    if r < L:
                        chi_set = int(j >= self.neu) % parameters.shape[1]
                        poly = poly_diff = poly_diff_2 = 0.0
                        for k in range(parameters.shape[0]):
                            p = parameters[k, chi_set]
                            poly += p * n_powers[i, j, k]
                            if k > 0:
                                poly_diff += k * p * n_powers[i, j, k-1]
                            if k > 1:
                                poly_diff_2 += k * (k-1) * p * n_powers[i, j, k-2]

                        res += (r-L)**C * (
                            C*(C - 1)/(r-L)**2 * poly + 2 * C/(r-L) * poly_diff + poly_diff_2 +
                            2 * (C/(r-L) * poly + poly_diff) / r
                        )
        return res

    def f_term_laplacian(self, e_powers, n_powers, e_vectors, n_vectors) -> float:
        """Jastrow f-term laplacian with respect to a e-coordinates
        f-term is a product of two spherically symmetric functions f(r_eI) and g(r_ee) so using
            ∇²(f*g) = ∇²(f)*g + 2*∇(f)*∇(g) + f*∇²(g)
        then Laplace operator of spherically symmetric function (in 3-D space) is
            ∇²(f) = d²f/dr² + 2/r * df/dr
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        res = 0.0
        if not self.f_cutoff.any():
            return res

        C = self.trunc
        for parameters, L, f_labels in zip(self.f_parameters, self.f_cutoff, self.f_labels):
            for i in f_labels:
                for j in range(1, self.neu + self.ned):
                    for k in range(j):
                        r_e1I_vec = n_vectors[i, j]
                        r_e2I_vec = n_vectors[i, k]
                        r_ee_vec = e_vectors[j, k]
                        r_e1I = n_powers[i, j, 1]
                        r_e2I = n_powers[i, k, 1]
                        r_ee = e_powers[j, k, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(j >= self.neu) + int(k >= self.neu)) % parameters.shape[3]
                            poly = poly_diff_e1I = poly_diff_e2I = 0.0
                            poly_diff_ee = poly_diff_e1I_2 = poly_diff_e2I_2 = 0.0
                            poly_diff_ee_2 = poly_diff_e1I_ee = poly_diff_e2I_ee = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        p = parameters[l, m, n, f_set]
                                        poly += n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n] * p
                                        if l > 0:
                                            poly_diff_e1I += l * n_powers[i, j, l-1] * n_powers[i, k, m] * e_powers[j, k, n] * p
                                        if m > 0:
                                            poly_diff_e2I += m * n_powers[i, j, l] * n_powers[i, k, m-1] * e_powers[j, k, n] * p
                                        if n > 0:
                                            poly_diff_ee += n * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n-1] * p
                                        if l > 1:
                                            poly_diff_e1I_2 += l * (l-1) * n_powers[i, j, l-2] * n_powers[i, k, m] * e_powers[j, k, n] * p
                                        if m > 1:
                                            poly_diff_e2I_2 += m * (m-1) * n_powers[i, j, l] * n_powers[i, k, m-2] * e_powers[j, k, n] * p
                                        if n > 1:
                                            poly_diff_ee_2 += n * (n-1) * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n-2] * p
                                        if l > 0 and n > 0:
                                            poly_diff_e1I_ee += l * n * n_powers[i, j, l-1] * n_powers[i, k, m] * e_powers[j, k, n-1] * p
                                        if m > 0 and n > 0:
                                            poly_diff_e2I_ee += m * n * n_powers[i, j, l] * n_powers[i, k, m-1] * e_powers[j, k, n-1] * p

                            diff_1 = (
                                (C/(r_e1I - L) * poly + poly_diff_e1I) / r_e1I +
                                (C/(r_e2I - L) * poly + poly_diff_e2I) / r_e2I +
                                2 * poly_diff_ee / r_ee
                            )
                            diff_2 = (
                                C * (C - 1) / (r_e1I - L) ** 2 * poly +
                                C * (C - 1) / (r_e2I - L) ** 2 * poly +
                                (poly_diff_e1I_2 + poly_diff_e2I_2 + 2 * poly_diff_ee_2) +
                                2 * C/(r_e1I - L) * poly_diff_e1I +
                                2 * C/(r_e2I - L) * poly_diff_e2I
                            )
                            dot_product = (
                                np.sum(r_e1I_vec * r_ee_vec) * (C/(r_e1I - L) * poly_diff_ee + poly_diff_e1I_ee) / r_e1I / r_ee -
                                np.sum(r_e2I_vec * r_ee_vec) * (C/(r_e2I - L) * poly_diff_ee + poly_diff_e2I_ee) / r_e2I / r_ee
                            )
                            res += (r_e1I - L) ** C * (r_e2I - L) ** C * (diff_2 + 2 * diff_1 + 2 * dot_product)
        return res

    def value(self, e_vectors, n_vectors) -> float:
        """Jastrow with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
        :return:
        """

        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term(e_powers) +
            self.chi_term(n_powers) +
            self.f_term(e_powers, n_powers)
        )

    def gradient(self, e_vectors, n_vectors):
        """Gradient with respect to e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term_gradient(e_powers, e_vectors) +
            self.chi_term_gradient(n_powers, n_vectors) +
            self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
        )

    def laplacian(self, e_vectors, n_vectors) -> float:
        """Laplacian with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term_laplacian(e_powers) +
            self.chi_term_laplacian(n_powers) +
            self.f_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)
        )

    def numerical_gradient(self, e_vectors, n_vectors):
        """Numerical gradient with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        delta = 0.00001

        res = np.zeros(shape=(self.neu + self.ned, 3))

        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res[i, j] -= self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res.ravel() / delta / 2

    def numerical_laplacian(self, e_vectors, n_vectors) -> float:
        """Numerical laplacian with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        delta = 0.00001

        res = -6 * (self.neu + self.ned) * self.value(e_vectors, n_vectors)
        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res += self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res += self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res / delta / delta

    def get_bounds(self):
        """"""
        lower_bonds = np.zeros((0,))
        upper_bonds = np.zeros((0,))

        if self.u_cutoff:
            size = self.u_mask.sum() + 1
            u_lower_bonds = - np.ones((size,)) * np.inf
            u_upper_bonds = np.ones((size,)) * np.inf
            u_lower_bonds[0] = 1
            u_upper_bonds[0] = 10
            lower_bonds = np.concatenate((lower_bonds, u_lower_bonds))
            upper_bonds = np.concatenate((upper_bonds, u_upper_bonds))

        if self.chi_cutoff.any():
            for chi_mask in self.chi_mask:
                size = chi_mask.sum() + 1
                chi_lower_bonds = - np.ones((size,)) * np.inf
                chi_upper_bonds = np.ones((size,)) * np.inf
                chi_lower_bonds[0] = 1
                chi_upper_bonds[0] = 10
                lower_bonds = np.concatenate((lower_bonds, chi_lower_bonds))
                upper_bonds = np.concatenate((upper_bonds, chi_upper_bonds))

        if self.f_cutoff.any():
            for f_mask in self.f_mask:
                size = f_mask.sum() + 1
                f_lower_bonds = - np.ones((size,)) * np.inf
                f_upper_bonds = np.ones((size,)) * np.inf
                f_lower_bonds[0] = 1
                f_upper_bonds[0] = 10
                lower_bonds = np.concatenate((lower_bonds, f_lower_bonds))
                upper_bonds = np.concatenate((upper_bonds, f_upper_bonds))

        return lower_bonds, upper_bonds

    def fix_u_parameters(self):
        """Fix u-term parameters"""
        C = self.trunc
        L = self.u_cutoff
        for i in range(3):
            self.u_cusp_const[i] = 1 / np.array([4, 2, 4])[i] / (-L) ** C + self.u_parameters[0, i % self.u_parameters.shape[1]] * C / L

    def fix_chi_parameters(self):
        """Fix chi-term parameters"""
        C = self.trunc
        for chi_parameters, L, chi_cusp in zip(self.chi_parameters, self.chi_cutoff, self.chi_cusp):
            chi_parameters[1] = chi_parameters[0] * C / L
            if chi_cusp:
                pass
                # FIXME: chi cusp not implemented
                # chi_parameters[1] -= charge / (-L) ** C

    def get_parameters(self):
        """Returns parameters in the following order:
        u-cutoff, u-linear parameters,
        for every chi-set: chi-cutoff, chi-linear parameters,
        for every f-set: f-cutoff, f-linear parameters.
        :return:
        """
        res = np.zeros(0)
        if self.u_cutoff:
            res = np.concatenate((
                res, np.array((self.u_cutoff, )), self.u_parameters.ravel()[self.u_mask.ravel()]
            ))

        if self.chi_cutoff.any():
            for chi_parameters, chi_mask, chi_cutoff in zip(self.chi_parameters, self.chi_mask, self.chi_cutoff):
                res = np.concatenate((
                    res, np.array((chi_cutoff, )), chi_parameters.ravel()[chi_mask.ravel()]
                ))

        if self.f_cutoff.any():
            for f_parameters, f_mask, f_cutoff in zip(self.f_parameters, self.f_mask, self.f_cutoff):
                res = np.concatenate((
                    res, np.array((f_cutoff, )), f_parameters.ravel()[f_mask.ravel()]
                ))

        return res

    def set_parameters(self, parameters):
        """
        u-cutoff, u-linear parameters,
        for every chi-set: chi-cutoff, chi-linear parameters,
        for every f-set: f-cutoff, f-linear parameters.
        :param parameters:
        :return:
        """
        n = -1
        if self.u_cutoff:
            n += 1
            self.u_cutoff = parameters[n]
            for j1 in range(self.u_parameters.shape[0]):
                for j2 in range(self.u_parameters.shape[1]):
                    if self.u_mask[j1, j2]:
                        n += 1
                        self.u_parameters[j1, j2] = parameters[n]
            self.fix_u_parameters()

        if self.chi_cutoff.any():
            for i, (chi_parameters, chi_mask) in enumerate(zip(self.chi_parameters, self.chi_mask)):
                n += 1
                # Sequence types is a pointer, but numeric types is not.
                self.chi_cutoff[i] = parameters[n]
                for j1 in range(chi_parameters.shape[0]):
                    for j2 in range(chi_parameters.shape[1]):
                        if chi_mask[j1, j2]:
                            n += 1
                            chi_parameters[j1, j2] = parameters[n]
            self.fix_chi_parameters()

        if self.f_cutoff.any():
            for i, (f_parameters, f_mask) in enumerate(zip(self.f_parameters, self.f_mask)):
                n += 1
                # Sequence types is a pointer, but numeric types is not.
                self.f_cutoff[i] = parameters[n]
                for j1 in range(f_parameters.shape[0]):
                    for j2 in range(f_parameters.shape[1]):
                        for j3 in range(f_parameters.shape[2]):
                            for j4 in range(f_parameters.shape[3]):
                                if f_mask[j1, j2, j3, j4]:
                                    n += 1
                                    f_parameters[j1, j2, j3, j4] = f_parameters[j2, j1, j3, j4] = parameters[n]
            # self.fix_f_parameters()

    def u_term_numerical_d1(self, e_powers):
        """Numerical first derivatives of logarithm wfn with respect to u-term parameters
        :param e_powers: powers of e-e distances
        """
        if not self.u_cutoff:
            return np.zeros((0,))

        delta = 0.00001
        size = self.u_mask.sum() + 1
        res = np.zeros((size,))

        n = 0
        self.u_cutoff -= delta
        self.fix_u_parameters()
        res[n] -= self.u_term(e_powers)
        self.u_cutoff += 2 * delta
        self.fix_u_parameters()
        res[n] += self.u_term(e_powers)
        self.u_cutoff -= delta

        for i in range(self.u_parameters.shape[0]):
            for j in range(self.u_parameters.shape[1]):
                if self.u_mask[i, j]:
                    n += 1
                    self.u_parameters[i, j] -= delta
                    self.fix_u_parameters()
                    res[n] -= self.u_term(e_powers)
                    self.u_parameters[i, j] += 2 * delta
                    self.fix_u_parameters()
                    res[n] += self.u_term(e_powers)
                    self.u_parameters[i, j] -= delta

        self.fix_u_parameters()
        return res / delta / 2

    def chi_term_numerical_d1(self, n_powers):
        """Numerical first derivatives of logarithm wfn with respect to chi-term parameters
        :param e_powers: powers of e-e distances
        """
        if not self.chi_cutoff.any():
            return np.zeros((0,))

        delta = 0.00001
        size = 0
        for chi_mask in self.chi_mask:
            size += chi_mask.sum() + 1
        res = np.zeros((size,))

        n = -1
        for i, (chi_parameters, chi_mask) in enumerate(zip(self.chi_parameters, self.chi_mask)):
            n += 1
            self.chi_cutoff[i] -= delta
            self.fix_chi_parameters()
            res[n] -= self.chi_term(n_powers)
            self.chi_cutoff[i] += 2 * delta
            self.fix_chi_parameters()
            res[n] += self.chi_term(n_powers)
            self.chi_cutoff[i] -= delta

            for i in range(chi_parameters.shape[0]):
                for j in range(chi_parameters.shape[1]):
                    if chi_mask[i, j]:
                        n += 1
                        chi_parameters[i, j] -= delta
                        self.fix_chi_parameters()
                        res[n] -= self.chi_term(n_powers)
                        chi_parameters[i, j] += 2 * delta
                        self.fix_chi_parameters()
                        res[n] += self.chi_term(n_powers)
                        chi_parameters[i, j] -= delta

        self.fix_chi_parameters()
        return res / delta / 2

    def f_term_numerical_d1(self, e_powers, n_powers):
        """Numerical first derivatives of logarithm wfn with respect to f-term parameters
        :param e_powers: powers of e-e distances
        """
        if not self.f_cutoff.any():
            return np.zeros((0,))

        delta = 0.00001
        size = 0
        for f_mask in self.f_mask:
            size += f_mask.sum() + 1
        res = np.zeros((size,))

        n = -1

        for i, (f_parameters, f_mask) in enumerate(zip(self.f_parameters, self.f_mask)):
            n += 1
            self.f_cutoff[i] -= delta
            # self.fix_f_parameters()
            res[n] -= self.f_term(e_powers, n_powers)
            self.f_cutoff[i] += 2 * delta
            # self.fix_f_parameters()
            res[n] += self.f_term(e_powers, n_powers)
            self.f_cutoff[i] -= delta

            for i in range(f_parameters.shape[0]):
                for j in range(f_parameters.shape[1]):
                    for k in range(f_parameters.shape[2]):
                        for l in range(f_parameters.shape[3]):
                            if f_mask[i, j, k, l]:
                                n += 1
                                f_parameters[i, j, k, l] -= delta
                                if i != j:
                                    f_parameters[j, i, k, l] -= delta
                                # self.fix_f_parameters()
                                res[n] -= self.f_term(e_powers, n_powers)
                                f_parameters[i, j, k, l] += 2 * delta
                                if i != j:
                                    f_parameters[j, i, k, l] += 2 * delta
                                # self.fix_f_parameters()
                                res[n] += self.f_term(e_powers, n_powers)
                                f_parameters[i, j, k, l] -= delta
                                if i != j:
                                    f_parameters[j, i, k, l] -= delta

        # self.fix_f_parameters()
        return res / delta / 2

    def parameters_numerical_d1(self, e_vectors, n_vectors):
        """Numerical first derivatives logarithm Jastrow with respect to the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return np.concatenate((
            self.u_term_numerical_d1(e_powers),
            self.chi_term_numerical_d1(n_powers),
            self.f_term_numerical_d1(e_powers, n_powers),
        ))

    def u_term_numerical_d2(self, e_powers):
        """Numerical second derivatives of logarithm wfn with respect to u-term parameters
        :param e_powers: powers of e-e distances
        """

        delta = 0.00001
        size = self.u_mask.sum() + 1
        res = -2 * self.u_term(e_powers) * np.eye(size)

        n = 0
        self.u_cutoff -= delta
        self.fix_u_parameters()
        res[n, n] += self.u_term(e_powers)
        self.u_cutoff += 2 * delta
        self.fix_u_parameters()
        res[n, n] += self.u_term(e_powers)
        self.u_cutoff -= delta

        # diagonal terms of linear parameters
        for i in range(self.u_parameters.shape[0]):
            for j in range(self.u_parameters.shape[1]):
                if self.u_mask[i, j]:
                    n += 1
                    self.u_parameters[i, j] -= delta
                    self.fix_u_parameters()
                    res[n, n] += self.u_term(e_powers)
                    self.u_parameters[i, j] += 2 * delta
                    self.fix_u_parameters()
                    res[n, n] += self.u_term(e_powers)
                    self.u_parameters[i, j] -= delta

        # partial derivatives on cutoff and linear parameters
        n = 0
        for i in range(self.u_parameters.shape[0]):
            for j in range(self.u_parameters.shape[1]):
                if self.u_mask[i, j]:
                    n += 1
                    self.u_parameters[i, j] -= delta
                    self.u_cutoff -= delta
                    self.fix_u_parameters()
                    res[0, n] += self.u_term(e_powers)
                    self.u_parameters[i, j] += 2 * delta
                    self.fix_u_parameters()
                    res[0, n] -= self.u_term(e_powers)
                    self.u_cutoff += 2 * delta
                    self.fix_u_parameters()
                    res[0, n] += self.u_term(e_powers)
                    self.u_parameters[i, j] -= 2 * delta
                    self.fix_u_parameters()
                    res[0, n] += self.u_term(e_powers)
                    self.u_parameters[i, j] += delta
                    self.u_cutoff -= delta
                    res[n, 0] = res[0, n]

        n = 0
        for i1 in range(self.u_parameters.shape[0]):
            for j1 in range(self.u_parameters.shape[1]):
                if self.u_mask[i1, j1]:
                    n += 1
                    m = 0
                    for i2 in range(self.u_parameters.shape[0]):
                        for j2 in range(self.u_parameters.shape[1]):
                            if self.u_mask[i2, j2]:
                                m += 1
                                if m > n:
                                    self.u_parameters[i1, j1] -= delta
                                    self.u_parameters[i2, j2] -= delta
                                    self.fix_u_parameters()
                                    res[n, m] += self.u_term(e_powers)
                                    self.u_parameters[i1, j1] += 2 * delta
                                    self.fix_u_parameters()
                                    res[n, m] -= self.u_term(e_powers)
                                    self.u_parameters[i2, j2] += 2 * delta
                                    self.fix_u_parameters()
                                    res[n, m] += self.u_term(e_powers)
                                    self.u_parameters[i1, j1] -= 2 * delta
                                    self.fix_u_parameters()
                                    res[n, m] += self.u_term(e_powers)
                                    self.u_parameters[i1, j1] += delta
                                    self.u_parameters[i2, j2] -= delta
                                    res[m, n] = res[n, m]

        self.fix_u_parameters()
        return res / delta / delta

    def chi_term_numerical_d2(self, n_powers):
        """Numerical second derivatives of logarithm wfn with respect to chi-term parameters
        :param n_powers: powers of e-n distances
        """
        if not self.chi_cutoff.any():
            return np.zeros((0, 0))

        delta = 0.00001
        size = 0
        for chi_mask in self.chi_mask:
            size += chi_mask.sum() + 1
        res = -2 * self.chi_term(n_powers) * np.eye(size)

        n = -1

        for i, (chi_parameters, chi_mask) in enumerate(zip(self.chi_parameters, self.chi_mask)):
            n += 1
            self.chi_cutoff[i] -= delta
            self.fix_chi_parameters()
            res[n, n] += self.chi_term(n_powers)
            self.chi_cutoff[i] += 2 * delta
            self.fix_chi_parameters()
            res[n, n] += self.chi_term(n_powers)
            self.chi_cutoff[i] -= delta

            # diagonal terms of linear parameters
            for i in range(chi_parameters.shape[0]):
                for j in range(chi_parameters.shape[1]):
                    if chi_mask[i, j]:
                        n += 1
                        chi_parameters[i, j] -= delta
                        self.fix_chi_parameters()
                        res[n, n] += self.chi_term(n_powers)
                        chi_parameters[i, j] += 2 * delta
                        self.fix_chi_parameters()
                        res[n, n] += self.chi_term(n_powers)
                        chi_parameters[i, j] -= delta

            # partial derivatives on cutoff and linear parameters
            n = 0
            for i in range(chi_parameters.shape[0]):
                for j in range(chi_parameters.shape[1]):
                    if chi_mask[i, j]:
                        n += 1
                        chi_parameters[i, j] -= delta
                        self.chi_cutoff[i] -= delta
                        self.fix_chi_parameters()
                        res[0, n] += self.u_term(n_powers)
                        chi_parameters[i, j] += 2 * delta
                        self.fix_chi_parameters()
                        res[0, n] -= self.chi_term(n_powers)
                        self.chi_cutoff[i] += 2 * delta
                        self.fix_chi_parameters()
                        res[0, n] += self.chi_term(n_powers)
                        chi_parameters[i, j] -= 2 * delta
                        self.fix_chi_parameters()
                        res[0, n] += self.chi_term(n_powers)
                        chi_parameters[i, j] += delta
                        self.chi_cutoff[i] -= delta
                        res[n, 0] = res[0, n]

            n = 0
            for i1 in range(chi_parameters.shape[0]):
                for j1 in range(chi_parameters.shape[1]):
                    if chi_mask[i1, j1]:
                        n += 1
                        m = 0
                        for i2 in range(chi_parameters.shape[0]):
                            for j2 in range(chi_parameters.shape[1]):
                                if chi_mask[i2, j2]:
                                    m += 1
                                    if m > n:
                                        chi_parameters[i1, j1] -= delta
                                        chi_parameters[i2, j2] -= delta
                                        self.fix_chi_parameters()
                                        res[n, m] += self.chi_term(n_powers)
                                        chi_parameters[i1, j1] += 2 * delta
                                        self.fix_u_parameters()
                                        res[n, m] -= self.chi_term(n_powers)
                                        chi_parameters[i2, j2] += 2 * delta
                                        self.fix_chi_parameters()
                                        res[n, m] += self.chi_term(n_powers)
                                        chi_parameters[i1, j1] -= 2 * delta
                                        self.fix_chi_parameters()
                                        res[n, m] += self.chi_term(n_powers)
                                        chi_parameters[i1, j1] += delta
                                        chi_parameters[i2, j2] -= delta
                                        res[m, n] = res[n, m]

        self.fix_chi_parameters()
        return res / delta / delta

    def f_term_numerical_d2(self, e_powers, n_powers):
        """Numerical second derivatives of logarithm wfn with respect to f-term parameters
        :param n_powers: powers of e-n distances
        """
        if not self.f_cutoff.any():
            return np.zeros((0, 0))

        delta = 0.00001
        size = 0
        for f_mask in self.f_mask:
            size += f_mask.sum() + 1
        res = -2 * self.f_term(e_powers, n_powers) * np.eye(size)

        n = -1

        for i, f_parameters, f_mask in enumerate(zip(self.f_parameters, self.f_mask)):
            n += 1
            self.f_cutoff[i] -= delta
            self.fix_f_parameters()
            res[n, n] += self.f_term(e_powers, n_powers)
            self.f_cutoff[i] += 2 * delta
            self.fix_f_parameters()
            res[n, n] += self.f_term(e_powers, n_powers)
            self.f_cutoff[i] -= delta

            # diagonal terms of linear parameters
            for i in range(f_parameters.shape[0]):
                for j in range(f_parameters.shape[1]):
                    for k in range(f_parameters.shape[2]):
                        for l in range(f_parameters.shape[3]):
                            if f_mask[i, j, k, l]:
                                n += 1
                                f_parameters[i, j, k, l] -= delta
                                if i != j:
                                    f_parameters[j, i, k, l] -= delta
                                self.fix_f_parameters()
                                res[n] += self.f_term(e_powers, n_powers)
                                f_parameters[i, j, k, l] += 2 * delta
                                if i != j:
                                    f_parameters[j, i, k, l] += 2 * delta
                                self.fix_f_parameters()
                                res[n] += self.f_term(e_powers, n_powers)
                                f_parameters[i, j, k, l] -= delta
                                if i != j:
                                    f_parameters[j, i, k, l] -= delta
            # partial derivatives on cutoff and linear parameters
            n = 0

        self.fix_f_parameters()
        return res / delta / delta

    def parameters_numerical_d2(self, e_vectors, n_vectors):
        """Numerical second derivatives with respect to the Jastrow parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        Using:
            ∂²exp(u(a) + chi(b))/∂a∂b = ∂(∂u(a)/∂a*exp(u(a) + chi(b)))/∂b = ∂u(a)/∂a * ∂chi(b)/∂b * exp(u(a) + chi(b))

        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        u_term = self.u_term_numerical_d2(e_powers)
        chi_term = self.chi_term_numerical_d2(n_powers)
        f_term = self.f_term_numerical_d2(e_powers, n_powers)

        # not supported by numba
        # res = np.block((
        #     (u_term, np.zeros((u_term.shape[0], chi_term.shape[0])), np.zeros((u_term.shape[0], f_term.shape[0]))),
        #     (np.zeros((chi_term.shape[0], u_term.shape[0])), chi_term, np.zeros((chi_term.shape[0], f_term.shape[0]))),
        #     (np.zeros((f_term.shape[0], u_term.shape[0])), np.zeros((f_term.shape[0], chi_term.shape[0])), f_term)
        # ))
        b = np.cumsum(np.array([0, u_term.shape[0], chi_term.shape[0], f_term.shape[0]]))
        res = np.zeros((b[3], b[3]))
        res[b[0]:b[1], b[0]:b[1]] = u_term
        res[b[1]:b[2], b[1]:b[2]] = chi_term
        res[b[2]:b[3], b[2]:b[3]] = f_term
        return res


@nb.jit(forceobj=True)
def initial_position(ne, atom_positions, atom_charges):
    """Initial positions of electrons."""
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
    return r_e


@nb.jit(nopython=True)
def random_step(dx, ne):
    """Random N-dim square distributed step"""
    return np.random.uniform(-dx, dx, ne * 3).reshape((ne, 3))


# @pool
@nb.jit(nopython=True, nogil=True)
def profiling_value(dx, neu, ned, steps, atom_positions, jastrow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        jastrow.value(e_vectors, n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def profiling_gradient(dx, neu, ned, steps, atom_positions, jastrow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        jastrow.gradient(e_vectors, n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def profiling_laplacian(dx, neu, ned, steps, atom_positions, jastrow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        jastrow.laplacian(e_vectors, n_vectors)


def main(casino):
    dx = 3.0

    jastrow = Jastrow(
        casino.input.neu, casino.input.ned,
        casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_mask, casino.jastrow.u_cutoff, casino.jastrow.u_cusp_const,
        casino.jastrow.chi_parameters, casino.jastrow.chi_mask, casino.jastrow.chi_cutoff, casino.jastrow.chi_labels,
        casino.jastrow.f_parameters, casino.jastrow.f_mask, casino.jastrow.f_cutoff, casino.jastrow.f_labels,
        casino.jastrow.no_dup_u_term, casino.jastrow.no_dup_chi_term, casino.jastrow.chi_cusp
    )

    r_initial = initial_position(casino.input.neu + casino.input.ned, casino.wfn.atom_positions, casino.wfn.atom_charges)

    start = default_timer()
    profiling_value(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, jastrow, r_initial)
    end = default_timer()
    logger.info(' value     %8.1f', end - start)

    start = default_timer()
    profiling_laplacian(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, jastrow, r_initial)
    end = default_timer()
    logger.info(' laplacian %8.1f', end - start)

    start = default_timer()
    profiling_gradient(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, jastrow, r_initial)
    end = default_timer()
    logger.info(' gradient  %8.1f', end - start)


if __name__ == '__main__':
    """
    He:
     value         25.6
     laplacian     30.4
     gradient      37.6
    Be:
     value         57.5
     laplacian     93.9
     gradient     112.4
    Ne:
     value        277.5
     laplacian    481.7
     gradient     536.5
    Ar:
     value        875.4
     laplacian   1612.5
     gradient    1771.5
    Kr:
     value       3174.8
    """

    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr'):
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Jastrow/'
        logger.info('%s:', mol)
        main(CasinoConfig(path))
