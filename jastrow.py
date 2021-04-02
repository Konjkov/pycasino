#!/usr/bin/env python3

import numpy as np
import numba as nb
# import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from readers.casino import Casino
from overload import subtract_outer

"""
https://github.com/numba/numba/issues/4522

So to summarize the current state (as of numba 0.45.1), jitclasses will (almost) always be slower
 than a functional njit equivalent routine because of the reference counting of class attributes.
I guess when using classes + numba, @selslack's suggestion of creating a normal class with njit methods,
 which is not the most ideal because it leads to some obfuscation, is the best option.
"""

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
        self, neu, ned, trunc, u_parameters, u_mask, u_cutoff, chi_parameters, chi_mask, chi_cutoff, chi_labels,
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
        self.u_cusp_const = np.zeros((3, ))
        if self.u_cutoff:
            self.fix_u_parameters()
        if self.chi_cutoff.any():
            self.fix_chi_parameters()
        if self.f_cutoff.any():
            self.fix_f_parameters()

    def ee_powers(self, e_vectors):
        """Powers of e-e distances
        :param e_vectors: e-e vectors - array(nelec, nelec, 3)
        :return: powers of e-e distances - array(nelec, nelec, max_ee_order)
        """
        res = np.zeros((e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(self.max_ee_order):
                    res[i, j, k] = res[j, i, k] = r_ee ** k
        return res

    def en_powers(self, n_vectors):
        """Powers of e-n distances
        :param n_vectors: e-n vectors - array(natom, nelec, 3)
        :return: powers of e-n distances - array(natom, nelec, max_en_order)
        """
        res = np.zeros((n_vectors.shape[0], n_vectors.shape[1], self.max_en_order))
        for i in range(n_vectors.shape[0]):
            for j in range(n_vectors.shape[1]):
                r_eI = np.linalg.norm(n_vectors[i, j])
                for k in range(self.max_en_order):
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
        res = np.zeros((e_vectors.shape[0], 3))

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

                    gradient = (C * (r-L) ** (C-1) * poly + (r-L) ** C * poly_diff) / r
                    res[i, :] += r_vec * gradient
                    res[j, :] -= r_vec * gradient
        return res.ravel()

    def chi_term_gradient(self, n_powers, n_vectors):
        """Jastrow chi-term gradient with respect to a e-coordinates
        :param n_powers: powers of e-n distances
        :param n_vectors: e-n vectors
        :return:
        """
        res = np.zeros((n_vectors.shape[1], 3))

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

                        gradient = (C * (r-L) ** (C-1) * poly + (r-L) ** C * poly_diff) / r
                        res[j, :] += r_vec * gradient
        return res.ravel()

    def f_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors):
        """Jastrow f-term gradient with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        res = np.zeros((e_vectors.shape[0], 3))

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

                            gradient = (
                                C * (r_e1I - L) ** (C-1) * (r_e2I - L) ** C * poly +
                                (r_e1I - L) ** C * (r_e2I - L) ** C * poly_diff_e1I
                            ) / r_e1I
                            res[j, :] += r_e1I_vec * gradient

                            gradient = (
                                (r_e1I - L) ** C * C * (r_e2I - L) ** (C-1) * poly +
                                (r_e1I - L) ** C * (r_e2I - L) ** C * poly_diff_e2I
                            ) / r_e2I
                            res[k, :] += r_e2I_vec * gradient

                            gradient = (r_e1I - L) ** C * (r_e2I - L) ** C * poly_diff_ee / r_ee
                            res[j, :] += r_ee_vec * gradient
                            res[k, :] -= r_ee_vec * gradient
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

                    res += (
                        C*(C - 1)*(r-L)**(C - 2) * poly +
                        2 * C*(r-L)**(C - 1) * poly_diff + (r-L)**C * poly_diff_2 +
                        2 * (C * (r-L)**(C-1) * poly + (r-L)**C * poly_diff) / r
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

                        res += (
                            C*(C - 1)*(r-L)**(C - 2) * poly +
                            2 * C*(r-L)**(C - 1) * poly_diff + (r-L)**C * poly_diff_2 +
                            2 * (C * (r-L)**(C-1) * poly + (r-L)**C * poly_diff) / r
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

                            gradient = (
                                (C * (r_e1I - L) ** (C-1) * (r_e2I - L) ** C * poly + (r_e1I - L) ** C * (r_e2I - L) ** C * poly_diff_e1I) / r_e1I +
                                ((r_e1I - L) ** C * C * (r_e2I - L) ** (C-1) * poly + (r_e1I - L) ** C * (r_e2I - L) ** C * poly_diff_e2I) / r_e2I +
                                2 * (r_e1I - L) ** C * (r_e2I - L) ** C * poly_diff_ee / r_ee
                            )

                            laplacian = (
                                    C * (C - 1) * (r_e1I - L) ** (C - 2) * (r_e2I - L) ** C * poly +
                                    (r_e1I - L) ** C * C * (C - 1) * (r_e2I - L) ** (C - 2) * poly +
                                    (r_e1I - L) ** C * (r_e2I - L) ** C * (poly_diff_e1I_2 + poly_diff_e2I_2 + 2 * poly_diff_ee_2) +
                                    2 * C * (r_e1I - L) ** (C - 1) * (r_e2I - L) ** C * poly_diff_e1I +
                                    2 * (r_e1I - L) ** C * C * (r_e2I - L) ** (C - 1) * poly_diff_e2I
                            )

                            dot_product = (
                                    np.sum(r_e1I_vec * r_ee_vec) * (
                                            C * (r_e1I - L) ** (C-1) * (r_e2I - L) ** C * poly_diff_ee +
                                            (r_e1I - L) ** C * (r_e2I - L) ** C * poly_diff_e1I_ee
                                    ) / r_e1I / r_ee -
                                    np.sum(r_e2I_vec * r_ee_vec) * (
                                            (r_e1I - L) ** C * C * (r_e2I - L) ** (C-1) * poly_diff_ee +
                                            (r_e1I - L) ** C * (r_e2I - L) ** C * poly_diff_e2I_ee
                                    ) / r_e2I / r_ee
                            )

                            res += laplacian + 2 * gradient + 2 * dot_product
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

        res = np.zeros((e_vectors.shape[0], 3))

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
                res, np.array((self.u_cutoff, )), self.u_parameters.flatten()[self.u_mask.flatten()]
            ))

        if self.chi_cutoff.any():
            for chi_parameters, chi_mask, chi_cutoff in zip(self.chi_parameters, self.chi_mask, self.chi_cutoff):
                res = np.concatenate((
                    res, np.array((chi_cutoff, )), chi_parameters.flatten()[chi_mask.flatten()]
                ))

        if self.f_cutoff.any():
            for f_parameters, f_mask, f_cutoff in zip(self.f_parameters, self.f_mask, self.f_cutoff):
                res = np.concatenate((
                    res, np.array((f_cutoff, )), f_parameters.flatten()[f_mask.flatten()]
                ))

        return res

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
                # chi_parameters[1] -= charge / (-L) ** C

    def fix_f_parameters(self):
        """Fix f-term parameters
        0 - zero value
        A - no electron–electron cusp constrains
        B - no electron–nucleus cusp constrains
        X - independent value

        n = 0            n = 1            n > 1
        -------------------------------------------------------
        B B B B B B B B  A A A A A A A B  ? X X X X X X B  <- m
        B X X X X X X X  A X X X X X A A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A A X X X X X A  X X X X X X X X
        B X X X X X X X  B A A A A A A A  B X X X X X X X
        ---------------- no_dup_u_term ------------------------
        0 B B B B B B B  0 A A A A A A B  0 X X X X X X B  <- m
        B B X X X X X X  A X X X X X A A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A A X X X X X A  X X X X X X X X
        B X X X X X X X  B A A A A A A A  B X X X X X X X
        ---------------- no_dup_chi_term ----------------------
        0 0 0 0 0 0 0 0  A A A A A A A B  X X X X X X X B  <- m
        0 B B B B B B B  A X X X X X A A  X X X X X X X X
        0 B X X X X X X  A X X X X X X A  X X X X X X X X
        0 B X X X X X X  A X X X X X X A  X X X X X X X X
        0 B X X X X X X  A X X X X X X A  X X X X X X X X
        0 B X X X X X X  A X X X X X X A  X X X X X X X X
        0 B X X X X X X  A A X X X X X A  X X X X X X X X
        0 B X X X X X X  B A A A A A A A  B X X X X X X X
        ^
        l
        """
        C = self.trunc
        for f_parameters, L, f_mask, no_dup_chi_term in zip(self.f_parameters, self.f_cutoff, self.f_mask, self.no_dup_chi_term):
            f_en_order = f_parameters.shape[0] - 1
            f_ee_order = f_parameters.shape[2] - 1
            for i in range(f_parameters.shape[0]):
                for j in range(f_parameters.shape[1]):
                    for k in range(f_parameters.shape[2]):
                        for l in range(f_parameters.shape[3]):
                            if f_mask[i, j, k, l] or f_mask[j, i, k, l]:
                                continue
                            f_parameters[i, j, k, l] = 0
            """fix 2 * f_en_order e–e cusp constrains"""
            for lm in range(2 * f_en_order + 1):
                lm_sum = np.zeros(f_parameters.shape[3])
                for l in range(f_en_order + 1):
                    for m in range(f_en_order + 1):
                        if l + m == lm:
                            lm_sum += f_parameters[l, m, 1, :]
                if lm < f_en_order:
                    f_parameters[0, lm, 1, :] = -lm_sum / 2
                    f_parameters[lm, 0, 1, :] = -lm_sum / 2
                elif lm == f_en_order:
                    sum_1 = -lm_sum / 2
                elif lm > f_en_order:
                    f_parameters[f_en_order, lm - f_en_order, 1, :] = -lm_sum / 2
                    f_parameters[lm - f_en_order, f_en_order, 1, :] = -lm_sum / 2
            """fix f_en_order+f_ee_order e–n cusp constrains"""
            for mn in range(f_en_order + f_ee_order, -1, -1):
                mn_sum = np.zeros(f_parameters.shape[3])
                for m in range(f_en_order + 1):
                    for n in range(f_ee_order + 1):
                        if m + n == mn:
                            mn_sum += self.trunc * f_parameters[0, m, n, :] - L * f_parameters[1, m, n, :]
                if mn > f_en_order:
                    f_parameters[0, f_en_order, mn - f_en_order, :] = -mn_sum / C
                    f_parameters[f_en_order, 0, mn - f_en_order, :] = -mn_sum / C
                elif mn == f_en_order:
                    sum_2 = -mn_sum
                elif mn < f_en_order:
                    if no_dup_chi_term:
                        f_parameters[1, mn, 0, :] = mn_sum / L
                        f_parameters[mn, 1, 0, :] = mn_sum / L
                    else:
                        f_parameters[0, mn, 0, :] = -mn_sum / C
                        f_parameters[mn, 0, 0, :] = -mn_sum / C
            """fix (l=en_order - 1, m=1, n=1) term"""
            f_parameters[f_en_order - 1, 1, 1, :] = sum_1 - f_parameters[f_en_order, 0, 1, :]
            f_parameters[1, f_en_order - 1, 1, :] = sum_1 - f_parameters[0, f_en_order, 1, :]

            sum_2 += L * f_parameters[f_en_order - 1, 1, 1, :]

            """fix (l=en_order, m=0, n=0) term"""
            if no_dup_chi_term:
                f_parameters[f_en_order, 1, 0, :] = - sum_2 / L
                f_parameters[1, f_en_order, 0, :] = - sum_2 / L
            else:
                f_parameters[f_en_order, 0, 0, :] = sum_2 / C
                f_parameters[0, f_en_order, 0, :] = sum_2 / C

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
            self.fix_f_parameters()

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
            self.fix_f_parameters()
            res[n] -= self.f_term(e_powers, n_powers)
            self.f_cutoff[i] += 2 * delta
            self.fix_f_parameters()
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
                                self.fix_f_parameters()
                                res[n] -= self.f_term(e_powers, n_powers)
                                f_parameters[i, j, k, l] += 2 * delta
                                if i != j:
                                    f_parameters[j, i, k, l] += 2 * delta
                                self.fix_f_parameters()
                                res[n] += self.f_term(e_powers, n_powers)
                                f_parameters[i, j, k, l] -= delta
                                if i != j:
                                    f_parameters[j, i, k, l] -= delta

        self.fix_f_parameters()
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

    def parameters_numerical_d2(self, e_vectors, n_vectors, neu):
        """Numerical second derivatives with respect to the Jastrow parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
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


if __name__ == '__main__':
    """Plot Jastrow terms
    """

    term = 'u'

    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_no_u_vmc/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_no_chi/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_no_u_no_chi_vmc/'
    path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_vmc_cbc/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/al/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/acetaldehyde/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'

    casino = Casino(path)
    jastrow = Jastrow(
        casino.input.neu, casino.input.ned,
        casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_mask, casino.jastrow.u_cutoff,
        casino.jastrow.chi_parameters, casino.jastrow.chi_mask, casino.jastrow.chi_cutoff, casino.jastrow.chi_labels,
        casino.jastrow.f_parameters, casino.jastrow.f_mask, casino.jastrow.f_cutoff, casino.jastrow.f_labels,
        casino.jastrow.no_dup_u_term, casino.jastrow.no_dup_chi_term, casino.jastrow.chi_cusp
    )

    steps = 100

    if term == 'u':
        x_min, x_max = 0, jastrow.u_cutoff
        x_grid = np.linspace(x_min, x_max, steps)
        for spin_dep in range(3):
            y_grid = np.zeros((steps, ))
            for i in range(100):
                r_e = np.array([[0.0, 0.0, 0.0], [x_grid[i], 0.0, 0.0]])
                e_vectors = subtract_outer(r_e, r_e)
                e_powers = jastrow.ee_powers(e_vectors)
                y_grid[i] = jastrow.u_term(e_powers, 2-spin_dep)
                if spin_dep == 1:
                    y_grid[i] /= 2.0
            plt.plot(x_grid, y_grid, label=['uu', 'ud/2', 'dd'][spin_dep])
        plt.xlabel('r_ee (au)')
        plt.ylabel('polynomial part')
        plt.title('JASTROW u-term')
    elif term == 'chi':
        for atom in range(casino.wfn.atom_positions.shape[0]):
            x_min, x_max = 0, jastrow.chi_cutoff[atom]
            x_grid = np.linspace(x_min, x_max, steps)
            for spin_dep in range(2):
                y_grid = np.zeros((steps, ))
                for i in range(100):
                    r_e = np.array([[x_grid[i], 0.0, 0.0]]) + casino.wfn.atom_positions[atom]
                    sl = slice(atom, atom+1)
                    jastrow.chi_parameters = nb.typed.List.empty_list(chi_parameters_type)
                    [jastrow.chi_parameters.append(p) for p in casino.jastrow.chi_parameters[sl]]
                    n_vectors = subtract_outer(casino.wfn.atom_positions[sl], r_e)
                    n_powers = jastrow.en_powers(n_vectors)
                    y_grid[i] = jastrow.chi_term(n_powers, 1-spin_dep)
                plt.plot(x_grid, y_grid, label=f'atom {atom} ' + ['u', 'd'][spin_dep])
        plt.xlabel('r_eN (au)')
        plt.ylabel('polynomial part')
        plt.title('JASTROW chi-term')
    elif term == 'f':
        figure = plt.figure()
        axis = figure.add_subplot(111, projection='3d')
        for atom in range(casino.wfn.atom_positions.shape[0]):
            x_min, x_max = -jastrow.f_cutoff[atom], jastrow.f_cutoff[atom]
            y_min, y_max = 0.0, np.pi
            x = np.linspace(x_min, x_max, steps)
            y = np.linspace(y_min, y_max, steps)
            x_grid, y_grid = np.meshgrid(x, y)
            for spin_dep in range(3):
                z_grid = np.zeros((steps, steps))
                for i in range(100):
                    for j in range(100):
                        r_e = np.array([
                            [x_grid[i, j] * np.cos(y_grid[i, j]), x_grid[i, j] * np.sin(y_grid[i, j]), 0.0],
                            [x_grid[i, j], 0.0, 0.0]
                        ]) + casino.wfn.atom_positions[atom]
                        sl = slice(atom, atom + 1)
                        jastrow.f_parameters = nb.typed.List.empty_list(f_parameters_type)
                        [jastrow.f_parameters.append(p) for p in casino.jastrow.f_parameters[sl]]
                        e_vectors = subtract_outer(r_e, r_e)
                        e_powers = jastrow.ee_powers(e_vectors)
                        n_vectors = subtract_outer(casino.wfn.atom_positions[sl], r_e)
                        n_powers = jastrow.en_powers(n_vectors)
                        z_grid[i, j] = jastrow.f_term(e_powers, n_powers, 2-spin_dep)
                axis.plot_wireframe(x_grid, y_grid, z_grid, label=f'atom {atom} ' + ['uu', 'ud', 'dd'][spin_dep])
        axis.set_xlabel('r_e1N (au)')
        axis.set_ylabel('r_e2N (au)')
        axis.set_zlabel('polynomial part')
        plt.title('JASTROW f-term')

    plt.grid(True)
    plt.legend()
    plt.show()
