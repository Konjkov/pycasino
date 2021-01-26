#!/usr/bin/env python3

import numpy as np
import numba as nb

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
u_parameters_type = nb.float64[:, :]
chi_parameters_type = nb.float64[:, :]
f_parameters_type = nb.float64[:, :, :, :]

spec = [
    ('enabled', nb.boolean),
    ('trunc', nb.int64),
    ('u_parameters', u_parameters_type),
    ('chi_parameters', nb.types.ListType(chi_parameters_type)),
    ('f_parameters', nb.types.ListType(f_parameters_type)),
    ('u_cutoff', nb.float64),
    ('chi_cutoff', nb.float64[:]),
    ('f_cutoff', nb.float64[:]),
    ('chi_labels', nb.types.ListType(labels_type)),
    ('f_labels', nb.types.ListType(labels_type)),
    ('u_spin_dep', nb.int64),
    ('chi_spin_dep', nb.int64[:]),
    ('f_spin_dep', nb.int64[:]),
    ('max_ee_order', nb.int64),
    ('max_en_order', nb.int64),
]


@nb.experimental.jitclass(spec)
class Jastrow:

    def __init__(self, trunc, u_parameters, u_cutoff, u_spin_dep, chi_parameters, chi_cutoff, chi_labels, chi_spin_dep, f_parameters, f_cutoff, f_labels, f_spin_dep):
        self.enabled = u_cutoff or chi_cutoff.any() or f_cutoff.any()
        self.trunc = trunc
        self.u_parameters = u_parameters
        self.chi_parameters = nb.typed.List.empty_list(chi_parameters_type)
        [self.chi_parameters.append(p) for p in chi_parameters]
        self.f_parameters = nb.typed.List.empty_list(f_parameters_type)
        [self.f_parameters.append(p) for p in f_parameters]
        self.u_cutoff = u_cutoff
        self.chi_cutoff = chi_cutoff
        self.f_cutoff = f_cutoff
        self.chi_labels = nb.typed.List.empty_list(labels_type)
        [self.chi_labels.append(p) for p in chi_labels]
        self.f_labels = nb.typed.List.empty_list(labels_type)
        [self.f_labels.append(p) for p in f_labels]
        self.u_spin_dep = u_spin_dep  # (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        self.chi_spin_dep = chi_spin_dep  # (0->u=d; 1->u/=d)
        self.f_spin_dep = f_spin_dep  # (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        self.max_ee_order = max((
            self.u_parameters.shape[0],
            max([p.shape[2] for p in self.f_parameters]),
        ))
        self.max_en_order = max((
            max([p.shape[0] for p in self.chi_parameters]),
            max([p.shape[0] for p in self.f_parameters]),
        ))

    def ee_powers(self, e_vectors):
        """Powers of e-e distances
        :param e_vectors: e-e vectors
        :return:
        """
        res = np.zeros((e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(e_vectors.shape[0] - 1):
            for j in range(i + 1, e_vectors.shape[1]):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(self.max_ee_order):
                    res[i, j, k] = r_ee ** k
        return res

    def en_powers(self, n_vectors):
        """Powers of e-n distances
        :param n_vectors: e-n vectors
        :return:
        """
        res = np.zeros((n_vectors.shape[1], n_vectors.shape[0], self.max_en_order))
        for i in range(n_vectors.shape[1]):
            for j in range(n_vectors.shape[0]):
                r_eI = np.linalg.norm(n_vectors[j, i])
                for k in range(self.max_en_order):
                    res[i, j, k] = r_eI ** k
        return res

    def u_term(self, e_powers, neu):
        """Jastrow u-term
        :param e_powers: powers of e-e distances
        :param neu: number of up electrons
        :return:
        """
        res = 0.0
        if not self.u_cutoff:
            return res

        parameters = self.u_parameters
        for i in range(e_powers.shape[0] - 1):
            for j in range(i + 1, e_powers.shape[1]):
                r = e_powers[i, j, 1]
                if r <= self.u_cutoff:
                    u_set = int(i >= neu) + int(j >= neu)
                    poly = 0.0
                    for k in range(parameters.shape[0]):
                        poly += parameters[k, u_set] * e_powers[i, j, k]
                    res += poly * (r - self.u_cutoff) ** self.trunc
        return res

    def chi_term(self, n_powers, neu):
        """Jastrow chi-term
        :param n_powers: powers of e-e distances
        :param neu: number of up electrons
        :return:
        """
        res = 0.0
        if not self.chi_cutoff.any():
            return res
        C = self.trunc
        for i, (parameters, L) in enumerate(zip(self.chi_parameters, self.chi_cutoff)):
            for j in range(n_powers.shape[1]):
                r = n_powers[i, j, 1]
                if r <= L:
                    chi_set = int(j >= neu)
                    poly = 0.0
                    for k in range(parameters.shape[0]):
                        poly += parameters[k, chi_set] * n_powers[i, j, k]
                    res += poly * (r - L) ** C
        return res

    def f_term(self, e_powers, n_powers, neu):
        """Jastrow f-term
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :param ned: number of up electrons
        :return:
        """
        res = 0.0
        if not self.f_cutoff.any():
            return res

        C = self.trunc
        for i, (parameters, L) in enumerate(zip(self.f_parameters, self.f_cutoff)):
            for j in range(n_powers.shape[1] - 1):
                for k in range(j+1, e_powers.shape[0]):
                    r_e1I = n_powers[i, j, 1]
                    r_e2I = n_powers[i, k, 1]
                    if r_e1I <= L and r_e2I <= L:
                        f_set = int(j >= neu) + int(k >= neu)
                        poly = 0.0
                        for l in range(parameters.shape[0]):
                            for m in range(parameters.shape[1]):
                                for n in range(parameters.shape[2]):
                                    poly += parameters[l, m, n, f_set] * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n]
                        res += poly * (r_e1I - L) ** C * (r_e2I - L) ** C
        return res

    def u_term_gradient(self, e_powers, e_vectors, neu):
        """Jastrow u-term gradient with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :param e_vectors: e-e vectors
        :param neu: number of up electrons
        :return:
        """
        res = np.zeros((e_vectors.shape[0], 3))

        if not self.u_cutoff:
            return res

        C = self.trunc
        L = self.u_cutoff
        parameters = self.u_parameters
        for i in range(e_powers.shape[0] - 1):
            for j in range(i + 1, e_powers.shape[1]):
                r_vec = e_vectors[i, j]
                r = e_powers[i, j, 1]
                if r <= L:
                    u_set = int(i >= neu) + int(j >= neu)
                    poly = poly_diff = 0.0
                    for k in range(parameters.shape[0]):
                        poly += parameters[k, u_set] * e_powers[i, j, k]
                        if k > 0:
                            poly_diff += parameters[k, u_set] * k * e_powers[i, j, k-1]

                    gradient = (C * (r-L) ** (C-1) * poly + (r-L) ** C * poly_diff) / r
                    res[i, :] += r_vec * gradient
                    res[j, :] -= r_vec * gradient
        return res

    def chi_term_gradient(self, n_powers, n_vectors, neu):
        """Jastrow chi-term gradient with respect to a e-coordinates
        :param n_powers: powers of e-n distances
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
        :return:
        """
        res = np.zeros((n_vectors.shape[0], 3))

        if not self.chi_cutoff.any():
            return res

        C = self.trunc
        for i, (parameters, L) in enumerate(zip(self.chi_parameters, self.chi_cutoff)):
            for j in range(n_powers.shape[1]):
                r_vec = n_vectors[j, i]
                r = n_powers[i, j, 1]
                if r <= L:
                    chi_set = int(j >= neu)
                    poly = poly_diff = 0.0
                    for k in range(parameters.shape[0]):
                        poly += parameters[k, chi_set] * n_powers[i, j, k]
                        if k > 0:
                            poly_diff += parameters[k, chi_set] * k * n_powers[i, j, k-1]

                    gradient = (C * (r-L) ** (C-1) * poly + (r-L) ** C * poly_diff) / r
                    res[j, :] += r_vec * gradient
        return res

    def f_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors, neu):
        """Jastrow f-term gradient with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
        :return:
        """
        res = np.zeros((e_vectors.shape[0], 3))

        if not self.f_cutoff.any():
            return res

        C = self.trunc
        for i, (parameters, L) in enumerate(zip(self.f_parameters, self.f_cutoff)):
            for j in range(n_powers.shape[1] - 1):
                for k in range(j+1, e_powers.shape[0]):
                    r_e1I_vec = n_vectors[j, i]
                    r_e2I_vec = n_vectors[k, i]
                    r_ee_vec = e_vectors[j, k]
                    r_e1I = n_powers[i, j, 1]
                    r_e2I = n_powers[i, k, 1]
                    r_ee = e_powers[j, k, 1]
                    if r_e1I <= L and r_e2I <= L:
                        f_set = int(j >= neu) + int(k >= neu)
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
        return res

    def u_term_laplacian(self, e_powers, neu):
        """Jastrow u-term laplacian with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :param neu: number of up electrons
        :return:
        """
        res = 0.0
        if not self.u_cutoff:
            return res

        C = self.trunc
        L = self.u_cutoff
        parameters = self.u_parameters
        for i in range(e_powers.shape[0] - 1):
            for j in range(i + 1, e_powers.shape[1]):
                r = e_powers[i, j, 1]
                if r <= L:
                    u_set = int(i >= neu) + int(j >= neu)
                    poly = poly_diff = poly_diff_2 = 0.0
                    for k in range(parameters.shape[0]):
                        poly += parameters[k, u_set] * e_powers[i, j, k]
                        if k > 0:
                            poly_diff += k * parameters[k, u_set] * e_powers[i, j, k-1]
                        if k > 1:
                            poly_diff_2 += k * (k-1) * parameters[k, u_set] * e_powers[i, j, k-2]

                    res += (
                        C*(C - 1)*(r-L)**(C - 2) * poly +
                        2 * C*(r-L)**(C - 1) * poly_diff + (r-L)**C * poly_diff_2 +
                        2 * (C * (r-L)**(C-1) * poly + (r-L)**C * poly_diff) / r
                    )
        return 2 * res

    def chi_term_laplacian(self, n_powers, neu):
        """Jastrow chi-term laplacian with respect to a e-coordinates
        :param n_powers: powers of e-n distances
        :param neu: number of up electrons
        :return:
        """
        res = 0.0
        if not self.chi_cutoff.any():
            return res

        C = self.trunc
        for i, (parameters, L) in enumerate(zip(self.chi_parameters, self.chi_cutoff)):
            for j in range(n_powers.shape[1]):
                r = n_powers[i, j, 1]
                if r <= L:
                    chi_set = int(j >= neu)
                    poly = poly_diff = poly_diff_2 = 0.0
                    for k in range(parameters.shape[0]):
                        poly += parameters[k, chi_set] * n_powers[i, j, k]
                        if k > 0:
                            poly_diff += k * parameters[k, chi_set] * n_powers[i, j, k-1]
                        if k > 1:
                            poly_diff_2 += k * (k-1) * parameters[k, chi_set] * n_powers[i, j, k-2]

                    res += (
                        C*(C - 1)*(r-L)**(C - 2) * poly +
                        2 * C*(r-L)**(C - 1) * poly_diff + (r-L)**C * poly_diff_2 +
                        2 * (C * (r-L)**(C-1) * poly + (r-L)**C * poly_diff) / r
                    )
        return res

    def f_term_laplacian(self, e_powers, n_powers, e_vectors, n_vectors, neu):
        """Jastrow f-term laplacian with respect to a e-coordinates
        f-term is a product of two spherically symmetric functions f(r_eI) and g(r_ee) so using
            ∇²(f*g) = ∇²(f)*g + 2*∇(f)*∇(g) + f*∇²(g)
        then Laplace operator of spherically symmetric function (in 3-D space) is
            ∇²(f) = d²f/dr² + 2/r * df/dr
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
        :return:
        """
        res = 0.0
        if not self.f_cutoff.any():
            return res

        C = self.trunc
        for i, (parameters, L) in enumerate(zip(self.f_parameters, self.f_cutoff)):
            for j in range(n_powers.shape[1] - 1):
                for k in range(j + 1, e_powers.shape[0]):
                    r_e1I_vec = n_vectors[j, i]
                    r_e2I_vec = n_vectors[k, i]
                    r_ee_vec = e_vectors[j, k]
                    r_e1I = n_powers[i, j, 1]
                    r_e2I = n_powers[i, k, 1]
                    r_ee = e_powers[j, k, 1]
                    if r_e1I <= L and r_e2I <= L:
                        f_set = int(j >= neu) + int(k >= neu)
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

    def value(self, e_vectors, n_vectors, neu):
        """Jastrow with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
        :return:
        """

        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term(e_powers, neu) +
            self.chi_term(n_powers, neu) +
            self.f_term(e_powers, n_powers, neu)
        )

    def numerical_gradient(self, e_vectors, n_vectors, neu):
        """Numerical gradient with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu:
        :return:
        """
        delta = 0.00001

        res = np.zeros((e_vectors.shape[0], 3))

        for i in range(e_vectors.shape[0]):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                n_vectors[i, :, j] -= delta
                res[i, j] -= self.value(e_vectors, n_vectors, neu)
                e_vectors[i, :, j] += 2 * delta
                n_vectors[i, :, j] += 2 * delta
                res[i, j] += self.value(e_vectors, n_vectors, neu)
                e_vectors[i, :, j] -= delta
                n_vectors[i, :, j] -= delta

        return res / delta / 2

    def numerical_laplacian(self, e_vectors, n_vectors, neu):
        """Numerical laplacian with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
        :return:
        """
        delta = 0.00001

        res = -2 * r_e.size * self.value(e_vectors, n_vectors, neu)
        for i in range(e_vectors.shape[0]):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                n_vectors[i, :, j] -= delta
                res += self.value(e_vectors, n_vectors, neu)
                e_vectors[i, :, j] += 2 * delta
                n_vectors[i, :, j] += 2 * delta
                res += self.value(e_vectors, n_vectors, neu)
                e_vectors[i, :, j] -= delta
                n_vectors[i, :, j] -= delta

        return res / delta / delta

    def gradient(self, e_vectors, n_vectors, neu):
        """Gradient with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term_gradient(e_powers, e_vectors, neu) +
            self.chi_term_gradient(n_powers, n_vectors, neu) +
            self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors, neu)
        )

    def laplacian(self, e_vectors, n_vectors, neu):
        """Laplacian with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term_laplacian(e_powers, neu) +
            self.chi_term_laplacian(n_powers, neu) +
            self.f_term_laplacian(e_powers, n_powers, e_vectors, n_vectors, neu)
        )

    def get_parameters(self):
        """"""
        return np.array([self.u_cutoff])

    def set_parameters(self, parameters):
        """
        :param parameters:
        :return:
        """
        self.u_cutoff = parameters[0]

    def u_term_cutoff_numerical_d1(self, e_powers, neu):
        """Numerical first derivatives of logarithm u-term with respect to u_cutoff
        :param e_powers: powers of e-e distances
        :param neu: number of up electrons
        """
        if not self.u_cutoff:
            return np.zeros((0,))

        delta = 0.00001
        res = np.zeros((1, ))

        self.u_cutoff -= delta
        res[0] -= self.u_term(e_powers, neu)
        self.u_cutoff += 2 * delta
        res[0] += self.u_term(e_powers, neu)
        self.u_cutoff -= delta

        return res / delta / 2

    def u_term_linear_parameters_numerical_d1(self, e_powers, neu):
        """Numerical first derivatives of logarithm u-term with respect to the linear Jastrow parameters
        :param e_powers: powers of e-e distances
        :param neu: number of up electrons
        """
        # if not self.u_cutoff:
        if True:
            return np.zeros((0,))

        delta = 0.00001
        res = np.zeros(((self.u_parameters.shape[0] - 1) * (self.u_spin_dep + 1),))

        n = -1
        for i in range(self.u_parameters.shape[0]):
            if i == 1:
                continue
            # (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
            if self.u_spin_dep == 0:
                n += 1
                self.u_parameters[i] -= delta
                res[n] -= self.u_term(e_powers, neu)
                self.u_parameters[i] += 2 * delta
                res[n] += self.u_term(e_powers, neu)
                self.u_parameters[i] -= delta
            elif self.u_spin_dep == 1:
                n += 1
                self.u_parameters[i, 0] -= delta
                self.u_parameters[i, 2] -= delta
                res[n] -= self.u_term(e_powers, neu)
                self.u_parameters[i, 0] += 2 * delta
                self.u_parameters[i, 2] += 2 * delta
                res[n] += self.u_term(e_powers, neu)
                self.u_parameters[i, 0] -= delta
                self.u_parameters[i, 2] -= delta
                n += 1
                self.u_parameters[i, 1] -= delta
                res[n] -= self.u_term(e_powers, neu)
                self.u_parameters[i, 1] += 2 * delta
                res[n] += self.u_term(e_powers, neu)
                self.u_parameters[i, 1] -= delta
            elif self.u_spin_dep == 2:
                for j in range(3):
                    n += 1
                    self.u_parameters[i, j] -= delta
                    res[n] -= self.u_term(e_powers, neu)
                    self.u_parameters[i, j] += 2 * delta
                    res[n] += self.u_term(e_powers, neu)
                    self.u_parameters[i, j] -= delta

        return res / delta / 2

    def chi_term_cutoff_numerical_d1(self, n_powers, neu):
        """Numerical first derivatives of logarithm chi-term with respect to chi_cutoff
        :param e_powers: powers of e-e distances
        :param neu: number of up electrons
        """
        if not self.chi_cutoff.any():
            return np.zeros((0,))

        delta = 0.00001
        res = np.zeros(len(self.chi_parameters),)

        n = -1
        for cutoff in self.chi_cutoff:
            n += 1
            cutoff -= delta
            res[n] -= self.chi_term(n_powers, neu)
            cutoff += 2 * delta
            res[n] += self.chi_term(n_powers, neu)
            cutoff -= delta

        return res / delta / 2

    def chi_term_linear_parameters_numerical_d1(self, n_powers, neu):
        """Numerical first derivatives of logarithm chi-term with respect to the linear Jastrow parameters
        :param n_powers: powers of e-n distances
        :param neu: number of up electrons
        """
        if not self.chi_cutoff.any():
            return np.zeros((0,))

        delta = 0.00001
        res = np.zeros((np.array(list([p.shape[0] * (sd + 1) for p, sd in zip(self.chi_parameters, self.chi_spin_dep)])).sum(),))

        n = -1
        for p in self.chi_parameters:
            for i in range(p.shape[0]):
                if i == 1:
                    continue
                # (0->u=d; 1->u/=d)
                if self.chi_spin_dep == 0:
                    n += 1
                    p[i] -= delta
                    res[n] -= self.chi_term(n_powers, neu)
                    p[i] += 2 * delta
                    res[n] += self.chi_term(n_powers, neu)
                    p[i] -= delta
                elif self.chi_spin_dep == 1:
                    for j in range(2):
                        n += 1
                        p[i, j] -= delta
                        res[n] -= self.chi_term(n_powers, neu)
                        p[i, j] += 2 * delta
                        res[n] += self.chi_term(n_powers, neu)
                        p[i, j] -= delta

        return res / delta / 2

    def f_term_cutoff_numerical_d1(self, e_powers, n_powers, neu):
        """Numerical first derivatives of logarithm f-term with respect to f_cutoff
        :param e_powers: powers of e-e distances
        :param neu: number of up electrons
        """
        if not self.f_cutoff.any():
            return np.zeros((0,))

        delta = 0.00001
        res = np.zeros(len(self.f_parameters),)

        n = -1
        for cutoff in self.f_cutoff:
            n += 1
            cutoff -= delta
            res[n] -= self.f_term(e_powers, n_powers, neu)
            cutoff += 2 * delta
            res[n] += self.f_term(e_powers, n_powers, neu)
            cutoff -= delta

        return res / delta / 2

    def f_term_linear_parameters_numerical_d1(self, e_powers, n_powers, neu):
        """Numerical first derivatives of logarithm f-term with respect to the linear Jastrow parameters
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :param neu: number of up electrons
        """
        if not self.f_cutoff.any():
            return np.zeros((0,))

        delta = 0.00001
        res = np.zeros((
            np.array(list([p.shape[0] * p.shape[1] * p.shape[2] * (sd + 1) for p, sd in zip(self.f_parameters, self.f_spin_dep)], )).sum(),
        ))

        n = -1
        for p in self.f_parameters:
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    for k in range(p.shape[2]):
                        # (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
                        if self.f_spin_dep == 0:
                            n += 1
                            p[i, j, k] -= delta
                            res[n] -= self.f_term(e_powers, n_powers, neu)
                            p[i, j, k] += 2 * delta
                            res[n] += self.f_term(e_powers, n_powers, neu)
                            p[i, j, k] -= delta
                        elif self.f_spin_dep == 1:
                            n += 1
                            p[i, j, k, 0] -= delta
                            p[i, j, k, 2] -= delta
                            res[n] -= self.f_term(e_powers, n_powers, neu)
                            p[i, j, k, 0] += 2 * delta
                            p[i, j, k, 2] += 2 * delta
                            res[n] += self.f_term(e_powers, n_powers, neu)
                            p[i, j, k, 0] -= delta
                            p[i, j, k, 2] -= delta
                            n += 1
                            p[i, j, k, 1] -= delta
                            res[n] -= self.f_term(e_powers, n_powers, neu)
                            p[i, j, k, 1] += 2 * delta
                            res[n] += self.f_term(e_powers, n_powers, neu)
                            p[i, j, k, 1] -= delta
                        elif self.f_spin_dep == 2:
                            for l in range(3):
                                n += 1
                                p[i, j, k, l] -= delta
                                res[n] -= self.f_term(e_powers, n_powers, neu)
                                p[i, j, k, l] += 2 * delta
                                res[n] += self.f_term(e_powers, n_powers, neu)
                                p[i, j, k, l] -= delta

        return res / delta / 2

    def parameters_numerical_d1(self, e_vectors, n_vectors, neu):
        """Numerical first derivatives logarithm Jastrow with respect to the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return np.concatenate((
            self.u_term_cutoff_numerical_d1(e_powers, neu),
            self.chi_term_cutoff_numerical_d1(n_powers, neu),
            self.f_term_cutoff_numerical_d1(e_powers, n_powers, neu),
            self.u_term_linear_parameters_numerical_d1(e_powers, neu),
            self.chi_term_linear_parameters_numerical_d1(n_powers, neu),
            self.f_term_linear_parameters_numerical_d1(e_powers, n_powers, neu)
        ))

    def u_term_cutoff_numerical_d2(self, e_powers, neu):
        """Numerical second derivatives of logarithm u-term with respect to u_cutoff
        :param e_powers: powers of e-e distances
        :param neu: number of up electrons
        """

        delta = 0.00001
        res = np.zeros((1, ))

        res[0] = -2 * self.u_term(e_powers, neu)
        self.u_cutoff -= delta
        res[0] += self.u_term(e_powers, neu)
        self.u_cutoff += 2 * delta
        res[0] += self.u_term(e_powers, neu)
        self.u_cutoff -= delta

        return res / delta / 2

    def u_term_parameters_numerical_d2(self, e_powers, neu):
        """Numerical second derivatives of logarithm u-term with respect to the Jastrow parameters
        :param e_powers: powers of e-e distances
        :param neu: number of up electrons
        """
        if not self.u_cutoff:
            return np.zeros((0, 0))

        delta = 0.00001
        res = -2 * self.u_term(e_powers, neu) * np.eye(self.u_parameters.size)
        # diagonal terms of linear parameters
        n = 0
        for i in range(self.u_parameters.shape[0]):
            for j in range(self.u_parameters.shape[1]):
                n += 1
                self.u_parameters[i, j] -= delta
                res[n, n] += self.u_term(e_powers, neu)
                self.u_parameters[i, j] += 2 * delta
                res[n, n] += self.u_term(e_powers, neu)
                self.u_parameters[i, j] -= delta

        n = 0
        for i in range(self.u_parameters.shape[0]):
            for j in range(self.u_parameters.shape[1]):
                n += 1
                self.u_parameters[i, j] -= delta
                self.u_cutoff -= delta
                res[0, n] += self.u_term(e_powers, neu)
                self.u_parameters[i, j] += 2 * delta
                res[0, n] -= self.u_term(e_powers, neu)
                self.u_cutoff += 2 * delta
                res[0, n] += self.u_term(e_powers, neu)
                self.u_parameters[i, j] -= 2 * delta
                res[0, n] += self.u_term(e_powers, neu)
                self.u_parameters[i, j] += delta
                self.u_cutoff -= delta
                res[n, 0] = res[0, n]

        n = m = 0
        for i1 in range(self.u_parameters.shape[0]):
            for j1 in range(self.u_parameters.shape[1]):
                n += 1
                for i2 in range(self.u_parameters.shape[0]):
                    for j2 in range(self.u_parameters.shape[1]):
                        m += 1
                        if m > n:
                            self.u_parameters[i1, j1] -= delta
                            self.u_parameters[i2, j2] -= delta
                            res[n, m] += self.u_term(e_powers, neu)
                            self.u_parameters[i1, j1] += 2 * delta
                            res[n, m] -= self.u_term(e_powers, neu)
                            self.u_parameters[i2, j2] += 2 * delta
                            res[n, m] += self.u_term(e_powers, neu)
                            self.u_parameters[i1, j1] -= 2 * delta
                            res[n, m] += self.u_term(e_powers, neu)
                            self.u_parameters[i1, j1] += delta
                            self.u_parameters[i2, j2] -= delta
                            res[m, n] = res[n, m]

        return res / delta / delta

    def chi_term_parameters_numerical_d2(self, n_powers, neu):
        """Numerical second derivatives of logarithm chi-term with respect to the Jastrow parameters
        :param n_powers: powers of e-n distances
        :param neu: number of up electrons
        """
        if not self.chi_cutoff.any():
            return np.zeros((0, 0))

        delta = 0.00001
        size = np.array(list([p.size + 1 for p in self.chi_parameters])).sum()
        res = -2 * self.chi_term(n_powers, neu) * np.eye(size)
        n = -1
        for parameters, cutoff in zip(self.chi_parameters, self.chi_cutoff):
            n += 1
            cutoff -= delta
            res[n, n] += self.chi_term(n_powers, neu)
            cutoff += 2 * delta
            res[n, n] += self.chi_term(n_powers, neu)
            cutoff -= delta

        return res / delta / delta

    def f_term_parameters_numerical_d2(self, e_powers, n_powers, neu):
        """Numerical second derivatives of logarithm f-term with respect to the Jastrow parameters
        :param n_powers: powers of e-n distances
        :param neu: number of up electrons
        """
        if not self.f_cutoff.any():
            return np.zeros((0, 0))

        delta = 0.00001
        size = np.array(list([p.size + 1 for p in self.f_parameters])).sum()
        res = -2 * self.f_term(e_powers, n_powers, neu) * np.eye(size)
        n = -1
        for parameters, cutoff in zip(self.f_parameters, self.f_cutoff):
            cutoff -= delta
            res[n, n] += self.f_term(e_powers, n_powers, neu)
            cutoff += 2 * delta
            res[n, n] += self.f_term(e_powers, n_powers, neu)
            cutoff -= delta

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

        self.u_term_parameters_numerical_d2(e_powers, neu)
        self.chi_term_parameters_numerical_d2(n_powers, neu)
        self.f_term_parameters_numerical_d2(e_powers, n_powers, neu)


if __name__ == '__main__':
    """Plot Jastrow terms
    """

    term = 'f'

    path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_no_u_vmc/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_no_chi/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_no_u_no_chi_vmc/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_vmc_cbc/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/al/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/acetaldehyde/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'

    casino = Casino(path)
    jastrow = Jastrow(
        casino.jastrow.trunc,
        casino.jastrow.u_parameters, casino.jastrow.u_cutoff, casino.jastrow.u_spin_dep,
        casino.jastrow.chi_parameters, casino.jastrow.chi_cutoff, casino.jastrow.chi_labels, casino.jastrow.chi_spin_dep,
        casino.jastrow.f_parameters, casino.jastrow.f_cutoff, casino.jastrow.f_labels, casino.jastrow.f_spin_dep
    )

    steps = 100

    if term == 'u':
        x_min, x_max = 0, jastrow.u_cutoff
        x_grid = np.linspace(x_min, x_max, steps)
        for spin_dep in range(3):
            y_grid = np.zeros(steps)
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
                y_grid = np.zeros(steps)
                for i in range(100):
                    r_e = np.array([[x_grid[i], 0.0, 0.0]]) + casino.wfn.atom_positions[atom]
                    sl = slice(atom, atom+1)
                    jastrow.chi_parameters = nb.typed.List.empty_list(chi_parameters_type)
                    [jastrow.chi_parameters.append(p) for p in casino.jastrow.chi_parameters[sl]]
                    n_vectors = subtract_outer(r_e, casino.wfn.atom_positions[sl])
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
                        n_vectors = subtract_outer(r_e, casino.wfn.atom_positions[sl])
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
