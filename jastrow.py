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

u_parameters_type = nb.float64[:, :]
chi_parameters_type = nb.float64[:, :]
f_parameters_type = nb.float64[:, :, :, :]

spec = [
    ('trunc', nb.int64),
    ('u_parameters', nb.types.ListType(u_parameters_type)),
    ('chi_parameters', nb.types.ListType(chi_parameters_type)),
    ('f_parameters', nb.types.ListType(f_parameters_type)),
    ('u_cutoff', nb.float64[:]),
    ('chi_cutoff', nb.float64[:]),
    ('f_cutoff', nb.float64[:]),
    ('max_ee_order', nb.int64),
    ('max_en_order', nb.int64),
]


@nb.experimental.jitclass(spec)
class Jastrow:

    def __init__(self, trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff):
        self.trunc = trunc
        self.u_parameters = nb.typed.List.empty_list(u_parameters_type)
        [self.u_parameters.append(p) for p in u_parameters]
        self.chi_parameters = nb.typed.List.empty_list(chi_parameters_type)
        [self.chi_parameters.append(p) for p in chi_parameters]
        self.f_parameters = nb.typed.List.empty_list(f_parameters_type)
        [self.f_parameters.append(p) for p in f_parameters]
        self.u_cutoff = u_cutoff
        self.chi_cutoff = chi_cutoff
        self.f_cutoff = f_cutoff
        self.max_ee_order = max((
            max([p.shape[0] for p in self.u_parameters]),
            max([p.shape[2] for p in self.f_parameters]),
        ))
        self.max_en_order = max((
            max([p.shape[0] for p in self.chi_parameters]),
            max([p.shape[0] for p in self.f_parameters]),
        ))

    def ee_powers(self, e_vectors):
        """powers of e-e distances"""
        res = np.zeros((e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(e_vectors.shape[0] - 1):
            for j in range(i + 1, e_vectors.shape[1]):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(self.max_ee_order):
                    res[i, j, k] = r_ee ** k
        return res

    def en_powers(self, n_vectors):
        """powers of e-n distances"""
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
        if not self.u_cutoff.any():
            return res

        p = self.u_parameters[0]
        for i in range(e_powers.shape[0] - 1):
            for j in range(i + 1, e_powers.shape[1]):
                r = e_powers[i, j, 1]
                if r <= self.u_cutoff[0]:
                    u_set = int(i >= neu) + int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, u_set] * e_powers[i, j, k]
                    res += poly * (r - self.u_cutoff[0]) ** self.trunc
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

        for i in range(n_powers.shape[0]):
            p = self.chi_parameters[i]
            for j in range(n_powers.shape[1]):
                r = n_powers[i, j, 1]
                if r <= self.chi_cutoff[i]:
                    chi_set = int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, chi_set] * n_powers[i, j, k]
                    res += poly * (r - self.chi_cutoff[i]) ** self.trunc
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

        for i in range(n_powers.shape[0]):
            p = self.f_parameters[i]
            for j in range(n_powers.shape[1] - 1):
                for k in range(j+1, e_powers.shape[0]):
                    r_e1I = n_powers[i, j, 1]
                    r_e2I = n_powers[i, k, 1]
                    if r_e1I <= self.f_cutoff[i] and r_e2I <= self.f_cutoff[i]:
                        f_set = int(j >= neu) + int(k >= neu)
                        poly = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly += p[l, m, n, f_set] * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n]
                        C = self.trunc
                        L = self.f_cutoff[i]
                        res += poly * (r_e1I - L) ** C * (r_e2I - L) ** C
        return res

    def u_term_gradient(self, e_powers, e_vectors, neu):
        """Jastrow u-term gradient with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :param e_vectors: electrons coordinates
        :param neu: number of up electrons
        :return:
        """
        res = np.zeros((e_vectors.shape[0], 3))

        if not self.u_cutoff.any():
            return res

        p = self.u_parameters[0]
        for i in range(e_powers.shape[0] - 1):
            for j in range(i + 1, e_powers.shape[1]):
                r_vec = e_vectors[i, j]
                r = e_powers[i, j, 1]
                if r <= self.u_cutoff[0]:
                    u_set = int(i >= neu) + int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, u_set] * e_powers[i, j, k]

                    poly_diff = 0.0
                    for k in range(1, p.shape[0]):
                        poly_diff += p[k, u_set] * k * e_powers[i, j, k-1]

                    C = self.trunc
                    L = self.u_cutoff[0]
                    gradient = (C * (r-L) ** (C-1) * poly + (r-L) ** C * poly_diff) / r
                    res[i, :] += r_vec * gradient
                    res[j, :] -= r_vec * gradient
        return res

    def chi_term_gradient(self, n_powers, n_vectors, neu):
        """Jastrow chi-term gradient with respect to a e-coordinates
        :param n_powers: powers of e-n distances
        :param n_vectors: nucleus coordinates
        :param neu: number of up electrons
        :return:
        """
        res = np.zeros((n_vectors.shape[0], 3))

        if not self.chi_cutoff.any():
            return res

        for i in range(n_powers.shape[0]):
            p = self.chi_parameters[i]
            for j in range(n_powers.shape[1]):
                r_vec = n_vectors[j, i]
                r = n_powers[i, j, 1]
                if r <= self.chi_cutoff[i]:
                    chi_set = int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, chi_set] * n_powers[i, j, k]

                    poly_diff = 0.0
                    for k in range(1, p.shape[0]):
                        poly_diff += p[k, chi_set] * k * n_powers[i, j, k-1]

                    C = self.trunc
                    L = self.chi_cutoff[i]
                    gradient = (C * (r-L) ** (C-1) * poly + (r-L) ** C * poly_diff) / r
                    res[j, :] += r_vec * gradient
        return res

    def f_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors, neu):
        """Jastrow f-term gradient with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :param e_vectors: electrons coordinates
        :param n_vectors: electrons coordinates
        :param neu: number of up electrons
        :return:
        """
        res = np.zeros((e_vectors.shape[0], 3))

        if not self.f_cutoff.any():
            return res

        for i in range(n_powers.shape[0]):
            p = self.f_parameters[i]
            for j in range(n_powers.shape[1] - 1):
                for k in range(j+1, e_powers.shape[0]):
                    r_e1I_vec = n_vectors[j, i]
                    r_e2I_vec = n_vectors[k, i]
                    r_ee_vec = e_vectors[j, k]
                    r_e1I = n_powers[i, j, 1]
                    r_e2I = n_powers[i, k, 1]
                    r_ee = e_powers[j, k, 1]
                    if r_e1I <= self.f_cutoff[i] and r_e2I <= self.f_cutoff[i]:
                        f_set = int(j >= neu) + int(k >= neu)
                        poly = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly += p[l, m, n, f_set] * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n]

                        poly_diff_e1I = 0.0
                        for l in range(1, p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e1I += p[l, m, n, f_set] * l * n_powers[i, j, l-1] * n_powers[i, k, m] * e_powers[j, k, n]

                        poly_diff_e2I = 0.0
                        for l in range(p.shape[0]):
                            for m in range(1, p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e2I += p[l, m, n, f_set] * m * n_powers[i, j, l] * n_powers[i, k, m-1] * e_powers[j, k, n]

                        poly_diff_ee = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(1, p.shape[2]):
                                    poly_diff_ee += p[l, m, n, f_set] * n * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n-1]

                        C = self.trunc
                        L = self.f_cutoff[i]
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
        if not self.u_cutoff.any():
            return res

        p = self.u_parameters[0]
        for i in range(e_powers.shape[0] - 1):
            for j in range(i + 1, e_powers.shape[1]):
                r = e_powers[i, j, 1]
                if r <= self.u_cutoff[0]:
                    u_set = int(i >= neu) + int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, u_set] * e_powers[i, j, k]

                    poly_diff = 0.0
                    for k in range(1, p.shape[0]):
                        poly_diff += k * p[k, u_set] * e_powers[i, j, k-1]

                    poly_diff_2 = 0.0
                    for k in range(2, p.shape[0]):
                        poly_diff_2 += k * (k-1) * p[k, u_set] * e_powers[i, j, k-2]

                    C = self.trunc
                    L = self.u_cutoff[0]
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

        for i in range(n_powers.shape[0]):
            p = self.chi_parameters[i]
            for j in range(n_powers.shape[1]):
                r = n_powers[i, j, 1]
                if r <= self.chi_cutoff[i]:
                    chi_set = int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, chi_set] * n_powers[i, j, k]

                    poly_diff = 0.0
                    for k in range(1, p.shape[0]):
                        poly_diff += k * p[k, chi_set] * n_powers[i, j, k-1]

                    poly_diff_2 = 0.0
                    for k in range(2, p.shape[0]):
                        poly_diff_2 += k * (k-1) * p[k, chi_set] * n_powers[i, j, k-2]

                    C = self.trunc
                    L = self.chi_cutoff[i]
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
        :param e_vectors: electrons coordinates
        :param n_vectors: nucleus coordinates
        :param neu: number of up electrons
        :return:
        """
        res = 0.0
        if not self.f_cutoff.any():
            return res

        for i in range(n_powers.shape[0]):
            p = self.f_parameters[i]
            for j in range(n_powers.shape[1] - 1):
                for k in range(j + 1, e_powers.shape[0]):
                    r_e1I_vec = n_vectors[j, i]
                    r_e2I_vec = n_vectors[k, i]
                    r_ee_vec = e_vectors[j, k]
                    r_e1I = n_powers[i, j, 1]
                    r_e2I = n_powers[i, k, 1]
                    r_ee = e_powers[j, k, 1]
                    if r_e1I <= self.f_cutoff[i] and r_e2I <= self.f_cutoff[i]:
                        f_set = int(j >= neu) + int(k >= neu)
                        poly = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly += p[l, m, n, f_set] * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n]

                        poly_diff_e1I = 0.0
                        for l in range(1, p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e1I += p[l, m, n, f_set] * l * n_powers[i, j, l-1] * n_powers[i, k, m] * e_powers[j, k, n]

                        poly_diff_e2I = 0.0
                        for l in range(p.shape[0]):
                            for m in range(1, p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e2I += p[l, m, n, f_set] * m * n_powers[i, j, l] * n_powers[i, k, m-1] * e_powers[j, k, n]

                        poly_diff_ee = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(1, p.shape[2]):
                                    poly_diff_ee += p[l, m, n, f_set] * n * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n-1]

                        poly_diff_e1I_2 = 0.0
                        for l in range(2, p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e1I_2 += p[l, m, n, f_set] * l * (l-1) * n_powers[i, j, l-2] * n_powers[i, k, m] * e_powers[j, k, n]

                        poly_diff_e2I_2 = 0.0
                        for l in range(p.shape[0]):
                            for m in range(2, p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e2I_2 += p[l, m, n, f_set] * m * (m-1) * n_powers[i, j, l] * n_powers[i, k, m-2] * e_powers[j, k, n]

                        poly_diff_ee_2 = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(2, p.shape[2]):
                                    poly_diff_ee_2 += p[l, m, n, f_set] * n * (n-1) * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n-2]

                        poly_diff_e1I_ee = 0.0
                        for l in range(1, p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(1, p.shape[2]):
                                    poly_diff_e1I_ee += p[l, m, n, f_set] * l * n * n_powers[i, j, l-1] * n_powers[i, k, m] * e_powers[j, k, n-1]

                        poly_diff_e2I_ee = 0.0
                        for l in range(p.shape[0]):
                            for m in range(1, p.shape[1]):
                                for n in range(1, p.shape[2]):
                                    poly_diff_e2I_ee += p[l, m, n, f_set] * m * n * n_powers[i, j, l] * n_powers[i, k, m-1] * e_powers[j, k, n-1]

                        C = self.trunc
                        L = self.f_cutoff[i]
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
        :param e_vectors: electrons coordinates
        :param n_vectors: nucleus coordinates
        :param neu: number of up electrons
        :return:
        """

        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.u_term(e_powers, neu) + self.chi_term(n_powers, neu) + self.f_term(e_powers, n_powers, neu)

    def numerical_gradient(self, e_vectors, n_vectors, neu):
        """Numerical gradient with respect to a e-coordinates
        :param e_vectors:
        :param n_vectors:
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
        :param e_vectors:
        :param n_vectors:
        :param neu:
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
        :param e_vectors:
        :param n_vectors:
        :param neu:
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.u_term_gradient(e_powers, e_vectors, neu) + self.chi_term_gradient(n_powers, n_vectors, neu) + self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors, neu)

    def laplacian(self, e_vectors, n_vectors, neu):
        """Laplacian with respect to a e-coordinates
        :param e_vectors:
        :param n_vectors:
        :param neu:
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.u_term_laplacian(e_powers, neu) + self.chi_term_laplacian(n_powers, neu) + self.f_term_laplacian(e_powers, n_powers, e_vectors, n_vectors, neu)


if __name__ == '__main__':
    """Plot Jastrow terms
    """

    term = 'u'

    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_no_u_vmc/'
    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_no_chi/'
    path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_no_u_no_chi_vmc/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term_vmc_cbc/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/al/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/acetaldehyde/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'

    casino = Casino(path)
    jastrow = Jastrow(
        casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_cutoff,
        casino.jastrow.chi_parameters, casino.jastrow.chi_cutoff,
        casino.jastrow.f_parameters, casino.jastrow.f_cutoff
    )

    steps = 100

    if term == 'u':
        x_min, x_max = 0, jastrow.u_cutoff[0]
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
