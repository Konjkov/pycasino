#!/usr/bin/env python3

import numpy as np
import numba as nb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from readers.casino import Casino

"""
https://github.com/numba/numba/issues/4522

So to summarize the current state (as of numba 0.45.1), jitclasses will (almost) always be slower
 than a functional njit equivalent routine because of the reference counting of class attributes.
I guess when using classes + numba, @selslack's suggestion of creating a normal class with njit methods,
 which is not the most ideal because it leads to some obfuscation, is the best option.
"""

u_parameters_type = nb.types.float64[:, :]
chi_parameters_type = nb.types.float64[:, :]
f_parameters_type = nb.types.float64[:, :, :, :]

spec = [
    ('trunc', nb.int64),
    ('u_parameters', nb.types.ListType(u_parameters_type)),
    ('chi_parameters', nb.types.ListType(chi_parameters_type)),
    ('f_parameters', nb.types.ListType(f_parameters_type)),
    ('u_cutoff', nb.float64[:]),
    ('chi_cutoff', nb.float64[:]),
    ('f_cutoff', nb.float64[:])
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

    def u_term(self, r_e, neu):
        """Jastrow u-term
        :param r_e: electrons coordinates
        :param neu: number of up electrons
        :return:
        """
        res = 0.0
        if not self.u_cutoff.any():
            return res

        p = self.u_parameters[0]
        for i in range(r_e.shape[0] - 1):
            for j in range(i + 1, r_e.shape[0]):
                r = np.linalg.norm(r_e[i] - r_e[j])  # FIXME to slow
                if r <= self.u_cutoff[0]:
                    u_set = int(i >= neu) + int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, u_set] * r ** k
                    res += poly * (r - self.u_cutoff[0]) ** self.trunc
        return res

    def chi_term(self, r_e, neu, r_I):
        """Jastrow chi-term
        :param r_e: electrons coordinates
        :param neu: number of up electrons
        :param r_I: nucleus coordinates
        :return:
        """
        res = 0.0
        if not self.chi_cutoff.any():
            return res

        for i in range(r_I.shape[0]):
            p = self.chi_parameters[i]
            for j in range(r_e.shape[0]):
                r = np.linalg.norm(r_e[j] - r_I[i])  # FIXME to slow
                if r <= self.chi_cutoff[i]:
                    chi_set = int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, chi_set] * r ** k
                    res += poly * (r - self.chi_cutoff[i]) ** self.trunc
        return res

    def f_term(self, r_e, neu, r_I):
        """Jastrow f-term
        :param r_e: electrons coordinates
        :param ned: number of up electrons
        :param r_I: nucleus coordinates
        :return:
        """
        res = 0.0
        if not self.f_cutoff.any():
            return res

        for i in range(r_I.shape[0]):
            p = self.f_parameters[i]
            for j in range(r_e.shape[0] - 1):
                for k in range(j+1, r_e.shape[0]):
                    r_ee = np.linalg.norm(r_e[j] - r_e[k])  # FIXME to slow
                    r_e1I = np.linalg.norm(r_e[j] - r_I[i])  # FIXME to slow
                    r_e2I = np.linalg.norm(r_e[k] - r_I[i])  # FIXME to slow
                    if r_e1I <= self.f_cutoff[i] and r_e2I <= self.f_cutoff[i]:
                        f_set = int(j >= neu) + int(k >= neu)
                        poly = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly += p[l, m, n, f_set] * r_e1I ** l * r_e2I ** m * r_ee ** n
                        res += poly * (r_e1I - self.f_cutoff[i]) ** self.trunc * (r_e2I - self.f_cutoff[i]) ** self.trunc
        return res

    def u_term_gradient(self, r_e, neu):
        """Jastrow u-term gradient
        :param r_e: electrons coordinates
        :param neu: number of up electrons
        :return:
        """
        res = np.zeros(r_e.shape)

        if not self.u_cutoff.any():
            return res

        p = self.u_parameters[0]
        for i in range(r_e.shape[0] - 1):
            for j in range(i + 1, r_e.shape[0]):
                r_vec = r_e[i] - r_e[j]  # FIXME to slow
                r = np.linalg.norm(r_vec)
                if r <= self.u_cutoff[0]:
                    u_set = int(i >= neu) + int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, u_set] * r ** k

                    poly_diff = 0.0
                    for k in range(1, p.shape[0]):
                        poly_diff += k * p[k, u_set] * r ** (k-1)

                    gradient = (self.trunc * (r-self.u_cutoff[0]) ** (self.trunc-1) * poly + (r-self.u_cutoff[0]) ** self.trunc * poly_diff) / r
                    res[i, :] += r_vec * gradient
                    res[j, :] -= r_vec * gradient
        return res

    def chi_term_gradient(self, r_e, neu, r_I):
        """Jastrow chi-term gradient
        :param r_e: electrons coordinates
        :param neu: number of up electrons
        :param r_I: nucleus coordinates
        :return:
        """
        res = np.zeros(r_e.shape)

        if not self.chi_cutoff.any():
            return res

        for i in range(r_I.shape[0]):
            p = self.chi_parameters[i]
            for j in range(r_e.shape[0]):
                r_vec = r_e[j] - r_I[i]  # FIXME to slow
                r = np.linalg.norm(r_vec)
                if r <= self.chi_cutoff[i]:
                    chi_set = int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, chi_set] * r ** k

                    poly_diff = 0.0
                    for k in range(1, p.shape[0]):
                        poly_diff += k * p[k, chi_set] * r ** (k-1)

                    gradient = (self.trunc * (r-self.chi_cutoff[i]) ** (self.trunc-1) * poly + (r-self.chi_cutoff[i]) ** self.trunc * poly_diff) / r
                    res[j, :] += r_vec * gradient
        return res

    def f_term_gradient(self, r_e, neu, r_I):
        """Jastrow f-term gradient
        :param r_e: electrons coordinates
        :param neu: number of up electrons
        :param r_I: nucleus coordinates
        :return:
        """
        res = np.zeros(r_e.shape)

        if not self.f_cutoff.any():
            return res

        for i in range(r_I.shape[0]):
            p = self.f_parameters[i]
            for j in range(r_e.shape[0] - 1):
                for k in range(j+1, r_e.shape[0]):
                    r_e1I_vec = r_e[j] - r_I[i]  # FIXME to slow
                    r_e2I_vec = r_e[k] - r_I[i]  # FIXME to slow
                    r_ee_vec = r_e[j] - r_e[k]  # FIXME to slow
                    r_e1I = np.linalg.norm(r_e1I_vec)
                    r_e2I = np.linalg.norm(r_e2I_vec)
                    r_ee = np.linalg.norm(r_ee_vec)
                    if r_e1I <= self.f_cutoff[i] and r_e2I <= self.f_cutoff[i]:
                        f_set = int(j >= neu) + int(k >= neu)
                        poly = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly += p[l, m, n, f_set] * r_e1I ** l * r_e2I ** m * r_ee ** n

                        poly_diff_e1I = 0.0
                        for l in range(1, p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e1I += p[l, m, n, f_set] * l * r_e1I ** (l-1) * r_e2I ** m * r_ee ** n

                        poly_diff_e2I = 0.0
                        for l in range(p.shape[0]):
                            for m in range(1, p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e2I += p[l, m, n, f_set] * r_e1I ** l * m * r_e2I ** (m-1) * r_ee ** n

                        poly_diff_ee = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(1, p.shape[2]):
                                    poly_diff_ee += p[l, m, n, f_set] * r_e1I ** l * r_e2I ** m * n * r_ee ** (n-1)

                        gradient = (
                            self.trunc * (r_e1I - self.f_cutoff[i]) ** (self.trunc-1) * (r_e2I - self.f_cutoff[i]) ** self.trunc * poly +
                            (r_e1I - self.f_cutoff[i]) ** self.trunc * (r_e2I - self.f_cutoff[i]) ** self.trunc * poly_diff_e1I
                        ) / r_e1I
                        res[j, :] += r_e1I_vec * gradient

                        gradient = (
                            (r_e1I - self.f_cutoff[i]) ** self.trunc * self.trunc * (r_e2I - self.f_cutoff[i]) ** (self.trunc-1) * poly +
                            (r_e1I - self.f_cutoff[i]) ** self.trunc * (r_e2I - self.f_cutoff[i]) ** self.trunc * poly_diff_e2I
                        ) / r_e2I
                        res[k, :] += r_e2I_vec * gradient

                        gradient = (r_e1I - self.f_cutoff[i]) ** self.trunc * (r_e2I - self.f_cutoff[i]) ** self.trunc * poly_diff_ee / r_ee
                        res[j, :] += r_ee_vec * gradient
                        res[k, :] -= r_ee_vec * gradient
        return res

    def u_term_laplacian(self, r_e, neu):
        """Jastrow u-term laplacian
        :param r_e: electrons coordinates
        :param neu: number of up electrons
        :return:
        """
        res = 0.0
        if not self.u_cutoff.any():
            return res

        p = self.u_parameters[0]
        for i in range(r_e.shape[0] - 1):
            for j in range(i + 1, r_e.shape[0]):
                r = np.linalg.norm(r_e[i] - r_e[j])  # FIXME to slow
                if r <= self.u_cutoff[0]:
                    u_set = int(i >= neu) + int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, u_set] * r ** k

                    poly_diff = 0.0
                    for k in range(1, p.shape[0]):
                        poly_diff += k * p[k, u_set] * r ** (k-1)

                    poly_diff_2 = 0.0
                    for k in range(2, p.shape[0]):
                        poly_diff_2 += k * (k-1) * p[k, u_set] * r ** (k-2)

                    res += (
                        self.trunc*(self.trunc - 1)*(r-self.u_cutoff[0])**(self.trunc - 2) * poly +
                        2 * self.trunc*(r-self.u_cutoff[0])**(self.trunc - 1) * poly_diff + (r-self.u_cutoff[0])**self.trunc * poly_diff_2 +
                        2 * (self.trunc * (r-self.u_cutoff[0])**(self.trunc-1) * poly + (r-self.u_cutoff[0])**self.trunc * poly_diff) / r
                    )
        return 2 * res

    def chi_term_laplacian(self, r_e, neu, r_I):
        """Jastrow chi-term laplacian
        :param r_e: electrons coordinates
        :param neu: number of up electrons
        :param r_I: nucleus coordinates
        :return:
        """
        res = 0.0
        if not self.chi_cutoff.any():
            return res

        for i in range(r_I.shape[0]):
            p = self.chi_parameters[i]
            for j in range(r_e.shape[0]):
                r = np.linalg.norm(r_e[j] - r_I[i])  # FIXME to slow
                if r <= self.chi_cutoff[i]:
                    chi_set = int(j >= neu)
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[k, chi_set] * r ** k

                    poly_diff = 0.0
                    for k in range(1, p.shape[0]):
                        poly_diff += k * p[k, chi_set] * r ** (k-1)

                    poly_diff_2 = 0.0
                    for k in range(2, p.shape[0]):
                        poly_diff_2 += k * (k-1) * p[k, chi_set] * r ** (k-2)

                    res += (
                        self.trunc*(self.trunc - 1)*(r-self.chi_cutoff[i])**(self.trunc - 2) * poly +
                        2 * self.trunc*(r-self.chi_cutoff[i])**(self.trunc - 1) * poly_diff + (r-self.chi_cutoff[i])**self.trunc * poly_diff_2 +
                        2 * (self.trunc * (r-self.chi_cutoff[i])**(self.trunc-1) * poly + (r-self.chi_cutoff[i])**self.trunc * poly_diff) / r
                    )
        return res

    def f_term_laplacian(self, r_e, neu, r_I):
        """Jastrow f-term laplacian
        f-term is a product of two spherically symmetric functions f(r_eI) and g(r_ee) so using
            ∇²(f*g) = ∇²(f)*g + 2*∇(f)*∇(g) + f*∇²(g)
        then Laplace operator of spherically symmetric function (in 3-D space) is
            ∇²(f) = d²f/dr² + 2/r * df/dr
        :param r_e: electrons coordinates
        :param neu: number of up electrons
        :param r_I: nucleus coordinates
        :return:
        """
        res = 0.0
        if not self.f_cutoff.any():
            return res

        for i in range(r_I.shape[0]):
            p = self.f_parameters[i]
            for j in range(r_e.shape[0] - 1):
                for k in range(j + 1, r_e.shape[0]):
                    r_e1I_vec = r_e[j] - r_I[i]  # FIXME to slow
                    r_e2I_vec = r_e[k] - r_I[i]  # FIXME to slow
                    r_ee_vec = r_e[j] - r_e[k]  # FIXME to slow
                    r_e1I = np.linalg.norm(r_e1I_vec)
                    r_e2I = np.linalg.norm(r_e2I_vec)
                    r_ee = np.linalg.norm(r_ee_vec)
                    if r_e1I <= self.f_cutoff[i] and r_e2I <= self.f_cutoff[i]:
                        f_set = int(j >= neu) + int(k >= neu)
                        poly = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly += p[l, m, n, f_set] * r_e1I ** l * r_e2I ** m * r_ee ** n

                        poly_diff_e1I = 0.0
                        for l in range(1, p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e1I += p[l, m, n, f_set] * l * r_e1I ** (l-1) * r_e2I ** m * r_ee ** n

                        poly_diff_e2I = 0.0
                        for l in range(p.shape[0]):
                            for m in range(1, p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e2I += p[l, m, n, f_set] * r_e1I ** l * m * r_e2I ** (m-1) * r_ee ** n

                        poly_diff_ee = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(1, p.shape[2]):
                                    poly_diff_ee += p[l, m, n, f_set] * r_e1I ** l * r_e2I ** m * n * r_ee ** (n-1)

                        poly_diff_e1I_2 = 0.0
                        for l in range(2, p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e1I_2 += p[l, m, n, f_set] * l * (l-1) * r_e1I ** (l-2) * r_e2I ** m * r_ee ** n

                        poly_diff_e2I_2 = 0.0
                        for l in range(p.shape[0]):
                            for m in range(2, p.shape[1]):
                                for n in range(p.shape[2]):
                                    poly_diff_e2I_2 += p[l, m, n, f_set] * r_e1I ** l * m * (m-1) * r_e2I ** (m-2) * r_ee ** n

                        poly_diff_ee_2 = 0.0
                        for l in range(p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(2, p.shape[2]):
                                    poly_diff_ee_2 += p[l, m, n, f_set] * r_e1I ** l * r_e2I ** m * n * (n-1) * r_ee ** (n-2)

                        poly_diff_e1I_ee = 0.0
                        for l in range(1, p.shape[0]):
                            for m in range(p.shape[1]):
                                for n in range(1, p.shape[2]):
                                    poly_diff_e1I_ee += p[l, m, n, f_set] * l * r_e1I ** (l-1) * r_e2I ** m * n * r_ee ** (n-1)

                        poly_diff_e2I_ee = 0.0
                        for l in range(p.shape[0]):
                            for m in range(1, p.shape[1]):
                                for n in range(1, p.shape[2]):
                                    poly_diff_e2I_ee += p[l, m, n, f_set] * r_e1I ** l * m * r_e2I ** (m-1) * n * r_ee ** (n-1)

                        C = self.trunc
                        L = self.f_cutoff
                        gradient = (
                            (C * (r_e1I - L[i]) ** (C-1) * (r_e2I - L[i]) ** C * poly + (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_e1I) / r_e1I +
                            ((r_e1I - L[i]) ** C * C * (r_e2I - L[i]) ** (C-1) * poly + (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_e2I) / r_e2I +
                            2 * (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_ee / r_ee
                        )

                        laplacian = (
                                C * (C - 1) * (r_e1I - L[i]) ** (C - 2) * (r_e2I - L[i]) ** C * poly +
                                (r_e1I - L[i]) ** C * C * (C - 1) * (r_e2I - L[i]) ** (C - 2) * poly +
                                (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * (poly_diff_e1I_2 + poly_diff_e2I_2 + 2 * poly_diff_ee_2) +
                                2 * C * (r_e1I - L[i]) ** (C - 1) * (r_e2I - L[i]) ** C * poly_diff_e1I +
                                2 * (r_e1I - L[i]) ** C * C * (r_e2I - L[i]) ** (C - 1) * poly_diff_e2I
                        )

                        dot_product = (
                                np.sum(r_e1I_vec * r_ee_vec) * (
                                        C * (r_e1I - L[i]) ** (C-1) * (r_e2I - L[i]) ** C * poly_diff_ee +
                                        (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_e1I_ee
                                ) / r_e1I / r_ee -
                                np.sum(r_e2I_vec * r_ee_vec) * (
                                        (r_e1I - L[i]) ** C * C * (r_e2I - L[i]) ** (C-1) * poly_diff_ee +
                                        (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_e2I_ee
                                ) / r_e2I / r_ee
                        )

                        res += laplacian + 2 * gradient + 2 * dot_product
        return res

    def value(self, r_e, neu, r_I):
        """Jastrow
        :param r_e: electrons coordinates
        :param neu: number of up electrons
        :param r_I: nucleus coordinates
        :return:
        """
        return self.u_term(r_e, neu) + self.chi_term(r_e, neu, r_I) + self.f_term(r_e, neu, r_I)

    def numerical_gradient(self, r_e, neu, r_I):
        delta = 0.00001

        res = np.zeros(r_e.shape)

        for i in range(r_e.shape[0]):
            for j in range(r_e.shape[1]):
                r_e[i, j] -= delta
                res[i, j] -= self.value(r_e, neu, r_I)
                r_e[i, j] += 2 * delta
                res[i, j] += self.value(r_e, neu, r_I)
                r_e[i, j] -= delta

        return res / delta / 2

    def numerical_laplacian(self, r_e, neu, r_I):
        delta = 0.00001

        res = -2 * r_e.size * self.value(r_e, neu, r_I)
        for i in range(r_e.shape[0]):
            for j in range(r_e.shape[1]):
                r_e[i, j] -= delta
                res += self.value(r_e, neu, r_I)
                r_e[i, j] += 2 * delta
                res += self.value(r_e, neu, r_I)
                r_e[i, j] -= delta

        return res / delta / delta

    def gradient(self, r_e, neu, r_I):
        return self.u_term_gradient(r_e, neu) + self.chi_term_gradient(r_e, neu, r_I) + self.f_term_gradient(r_e, neu, r_I)

    def laplacian(self, r_e, neu, r_I):
        return self.u_term_laplacian(r_e, neu) + self.chi_term_laplacian(r_e, neu, r_I) + self.f_term_laplacian(r_e, neu, r_I)


if __name__ == '__main__':
    """
    """

    term = 'chi'

    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/casl/8_8_44/10000/'
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
                y_grid[i] = jastrow.u_term(r_e, 2-spin_dep)
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
                    y_grid[i] = jastrow.chi_term(r_e, 1-spin_dep, casino.wfn.atom_positions[sl])
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
                        z_grid[i, j] = jastrow.f_term(r_e, 2-spin_dep, casino.wfn.atoms[sl])
                axis.plot_wireframe(x_grid, y_grid, z_grid, label=f'atom {atom} ' + ['uu', 'ud', 'dd'][spin_dep])
        axis.set_xlabel('r_e1N (au)')
        axis.set_ylabel('r_e2N (au)')
        axis.set_zlabel('polynomial part')
        plt.title('JASTROW f-term')

    plt.grid(True)
    plt.legend()
    plt.show()
