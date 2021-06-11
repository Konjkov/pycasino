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

from readers.casino import Casino
from overload import subtract_outer
from logger import logging

logger = logging.getLogger('vmc')


labels_type = nb.int64[:]
eta_parameters_type = nb.float64[:, :]
mu_parameters_type = nb.float64[:, :]
phi_parameters_type = nb.float64[:, :, :, :]
theta_parameters_type = nb.float64[:, :, :, :]


spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('trunc', nb.int64),
    ('eta_parameters', eta_parameters_type),
    ('mu_parameters', nb.types.ListType(mu_parameters_type)),
    ('phi_parameters', nb.types.ListType(phi_parameters_type)),
    ('theta_parameters', nb.types.ListType(theta_parameters_type)),
    ('eta_cutoff', nb.float64[:]),
    ('mu_cutoff', nb.float64[:]),
    ('phi_cutoff', nb.float64[:]),
    ('mu_labels', nb.types.ListType(labels_type)),
    ('phi_labels', nb.types.ListType(labels_type)),
    ('max_ee_order', nb.int64),
    ('max_en_order', nb.int64),
    ('phi_irrotational', nb.boolean[:]),
    ('ae_cutoff', nb.float64[:]),
]


@nb.experimental.jitclass(spec)
class Backflow:

    def __init__(
        self, neu, ned, trunc, eta_parameters, eta_cutoff, mu_parameters, mu_cutoff, mu_labels, phi_parameters,
        theta_parameters, phi_cutoff, phi_labels, phi_irrotational, ae_cutoff
    ):
        self.neu = neu
        self.ned = ned
        self.trunc = trunc
        self.eta_parameters = eta_parameters
        self.mu_parameters = nb.typed.List.empty_list(mu_parameters_type)
        [self.mu_parameters.append(p) for p in mu_parameters]
        self.phi_parameters = nb.typed.List.empty_list(phi_parameters_type)
        [self.phi_parameters.append(p) for p in phi_parameters]
        self.theta_parameters = nb.typed.List.empty_list(theta_parameters_type)
        [self.theta_parameters.append(p) for p in theta_parameters]
        self.mu_labels = nb.typed.List.empty_list(labels_type)
        [self.mu_labels.append(p) for p in mu_labels]
        self.phi_labels = nb.typed.List.empty_list(labels_type)
        [self.phi_labels.append(p) for p in phi_labels]
        self.max_ee_order = max((
            self.eta_parameters.shape[0],
            max([p.shape[2] for p in self.phi_parameters]) if self.phi_parameters else 0,
        ))
        self.max_en_order = max((
            max([p.shape[0] for p in self.mu_parameters]) if self.mu_parameters else 0,
            max([p.shape[0] for p in self.phi_parameters]) if self.phi_parameters else 0,
            2
        ))
        self.eta_cutoff = eta_cutoff
        self.mu_cutoff = mu_cutoff
        self.phi_cutoff = phi_cutoff
        self.ae_cutoff = ae_cutoff

    def ee_powers(self, e_vectors):
        """Powers of e-e distances
        :param e_vectors: e-e vectors - array(nelec, nelec, 3)
        :return:
        """
        res = np.ones((e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(1, e_vectors.shape[0]):
            for j in range(i):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(1, self.max_ee_order):
                    res[i, j, k] = res[j, i, k] = r_ee ** k
        return res

    def en_powers(self, n_vectors):
        """Powers of e-n distances
        :param n_vectors: e-n vectors
        :return:
        """
        res = np.ones((n_vectors.shape[0], n_vectors.shape[1], self.max_en_order))
        for i in range(n_vectors.shape[0]):
            for j in range(n_vectors.shape[1]):
                r_eI = np.linalg.norm(n_vectors[i, j])
                for k in range(1, self.max_en_order):
                    res[i, j, k] = r_eI ** k
        return res

    def ae_multiplier(self, n_vectors, n_powers):
        """Zeroing the backflow displacement at AE atoms."""
        res = np.ones((self.neu + self.ned, 3))
        for i in range(n_vectors.shape[0]):
            Lg = self.ae_cutoff[i]
            for j in range(self.neu + self.ned):
                r = n_powers[i, j, 1]
                if r < Lg:
                    res[j] = (r/Lg)**2 * (6 - 8 * (r/Lg) + 3 * (r/Lg)**2)
        return res

    def ae_multiplier_gradient(self, n_vectors, n_powers):
        """Zeroing the backflow displacement at AE atoms.
        Gradient of spherically symmetric function (in 3-D space) is:
            ∇(f) = df/dr * r_vec/r
        """
        res = np.zeros((self.neu + self.ned, 3, self.neu + self.ned, 3))
        for i in range(n_vectors.shape[0]):
            Lg = self.ae_cutoff[i]
            for j in range(self.neu + self.ned):
                r_vec = n_vectors[i, j]
                r = n_powers[i, j, 1]
                if r < Lg:
                    res[j, :, j, :] = 12*r_vec/Lg**2 * (1 - r/Lg)**2
        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    def ae_multiplier_laplacian(self, n_vectors, n_powers):
        """Zeroing the backflow displacement at AE atoms.
        Laplace operator of spherically symmetric function (in 3-D space) is:
            ∇²(f) = d²f/dr² + 2/r * df/dr
        """
        res = np.zeros((self.neu + self.ned, 3))
        for i in range(n_vectors.shape[0]):
            Lg = self.ae_cutoff[i]
            for j in range(self.neu + self.ned):
                r = n_powers[i, j, 1]
                if r < Lg:
                    res[j] = 12/Lg**2 * (3 - 8 * (r/Lg) + 5 * (r/Lg)**2)
        return res.ravel()

    def eta_term(self, e_vectors, e_powers):
        """
        :param e_vectors: e-e vectors
        :param e_powers: powers of e-e distances
        :return: displacements of electrons - array(nelec, 3)
        """
        res = np.zeros((self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res

        C = self.trunc
        parameters = self.eta_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_vec = e_vectors[i, j]
                r = e_powers[i, j, 1]
                eta_set = (int(i >= self.neu) + int(j >= self.neu)) % parameters.shape[1]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    poly = 0
                    for k in range(parameters.shape[0]):
                        poly += parameters[k, eta_set] * e_powers[i, j, k]
                    bf = (1 - r/L) ** C * poly * r_vec
                    res[i] += bf
                    res[j] -= bf
        return res

    def mu_term(self, n_vectors, n_powers):
        """
        :param n_vectors: e-n vectors
        :param n_powers: powers of e-n distances
        :return: displacements of electrons - array(nelec, 3)
        """
        res = np.zeros((self.neu + self.ned, 3))
        if not self.mu_cutoff.any():
            return res

        C = self.trunc
        for parameters, L, mu_labels in zip(self.mu_parameters, self.mu_cutoff, self.mu_labels):
            for i in mu_labels:
                for j in range(self.neu + self.ned):
                    r_vec = n_vectors[i, j]
                    r = n_powers[i, j, 1]
                    if r < L:
                        mu_set = int(j >= self.neu) % parameters.shape[1]
                        poly = 0.0
                        for k in range(parameters.shape[0]):
                            poly += parameters[k, mu_set] * n_powers[i, j, k]
                        res[j] += poly * (1 - r/L) ** C * r_vec
        return res

    def phi_term(self, e_powers, n_powers, e_vectors, n_vectors):
        """
        :param e_vectors:
        :param n_vectors:
        :param e_powers:
        :param n_powers:
        :return: displacements of electrons - array(nelec, 3)
        """
        res = np.zeros((self.neu + self.ned, 3))
        if not self.phi_cutoff.any():
            return res

        C = self.trunc
        for phi_parameters, theta_parameters, L, phi_labels in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_labels):
            for i in phi_labels:
                for j1 in range(self.neu + self.ned):
                    for j2 in range(self.neu + self.ned):
                        if j1 == j2:
                            continue
                        r_e1I_vec = n_vectors[i, j1]
                        r_ee_vec = e_vectors[j1, j2]
                        r_e1I = n_powers[i, j1, 1]
                        r_e2I = n_powers[i, j2, 1]
                        if r_e1I < L and r_e2I < L:
                            phi_set = (int(j1 >= self.neu) + int(j2 >= self.neu)) % phi_parameters.shape[3]
                            phi_poly = theta_poly = 0.0
                            for k in range(phi_parameters.shape[0]):
                                for l in range(phi_parameters.shape[1]):
                                    for m in range(phi_parameters.shape[2]):
                                        poly = n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m]
                                        phi_poly += phi_parameters[k, l, m, phi_set] * poly
                                        theta_poly += theta_parameters[k, l, m, phi_set] * poly
                            res[j1] += (1-r_e1I/L) ** C * (1-r_e2I/L) ** C * (phi_poly * r_ee_vec + theta_poly * r_e1I_vec)

        return res

    def eta_term_gradient(self, e_powers, e_vectors):
        """
        :param e_powers:
        :param e_vectors:
        Gradient of spherically symmetric function (in 3-D space) is df/dr * (x, y, z)
        :return: partial derivatives of displacements of electrons - array(nelec * 3, nelec * 3)
        """
        res = np.zeros((self.neu + self.ned, 3, self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

        C = self.trunc
        parameters = self.eta_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_vec = e_vectors[i, j]
                r = e_powers[i, j, 1]
                eta_set = (int(i >= self.neu) + int(j >= self.neu)) % parameters.shape[1]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    poly = poly_diff = 0
                    for k in range(parameters.shape[0]):
                        p = parameters[k, eta_set]
                        poly += p * e_powers[i, j, k]
                        if k > 0:
                            poly_diff += p * k * e_powers[i, j, k - 1]

                    bf = (1 - r/L)**C * (
                        (poly_diff - C/(L - r)*poly) * np.outer(r_vec, r_vec)/r + poly * np.eye(3)
                    )
                    res[i, :, i, :] += bf
                    res[i, :, j, :] -= bf
                    res[j, :, i, :] -= bf
                    res[j, :, j, :] += bf

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    def mu_term_gradient(self, n_powers, n_vectors):
        """
        :param n_powers:
        :param n_vectors:
        :return: partial derivatives of displacements of electrons - array(nelec * 3, nelec * 3)
        """
        res = np.zeros((self.neu + self.ned, 3, self.neu + self.ned, 3))
        if not self.mu_cutoff.any():
            return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

        C = self.trunc
        for parameters, L, mu_labels in zip(self.mu_parameters, self.mu_cutoff, self.mu_labels):
            for i in mu_labels:
                for j in range(self.neu + self.ned):
                    r_vec = n_vectors[i, j]
                    r = n_powers[i, j, 1]
                    if r < L:
                        mu_set = int(j >= self.neu) % parameters.shape[1]
                        poly = poly_diff = 0.0
                        for k in range(parameters.shape[0]):
                            p = parameters[k, mu_set]
                            poly += p * n_powers[i, j, k]
                            if k > 0:
                                poly_diff += k * p * n_powers[i, j, k-1]

                        res[j, :, j, :] += (1 - r/L)**C * (
                            (poly_diff - C/(L - r)*poly) * np.outer(r_vec, r_vec)/r + poly * np.eye(3)
                        )

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    def phi_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors):
        """
        :param e_powers:
        :param e_vectors:
        :return: partial derivatives of displacements of electrons - array(nelec * 3, nelec * 3)
        """
        res = np.zeros((self.neu + self.ned, 3, self.neu + self.ned, 3))
        if not self.mu_cutoff.any():
            return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

        C = self.trunc
        for phi_parameters, theta_parameters, L, phi_labels in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_labels):
            for i in phi_labels:
                for j1 in range(self.neu + self.ned):
                    for j2 in range(self.neu + self.ned):
                        if j1 == j2:
                            continue
                        r_e1I_vec = n_vectors[i, j1]
                        r_e2I_vec = n_vectors[i, j2]
                        r_ee_vec = e_vectors[j1, j2]
                        r_e1I = n_powers[i, j1, 1]
                        r_e2I = n_powers[i, j2, 1]
                        r_ee = e_powers[j1, j2, 1]
                        if r_e1I < L and r_e2I < L:
                            phi_set = (int(j1 >= self.neu) + int(j2 >= self.neu)) % phi_parameters.shape[3]
                            phi_poly = phi_poly_diff_e1I = phi_poly_diff_e2I = phi_poly_diff_ee = 0.0
                            theta_poly = theta_poly_diff_e1I = theta_poly_diff_e2I = theta_poly_diff_ee = 0.0
                            for k in range(phi_parameters.shape[0]):
                                for l in range(phi_parameters.shape[1]):
                                    for m in range(phi_parameters.shape[2]):
                                        phi_p = phi_parameters[k, l, m, phi_set]
                                        theta_p = theta_parameters[k, l, m, phi_set]
                                        phi_poly += n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m] * phi_p
                                        theta_poly += n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m] * theta_p
                                        if k > 0:
                                            poly_diff_e1I = k * n_powers[i, j1, k-1] * n_powers[i, j2, l] * e_powers[j1, j2, m]
                                            phi_poly_diff_e1I += poly_diff_e1I * phi_p
                                            theta_poly_diff_e1I += poly_diff_e1I * theta_p
                                        if l > 0:
                                            poly_diff_e2I = l * n_powers[i, j1, k] * n_powers[i, j2, l-1] * e_powers[j1, j2, m]
                                            phi_poly_diff_e2I += poly_diff_e2I * phi_p
                                            theta_poly_diff_e2I += poly_diff_e2I * theta_p
                                        if m > 0:
                                            poly_diff_ee = m * n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m-1]
                                            phi_poly_diff_ee += poly_diff_ee * phi_p
                                            theta_poly_diff_ee += poly_diff_ee * theta_p

                            res[j1, :, j1, :] += (1-r_e1I/L) ** C * (1-r_e2I/L) ** C * (
                                (phi_poly_diff_e1I - C/(L - r_e1I)*phi_poly) * np.outer(r_ee_vec, r_e1I_vec)/r_e1I +
                                phi_poly_diff_ee * np.outer(r_ee_vec, r_ee_vec) / r_ee +
                                phi_poly * np.eye(3) +
                                (theta_poly_diff_e1I - C / (L - r_e1I) * theta_poly) * np.outer(r_e1I_vec, r_e1I_vec) / r_e1I +
                                theta_poly_diff_ee * np.outer(r_e1I_vec, r_ee_vec) / r_ee +
                                theta_poly * np.eye(3)
                            )
                            res[j1, :, j2, :] += (1-r_e1I/L) ** C * (1-r_e2I/L) ** C * (
                                (phi_poly_diff_e2I - C/(L - r_e2I)*phi_poly) * np.outer(r_ee_vec, r_e2I_vec)/r_e2I -
                                phi_poly_diff_ee * np.outer(r_ee_vec, r_ee_vec) / r_ee -
                                phi_poly * np.eye(3) +
                                (theta_poly_diff_e2I - C / (L - r_e2I) * theta_poly) * np.outer(r_e1I_vec, r_e2I_vec) / r_e2I -
                                theta_poly_diff_ee * np.outer(r_e1I_vec, r_ee_vec) / r_ee
                            )

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    def eta_term_laplacian(self, e_powers, e_vectors):
        """
        :param e_powers:
        :param e_vectors:
        Laplace operator of spherically symmetric function (in 3-D space) is
            ∇²(f) = d²f/dr² + 2/r * df/dr
        :return: vector laplacian - array(nelec * 3)
        """
        res = np.zeros((self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res.ravel()

        C = self.trunc
        parameters = self.eta_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_vec = e_vectors[i, j]
                r = e_powers[i, j, 1]
                eta_set = (int(i >= self.neu) + int(j >= self.neu)) % parameters.shape[1]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    poly = poly_diff = poly_diff_2 = 0
                    for k in range(parameters.shape[0]):
                        p = parameters[k, eta_set]
                        poly += p * e_powers[i, j, k]
                        if k > 0:
                            poly_diff += k * p * e_powers[i, j, k-1]
                        if k > 1:
                            poly_diff_2 += k * (k - 1) * p * e_powers[i, j, k-2]

                    bf = 2 * (1 - r/L)**C * (
                        4*(poly_diff - C/(L - r) * poly) +
                        r*(C*(C - 1)/(L - r)**2*poly - 2*C/(L - r)*poly_diff + poly_diff_2)
                    ) * r_vec/r
                    res[i] += bf
                    res[j] -= bf

        return res.ravel()

    def mu_term_laplacian(self, n_powers, n_vectors):
        """
        :param e_powers:
        :param e_vectors:
        Laplace operator of spherically symmetric function (in 3-D space) is
            ∇²(f) = d²f/dr² + 2/r * df/dr
        :return: vector laplacian - array(nelec * 3)
        """
        res = np.zeros((self.neu + self.ned, 3))
        if not self.mu_cutoff.any():
            return res.ravel()

        C = self.trunc
        for parameters, L, mu_labels in zip(self.mu_parameters, self.mu_cutoff, self.mu_labels):
            for i in mu_labels:
                for j in range(self.neu + self.ned):
                    r_vec = n_vectors[i, j]
                    r = n_powers[i, j, 1]
                    if r < L:
                        mu_set = int(j >= self.neu) % parameters.shape[1]
                        poly = poly_diff = poly_diff_2 = 0.0
                        for k in range(parameters.shape[0]):
                            p = parameters[k, mu_set]
                            poly += p * n_powers[i, j, k]
                            if k > 0:
                                poly_diff += k * p * n_powers[i, j, k-1]
                            if k > 1:
                                poly_diff_2 += k * (k-1) * p * n_powers[i, j, k-2]

                        res[j] += (1 - r/L)**C * (
                            4*(poly_diff - C/(L - r) * poly) +
                            r*(C*(C - 1)/(L - r)**2*poly - 2*C/(L - r)*poly_diff + poly_diff_2)
                        ) * r_vec/r

        return res.ravel()

    def phi_term_laplacian(self, e_powers, n_powers, e_vectors, n_vectors):
        """
        :param e_powers:
        :param e_vectors:
        phi-term is a product of two spherically symmetric functions f(r_eI) and g(r_ee) so using
            ∇²(f*g) = ∇²(f)*g + 2*∇(f)*∇(g) + f*∇²(g)
        Laplace operator of spherically symmetric function (in 3-D space) is
            ∇²(f) = d²f/dr² + 2/r * df/dr
        :return: vector laplacian - array(nelec * 3)
        """
        res = np.zeros((self.neu + self.ned, 3))
        if not self.phi_cutoff.any():
            return res.ravel()

        C = self.trunc
        for phi_parameters, theta_parameters, L, phi_labels in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_labels):
            for i in phi_labels:
                for j1 in range(self.neu + self.ned):
                    for j2 in range(self.neu + self.ned):
                        if j1 == j2:
                            continue
                        r_e1I_vec = n_vectors[i, j1]
                        r_e2I_vec = n_vectors[i, j2]
                        r_ee_vec = e_vectors[j1, j2]
                        r_e1I = n_powers[i, j1, 1]
                        r_e2I = n_powers[i, j2, 1]
                        r_ee = e_powers[j1, j2, 1]
                        if r_e1I < L and r_e2I < L:
                            phi_set = (int(j1 >= self.neu) + int(j2 >= self.neu)) % phi_parameters.shape[3]
                            phi_poly = phi_poly_diff_e1I = phi_poly_diff_e2I = phi_poly_diff_ee = 0.0
                            phi_poly_diff_e1I_2 = phi_poly_diff_e2I_2 = phi_poly_diff_ee_2 = 0.0
                            phi_poly_diff_e1I_ee = phi_poly_diff_e2I_ee = 0.0
                            theta_poly = theta_poly_diff_e1I = theta_poly_diff_e2I = theta_poly_diff_ee = 0.0
                            theta_poly_diff_e1I_2 = theta_poly_diff_e2I_2 = theta_poly_diff_ee_2 = 0.0
                            theta_poly_diff_e1I_ee = theta_poly_diff_e2I_ee = 0.0
                            for k in range(phi_parameters.shape[0]):
                                for l in range(phi_parameters.shape[1]):
                                    for m in range(phi_parameters.shape[2]):
                                        phi_p = phi_parameters[k, l, m, phi_set]
                                        theta_p = theta_parameters[k, l, m, phi_set]
                                        poly = n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m]
                                        phi_poly += poly * phi_p
                                        theta_poly += poly * theta_p
                                        if k > 0:
                                            poly_diff_e1I = k * n_powers[i, j1, k-1] * n_powers[i, j2, l] * e_powers[j1, j2, m]
                                            phi_poly_diff_e1I += poly_diff_e1I * phi_p
                                            theta_poly_diff_e1I += poly_diff_e1I * theta_p
                                        if l > 0:
                                            poly_diff_e2I = l * n_powers[i, j1, k] * n_powers[i, j2, l-1] * e_powers[j1, j2, m]
                                            phi_poly_diff_e2I += poly_diff_e2I * phi_p
                                            theta_poly_diff_e2I += poly_diff_e2I * theta_p
                                        if m > 0:
                                            poly_diff_ee = m * n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m-1]
                                            phi_poly_diff_ee += poly_diff_ee * phi_p
                                            theta_poly_diff_ee += poly_diff_ee * theta_p
                                        if k > 1:
                                            poly_diff_e1I_2 = k * (k-1) * n_powers[i, j1, k-2] * n_powers[i, j2, l] * e_powers[j1, j2, m]
                                            phi_poly_diff_e1I_2 += poly_diff_e1I_2 * phi_p
                                            theta_poly_diff_e1I_2 += poly_diff_e1I_2 * theta_p
                                        if l > 1:
                                            poly_diff_e2I_2 = l * (l-1) * n_powers[i, j1, k] * n_powers[i, j2, l-2] * e_powers[j1, j2, m]
                                            phi_poly_diff_e2I_2 += poly_diff_e2I_2 * phi_p
                                            theta_poly_diff_e2I_2 += poly_diff_e2I_2 * theta_p
                                        if m > 1:
                                            poly_diff_ee_2 = m * (m-1) * n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m-2]
                                            phi_poly_diff_ee_2 += poly_diff_ee_2 * phi_p
                                            theta_poly_diff_ee_2 += poly_diff_ee_2 * theta_p
                                        if k > 0 and m > 0:
                                            poly_diff_e1I_ee = k * m * n_powers[i, j1, k-1] * n_powers[i, j2, l] * e_powers[j1, j2, m-1]
                                            phi_poly_diff_e1I_ee += poly_diff_e1I_ee * phi_p
                                            theta_poly_diff_e1I_ee += poly_diff_e1I_ee * theta_p
                                        if l > 0 and m > 0:
                                            poly_diff_e2I_ee = l * m * n_powers[i, j1, k] * n_powers[i, j2, l-1] * e_powers[j1, j2, m-1]
                                            phi_poly_diff_e2I_ee += poly_diff_e2I_ee * phi_p
                                            theta_poly_diff_e2I_ee += poly_diff_e2I_ee * theta_p

                            phi_diff_1 = (
                                (phi_poly_diff_e1I - C*phi_poly/(L - r_e1I))/r_e1I +
                                (phi_poly_diff_e2I - C*phi_poly/(L - r_e2I))/r_e2I +
                                4 * phi_poly_diff_ee/r_ee
                            )
                            phi_diff_2 = (
                                (C*(C - 1)*phi_poly/(L - r_e1I)**2 - 2*C*phi_poly_diff_e1I/(L - r_e1I) + phi_poly_diff_e1I_2) +
                                (C*(C - 1)*phi_poly/(L - r_e2I)**2 - 2*C*phi_poly_diff_e2I/(L - r_e2I) + phi_poly_diff_e2I_2) +
                                2 * phi_poly_diff_ee_2
                            )
                            phi_dot_product = (
                                (phi_poly_diff_e1I - C*phi_poly/(L - r_e1I)) * np.eye(3) @ r_e1I_vec/r_e1I -
                                (phi_poly_diff_e2I - C*phi_poly/(L - r_e2I)) * np.eye(3) @ r_e2I_vec/r_e2I +
                                (phi_poly_diff_e1I_ee - C*phi_poly_diff_ee/(L - r_e1I)) * np.outer(r_ee_vec, r_ee_vec)/r_ee @ r_e1I_vec/r_e1I -
                                (phi_poly_diff_e2I_ee - C*phi_poly_diff_ee/(L - r_e2I)) * np.outer(r_ee_vec, r_ee_vec)/r_ee @ r_e2I_vec/r_e2I
                            )
                            theta_diff_1 = (
                                2 * (theta_poly_diff_e1I - C*theta_poly/(L - r_e1I))/r_e1I +
                                (theta_poly_diff_e2I - C*theta_poly/(L - r_e2I))/r_e2I +
                                2 * theta_poly_diff_ee/r_ee
                            )
                            theta_diff_2 = (
                                (C*(C - 1)*theta_poly/(L - r_e1I)**2 - 2*C*theta_poly_diff_e1I/(L - r_e1I) + theta_poly_diff_e1I_2) +
                                (C*(C - 1)*theta_poly/(L - r_e2I)**2 - 2*C*theta_poly_diff_e2I/(L - r_e2I) + theta_poly_diff_e2I_2) +
                                2 * theta_poly_diff_ee_2
                            )
                            theta_dot_product = (
                                (theta_poly_diff_e1I_ee - C*theta_poly_diff_ee/(L - r_e1I)) * np.outer(r_e1I_vec, r_e1I_vec)/r_e1I @ r_ee_vec/r_ee -
                                (theta_poly_diff_e2I_ee - C*theta_poly_diff_ee/(L - r_e2I)) * np.outer(r_e1I_vec, r_e2I_vec)/r_e2I @ r_ee_vec/r_ee +
                                theta_poly_diff_ee * np.eye(3) @ r_ee_vec/r_ee
                            )
                            res[j1] += (1-r_e1I/L)**C * (1-r_e2I/L) ** C * (
                                (phi_diff_2 + 2 * phi_diff_1) * r_ee_vec + 2 * phi_dot_product +
                                (theta_diff_2 + 2 * theta_diff_1) * r_e1I_vec + 2 * theta_dot_product
                            )

        return res.ravel()

    def value(self, e_vectors, n_vectors):
        """Backflow displacemets
        :param e_vectors:
        :param n_vectors:
        :return: backflow displacement array(nelec * 3)
        """

        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.eta_term(e_vectors, e_powers) * self.ae_multiplier(n_vectors, n_powers) +
            self.mu_term(n_vectors, n_powers) +
            self.phi_term(e_powers, n_powers, e_vectors, n_vectors)
        )

    def numerical_gradient(self, e_vectors, n_vectors):
        """Numerical gradient with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return: partial derivatives of displacements of electrons - array(nelec * 3, nelec * 3)
        """
        delta = 0.00001

        res = np.zeros((self.neu + self.ned, 3, self.neu + self.ned, 3))

        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res[:, :, i, j] -= self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res[:, :, i, j] += self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3) / delta / 2

    def numerical_laplacian(self, e_vectors, n_vectors):
        """Numerical laplacian with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return: vector laplacian - array(nelec * 3)
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

        return res.ravel() / delta / delta

    def gradient(self, e_vectors, n_vectors):
        """Gradient with respect to e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.ae_multiplier_gradient(n_vectors, n_powers) * self.eta_term(e_vectors, e_powers).reshape((-1, 1)) +
            self.eta_term_gradient(e_powers, e_vectors) * self.ae_multiplier(n_vectors, n_powers).reshape((-1, 1)) +
            self.mu_term_gradient(n_powers, n_vectors) +
            self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
        )

    def laplacian(self, e_vectors, n_vectors):
        """Laplacian with respect to e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        a = self.ae_multiplier_laplacian(n_vectors, n_powers) * self.eta_term(e_vectors, e_powers).ravel()
        b = np.zeros(((self.neu + self.ned) * 3))
        term_gradient = self.eta_term_gradient(e_powers, e_vectors)
        cutoff_gradient = self.ae_multiplier_gradient(n_vectors, n_powers)
        for i in range((self.neu + self.ned) * 3):
            b[i] = np.sum(term_gradient[i] * cutoff_gradient[i])
        c = self.eta_term_laplacian(e_powers, e_vectors) * self.ae_multiplier(n_vectors, n_powers).ravel()

        return (
            a + 2 * b + c +
            self.mu_term_laplacian(n_powers, n_vectors) +
            self.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)
        )


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
def profiling_value(dx, neu, ned, steps, atom_positions, backflow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        backflow.value(e_vectors, n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def profiling_gradient(dx, neu, ned, steps, atom_positions, backflow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        backflow.gradient(e_vectors, n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def profiling_laplacian(dx, neu, ned, steps, atom_positions, backflow, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = subtract_outer(atom_positions, r_e)
        backflow.laplacian(e_vectors, n_vectors)


def main(casino):
    dx = 3.0

    backflow = Backflow(
        casino.input.neu, casino.input.ned,
        casino.backflow.trunc, casino.backflow.eta_parameters, casino.backflow.eta_cutoff,
        casino.backflow.mu_parameters, casino.backflow.mu_cutoff, casino.backflow.mu_labels,
        casino.backflow.phi_parameters, casino.backflow.theta_parameters, casino.backflow.phi_cutoff,
        casino.backflow.phi_labels, casino.backflow.phi_irrotational, casino.backflow.ae_cutoff
    )

    r_initial = initial_position(casino.input.neu + casino.input.ned, casino.wfn.atom_positions, casino.wfn.atom_charges)

    start = default_timer()
    profiling_value(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, backflow, r_initial)
    end = default_timer()
    logger.info(' value     %8.1f', end - start)

    start = default_timer()
    profiling_gradient(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, backflow, r_initial)
    end = default_timer()
    logger.info(' gradient  %8.1f', end - start)

    start = default_timer()
    profiling_laplacian(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, backflow, r_initial)
    end = default_timer()
    logger.info(' laplacian %8.1f', end - start)


if __name__ == '__main__':
    """
    He:
     value         40.0
     gradient     121.6
     laplacian    138.8
    Be:
     value         99.5
     gradient     481.4
     laplacian    573.4
    Ne:
     value        415.0
     gradient    1897.3
     laplacian   2247.9
    Ar:
     value       1501.9
    """

    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr'):
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Backflow/'
        logger.info('%s:', mol)
        main(Casino(path))
