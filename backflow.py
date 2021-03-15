#!/usr/bin/env python3

import numpy as np
import numba as nb
# import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from readers.casino import Casino

eta_parameters_type = nb.float64[:, :]
mu_parameters_type = nb.float64[:, :]
phi_parameters_type = nb.float64[:, :, :, :]
theta_parameters_type = nb.float64[:, :, :, :]


spec = [
    ('trunc', nb.int64),
    ('eta_parameters', eta_parameters_type),
    ('mu_parameters', nb.types.ListType(mu_parameters_type)),
    ('phi_parameters', nb.types.ListType(phi_parameters_type)),
    ('theta_parameters', nb.types.ListType(theta_parameters_type)),
    ('eta_cutoff', nb.float64[:]),
    ('mu_cutoff', nb.float64[:]),
    ('phi_cutoff', nb.float64[:]),
    ('max_ee_order', nb.int64),
    ('max_en_order', nb.int64),
]


@nb.experimental.jitclass(spec)
class Backflow:

    def __init__(
        self, trunc, eta_parameters, eta_cutoff, mu_parameters, mu_cutoff, phi_parameters, theta_parameters, phi_cutoff
    ):
        self.trunc = trunc
        self.eta_parameters = eta_parameters
        self.mu_parameters = nb.typed.List.empty_list(mu_parameters_type)
        [self.mu_parameters.append(p) for p in mu_parameters]
        self.phi_parameters = nb.typed.List.empty_list(phi_parameters_type)
        [self.phi_parameters.append(p) for p in phi_parameters]
        self.theta_parameters = nb.typed.List.empty_list(theta_parameters_type)
        [self.theta_parameters.append(p) for p in theta_parameters]
        self.max_ee_order = max((
            self.eta_parameters.shape[0],
            max([p.shape[2] for p in self.phi_parameters]) if self.phi_parameters else 0,
        ))
        self.max_en_order = max((
            max([p.shape[0] for p in self.mu_parameters]) if self.mu_parameters else 0,
            max([p.shape[0] for p in self.phi_parameters]) if self.phi_parameters else 0,
        ))
        self.eta_cutoff = eta_cutoff
        self.mu_cutoff = mu_cutoff
        self.phi_cutoff = phi_cutoff

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

    def mu_term(self, e_vectors, n_vectors, e_powers, neu):
        C = self.trunc
        parameters = self.mu_parameters
        for i in range(e_powers.shape[0] - 1):
            for j in range(i + 1, e_powers.shape[1]):
                r = e_powers[i, j, 1]

    def value(self, e_vectors, n_vectors, neu):
        """Backflow displacemets
        :param e_vectors:
        :param n_vectors:
        :param neu:
        :return:
        """

        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.mu_term(e_vectors, n_vectors, e_powers, neu)


if __name__ == '__main__':
    """Plot Backflow terms
    """

    term = 'eta'

    path = 'test/stowfn/ne/HF/QZ4P/VMC_OPT_BF/emin_BF/8_8_44__9_9_33/'

    casino = Casino(path)
    backflow = Backflow(
        casino.backflow.trunc, casino.backflow.eta_parameters, casino.backflow.eta_cutoff,
        casino.backflow.mu_parameters, casino.backflow.mu_cutoff,
        casino.backflow.phi_parameters, casino.backflow.theta_parameters, casino.backflow.phi_cutoff,
    )
