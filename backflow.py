#!/usr/bin/env python3

import numpy as np
import numba as nb
# import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from readers.casino import Casino
from overload import subtract_outer

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
        ))
        self.eta_cutoff = eta_cutoff
        self.mu_cutoff = mu_cutoff
        self.phi_cutoff = phi_cutoff
        self.phi_irrotational = phi_irrotational
        self.ae_cutoff = ae_cutoff
        if self.eta_cutoff.any():
            self.fix_eta_parameters()
        if self.mu_cutoff.any():
            self.fix_mu_parameters()
        if self.phi_cutoff.any():
            self.fix_phi_parameters()

    def ee_powers(self, e_vectors):
        """Powers of e-e distances
        :param e_vectors: e-e vectors - array(nelec, nelec, 3)
        :return:
        """
        res = np.zeros((e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(1, e_vectors.shape[0]):
            for j in range(i):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(self.max_ee_order):
                    res[i, j, k] = res[j, i, k] = r_ee ** k
        return res

    def en_powers(self, n_vectors):
        """Powers of e-n distances
        :param n_vectors: e-n vectors
        :return:
        """
        res = np.zeros((n_vectors.shape[0], n_vectors.shape[1], self.max_en_order))
        for i in range(n_vectors.shape[0]):
            for j in range(n_vectors.shape[1]):
                r_eI = np.linalg.norm(n_vectors[i, j])
                for k in range(self.max_en_order):
                    res[i, j, k] = r_eI ** k
        return res

    def ae_cutoffs(self, n_vectors, n_powers):
        """Zeroing the backflow displacement at AE atoms."""
        res = np.ones((self.neu + self.ned, 3))
        for i in range(n_vectors.shape[0]):
            Lg = self.ae_cutoff[i]
            for j in range(self.neu + self.ned):
                r = n_powers[i, j, 1]
                if r < Lg:
                    res[j] *= (r/Lg)**2 * (6 - 8 * (r/Lg) + 3 * (r/Lg)**2)
        return res

    def eta_term(self, e_vectors, e_powers):
        """
        :param e_vectors:
        :param e_powers:
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
                # I don't think it works fast if NO SPIN-DEP
                L = self.eta_cutoff[eta_set] or self.eta_cutoff[0]
                if r < L:
                    poly = 0
                    for k in range(parameters.shape[0]):
                        poly += parameters[k, eta_set] * e_powers[i, j, k]
                    bf = (1 - r/L) ** C * poly * r_vec
                    res[i] += bf
                    # res[j] -= bf
        return res

    def mu_term(self, n_vectors, n_powers):
        """
        :param n_vectors:
        :param n_powers:
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

    def phi_term(self, e_vectors, n_vectors, e_powers, n_powers):
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
        for parameters, L, phi_labels, phi_irrotational in zip(self.phi_parameters, self.phi_cutoff, self.phi_labels, self.phi_irrotational):
            for i in phi_labels:
                for j in range(self.neu + self.ned):
                    for k in range(j):
                        r_e1I_vec = n_vectors[i, j]
                        r_e2I_vec = n_vectors[i, k]
                        r_ee_vec = e_vectors[j, k]
                        r_e1I = n_powers[i, j, 1]
                        r_e2I = n_powers[i, k, 1]
                        if r_e1I < L and r_e2I < L:
                            phi_set = (int(j >= self.neu) + int(k >= self.neu)) % parameters.shape[3]
                            poly = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        poly += parameters[l, m, n, phi_set] * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n]
                            bf = poly * (1-r_e1I/L) ** C * (1-r_e2I/L) ** C * r_ee_vec
                            res[j] += bf
                            res[k] -= bf
                            if phi_irrotational:
                                continue
                            poly = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        poly += parameters[l, m, n, phi_set] * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n]
                            bf = poly * (1-r_e1I/L) ** C * (1-r_e2I/L) ** C
                            res[j] += bf * r_e1I_vec
                            res[k] += bf * r_e2I_vec

        return res

    def eta_term_gradient(self, e_powers, e_vectors):
        """
        :param e_powers:
        :param e_vectors:
        :return: partial derivatives of displacements of electrons - array(nelec * 3, nelec * 3)
        """
        res = np.zeros((self.neu + self.ned, 3, self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

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
                                poly_diff += p * k * n_powers[i, j, k-1]

                        res[j, :, j, :] += (1 - r/L)**C * (
                            ((L - r)*poly_diff - C*poly)/(r*(L - r)) * np.outer(r_vec, r_vec) + poly * np.eye(3)
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

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    def eta_term_laplacian(self, e_powers, e_vectors):
        """
        :param e_powers:
        :param e_vectors:
        :return: vector laplacian - array(nelec * 3)
        """
        res = np.zeros((self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res.ravel()

        return res.ravel()

    def mu_term_laplacian(self, n_powers, n_vectors):
        """
        :param e_powers:
        :param e_vectors:
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
                                poly_diff += p * k * n_powers[i, j, k-1]
                            if k > 1:
                                poly_diff_2 += k * (k-1) * p * n_powers[i, j, k - 2]

                        res[j] += r_vec

        return res.ravel()

    def phi_term_laplacian(self, e_powers, n_powers, e_vectors, n_vectors):
        """
        :param e_powers:
        :param e_vectors:
        :return: vector laplacian - array(nelec * 3)
        """
        res = np.zeros((self.neu + self.ned, 3))
        if not self.phi_cutoff.any():
            return res.ravel()

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
            self.eta_term(e_vectors, e_powers) * self.ae_cutoffs(n_vectors, n_powers) +
            self.mu_term(n_vectors, n_powers) +
            self.phi_term(e_vectors, n_vectors, e_powers, n_powers)
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
                res[i, j] -= self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.value(e_vectors, n_vectors)
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
            self.eta_term_gradient(e_powers, e_vectors) +
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

        return (
            self.eta_term_laplacian(e_powers, e_vectors) +
            self.mu_term_laplacian(n_powers, n_vectors) +
            self.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)
        )

    def fix_eta_parameters(self):
        """Fix eta-term parameters"""
        C = self.trunc
        L = self.eta_cutoff[0]
        self.eta_parameters[1, 0] = C * self.eta_parameters[0, 0] / L
        if self.eta_parameters.shape[1] == 3:
            L = self.eta_cutoff[2] or self.eta_cutoff[0]
            self.eta_parameters[1, 2] = C * self.eta_parameters[0, 2] / L

    def fix_mu_parameters(self):
        """Fix mu-term parameters"""

    def fix_phi_parameters(self):
        """Fix phi-term parameters
        0 - zero value
        A - no electron–electron cusp constrains
        B - no electron–nucleus cusp constrains
        X - independent value

        m = 0            m = 1            m = 2            m > 2
        --------------------------- same spin ---------------------------------
        . . . . . . . .  . . . . . . . .  . X X X X X . .  X X X X X X . . <- l
        . . . . . . . .  . . X X X X X .  X X X X X X X .  X X X X X X X .
        . . X X X X X X  . . X X X X X X  X X X X X X X X  X X X X X X X X
        . . X X X X X X  . . X X X X X X  X X X X X X X X  X X X X X X X X
        . . X X X X X X  . . X X X X X X  X X X X X X X X  X X X X X X X X
        . . X X X X X X  . . X X X X X X  X X X X X X X X  X X X X X X X X
        . . X X X X X X  . . . X X X X X  . X X X X X X X  . X X X X X X X
        . . X X X X X X  . . . . . . . .  . . X X X X X X  . . X X X X X X

        ------------------------- opposite spin -------------------------------
        . . . . . . . .  . . . . . . . .  X X X X X X . .  X X X X X X . . <- l
        . . . . . . . .  . X X X X X X .  X X X X X X X .  X X X X X X X .
        . X X X X X X X  . X X X X X X X  X X X X X X X X  X X X X X X X X
        . X X X X X X X  . X X X X X X X  X X X X X X X X  X X X X X X X X
        . X X X X X X X  . X X X X X X X  X X X X X X X X  X X X X X X X X
        . X X X X X X X  . X X X X X X X  X X X X X X X X  X X X X X X X X
        . X X X X X X X  . X X X X X X X  . X X X X X X X  . X X X X X X X
        . X X X X X X X  . . X X X X X X  . . X X X X X X  . . X X X X X X

        --------------------- same spin irrotational---------------------------
        . . . . . . . .  . . . . . . . .  . . . . . . . .  . . . . . . . . <- l
        . . . . . . . .  . . . . . . . .  . . . . . . . .  . . X X X X . .
        . X X X X X X X  . . . . . . . .  . . X X X X . .  . . X X X X X .
        . X X X X X X X  . . . . . . . .  . . X X X X X X  . . X X X X X X
        . X X X X X X X  . . . . . . . .  . . X X X X X X  . . X X X X X X
        . X X X X X X X  . . . . . . . .  . . X X X X X X  . . X X X X X X
        . X X X X X X X  . . . . . . . .  . . . . . . . .  . . X X X X X X
        . X X X X X X X  . . . . . . . .  . . . . . . . .  . . X X X X X X

        ^
        k
        """

    def fix_theta_parameters(self):
        """Fix theta-term parameters
        0 - zero value
        A - no electron–electron cusp constrains
        B - no electron–nucleus cusp constrains
        X - independent value

        m = 0            m = 1            m = 2            m > 2
        -----------------------------------------------------------------------
        . . . . . . . .  . . . . . . . .  . X X X X X . .  X X X X X X . . <- l
        . . . . . . . .  . . X X X X X .  X X X X X X X .  X X X X X X X .
        . X X X X X X X  . . X X X X X X  X X X X X X X X  X X X X X X X X
        . X X X X X X X  . . X X X X X X  X X X X X X X X  X X X X X X X X
        . X X X X X X X  . . X X X X X X  X X X X X X X X  X X X X X X X X
        . X X X X X X X  . . X X X X X X  X X X X X X X X  X X X X X X X X
        . X X X X X X X  . . . X X X X X  X X X X X X X X  X X X X X X X X
        . X X X X X X X  . . . . . . . .  . . X X X X X X  . . X X X X X X
        ^
        k
        """


if __name__ == '__main__':
    """Plot Backflow terms
    """

    term = 'eta'

    path = 'test/stowfn/ne/HF/QZ4P/VMC_OPT_BF/emin_BF/8_8_44__9_9_33_2/'

    casino = Casino(path)
    backflow = Backflow(
        casino.input.neu, casino.input.ned,
        casino.backflow.trunc, casino.backflow.eta_parameters, casino.backflow.eta_cutoff,
        casino.backflow.mu_parameters, casino.backflow.mu_cutoff, casino.backflow.mu_labels,
        casino.backflow.phi_parameters, casino.backflow.theta_parameters, casino.backflow.phi_cutoff,
        casino.backflow.phi_labels, casino.backflow.phi_irrotational, casino.backflow.ae_cutoff
    )

    steps = 100

    if term == 'eta':
        x_min, x_max = 0, np.max(backflow.eta_cutoff)
        x_grid = np.linspace(x_min, x_max, steps)
        for spin_dep in range(3):
            backflow.neu = 2-spin_dep
            backflow.ned = spin_dep
            y_grid = np.zeros((steps, ))
            for i in range(100):
                r_e = np.array([[0.0, 0.0, 0.0], [x_grid[i], 0.0, 0.0]])
                e_vectors = subtract_outer(r_e, r_e)
                e_powers = backflow.ee_powers(e_vectors)
                y_grid[i] = backflow.eta_term(e_vectors, e_powers)[1, 0]
            plt.plot(x_grid, y_grid, label=['uu', 'ud', 'dd'][spin_dep])
        plt.xlabel('r_ee (au)')
        plt.ylabel('polynomial part')
        plt.title('Backflow eta-term')
    elif term == 'mu':
        for atom in range(casino.wfn.atom_positions.shape[0]):
            x_min, x_max = 0, backflow.eta_cutoff[atom]
            x_grid = np.linspace(x_min, x_max, steps)
            for spin_dep in range(2):
                backflow.neu = 1 - spin_dep
                backflow.ned = spin_dep
                y_grid = np.zeros((steps, ))
                for i in range(100):
                    r_e = np.array([[x_grid[i], 0.0, 0.0]]) + casino.wfn.atom_positions[atom]
                    sl = slice(atom, atom+1)
                    backflow.chi_parameters = nb.typed.List.empty_list(mu_parameters_type)
                    [backflow.mu_parameters.append(p) for p in casino.jastrow.mu_parameters[sl]]
                    n_vectors = subtract_outer(casino.wfn.atom_positions[sl], r_e)
                    n_powers = backflow.en_powers(n_vectors)
                    y_grid[i] = backflow.mu_term(n_powers)
                plt.plot(x_grid, y_grid, label=f'atom {atom} ' + ['u', 'd'][spin_dep])

    plt.grid(True)
    plt.legend()
    plt.show()
