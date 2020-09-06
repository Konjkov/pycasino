#!/usr/bin/env python3

import numpy as np
import numba as nb

import matplotlib.pyplot as plt

from readers.casino import Casino
from overload import subtract_outer


parameters_type = nb.types.float64[:]
linear_parameters_type = nb.types.float64[:, :]

spec = [
    ('ee_basis_type', nb.types.unicode_type),
    ('en_basis_type', nb.types.unicode_type),
    ('ee_cutoff_type', nb.types.unicode_type),
    ('en_cutoff_type', nb.types.unicode_type),
    ('e_trunc', nb.types.int64),
    ('n_trunc', nb.types.int64),
    ('e_parameters', parameters_type),
    ('n_parameters', parameters_type),
    ('linear_parameters', linear_parameters_type),
]


@nb.experimental.jitclass(spec)
class Gjastrow:

    def __init__(self, ee_basis_type, en_basis_type, ee_cutoff_type, en_cutoff_type, e_trunc, n_trunc, e_parameters, n_parameters, linear_parameters):
        self.ee_basis_type = ee_basis_type
        self.en_basis_type = en_basis_type
        self.ee_cutoff_type = ee_cutoff_type
        self.en_cutoff_type = en_cutoff_type
        self.e_trunc = e_trunc
        self.n_trunc = n_trunc
        self.e_parameters = e_parameters
        self.n_parameters = n_parameters
        self.linear_parameters = linear_parameters

    def ee_powers(self, e_vectors):
        res = np.zeros((e_vectors.shape[0], e_vectors.shape[1], self.linear_parameters.shape[1]))
        for i in range(e_vectors.shape[0] - 1):
            for j in range(i + 1, e_vectors.shape[1]):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(self.linear_parameters.shape[0]):
                    res[i, j, k] = r_ee ** k
        return res

    def en_powers(self, n_vectors):
        res = np.zeros((n_vectors.shape[1], n_vectors.shape[0], self.linear_parameters.shape[1]))
        for i in range(n_vectors.shape[1]):
            for j in range(n_vectors.shape[0]):
                r_eI = np.linalg.norm(n_vectors[j, i])
                for k in range(self.linear_parameters.shape[0]):
                    res[i, j, k] = r_eI ** k
        return res

    def term_2_0(self, e_powers, neu):
        """Jastrow term rank [2, 0]
        :param e_powers: electrons coordinates
        :param neu: number of up electrons
        :return:
        """
        res = 0.0

        p = self.linear_parameters
        for i in range(e_powers.shape[0] - 1):
            for j in range(i + 1, e_powers.shape[1]):
                r = e_powers[i, j, 1]
                u_set = int(i >= neu) + int(j >= neu)
                if r <= self.e_parameters[u_set]:
                    poly = 0.0
                    for k in range(p.shape[0]):
                        poly += p[u_set, k] * e_powers[i, j, k]
                    res += poly * (r - self.e_parameters[u_set]) ** self.e_trunc
        return res

    def value(self, e_vectors, n_vectors, neu):
        """Jastrow
        :param e_vectors: electrons coordinates
        :param n_vectors: nucleus coordinates
        :param neu: number of up electrons
        :return:
        """

        e_powers = self.ee_powers(e_vectors)
        # n_powers = self.en_powers(n_vectors)

        return self.term_2_0(e_powers, neu)


if __name__ == '__main__':
    """
    """

    term = 'chi'

    path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/casl/8__1/'

    casino = Casino(path)
    gjastrow = Gjastrow(
        casino.jastrow.ee_basis_type, casino.jastrow.en_basis_type,
        casino.jastrow.ee_cutoff_type, casino.jastrow.en_cutoff_type,
        casino.jastrow.e_trunc, casino.jastrow.n_trunc,
        casino.jastrow.e_parameters, casino.jastrow.n_parameters, casino.jastrow.linear_parameters
    )

    steps = 100

    if True:
        x_min, x_max = 0, gjastrow.e_parameters[0]
        x_grid = np.linspace(x_min, x_max, steps)
        for spin_dep in range(3):
            y_grid = np.zeros(steps)
            for i in range(100):
                r_e = np.array([[0.0, 0.0, 0.0], [x_grid[i], 0.0, 0.0]])
                e_vectors = subtract_outer(r_e, r_e)
                e_powers = gjastrow.ee_powers(e_vectors)
                y_grid[i] = gjastrow.term_2_0(e_powers, 2-spin_dep)
                if spin_dep == 1:
                    y_grid[i] /= 2.0
            plt.plot(x_grid, y_grid, label=['uu', 'ud/2', 'dd'][spin_dep])
        plt.xlabel('r_ee (au)')
        plt.ylabel('polynomial part')
        plt.title('JASTROW term [2, 0]')

    plt.grid(True)
    plt.legend()
    plt.show()
