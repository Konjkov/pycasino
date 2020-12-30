#!/usr/bin/env python3

import numpy as np
import numba as nb

import matplotlib.pyplot as plt

from readers.casino import Casino
from overload import subtract_outer


constants_type = nb.types.DictType(nb.types.unicode_type, nb.float64)
parameters_type = nb.types.ListType(nb.types.DictType(nb.types.unicode_type, nb.float64))
linear_parameters_type = nb.float64[:, :]

spec = [
    ('e_rank', nb.types.ListType(nb.int64)),
    ('n_rank', nb.types.ListType(nb.int64)),
    ('ee_basis_type', nb.types.ListType(nb.types.unicode_type)),
    ('en_basis_type', nb.types.ListType(nb.types.unicode_type)),
    ('ee_cutoff_type', nb.types.ListType(nb.types.unicode_type)),
    ('en_cutoff_type', nb.types.ListType(nb.types.unicode_type)),
    ('ee_constants', nb.types.ListType(constants_type)),
    ('en_constants', nb.types.ListType(constants_type)),
    ('ee_basis_parameters', parameters_type),
    ('en_basis_parameters', parameters_type),
    ('ee_cutoff_parameters', parameters_type),
    ('en_cutoff_parameters', parameters_type),
    ('linear_parameters', linear_parameters_type),
]


@nb.experimental.jitclass(spec)
class Gjastrow:

    def __init__(
            self, e_rank, n_rank, ee_basis_type, en_basis_type, ee_cutoff_type, en_cutoff_type,
            ee_constants, en_constants, ee_basis_parameters, en_basis_parameters, ee_cutoff_parameters,
            en_cutoff_parameters, linear_parameters):
        self.e_rank = e_rank
        self.n_rank = n_rank
        self.ee_basis_type = ee_basis_type
        self.en_basis_type = en_basis_type
        self.ee_cutoff_type = ee_cutoff_type
        self.en_cutoff_type = en_cutoff_type
        self.ee_constants = ee_constants
        self.en_constants = en_constants
        self.ee_basis_parameters = ee_basis_parameters
        self.en_basis_parameters = en_basis_parameters
        self.ee_cutoff_parameters = ee_cutoff_parameters
        self.en_cutoff_parameters = en_cutoff_parameters
        self.linear_parameters = linear_parameters

    def ee_powers(self, e_vectors):
        res = np.zeros((e_vectors.shape[0], e_vectors.shape[1], self.linear_parameters.shape[1], self.linear_parameters.shape[0]))
        if self.ee_basis_parameters:
            a = self.ee_basis_parameters[channel].get('a')
            b = self.ee_basis_parameters[channel].get('b')
        for i in range(e_vectors.shape[0] - 1):
            for j in range(i + 1, e_vectors.shape[1]):
                r = np.linalg.norm(e_vectors[i, j])
                for k in range(self.linear_parameters.shape[1]):
                    for l in range(self.linear_parameters.shape[0]):
                        if self.ee_basis_type[0] == 'natural power':
                            res[i, j, k, l] = r ** k
                        elif self.ee_basis_type[0] == 'r/(r^b+a) power':
                            res[i, j, k, l] = (r/(r**b + a)) ** k
                        elif self.ee_basis_type[0] == 'r/(r+a) power':
                            res[i, j, k, l] = (r/(r + a)) ** k
                        elif self.ee_basis_type[0] == '1/(r+a) power':
                            res[i, j, k, l] = (1/(r + a)) ** k
        return res

    def en_powers(self, n_vectors):
        res = np.zeros((n_vectors.shape[1], n_vectors.shape[0], self.linear_parameters.shape[1], self.linear_parameters.shape[0]))
        if self.ee_basis_parameters:
            a = self.en_basis_parameters[channel].get('a')
            b = self.en_basis_parameters[channel].get('b')
        for i in range(n_vectors.shape[1]):
            for j in range(n_vectors.shape[0]):
                r = np.linalg.norm(n_vectors[j, i])
                for k in range(self.linear_parameters.shape[1]):
                    for l in range(self.linear_parameters.shape[0]):
                        if self.en_basis_type[0] == 'natural power':
                            res[i, j, k, l] = r ** k
                        elif self.en_basis_type[0] == 'r/(r^b+a) power':
                            res[i, j, k, l] = (r/(r**b + a)) ** k
                        elif self.en_basis_type[0] == 'r/(r+a) power':
                            res[i, j, k, l] = (r/(r + a)) ** k
                        elif self.en_basis_type[0] == '1/(r+a) power':
                            res[i, j, k, l] = (1/(r + a)) ** k
        return res

    def term_2_0(self, e_powers, e_vectors, neu):
        """Jastrow term rank [2, 0]
        :param e_powers: electrons coordinates
        :param neu: number of up electrons
        :return:
        """
        res = 0.0

        p = self.linear_parameters
        C = self.ee_constants[0]['C']  # FIXME: first term hardcoded
        for i in range(e_powers.shape[0] - 1):
            for j in range(i + 1, e_powers.shape[1]):
                r = np.linalg.norm(e_vectors[i, j])
                channel = int(i >= neu) + int(j >= neu)
                L = self.ee_cutoff_parameters[channel]['L']
                L_hard = self.ee_cutoff_parameters[channel].get('L_hard')

                poly = 0.0
                for k in range(p.shape[0]):
                    poly += p[channel, k] * e_powers[i, j, k, channel]

                if self.ee_cutoff_type == 'gaussian':
                    if r <= L_hard:
                        res += poly * np.exp(-(r/L) ** 2)
                elif r <= L:
                    if self.ee_cutoff_type[0] == 'polynomial':
                        res += poly * (1 - r/L) ** C
                    elif self.ee_cutoff_type == 'alt polynomial':
                        res += poly * (r - L) ** C
                    elif self.ee_cutoff_type == 'spline':
                        pass
                    elif self.ee_cutoff_type == 'anisotropic polynomial':
                        pass
        return res

    def value(self, e_vectors, n_vectors, neu):
        """Jastrow
        :param e_vectors: electrons coordinates
        :param n_vectors: nucleus coordinates
        :param neu: number of up electrons
        :return:
        """

        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.term_2_0(e_powers, e_vectors, neu)


if __name__ == '__main__':
    """
    """

    rank = [2, 0]

    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/casl/8__1/'
    path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/casl/8__1/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/casl/8__4/'

    casino = Casino(path)
    gjastrow = Gjastrow(
        casino.jastrow.e_rank, casino.jastrow.n_rank,
        casino.jastrow.ee_basis_type, casino.jastrow.en_basis_type,
        casino.jastrow.ee_cutoff_type, casino.jastrow.en_cutoff_type,
        casino.jastrow.ee_constants, casino.jastrow.en_constants,
        casino.jastrow.ee_basis_parameters, casino.jastrow.en_basis_parameters,
        casino.jastrow.ee_cutoff_parameters, casino.jastrow.en_cutoff_parameters,
        casino.jastrow.linear_parameters
    )

    steps = 100

    if rank == [2, 0]:
        x_min, x_max = 0, gjastrow.ee_cutoff_parameters[0]['L']
        x_grid = np.linspace(x_min, x_max, steps)
        for channel in range(gjastrow.linear_parameters.shape[0]):
            y_grid = np.zeros(steps)
            for i in range(100):
                r_e = np.array([[0.0, 0.0, 0.0], [x_grid[i], 0.0, 0.0]])
                e_vectors = subtract_outer(r_e, r_e)
                e_powers = gjastrow.ee_powers(e_vectors)
                y_grid[i] = gjastrow.term_2_0(e_powers, e_vectors, 2-channel)
            plt.plot(x_grid, y_grid, label=['1-1', '1-2', '2-2'][channel])
        plt.xlabel('r_ee (au)')
        plt.ylabel('polynomial part')
        plt.title(f'JASTROW term {rank}')

    plt.grid(True)
    plt.legend()
    plt.show()
