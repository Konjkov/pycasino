#!/usr/bin/env python3

import numpy as np
from yaml import safe_load


class Gjastrow:
    """Jastrow reader from file.
    CASINO manual: 7.8 Wave function parameter file: parameters.casl

    Framework for constructing generic Jastrow correlation factors
    P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
    Phys. Rev. E 86, 036703
    """

    def order(self, term):
        try:
            return term['Order']
        except TypeError:
            return term[1]['Order']

    def rank(self, term):
        if isinstance(term['Rank'], list):
            return term['Rank']
        elif isinstance(term['Rank'], str):
            return list(map(int, term['Rank'].split()))

    def channels(self, term):
        return term['Linear parameters']

    def trunc(self, term):
        try:
            return term['e-e cutoff']['Constants']['C']
        except TypeError:
            return term['e-e cutoff']['Constants'][0]['C']

    def parameters(self, term, channel):
        return term['e-e cutoff']['Parameters'][channel]['L'][0]

    def linear_parameters(self, term):
        """Load linear parameters into multidimensional array."""
        e_rank, n_rank = self.rank(term)
        dims = [len(self.channels(term))]
        if e_rank > 1:
            e_order = self.order(term['e-e basis'])
            dims += [e_order] * (e_rank * (e_rank-1) // 2)
        if n_rank > 0:
            n_order = self.order(term['e-n basis'])
            dims += [n_order] * (e_rank * n_rank)

        linear_parameters = np.zeros(dims, np.float)
        for i, channel in enumerate(self.channels(term).values()):
            for key, val in channel.items():
                index = tuple(map(lambda x: x-1, map(int, key.split('_')[1].split(','))))
                linear_parameters[i, index] = val[0]
        return linear_parameters

    def __init__(self, file, atom_charges):
        with open(file, 'r') as f:
            self._jastrow_data = safe_load(f)['JASTROW']
        for key, term in self._jastrow_data.items():
            if key.startswith('TERM'):
                self.trunc = self.trunc(term)
                self.u_parameters = self.linear_parameters(term)
                self.u_cutoff = self.parameters(term, 'Channel 2-2')
                self.chi_parameters = False
                self.chi_cutoff = False
                self.f_parameters = False
                self.f_cutoff = False


if __name__ == '__main__':
    """
    """
    path = 'parameters.casl'

    Gjastrow(path)
