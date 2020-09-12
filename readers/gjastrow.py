#!/usr/bin/env python3

from collections import ChainMap

import numpy as np
from yaml import safe_load


class Gjastrow:
    """Jastrow reader from file.
    CASINO manual: 7.8 Wave function parameter file: parameters.casl

    Framework for constructing generic Jastrow correlation factors
    P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
    Phys. Rev. E 86, 036703
    """

    def list_or_dict(self, node):
        return dict(ChainMap(*node)) if isinstance(node, list) else node

    def get_ee_cusp(self, term):
        """Load e-e cusp.
        """
        e_rank, n_rank = term['Rank']
        ee_cusp = False
        if e_rank > 1:
            ee_cusp = term.get('e-e cusp') == 'T'
        return ee_cusp

    def get_cutoff_type(self, term):
        """Load cutoff type.
        """
        e_rank, n_rank = term['Rank']
        e_cutoff_type = n_cutoff_type = ''
        if e_rank > 1:
            e_cutoff_type = self.list_or_dict(term['e-e cutoff'])['Type']
        if n_rank > 0:
            n_cutoff_type = self.list_or_dict(term['e-n cutoff'])['Type']
        return e_cutoff_type, n_cutoff_type

    def get_trunc(self, term):
        """Load truncation constant.
        """
        e_rank, n_rank = term['Rank']
        e_trunc = n_trunc = 0
        if e_rank > 1 and term.get('e-e cutoff'):
            e_trunc = self.list_or_dict(term['e-e cutoff']['Constants'])['C']
        if n_rank > 0 and term.get('e-n cutoff'):
            n_trunc = self.list_or_dict(term['e-n cutoff']['Constants'])['C']
        return e_trunc, n_trunc

    def get_parameters(self, term):
        """Load parameters into 1-dimensional array.
        """
        e_rank, n_rank = term['Rank']
        e_parameters = n_parameters = np.zeros((0,))
        if e_rank > 1 and term.get('e-e cutoff'):
            parameters = term['e-e cutoff']['Parameters']
            e_parameters = np.array([self.list_or_dict(channel)['L'][0] for channel in parameters.values()], np.float)
        if n_rank > 0 and term.get('e-n cutoff'):
            parameters = term['e-n cutoff']['Parameters']
            n_parameters = np.array([self.list_or_dict(channel)['L'][0] for channel in parameters.values()], np.float)
        return e_parameters, n_parameters

    def get_basis_type(self, term):
        """Load basis type.
        """
        e_rank, n_rank = term['Rank']
        e_basis_type = n_basis_type = ''
        if e_rank > 1:
            e_basis_type = self.list_or_dict(term['e-e basis'])['Type']
        if n_rank > 0:
            n_basis_type = self.list_or_dict(term['e-n basis'])['Type']
        return e_basis_type, n_basis_type

    def get_linear_parameters(self, term):
        """Load linear parameters into multidimensional array.
        """
        e_rank, n_rank = term['Rank']
        linear_parameters = term['Linear parameters']
        dims = [len(linear_parameters)]
        if e_rank > 1:
            e_order = self.list_or_dict(term['e-e basis'])['Order']
            dims += [e_order] * (e_rank * (e_rank-1) // 2)
        if n_rank > 0:
            n_order = self.list_or_dict(term['e-n basis'])['Order']
            dims += [n_order] * (e_rank * n_rank)

        res = np.zeros(dims, np.float)
        for i, channel in enumerate(linear_parameters.values()):
            for key, val in channel.items():
                j = tuple(map(lambda x: x-1, map(int, key.split('_')[1].split(','))))
                res[i, j] = val[0]
        return res

    def __init__(self, file, atom_charges):
        with open(file, 'r') as f:
            self._jastrow_data = safe_load(f)['JASTROW']
        for key, term in self._jastrow_data.items():
            if key.startswith('TERM 1'):
                self.ee_cusp = self.get_ee_cusp(term)
                self.ee_basis_type, self.en_basis_type = self.get_basis_type(term)
                self.ee_cutoff_type, self.en_cutoff_type = self.get_cutoff_type(term)
                self.e_trunc, self.n_trunc = self.get_trunc(term)
                self.e_parameters, self.n_parameters = self.get_parameters(term)
                self.linear_parameters = self.get_linear_parameters(term)
                for i, channel in enumerate(term['Linear parameters']):
                    ch1, ch2 = channel[8:].split('-')
                    G = 1/4 if ch1 == ch2 else 1/2
                    if self.linear_parameters[i, 0]:
                        continue
                    C = self.e_parameters[i] / self.e_trunc
                    if self.ee_cutoff_type == 'polynomial':
                        self.linear_parameters[i, 0] = C * (self.linear_parameters[i, 1] - G)
                    elif self.ee_cutoff_type == 'alt polynomial':
                        self.linear_parameters[i, 0] = C * (self.linear_parameters[i, 1] - G/(-self.e_parameters[i])**self.e_trunc)
                self.e_permutation = np.zeros((0,))


if __name__ == '__main__':
    """
    """
    path = 'parameters.casl'

    Gjastrow(path)
