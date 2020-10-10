#!/usr/bin/env python3

from collections import ChainMap

import numpy as np
import numba as nb
from yaml import safe_load


def dict_to_typed_dict(_dict, _type=nb.types.float64):
    result = nb.typed.Dict.empty(nb.types.string, _type)
    for k, v in _dict.items():
        result[k] = v
    return result


class Gjastrow:
    """Jastrow reader from file.
    CASINO manual: 7.8 Wave function parameter file: parameters.casl

    Framework for constructing generic Jastrow correlation factors
    P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
    Phys. Rev. E 86, 036703
    """

    def get(self, term, *args):
        def list_or_dict(node):
            return dict(ChainMap(*node)) if isinstance(node, list) else node
        res = term
        for arg in args:
            if arg not in ['a', 'b', 'L']:
                res = list_or_dict(res.get(arg))
            else:
                res = res.get(arg)
        return res

    def get_ee_cusp(self, term):
        """Load e-e cusp.
        """
        e_rank, n_rank = term['Rank']
        ee_cusp = False
        if e_rank > 1:
            ee_cusp = term.get('e-e cusp') == 'T'
        return ee_cusp

    def get_basis_type(self, term):
        """Load basis type.
        """
        e_rank, n_rank = term['Rank']
        ee_basis_type = en_basis_type = ''
        if e_rank > 1:
            ee_basis_type = self.get(term, 'e-e basis')['Type']
        if n_rank > 0:
            en_basis_type = self.get(term, 'e-n basis')['Type']
        return ee_basis_type, en_basis_type

    def get_cutoff_type(self, term):
        """Load cutoff type.
        """
        e_rank, n_rank = term['Rank']
        e_cutoff_type = n_cutoff_type = ''
        if e_rank > 1:
            e_cutoff_type = self.get(term, 'e-e cutoff')['Type']
        if n_rank > 0:
            n_cutoff_type = self.get(term, 'e-n cutoff')['Type']
        return e_cutoff_type, n_cutoff_type

    def get_constants(self, term):
        """Load truncation constant.
        """
        e_rank, n_rank = term['Rank']
        ee_constants = en_constants = dict()
        if e_rank > 1 and term.get('e-e cutoff'):
            ee_constants = self.get(term, 'e-e cutoff', 'Constants')
        if n_rank > 0 and term.get('e-n cutoff'):
            en_constants = self.get(term, 'e-n cutoff', 'Constants')
        return ee_constants, en_constants

    def get_basis_parameters(self, term):
        """Load basis parameters into 1-dimensional array.
        """
        e_rank, n_rank = term['Rank']
        ee_parameters = en_parameters = list()
        if e_rank > 1:
            if parameters := self.get(term, 'e-e basis', 'Parameters'):
                for channel in parameters:
                    en_parameters.append(self.get(term, 'e-e basis', 'Parameters', channel, 'L')[0])
        if n_rank > 0 and term.get('e-n basis'):
            if parameters := self.get(term, 'e-n basis', 'Parameters'):
                for channel in parameters:
                    en_parameters.append(self.get(term, 'e-n basis', 'Parameters', channel, 'L')[0])
        return np.array(ee_parameters, np.float), np.array(en_parameters, np.float)

    def get_cutoff_parameters(self, term):
        """Load cutoff parameters into 1-dimensional array.
        """
        e_rank, n_rank = term['Rank']
        ee_parameters = en_parameters = list()
        if e_rank > 1 and term.get('e-e cutoff'):
            if parameters := self.get(term, 'e-e cutoff', 'Parameters'):
                for channel in parameters:
                    ee_parameters.append(self.get(term, 'e-e cutoff', 'Parameters', channel, 'L')[0])
        if n_rank > 0 and term.get('e-n cutoff'):
            if parameters := self.get(term, 'e-n cutoff', 'Parameters'):
                for channel in parameters:
                    en_parameters.append(self.get(term, 'e-n cutoff', 'Parameters', channel, 'L')[0])
        return np.array(ee_parameters, np.float), np.array(en_parameters, np.float)

    def get_linear_parameters(self, term):
        """Load linear parameters into multidimensional array.
        """
        e_rank, n_rank = term['Rank']
        linear_parameters = term['Linear parameters']
        dims = [len(linear_parameters)]
        if e_rank > 1:
            e_order = self.get(term, 'e-e basis')['Order']
            dims += [e_order] * (e_rank * (e_rank-1) // 2)
        if n_rank > 0:
            n_order = self.get(term, 'e-n basis')['Order']
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
                ee_constants, en_constants = self.get_constants(term)
                self.ee_constants = dict_to_typed_dict(ee_constants)
                self.en_constants = dict_to_typed_dict(en_constants)
                self.ee_basis_parameters, self.en_basis_parameters = self.get_basis_parameters(term)
                self.ee_cutoff_parameters, self.en_cutoff_parameters = self.get_cutoff_parameters(term)
                self.linear_parameters = self.get_linear_parameters(term)
                for i, channel in enumerate(term['Linear parameters']):
                    ch1, ch2 = channel[8:].split('-')
                    G = 1/4 if ch1 == ch2 else 1/2
                    if self.linear_parameters[i, 0]:
                        continue
                    C = self.ee_cutoff_parameters[i] / self.ee_constants['C']
                    if self.ee_cutoff_type == 'polynomial':
                        self.linear_parameters[i, 0] = C * (self.linear_parameters[i, 1] - G)
                    elif self.ee_cutoff_type == 'alt polynomial':
                        self.linear_parameters[i, 0] = C * (self.linear_parameters[i, 1] - G/(-self.ee_cutoff_parameters[i])**self.ee_constants['C'])
                self.e_permutation = np.zeros((0,))


if __name__ == '__main__':
    """
    """
    path = 'parameters.casl'

    Gjastrow(path)
