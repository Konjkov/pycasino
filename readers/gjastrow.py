#!/usr/bin/env python3

from collections import ChainMap
from typing import Tuple, Dict

import numpy as np
import numba as nb
from yaml import safe_load


def dict_to_typed_dict(_dict, _type=nb.types.float64):
    result = nb.typed.Dict.empty(nb.types.string, _type)
    for k, v in _dict.items():
        result[k] = v
    return result


parameters_type = nb.types.DictType(nb.types.unicode_type, nb.types.float64)


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

    def get_terms(self):
        return [term for key, term in self._jastrow_data.items() if key.startswith('TERM')]

    def get_rank(self, terms):
        return (
            nb.typed.List([term['Rank'][0] for term in terms]),
            nb.typed.List([term['Rank'][1] for term in terms])
        )

    def get_rules(self, term):
        return term['Rules'] or list()

    def get_ee_cusp(self, term) -> bool:
        """Load e-e cusp.
        """
        e_rank, n_rank = term['Rank']
        ee_cusp = False
        if e_rank > 1:
            ee_cusp = term.get('e-e cusp') == 'T'
        return ee_cusp

    def get_basis_type(self, terms) -> Tuple[nb.typed.List, nb.typed.List]:
        """Load basis type.
        """
        ee_basis_type = nb.typed.List.empty_list(nb.types.unicode_type)
        en_basis_type = nb.typed.List.empty_list(nb.types.unicode_type)
        for term in terms:
            e_rank, n_rank = term['Rank']
            ee_basis_type.append(self.get(term, 'e-e basis')['Type'] if e_rank > 1 else '')
            en_basis_type.append(self.get(term, 'e-n basis')['Type'] if n_rank > 0 else '')
        return ee_basis_type, en_basis_type

    def get_cutoff_type(self, terms) -> Tuple[nb.typed.List, nb.typed.List]:
        """Load cutoff type.
        """
        e_cutoff_type = nb.typed.List.empty_list(nb.types.unicode_type)
        n_cutoff_type = nb.typed.List.empty_list(nb.types.unicode_type)
        for term in terms:
            e_rank, n_rank = term['Rank']
            e_cutoff_type.append(self.get(term, 'e-e cutoff')['Type'] if e_rank > 1 else '')
            n_cutoff_type.append(self.get(term, 'e-n cutoff')['Type'] if n_rank > 0 else '')
        return e_cutoff_type, n_cutoff_type

    def get_constants(self, terms) -> Tuple[nb.typed.List, nb.typed.List]:
        """Load truncation constant.
        """
        ee_constants = nb.typed.List.empty_list(parameters_type)
        en_constants = nb.typed.List.empty_list(parameters_type)
        for term in terms:
            e_rank, n_rank = term['Rank']
            if e_rank > 1 and term.get('e-e cutoff'):
                ee_constants.append(dict_to_typed_dict(self.get(term, 'e-e cutoff', 'Constants')))
            else:
                ee_constants.append(nb.typed.Dict.empty(nb.types.string, nb.types.float64))
            if n_rank > 0 and term.get('e-n cutoff'):
                en_constants.append(dict_to_typed_dict(self.get(term, 'e-n cutoff', 'Constants')))
            else:
                en_constants.append(nb.typed.Dict.empty(nb.types.string, nb.types.float64))
        return ee_constants, en_constants

    def get_basis_parameters(self, term) -> Tuple[nb.typed.List, nb.typed.List]:
        """Load basis parameters into 1-dimensional array.
        """
        e_rank, n_rank = term['Rank']
        ee_term_parameters = nb.typed.List.empty_list(parameters_type)
        en_term_parameters = nb.typed.List.empty_list(parameters_type)
        if e_rank > 1:
            for channel in self.get(term, 'e-e basis', 'Parameters') or []:
                ee_term_parameters.append(
                    dict_to_typed_dict(
                        {parameter: self.get(term, 'e-e basis', 'Parameters', channel, parameter)[0]
                         for parameter in self.get(term, 'e-e basis', 'Parameters', channel)}
                    )
                )
        if n_rank > 0:
            for channel in self.get(term, 'e-n basis', 'Parameters') or []:
                en_term_parameters.append(
                    dict_to_typed_dict(
                        {parameter: self.get(term, 'e-n basis', 'Parameters', channel, parameter)[0]
                         for parameter in self.get(term, 'e-n basis', 'Parameters', channel)}
                    )
                )
        return ee_term_parameters, en_term_parameters

    def get_cutoff_parameters(self, term) -> Tuple[nb.typed.List, nb.typed.List]:
        """Load cutoff parameters into 1-dimensional array.
        """
        e_rank, n_rank = term['Rank']
        ee_term_parameters = nb.typed.List.empty_list(parameters_type)
        en_term_parameters = nb.typed.List.empty_list(parameters_type)
        if e_rank > 1:
            for channel in self.get(term, 'e-e cutoff', 'Parameters') or []:
                ee_term_parameters.append(
                    dict_to_typed_dict(
                        {parameter: self.get(term, 'e-e cutoff', 'Parameters', channel, parameter)[0]
                         for parameter in self.get(term, 'e-e cutoff', 'Parameters', channel)}
                    )
                )
        if n_rank > 0:
            for channel in self.get(term, 'e-n cutoff', 'Parameters') or []:
                en_term_parameters.append(
                    dict_to_typed_dict(
                        {parameter: self.get(term, 'e-e cutoff', 'Parameters', channel, parameter)[0]
                         for parameter in self.get(term, 'e-e cutoff', 'Parameters', channel)}
                    )
                )
        return ee_term_parameters, en_term_parameters

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

        terms = self.get_terms()

        self.e_rank, self.n_rank = self.get_rank(terms)
        self.ee_basis_type, self.en_basis_type = self.get_basis_type(terms)
        self.ee_cutoff_type, self.en_cutoff_type = self.get_cutoff_type(terms)
        self.ee_constants, self.en_constants = self.get_constants(terms)

        for i, term in enumerate(terms):
            self.rules = self.get_rules(term)
            self.ee_cusp = self.get_ee_cusp(term)
            self.ee_basis_parameters, self.en_basis_parameters = self.get_basis_parameters(term)
            self.ee_cutoff_parameters, self.en_cutoff_parameters = self.get_cutoff_parameters(term)
            self.linear_parameters = self.get_linear_parameters(term)
            for j, channel in enumerate(term['Linear parameters']):
                ch1, ch2 = channel[8:].split('-')
                G = 1/4 if ch1 == ch2 else 1/2
                if self.linear_parameters[j, 0]:
                    continue
                C = self.ee_cutoff_parameters[j]['L'] / self.ee_constants[i]['C']
                if self.ee_cutoff_type[0] == 'polynomial':
                    self.linear_parameters[j, 0] = C * (self.linear_parameters[j, 1] - G)
                elif self.ee_cutoff_type[0] == 'alt polynomial':
                    self.linear_parameters[j, 0] = C * (self.linear_parameters[j, 1] - G/(-self.ee_cutoff_parameters[j]['L'])**self.ee_constants[i]['C'])
            # self.e_permutation = np.zeros((0,))
