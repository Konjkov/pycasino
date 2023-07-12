import os
import numpy as np
import numba as nb

from collections import ChainMap
from yaml import safe_load


def dict_to_typed_dict(_dict, _type=nb.types.float64):
    result = nb.typed.Dict.empty(nb.types.string, _type)
    for k, v in _dict.items():
        result[k] = v
    return result


shape_type = nb.types.ListType(nb.int64)
parameters_type = nb.types.DictType(nb.types.unicode_type, nb.types.float64)
linear_parameters_type = nb.float64[:]


class Gjastrow:
    """Jastrow reader from file.
    CASINO manual: 7.8 Wave function parameter file: parameters.casl

    Framework for constructing generic Jastrow correlation factors
    P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
    Phys. Rev. E 86, 036703
    """

    def __init__(self):
        """init method"""
        self.terms = []
        self.rank = np.zeros(shape=(0, 2))
        self.rules = []
        self.cusp = np.zeros(shape=(0, 2))
        self.ee_basis_type = nb.typed.List.empty_list(nb.types.unicode_type)
        self.en_basis_type = nb.typed.List.empty_list(nb.types.unicode_type)
        self.ee_cutoff_type = nb.typed.List.empty_list(nb.types.unicode_type)
        self.en_cutoff_type = nb.typed.List.empty_list(nb.types.unicode_type)
        self.ee_constants = nb.typed.List.empty_list(parameters_type)
        self.en_constants = nb.typed.List.empty_list(parameters_type)
        self.ee_basis_parameters = nb.typed.List.empty_list(parameters_type)
        self.en_basis_parameters = nb.typed.List.empty_list(parameters_type)
        self.ee_cutoff_parameters = nb.typed.List.empty_list(parameters_type)
        self.en_cutoff_parameters = nb.typed.List.empty_list(parameters_type)
        self.linear_parameters = nb.typed.List.empty_list(linear_parameters_type)
        self.linear_parameters_shape = nb.typed.List.empty_list(shape_type)

    def get(self, term, *args):
        def list_or_dict(node):
            return dict(ChainMap(*node)) if isinstance(node, list) else node
        res = term
        for arg in args:
            if res := res.get(arg):
                if arg not in ['a', 'b', 'L']:
                    res = list_or_dict(res)
            else:
                return ''
        return res

    def get_terms(self, jastrow_data):
        """Load terms."""
        self.terms = [term for key, term in jastrow_data.items() if key.startswith('TERM')]

    def get_rank(self):
        """Load terms rank."""
        self.rank = np.array([term['Rank'] for term in self.terms])

    def get_rules(self):
        """Load terms rules."""
        self.rules = [term['Rules'] or list() for term in self.terms]

    def get_cusp(self):
        """Load cusp."""
        self.cusp = np.array(
            [[term.get('e-e cusp') == 'T', term.get('e-n cusp') == 'T'] for term in self.terms]
        )

    def get_basis_type(self):
        """Load basis type."""
        for term in self.terms:
            e_rank, n_rank = term['Rank']
            self.ee_basis_type.append(self.get(term, 'e-e basis', 'Type') if e_rank > 1 else '')
            self.en_basis_type.append(self.get(term, 'e-n basis', 'Type') if n_rank > 0 else '')

    def get_cutoff_type(self):
        """Load cutoff type."""
        for term in self.terms:
            e_rank, n_rank = term['Rank']
            self.ee_cutoff_type.append(self.get(term, 'e-e cutoff', 'Type') if e_rank > 1 else '')
            self.en_cutoff_type.append(self.get(term, 'e-n cutoff', 'Type') if n_rank > 0 else '')

    def get_constants(self):
        """Load truncation constant.
        """
        for term in self.terms:
            e_rank, n_rank = term['Rank']
            if e_rank > 1 and term.get('e-e cutoff'):
                self.ee_constants.append(dict_to_typed_dict(self.get(term, 'e-e cutoff', 'Constants')))
            else:
                self.ee_constants.append(dict_to_typed_dict({}))
            if n_rank > 0 and term.get('e-n cutoff'):
                self.en_constants.append(dict_to_typed_dict(self.get(term, 'e-n cutoff', 'Constants')))
            else:
                self.en_constants.append(dict_to_typed_dict({}))

    def get_basis_parameters(self):
        """Load basis parameters into 1-dimensional array."""
        for term in self.terms:
            e_rank, n_rank = term['Rank']
            if e_rank > 1:
                for channel in self.get(term, 'e-e basis', 'Parameters') or []:
                    self.ee_basis_parameters.append(
                        dict_to_typed_dict(
                            {parameter: self.get(term, 'e-e basis', 'Parameters', channel, parameter)[0]
                             for parameter in self.get(term, 'e-e basis', 'Parameters', channel)}
                        )
                    )
            if n_rank > 0:
                for channel in self.get(term, 'e-n basis', 'Parameters') or []:
                    self.en_basis_parameters.append(
                        dict_to_typed_dict(
                            {parameter: self.get(term, 'e-n basis', 'Parameters', channel, parameter)[0]
                             for parameter in self.get(term, 'e-n basis', 'Parameters', channel)}
                        )
                    )

    def get_cutoff_parameters(self):
        """Load cutoff parameters into 1-dimensional array.
        """
        for term in self.terms:
            e_rank, n_rank = term['Rank']
            if e_rank > 1 and self.get(term, 'e-e cutoff'):
                for channel in self.get(term, 'e-e cutoff', 'Parameters') or []:
                    self.ee_cutoff_parameters.append(
                        dict_to_typed_dict(
                            {parameter: self.get(term, 'e-e cutoff', 'Parameters', channel, parameter)[0]
                             for parameter in self.get(term, 'e-e cutoff', 'Parameters', channel)}
                        )
                    )
            else:
                self.ee_cutoff_parameters.append(dict_to_typed_dict({}))
            if n_rank > 0 and self.get(term, 'e-n cutoff'):
                for channel in self.get(term, 'e-n cutoff', 'Parameters') or []:
                    self.en_cutoff_parameters.append(
                        dict_to_typed_dict(
                            {parameter: self.get(term, 'e-n cutoff', 'Parameters', channel, parameter)[0]
                             for parameter in self.get(term, 'e-n cutoff', 'Parameters', channel)}
                        )
                    )
            else:
                self.en_cutoff_parameters.append(dict_to_typed_dict({}))

    def get_linear_parameters(self):
        """Load linear parameters into multidimensional array."""
        for term in self.terms:
            e_rank, n_rank = term['Rank']
            linear_parameters_data = term['Linear parameters']
            channels = len(linear_parameters_data)
            shape = [channels]
            if e_rank > 1:
                e_order = self.get(term, 'e-e basis', 'Order')
                shape += [e_order] * (e_rank * (e_rank-1) // 2)
            if n_rank > 0:
                n_order = self.get(term, 'e-n basis', 'Order')
                shape += [n_order] * (e_rank * n_rank)

            self.linear_parameters_shape.append(nb.typed.List(shape))
            linear_parameters = np.zeros(shape=shape, dtype=float)
            for i, channel in enumerate(linear_parameters_data.values()):
                for key, val in channel.items():
                    j = tuple(map(lambda x: x-1, map(int, key.split('_')[1].split(','))))
                    linear_parameters[i, j] = val[0]
            self.linear_parameters.append(linear_parameters.ravel())

    def fix_terns(self):
        """Fix dependent parameters."""
        # FIXME: not works
        for i, term in enumerate(self.terms):
            for j, channel in enumerate(term['Linear parameters']):
                ch1, ch2 = channel[8:].split('-')
                G = 1/4 if ch1 == ch2 else 1/2
                if self.linear_parameters[j, 0]:
                    continue
                if self.ee_cutoff_type[i] == 'polynomial':
                    C = self.ee_cutoff_parameters[j]['L'] / self.ee_constants[i]['C']
                    self.linear_parameters[j, 0] = C * (self.linear_parameters[j, 1] - G)
                elif self.ee_cutoff_type[i] == 'alt polynomial':
                    C = self.ee_cutoff_parameters[j]['L'] / self.ee_constants[i]['C']
                    self.linear_parameters[j, 0] = C * (self.linear_parameters[j, 1] - G/(-self.ee_cutoff_parameters[j]['L'])**self.ee_constants[i]['C'])

    def read(self, base_path):
        file_path = os.path.join(base_path, 'parameters.casl')
        if not os.path.isfile(file_path):
            return

        with open(file_path, 'r') as f:
            jastrow_data = safe_load(f)['JASTROW']
            self.get_terms(jastrow_data)
            self.get_rank()
            self.get_rules()
            self.get_cusp()
            self.get_basis_type()
            self.get_cutoff_type()
            self.get_constants()
            self.get_basis_parameters()
            self.get_cutoff_parameters()
            self.get_linear_parameters()
            # self.fix_terns()
