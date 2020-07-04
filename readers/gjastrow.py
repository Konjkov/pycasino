

import numpy as np
from numpy.polynomial.polynomial import polyval, polyval2d, polyval3d
import sys
from math import exp
from yaml import safe_load

NONE = 0
NATURAL_POWER = 1


class Jastrow:
    """Jastrow reader from file.
    Framework for constructing generic Jastrow correlation factors
    P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
    Phys. Rev. E 86, 036703 – Published 27 September 2012
    """

    def __init__(self, file):
        with open(file, 'r') as f:
            self._jastrow_data = safe_load(f)['JASTROW']

        _terms = []
        for key, term in self._jastrow_data.items():
            if key.startswith('TERM'):
                for channel in self.channels(term):
                    prefix, suffix = channel.split(' ')
                    channels = [int(''.join(filter(str.isdigit, channel))) for channel in suffix.split('-')]
                    channels += [0] * (3 - len(channels))
                    e_rank, n_rank = self.rank(term)
                    _terms.append((
                        e_rank,
                        n_rank,
                        channels,
                        self.e_basis_type(term),
                        self.e_basis_order(term),
                        self.e_cusp(term),
                        self.n_basis_type(term),
                        self.n_basis_order(term),
                        self.n_cutoff_type(term),
                        self.n_cutoff_constant(term),
                        self.n_cutoff_parameters(term),
                        # self.linear_parameters(term, channel),
                    ))
        print(_terms)
        self.terms = np.array(_terms, dtype=[
            ('e_rank', np.int),
            ('n_rank', np.int),
            ('channels', np.int, (3,)),
            ('e_basis_type', np.int),
            ('e_basis_order', np.int),
            ('e_cusp', np.bool),
            ('n_basis_type', np.int),
            ('n_basis_order', np.int),
            ('n_cutoff_type', np.int),
            ('n_cutoff_constant', np.int),
            ('n_cutoff_parameters', np.int),
            # ('linear_parameters', np.int, (8, 8, 0)),
        ])

    def term(self, term_num):
        try:
            return self._jastrow_data['TERM {}'.format(term_num)]
        except KeyError:
            print('TERM {} not found in the input file'.format(term_num))

    def channels(self, term):
        return term['Linear parameters'].keys()

    def rank(self, term):
        """"""
        if isinstance(term['Rank'], list):
            return term['Rank']
        elif isinstance(term['Rank'], str):
            return list(map(int, term['Rank'].split()))

    def type(self, term):
        try:
            return term['Type']
        except TypeError:
            return term[0]['Type']

    def order(self, term):
        try:
            return term['Order']
        except TypeError:
            return term[1]['Order']

    def rules(self, term):
        if isinstance(term['Rules'], list):
            return term['Rules']
        elif isinstance(term['Rules'], str):
            return list(map(int, term['Rules'].split()))

    def C(self, term):
        try:
            return term['C']
        except TypeError:
            return term[0]['C']

    def e_basis_type(self, term):
        if term.get('e-e basis'):
            if self.type(term['e-e basis']) == 'natural power':
                return NATURAL_POWER
        else:
            return NONE

    def n_basis_type(self, term):
        if term.get('e-n basis'):
            if self.type(term['e-n basis']) == 'natural power':
                return NATURAL_POWER
        else:
            return NONE

    def n_cutoff_type(self, term):
        if term.get('e-n cutoff'):
            if self.type(term['e-n cutoff']) == 'polynomial':
                return NATURAL_POWER
            elif self.type(term['e-n cutoff']) == 'anisotropic polynomial':
                return NATURAL_POWER
            elif self.type(term['e-n cutoff']) == 'alt polynomial':
                return NATURAL_POWER
            elif self.type(term['e-n cutoff']) == 'gaussian':
                return NATURAL_POWER
        else:
            return NONE

    def n_cutoff_constant(self, term):
        if term.get('e-n cutoff'):
            return self.C(term['e-n cutoff']['Constants'])
        else:
            return 0

    def e_basis_order(self, term):
        if term.get('e-e basis'):
            return self.order(term['e-e basis'])
        else:
            return 0

    def n_basis_order(self, term):
        if term.get('e-n basis'):
            return self.order(term['e-n basis'])
        else:
            return 0

    def e_cusp(self, term):
        return term.get('e-e cusp') == 'T'

    def linear_parameters(self, term, channel):
        """Load linear parameters into multidimensional array."""
        e_rank, n_rank = self.rank(term)
        dims = []
        if e_rank > 1:
            e_order = self.e_basis_order(term)
            dims += [e_order] * (e_rank * (e_rank-1) // 2)
        if n_rank > 0:
            n_order = self.n_basis_order(term)
            dims += [n_order] * e_rank * n_rank

        linear_parameters = np.zeros(dims, 'd')
        for key, val in term['Linear parameters'][channel].items():
            index = map(lambda x: x-1, map(int, key.split('_')[1].split(',')))
            linear_parameters[tuple(index)] = val[0]
        return linear_parameters

    def basis(self, term, channel):
        """type of functional bases should be:
        natural_power
        cosine
        cosine with k-cutoff
        r/(r^b+a) power
        1/(r+a) power
        r/(r+a) power
        """
        if self.type(term) == 'natural power':
            return lambda r: r
        elif self.type(term) == 'r/(r^b+a) power':
            parameters = term['Parameters']
            a = parameters[channel]['a'][0]
            b = parameters[channel]['b'][0]
            return lambda r: r/(r**b+a)
        elif self.type(term) == '1/(r+a) power':
            parameters = term['Parameters']
            a = parameters[channel]['a'][0]
            return lambda r: 1/(r+a)
        elif self.type(term) == 'r/(r+a) power':
            parameters = term['Parameters']
            a = parameters[channel]['a'][0]
            return lambda r: r/(r+a)
        else:
            print('basis with a {} type is not supported'.format(self.type(term)))
            sys.exit(0)

    def cutoff(self, term, channel):
        """type of cutoff functions should be:
        polynomial
        alt polynomial
        gaussian
        anisotropic polynomial
        """
        if term is None:
            return lambda r: 1.0
        elif self.type(term) in ('polynomial', 'anisotropic polynomial'):
            C = self.C(term['Constants'])
            L = term['Parameters'][channel]['L'][0]
            return lambda r: (1-r/L) ** C * np.heaviside(L-r, 0.0)
        elif self.type(term) == 'alt polynomial':
            C = self.C(term['Constants'])
            L = term['Parameters'][channel]['L'][0]
            return lambda r: (r-L) ** C * np.heaviside(L-r, 0.0)
        elif self.type(term) == 'gaussian':
            L_hard = term['Parameters'][channel]['L_hard'][0]
            return lambda r: exp(-(r/L)**2) * np.heaviside(L_hard-r/L, 0.0)
        else:
            print('cutoff with {} type is not supported'.format(self.type(term)))
            sys.exit(0)

    def cutoff_channel(self, term, channel, i, j):
        """make cutoff Parameters channel from Linear parameters channel"""
        prefix, suffix = channel.split(' ')
        split_suffix = suffix.split('-')
        if self.rank(term) == [1, 2]:
            return prefix + ' ' + split_suffix[i] + '-' + split_suffix[1]
        else:
            return prefix + ' ' + split_suffix[i] + '-' + split_suffix[j]

    def jastrow(self, term, channel, *args):
        u"""JASTROW
        :param term: jastrow term
        :param channel:
        :param args: [ri1, ... rin, rI1, ...rIm]
        :return:
        """
        e_rank, n_rank = self.rank(term)
        p_args = []
        cutoff = 1.0
        if e_rank > 1:
            ee_basis = self.basis(term['e-e basis'], channel)
            for i, ri, in enumerate(args[:e_rank]):
                for j, rj in enumerate(args[:e_rank]):
                    if i < j:
                        rij = np.linalg.norm(ri - rj)
                        p_args.append(ee_basis(rij))
                        ee_cutoff = self.cutoff(term.get('e-e cutoff'), self.cutoff_channel(term, channel, i, j))
                        cutoff *= ee_cutoff(rij)

        if n_rank > 0:
            en_basis = self.basis(term['e-n basis'], channel)
            for i, ri in enumerate(args[:e_rank]):
                for j, rI in enumerate(args[e_rank:], e_rank):
                    riI = np.linalg.norm(ri - rI)
                    p_args.append(en_basis(riI))
                    en_cutoff = self.cutoff(term.get('e-n cutoff'), self.cutoff_channel(term, channel, i, j))
                    cutoff *= en_cutoff(riI)

        if len(p_args) == 1:
            result = polyval(*p_args, self.linear_parameters(term, channel))
        elif len(p_args) == 2:
            result = polyval2d(*p_args, self.linear_parameters(term, channel))
        elif len(p_args) == 3:
            result = polyval3d(*p_args, self.linear_parameters(term, channel))
        return np.exp(result, self.linear_parameters(term, channel) * cutoff)
