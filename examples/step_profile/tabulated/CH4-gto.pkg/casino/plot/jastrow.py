#!/usr/bin/env python3

import numpy as np
from numpy.polynomial.polynomial import polyval, polyval2d, polyval3d
import sys
from itertools import combinations
from math import exp
from yaml import safe_load


class Jastrow:
    """Jastrow reader from file.
    Framework for constructing generic Jastrow correlation factors
    P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
    Phys. Rev. E 86, 036703 – Published 27 September 2012
    """

    def __init__(self, term_num, file):
        self.file = file
        self.term_num = term_num
        self.jastrow_data = self.read_input_file()
        self.channels = self.term['Linear parameters'].keys()

    def read_input_file(self):
        with open(self.file, 'r') as f:
            casl = safe_load(f)
            return casl['JASTROW']

    @property
    def term(self):
        try:
            return self.jastrow_data['TERM {}'.format(self.term_num)]
        except KeyError:
            print('TERM {} not found in the input file'.format(self.term_num))

    def rank(self, term):
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

    def L_hard(self, term):
        try:
            return term['L_hard']
        except TypeError:
            return term[0]['L_hard']

    def ee_cusp(self, term):
        return term.get('e-e cusp') == 'T'

    def parameters(self, term, channel):
        return term['Parameters'][channel]

    def linear_parameters(self, term, channel):
        """Load linear parameters into multidimensional array."""
        e_rank, n_rank = self.rank(term)
        dims = []
        if e_rank > 1:
            e_order = self.order(term['e-e basis'])
            dims += [e_order] * (e_rank * (e_rank-1) // 2)
        if n_rank > 0:
            n_order = self.order(term['e-n basis'])
            dims += [n_order] * e_rank * n_rank

        linear_parameters = np.zeros(dims, 'd')
        for key, val in term['Linear parameters'][channel].items():
            index = map(lambda x: x-1, map(int, key.split('_')[1].split(',')))
            linear_parameters[tuple(index)] = val[0]
        return linear_parameters

    def cusp(self):
        return lambda r: 1/r

    def basis(self, term, channel):
        """type of functional bases should be:
        natural power
        cosine
        cosine with k-cutoff
        r/(r^b+a) power
        1/(r+a) power
        r/(r+a) power
        natural polynomial
        """
        if self.type(term) == 'natural power':
            return lambda r: r
        elif self.type(term) == 'r/(r^b+a) power':
            a = self.parameters(term, channel)['a'][0]
            b = self.parameters(term, channel)['b'][0]
            return lambda r: r/(r**b+a)
        elif self.type(term) == '1/(r+a) power':
            a = self.parameters(term, channel)['a'][0]
            return lambda r: 1/(r+a)
        elif self.type(term) == 'r/(r+a) power':
            a = self.parameters(term, channel)['a'][0]
            return lambda r: r/(r+a)
        else:
            print('basis with a {} type is not supported'.format(self.type(term)))
            sys.exit(0)

    def cutoff(self, term, channel):
        """type of cutoff functions should be:
        polynomial
        alt polynomial
        gaussian
        spline
        anisotropic polynomial
        """
        if term is None:
            return lambda r: 1.0
        elif self.type(term) in ('polynomial', 'anisotropic polynomial'):
            C = self.C(term['Constants'])
            L = self.parameters(term, channel)['L'][0]
            return lambda r: (1-r/L) ** C * np.heaviside(L-r, 0.0)
        elif self.type(term) == 'alt polynomial':
            C = self.C(term['Constants'])
            L = self.parameters(term, channel)['L'][0]
            return lambda r: (r-L) ** C * np.heaviside(L-r, 0.0)
        elif self.type(term) == 'gaussian':
            L_hard = self.L_hard(term['Constants'])
            L = self.parameters(term, channel)['L'][0]
            return lambda r: np.exp(-(r/L)**2) * np.heaviside(L_hard-r/L, 0.0)
        elif self.type(term) == 'spline':
            print('cutoff with {} type is not supported'.format(self.type(term)))
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
        :param rank: jastrow rank
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
                        rij = np.hypot(*(ri - rj))
                        p_args.append(ee_basis(rij))
                        ee_cutoff = self.cutoff(
                            term.get('e-e cutoff'),
                            self.cutoff_channel(term, channel, i, j)
                        )
                        cutoff *= ee_cutoff(rij)

        if n_rank > 0:
            en_basis = self.basis(term['e-n basis'], channel)
            for i, ri in enumerate(args[:e_rank]):
                for j, rI in enumerate(args[e_rank:], e_rank):
                    riI = np.hypot(*(ri - rI))
                    p_args.append(en_basis(riI))
                    en_cutoff = self.cutoff(
                        term.get('e-n cutoff'),
                        self.cutoff_channel(term, channel, i, j)
                    )
                    cutoff *= en_cutoff(riI)

        if len(p_args) == 1:
            basis = polyval(*p_args, self.linear_parameters(term, channel))
        elif len(p_args) == 2:
            basis = polyval2d(*p_args, self.linear_parameters(term, channel))
        elif len(p_args) == 3:
            basis = polyval3d(*p_args, self.linear_parameters(term, channel))
        return np.exp(basis * cutoff)
