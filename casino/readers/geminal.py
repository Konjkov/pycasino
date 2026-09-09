import re

import numpy as np

from casino.readers import casl

constraint_re = re.compile(r'^(\d+)\^([cgu])(?:_(\d+),(\d+))?$')


class Geminal:
    """Multi-geminal (MAGP/AGP) pairing wave function.

    Psi = sum_n c_n det[M_n], with the neu x neu matrix
        M_n[i, j]        = sum_{p,q} phi_p(r_i^up) g_n[p, q] phi_q(r_j^down)   (j < ned)
        M_n[i, ned + k]  = sum_p phi_p(r_i^up) u_n[p, k]                       (unpaired)
    """

    def __init__(self, neu, ned):
        self.title = 'no title given'
        self.neu = neu
        self.ned = ned
        self.nunpaired = neu - ned
        self.norb = neu
        # Hartree-Fock default: one geminal, diagonal g on the paired orbitals,
        # one unpaired column per singly occupied orbital.
        g = np.zeros(shape=(1, neu, neu))
        u = np.zeros(shape=(1, neu, self.nunpaired))
        for i in range(ned):
            g[0, i, i] = 1.0
        for k in range(self.nunpaired):
            u[0, ned + k, k] = 1.0
        self.c = np.ones(1)
        self.g = g
        self.u = u
        self.g_mask = np.zeros(shape=g.shape, dtype=bool)
        self.u_mask = np.zeros(shape=u.shape, dtype=bool)
        self.c_mask = np.zeros(shape=1, dtype=bool)
        self.g_available = np.zeros(shape=g.shape, dtype=bool)
        self.u_available = np.zeros(shape=u.shape, dtype=bool)
        for i in range(ned):
            self.g_available[0, i, i] = True
        for k in range(self.nunpaired):
            self.u_available[0, ned + k, k] = True
        self.constraints = []
        self.c_ties = np.zeros(shape=(0, 2), dtype=int)
        self.g_ties = np.zeros(shape=(0, 6), dtype=int)
        self.u_ties = np.zeros(shape=(0, 6), dtype=int)

    def read(self, base_path):
        tree = casl.read(base_path)
        block = tree and tree.item('GEMINAL')
        if not block:
            return
        g_default = block.item('Default g optimizability') == 'optimizable'
        c_default = block.item('Default c optimizability') == 'optimizable'
        geminals = []
        constraints = []
        for name, item in block.items():
            if casl.unique(name).startswith('geminal'):
                geminals.append(self.parameters(item.item('Parameters')))
            elif casl.unique(name) == 'constraints':
                constraints = item.implicit()
        if not geminals:
            return
        groups = [self.parse_constraint(line) for line in constraints]
        norb = self.neu
        for gem in geminals:
            for row, col in gem['g']:
                norb = max(norb, row + 1, col + 1)
            for row, col in gem['u']:
                norb = max(norb, row + 1)
        for group in groups:
            for key in group:
                if key[0] == 'g':
                    norb = max(norb, key[2] + 1, key[3] + 1)
                elif key[0] == 'u':
                    norb = max(norb, key[2] + 1)
        n_gem = len(geminals)
        self.norb = norb
        self.constraints = constraints
        self.c = np.array([gem['c'][0] for gem in geminals])
        self.g = np.zeros(shape=(n_gem, norb, norb))
        self.u = np.zeros(shape=(n_gem, norb, self.nunpaired))
        self.c_mask = np.array([self.optimizable(gem['c'][1], c_default) for gem in geminals])
        self.g_mask = np.zeros(shape=self.g.shape, dtype=bool)
        self.u_mask = np.zeros(shape=self.u.shape, dtype=bool)
        self.g_available = np.zeros(shape=self.g.shape, dtype=bool)
        self.u_available = np.zeros(shape=self.u.shape, dtype=bool)
        for n, gem in enumerate(geminals):
            for (row, col), (value, flag) in gem['g'].items():
                self.g[n, row, col] = self.g[n, col, row] = value
                self.g_mask[n, row, col] = self.g_mask[n, col, row] = self.optimizable(flag, g_default)
                self.g_available[n, row, col] = self.g_available[n, col, row] = True
            for (row, col), (value, flag) in gem['u'].items():
                self.u[n, row, col] = value
                self.u_mask[n, row, col] = self.optimizable(flag, g_default)
                self.u_available[n, row, col] = True
        self.apply_constraints(groups, geminals, g_default, c_default)

    @staticmethod
    def parameters(block):
        """The c, g and u parameters of one geminal, each as its value and the optimizability flag
        it was given, None if it was given none.
        """
        geminal = {'c': (1.0, None), 'g': {}, 'u': {}}
        for name, item in (block or {}).items():
            values = item.implicit() if isinstance(item, casl.Block) else [item]
            value = casl.to_float(values[0])
            flag = values[1] if len(values) > 1 else None
            if casl.unique(name) == 'c':
                geminal['c'] = (value, flag)
            elif name[0] in 'gu':
                row, col = (int(x) - 1 for x in name[2:].split(','))
                if name[0] == 'g':
                    row, col = min(row, col), max(row, col)
                geminal[name[0]][row, col] = (value, flag)
        return geminal

    @staticmethod
    def optimizable(flag, default):
        """Optimizability of a parameter, the block default being what a flagless one takes."""
        if flag is None:
            return default
        return flag == 'optimizable'

    @staticmethod
    def parse_constraint(line):
        """One tied group, as the list of the parameters it equates. Off-diagonal g elements are
        canonicalized to the upper triangle, where the matrix is stored.
        """
        group = []
        for item in line.split('='):
            match = constraint_re.match(item.strip())
            if match is None:
                raise ValueError(f'unrecognized geminal constraint: {line}')
            n, kind = int(match.group(1)) - 1, match.group(2)
            if kind == 'c':
                key = ('c', n)
            else:
                row, col = int(match.group(3)) - 1, int(match.group(4)) - 1
                if kind == 'g':
                    row, col = min(row, col), max(row, col)
                key = (kind, n, row, col)
            if key not in group:
                group.append(key)
        return group

    def apply_constraints(self, groups, geminals, g_default, c_default):
        """Tie every group to the one member that carries an explicit optimizability flag: its
        value and flag are copied onto the rest, which stop being independent parameters and are
        restored from it after every parameter update.
        """
        values = {'c': self.c, 'g': self.g, 'u': self.u}
        masks = {'c': self.c_mask, 'g': self.g_mask, 'u': self.u_mask}
        available = {'g': self.g_available, 'u': self.u_available}
        ties = {'c': [], 'g': [], 'u': []}
        for group in groups:
            declared = [key for key in group if self.declared(key, geminals)]
            if len(declared) > 1:
                raise ValueError(f'more than one declared parameter in the geminal constraint: {declared}')
            reference = declared[0] if declared else group[0]
            kind = reference[0]
            value = values[kind][reference[1:]]
            optimizable = masks[kind][reference[1:]] if declared else (c_default if kind == 'c' else g_default)
            for key in group:
                values[key[0]][key[1:]] = value
                masks[key[0]][key[1:]] = optimizable and key == reference
                if key[0] != 'c':
                    available[key[0]][key[1:]] = True
                if key[0] == 'g':
                    values['g'][key[1], key[3], key[2]] = value
                    masks['g'][key[1], key[3], key[2]] = masks['g'][key[1:]]
                    available['g'][key[1], key[3], key[2]] = True
                if key != reference:
                    ties[kind].append(list(key[1:]) + list(reference[1:]))
        self.c_ties = np.array(ties['c'], dtype=int).reshape(-1, 2)
        self.g_ties = np.array(ties['g'], dtype=int).reshape(-1, 6)
        self.u_ties = np.array(ties['u'], dtype=int).reshape(-1, 6)

    @staticmethod
    def declared(key, geminals):
        """A group member is the reference of its group if it was given an explicit flag in a
        Parameters block. Every other member is determined and must be left undeclared.
        """
        gem = geminals[key[1]]
        if key[0] == 'c':
            return gem['c'][1] is not None
        return gem[key[0]].get(key[2:], (0.0, None))[1] is not None

    def write(self):
        block = casl.Block()
        block.add('Default g optimizability', 'fixed')
        block.add('Default c optimizability', 'fixed')
        # a determined parameter is regenerated from its reference on every read, and casino
        # errstops if it is declared as well, so it is left out of the Parameters block
        c_determined = {tie[0] for tie in self.c_ties}
        g_determined = {tuple(tie[:3]) for tie in self.g_ties}
        u_determined = {tuple(tie[:3]) for tie in self.u_ties}
        for n in range(self.c.size):
            parameters = casl.Block()
            if n not in c_determined:
                parameters.add('c', self.parameter(self.c[n], self.c_mask[n]))
            for row in range(self.norb):
                for col in range(row, self.norb):
                    if (self.g[n, row, col] != 0.0 or self.g_mask[n, row, col]) and (n, row, col) not in g_determined:
                        parameters.add(f'g_{row + 1},{col + 1}', self.parameter(self.g[n, row, col], self.g_mask[n, row, col]))
            for col in range(self.nunpaired):
                for row in range(self.norb):
                    if (self.u[n, row, col] != 0.0 or self.u_mask[n, row, col]) and (n, row, col) not in u_determined:
                        parameters.add(f'u_{row + 1},{col + 1}', self.parameter(self.u[n, row, col], self.u_mask[n, row, col]))
            block.add(f'Geminal {n + 1}', casl.Block([('Parameters', parameters)]))
        if self.constraints:
            block.add('Constraints', casl.Block([(f'%u{i}', line) for i, line in enumerate(self.constraints, 1)]))
        return casl.dumps(casl.Block([('GEMINAL', block)]))

    @staticmethod
    def parameter(value, optimizable):
        """One parameter as casino writes it, value and optimizability flag."""
        return casl.inline(f'{value: .8e}', 'optimizable' if optimizable else 'fixed')
