import os
import re

import numpy as np

geminal_template = """\
GEMINAL:
  Default g optimizability: fixed
  Default c optimizability: fixed
{geminals}"""

geminal_block_template = """\
  Geminal {n}:
    Parameters:
{parameters}
"""

parameter_re = re.compile(r'^([cgu][_\d,]*)\s*:\s*\[\s*([^,\]]+?)\s*(?:,\s*([a-z]+)\s*)?\]')
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
        file_path = os.path.join(base_path, 'parameters.casl')
        if not os.path.isfile(file_path):
            return
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # locate the GEMINAL top-level block
        start = None
        for i, line in enumerate(lines):
            if line.rstrip() == 'GEMINAL:':
                start = i + 1
                break
        if start is None:
            return
        geminals = []
        constraints = []
        g_default = c_default = False
        current = None
        for line in lines[start:]:
            if line and not line[0].isspace():
                break
            stripped = line.strip()
            if stripped.startswith('Default g optimizability'):
                g_default = stripped.split(':')[-1].strip() == 'optimizable'
            elif stripped.startswith('Default c optimizability'):
                c_default = stripped.split(':')[-1].strip() == 'optimizable'
            elif stripped.startswith('Geminal '):
                current = {'c': (1.0, None), 'g': {}, 'u': {}}
                geminals.append(current)
            elif stripped.startswith('Constraints'):
                current = None
            elif current is None:
                if '=' in stripped:
                    constraints.append(stripped)
            else:
                match = parameter_re.match(stripped)
                if match is None:
                    continue
                key, value, flag = match.group(1), float(match.group(2)), match.group(3)
                if key == 'c':
                    current['c'] = (value, flag)
                elif key.startswith('g_'):
                    row, col = (int(x) - 1 for x in key[2:].split(','))
                    current['g'][min(row, col), max(row, col)] = (value, flag)
                elif key.startswith('u_'):
                    row, col = (int(x) - 1 for x in key[2:].split(','))
                    current['u'][row, col] = (value, flag)
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
        blocks = []
        # a determined parameter is regenerated from its reference on every read, and casino
        # errstops if it is declared as well, so it is left out of the Parameters block
        c_determined = {tie[0] for tie in self.c_ties}
        g_determined = {tuple(tie[:3]) for tie in self.g_ties}
        u_determined = {tuple(tie[:3]) for tie in self.u_ties}
        for n in range(self.c.size):
            parameters = []
            if n not in c_determined:
                parameters.append(f'      c: [ {self.c[n]: .8e}, {"optimizable" if self.c_mask[n] else "fixed"} ]')
            for row in range(self.norb):
                for col in range(row, self.norb):
                    if (self.g[n, row, col] != 0.0 or self.g_mask[n, row, col]) and (n, row, col) not in g_determined:
                        flag = 'optimizable' if self.g_mask[n, row, col] else 'fixed'
                        parameters.append(f'      g_{row + 1},{col + 1}: [ {self.g[n, row, col]: .8e}, {flag} ]')
            for col in range(self.nunpaired):
                for row in range(self.norb):
                    if (self.u[n, row, col] != 0.0 or self.u_mask[n, row, col]) and (n, row, col) not in u_determined:
                        flag = 'optimizable' if self.u_mask[n, row, col] else 'fixed'
                        parameters.append(f'      u_{row + 1},{col + 1}: [ {self.u[n, row, col]: .8e}, {flag} ]')
            blocks.append(geminal_block_template.format(n=n + 1, parameters='\n'.join(parameters)))
        res = geminal_template.format(geminals=''.join(blocks))
        if self.constraints:
            res += '  Constraints:\n' + ''.join(f'    {line}\n' for line in self.constraints)
        return res
