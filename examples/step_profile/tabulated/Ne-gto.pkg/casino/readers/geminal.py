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
        current = None
        for line in lines[start:]:
            if line and not line[0].isspace():
                break
            stripped = line.strip()
            if stripped.startswith('Geminal '):
                current = {'c': 1.0, 'g': {}, 'u': {}, 'c_opt': False, 'g_opt': {}, 'u_opt': {}}
                geminals.append(current)
            elif stripped.startswith('Constraints'):
                current = None
            elif current is not None:
                match = parameter_re.match(stripped)
                if match is None:
                    continue
                key, value, flag = match.group(1), float(match.group(2)), match.group(3)
                optimizable = flag == 'optimizable'
                if key == 'c':
                    current['c'] = value
                    current['c_opt'] = optimizable
                elif key.startswith('g_'):
                    row, col = (int(x) - 1 for x in key[2:].split(','))
                    current['g'][row, col] = value
                    current['g_opt'][row, col] = optimizable
                elif key.startswith('u_'):
                    row, col = (int(x) - 1 for x in key[2:].split(','))
                    current['u'][row, col] = value
                    current['u_opt'][row, col] = optimizable
        if not geminals:
            return
        norb = self.neu
        for gem in geminals:
            for row, col in gem['g']:
                norb = max(norb, row + 1, col + 1)
            for row, col in gem['u']:
                norb = max(norb, row + 1)
        n_gem = len(geminals)
        self.norb = norb
        self.c = np.array([gem['c'] for gem in geminals])
        self.g = np.zeros(shape=(n_gem, norb, norb))
        self.u = np.zeros(shape=(n_gem, norb, self.nunpaired))
        self.g_mask = np.zeros(shape=self.g.shape, dtype=bool)
        self.u_mask = np.zeros(shape=self.u.shape, dtype=bool)
        self.c_mask = np.array([gem['c_opt'] for gem in geminals])
        for n, gem in enumerate(geminals):
            for (row, col), value in gem['g'].items():
                self.g[n, row, col] = self.g[n, col, row] = value
                self.g_mask[n, row, col] = self.g_mask[n, col, row] = gem['g_opt'][row, col]
            for (row, col), value in gem['u'].items():
                self.u[n, row, col] = value
                self.u_mask[n, row, col] = gem['u_opt'][row, col]

    def write(self):
        blocks = []
        for n in range(self.c.size):
            parameters = [f'      c: [ {self.c[n]: .8e}, {"optimizable" if self.c_mask[n] else "fixed"} ]']
            for row in range(self.norb):
                for col in range(row, self.norb):
                    if self.g[n, row, col] != 0.0 or self.g_mask[n, row, col]:
                        flag = 'optimizable' if self.g_mask[n, row, col] else 'fixed'
                        parameters.append(f'      g_{row + 1},{col + 1}: [ {self.g[n, row, col]: .8e}, {flag} ]')
            for col in range(self.nunpaired):
                for row in range(self.norb):
                    if self.u[n, row, col] != 0.0 or self.u_mask[n, row, col]:
                        flag = 'optimizable' if self.u_mask[n, row, col] else 'fixed'
                        parameters.append(f'      u_{row + 1},{col + 1}: [ {self.u[n, row, col]: .8e}, {flag} ]')
            blocks.append(geminal_block_template.format(n=n + 1, parameters='\n'.join(parameters)))
        return geminal_template.format(geminals=''.join(blocks))
