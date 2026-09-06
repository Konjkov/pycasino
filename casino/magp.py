"""Build a MAGP parameters.casl from a correlated natural orbital spectrum.

The geminal expansion is fixed by the natural orbital occupations alone:

    G1        reference, g_m,m = 1 on the doubly occupied orbitals,
              one u column per singly occupied orbital
    G2(m)     doubly occupied orbitals except m, plus an optimizable g on
              every degenerate group of virtual natural orbitals
    G3(m)     the same without the core, c = -1, cancelling the terms in
              which the core pair is emptied
    G4(s,m)   the u column of the singly occupied orbital s moved onto the
              doubly occupied orbital m, the pair taking g_a,b and g_b,a on
              every pair of orbitals whose representations multiply to that
              of the m -> s replacement
    G5(s,m)   its c = -1 partner

Moving the u column alone would only rotate the singly occupied orbital,
which is a one-body freedom the Jastrow factor already covers. What the
pair function cannot otherwise produce is s occupied singly alongside a
hole in m, and that is what the second family adds. Symmetry is the only
selection rule available: the block must carry g(a) x g(b) = g(m) x g(s),
which for a = s leaves the single excitations m -> b, and otherwise the
double excitations of the same symmetry. The point group is assumed D2h,
as ORCA uses for an atom under UseSym.

Orbitals outside a CASSCF active space are not natural orbitals and carry
no occupation, which is why an active space cannot seed a geminal beyond
its own size.

Take the ORCA output from the run that produced gwfn.data, so that the
orbital indices match. A correlated density is required, for example

    ! CCSD ano-pVDZ AutoAux VeryTightSCF UseSym
    %mdci  density unrelaxed
           natorbs true
    end

or the same with CCSD(T), OO-RI-MP2 or RI-MP2.
"""

import argparse
import math
import re
import sys

from casino.readers.geminal import geminal_block_template, geminal_template

DOCC_MIN = 1.5
SOMO_MIN = 0.5
DEGENERACY = 1e-6

# D2h irreducible representations as the parity of x, y, z, so that the
# product of two of them is the exclusive or of their codes.
irrep = {'Ag': 0, 'B3u': 1, 'B2u': 2, 'B1g': 3, 'B1u': 4, 'B2g': 5, 'B3g': 6, 'Au': 7}

occupation_re = re.compile(r'^\s*N\[\s*\d+\](?:\(\s*(\S+)\s*\))?\s*=\s*([-\d.]+)')


def read_natural_orbitals(file_path):
    """Natural orbital symmetry labels and occupations from an ORCA output."""
    symmetry, occupation = [], []
    found = False
    with open(file_path, 'r') as f:
        for line in f:
            if 'Natural Orbital Occupation Numbers:' in line:
                found = True
                continue
            if found:
                match = occupation_re.match(line)
                if match is None:
                    break
                symmetry.append(match.group(1))
                occupation.append(float(match.group(2)))
    return symmetry, occupation


def degenerate_groups(orbitals, occupation):
    groups = []
    for orb in orbitals:
        if groups and abs(occupation[orb - 1] - occupation[groups[-1][0] - 1]) < DEGENERACY:
            groups[-1].append(orb)
        else:
            groups.append([orb])
    return groups


class MultiGeminal:
    """MAGP expansion generated from a natural orbital spectrum."""

    def __init__(self, file_path, neu, ned, scale=1.0):
        self.scale = scale
        self.symmetry, self.occupation = read_natural_orbitals(file_path)
        if not self.occupation:
            raise ValueError(f'no natural orbital occupations found in {file_path}')
        norb = len(self.occupation)
        self.docc = [i for i in range(1, norb + 1) if self.occupation[i - 1] > DOCC_MIN]
        self.somo = [i for i in range(1, norb + 1) if SOMO_MIN < self.occupation[i - 1] <= DOCC_MIN]
        self.virtual = [i for i in range(1, norb + 1) if self.occupation[i - 1] <= SOMO_MIN]
        if len(self.docc) != ned or len(self.docc) + len(self.somo) != neu:
            raise ValueError(f'occupations give {len(self.docc)} paired and {len(self.somo)} unpaired orbitals, expected {ned} and {neu - ned}')
        self.groups = degenerate_groups(self.virtual, self.occupation)
        self.shells = degenerate_groups(self.docc[1:], self.occupation)
        self.reference = self.occupation[self.docc[-1] - 1]
        self.blocks = []
        self.constraints = []

    def unpaired_columns(self, moved=None, orbital=None):
        columns = []
        for k, s in enumerate(self.somo):
            orb = orbital if k == moved else s
            columns.append(f'u_{orb},{k + 1}: [ 1.0, fixed ]')
        return columns

    def seed(self, orb):
        return -self.scale * math.sqrt(self.occupation[orb - 1] / self.reference)

    def parameter(self, pair):
        row, col = pair
        return f'g_{row},{col}: [ {self.seed(col):.6f}, optimizable ]'

    def channel(self, s, m):
        """Orbital pairs carrying the symmetry of the m -> s replacement, by degeneracy."""
        pool = [s] + self.virtual
        target = irrep[self.symmetry[m - 1]] ^ irrep[self.symmetry[s - 1]]
        groups = {}
        for i, a in enumerate(pool):
            for b in pool[i + 1 :]:
                if irrep[self.symmetry[a - 1]] ^ irrep[self.symmetry[b - 1]] == target:
                    key = (round(self.occupation[a - 1] / DEGENERACY), round(self.occupation[b - 1] / DEGENERACY))
                    groups.setdefault(key, []).extend(((a, b), (b, a)))
        return list(groups.values())

    def pair_family(self, shell, groups, moved=None):
        first = len(self.blocks) + 1
        for m in shell:
            keep = [d for d in self.docc if d != m]
            fixed = [f'g_{d},{d}: [ 1.0, fixed ]' for d in keep]
            columns = self.unpaired_columns(moved, m)
            if len(self.blocks) + 1 == first:
                optimizable = [self.parameter(group[0]) for group in groups]
            else:
                optimizable = []
            self.blocks.append(['c: [ 1.0, fixed ]'] + fixed + optimizable + columns)
            self.blocks.append(['c: [ -1.0, fixed ]'] + fixed[1:] + columns)
        last = len(self.blocks)
        for group in groups:
            self.constraints.append('='.join(f'{n}^g_{row},{col}' for n in range(first, last + 1) for row, col in group))

    def build(self, with_unpaired=True):
        paired = [f'g_{m},{m}: [ 1.0, fixed ]' for m in self.docc]
        self.blocks.append(['c: [ 1.0, fixed ]'] + paired + self.unpaired_columns())
        diagonal = [[(orb, orb) for orb in group] for group in self.groups]
        for shell in self.shells:
            self.pair_family(shell, diagonal)
        if with_unpaired and self.shells:
            for k, s in enumerate(self.somo):
                for m in self.shells[-1]:
                    groups = self.channel(s, m)
                    if groups:
                        self.pair_family([m], groups, k)

    def __str__(self):
        geminals = ''.join(
            geminal_block_template.format(n=n, parameters='\n'.join(f'      {p}' for p in block)) for n, block in enumerate(self.blocks, 1)
        )
        text = geminal_template.format(geminals=geminals)
        if self.constraints:
            text += '  Constraints:\n' + ''.join(f'      {c}\n' for c in self.constraints)
        return text


def main():
    parser = argparse.ArgumentParser(
        description='This script builds parameters.casl from ORCA natural orbitals.', formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('orca_output', type=str, help='path to the ORCA output file')
    parser.add_argument('neu', type=int, help='number of up electrons')
    parser.add_argument('ned', type=int, help='number of down electrons')
    parser.add_argument('--no-unpaired', action='store_true', help='omit the geminals correlating the unpaired electrons')
    parser.add_argument('--scale', type=float, default=1.0, help='factor on the seeds, to keep the excited weight down when the families are many')
    args = parser.parse_args()

    magp = MultiGeminal(args.orca_output, args.neu, args.ned, args.scale)
    magp.build(with_unpaired=not args.no_unpaired)
    sys.stdout.write(str(magp))
