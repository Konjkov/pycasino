#!/usr/bin/env python3
import os
import argparse
import numpy as np

periodic = ['', 'H', 'He']
periodic += ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
periodic += ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
periodic += ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']
periodic += ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe']
periodic += ['Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']
periodic += ['Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
periodic[58:58] = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
periodic[90:90] = ['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']


header_template = """\
Pseudopotential in real space for {symbol}
Atomic number and pseudo-charge
{atomic_number} {pseudo_charge}
Energy units (rydberg/hartree/ev):
rydberg
Angular momentum of local component (0=s,1=p,2=d..)
{max_angular_momentum}
NLRULE override (1) VMC/DMC (2) config gen (0 ==> input/default value)
0 0
Number of grid points
{n_grid}
"""


class PPotential:
    """Pseudopotential converter from ORCA to CASINO."""

    def read_str(self):
        return str(self.f.readline())

    def read_int(self):
        return int(self.f.readline())

    def read_float(self):
        return float(self.f.readline())

    def __init__(self):
        self.atom_symbol = ''
        self.pseudo_charge = 0
        self.atomic_number = 0
        self.d = np.zeros(shape=(0, 0))
        self.n = np.zeros(shape=(0, 0))
        self.alpha = np.zeros(shape=(0, 0))
        self.ppotential = np.zeros(shape=(0, 0), dtype=float)

    def nuclear_charge(self):
        """Find nuclear charge from atomic symbol"""
        return periodic.index(self.atom_symbol.capitalize())

    def r_i(self):
        """R(i) grid in atomic units
        copy past from CASINO ptm.f90 source code
        """
        rmax = 50
        aa = 8
        bb = 80
        a = bb * np.exp(-aa * np.log(10)) / self.pseudo_charge
        b = 1 / bb
        i = 0
        r_list = []
        while True:
            r = a * (np.exp(b * i) - 1)
            r_list.append(r)
            i += 1
            if r > rmax:
                break
        return np.array(r_list)

    def read(self, file_path):
        """Read Pseudopotential in ORCA format."""
        if not os.path.isfile(file_path):
            return
        with open(file_path, 'r') as self.f:
            for line in self.f:
                line = line.strip()
                if line.startswith('newecp'):
                    self.atom_symbol = line.split()[1]
                elif line.startswith('N_core'):
                    self.atomic_number = self.nuclear_charge()
                    self.pseudo_charge = self.atomic_number - float(line.split()[1])
                elif line.startswith('lmax'):
                    l_max = line.split()[1]
                    l_max = dict(s=0, p=1, d=2)[l_max]
                    max_primitives = 4
                    self.d = np.zeros(shape=(l_max + 1, max_primitives))
                    self.n = np.zeros(shape=(l_max + 1, max_primitives))
                    self.alpha = np.zeros(shape=(l_max + 1, max_primitives))
                    for i in range(l_max + 1):
                        l, primitives = self.f.readline().split()
                        l = dict(s=0, p=1, d=2)[l]
                        for k in range(int(primitives)):
                            _, self.alpha[l, k], self.d[l, k], self.n[l, k] = map(float, self.f.readline().split())

        r_i = self.r_i()
        self.ppotential = np.zeros(shape=(l_max + 2, r_i.size))
        self.ppotential[0] = r_i
        for l in range(1, self.ppotential.shape[0]):
            for i in range(self.ppotential.shape[1]):
                r = self.ppotential[0, i]
                self.ppotential[l, i] = np.nan_to_num(np.sum(self.d[l-1] * r ** (self.n[l-1] - 1) * np.exp(-self.alpha[l-1] * r**2)))
        self.ppotential[-1] -= self.pseudo_charge
        self.ppotential[1:-1] += self.ppotential[-1]

    def write(self):
        """Write Pseudopotential in CASINO format."""
        header = header_template.format(
            symbol=self.atom_symbol,
            n_grid=self.ppotential.shape[1],
            atomic_number=self.atomic_number,
            pseudo_charge=self.pseudo_charge,
            max_angular_momentum=self.ppotential.shape[0] - 2
        )
        with open(f'{self.atom_symbol.lower()}_pp.data', 'w') as f:
            f.write(header)
            for i in range(self.ppotential.shape[0]):
                if i == 0:
                    print('R(i) in atomic units', file=f)
                else:
                    print(f'r*potential (L={i - 1}) in Ry', file=f)
                for j in range(self.ppotential.shape[1]):
                    if i == 0:
                        print(f'{self.ppotential[i, j]: 26.15e}', file=f)
                    else:
                        # convert to rydberg
                        print(f'{self.ppotential[i, j] * 2: 26.15e}', file=f)


def main():
    parser = argparse.ArgumentParser('orca_pp2casino')
    parser.add_argument("input_file", help="orca pseudopotential", type=str)
    args = parser.parse_args()
    pp = PPotential()
    pp.read(args.input_file)
    pp.write()


if __name__ == "__main__":
    main()
