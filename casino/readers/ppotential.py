import os

import numpy as np


class PPotential:
    """Pseudopotential reader from file."""

    def read_str(self):
        return str(self.f.readline())

    def read_int(self):
        return int(self.f.readline())

    def read_float(self):
        return float(self.f.readline())

    def __init__(self):
        self.atomic_number = 0
        self.pseudo_charge = 0
        self.vmc_nonlocal_grid = 0
        self.dmc_nonlocal_grid = 0
        self.ppotential = np.zeros(shape=(0, 0), dtype=float)

    @staticmethod
    def atom_charge(symbol):
        """Find atomic number from atomic symbol"""
        periodic = ['X', 'H', 'He']
        periodic += ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
        periodic += ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
        periodic += ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']
        periodic += ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe']
        periodic += ['Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']
        periodic += ['Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
        periodic[58:58] = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        periodic[90:90] = ['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        return periodic.index(symbol)

    def read(self, base_path):
        """Read Pseudopotential from file."""
        for file_name in os.listdir(base_path):
            if not file_name.endswith('_pp.data'):
                continue
            file_path = os.path.join(base_path, file_name)
            if not os.path.isfile(file_path):
                return
            with open(file_path, 'r') as f:
                self.f = f
                for line in f:
                    line = line.strip()
                    if line.startswith('Atomic number and pseudo-charge'):
                        atomic_number, pseudo_charge = self.f.readline().split()
                        self.atomic_number = int(atomic_number)
                        self.pseudo_charge = float(pseudo_charge)
                    elif line.startswith('Energy units (rydberg/hartree/ev)'):
                        units = self.read_str()[:-1]
                        if units == 'rydberg':
                            scale = 0.5
                        elif units == 'hartree':
                            scale = 1
                        elif units == 'ev':
                            scale = 27.2114079527
                    elif line.startswith('Angular momentum of local component (0=s,1=p,2=d..)'):
                        # FIXME: local channel
                        max_angular_momentum = self.read_int()
                    elif line.startswith('NLRULE override (1) VMC/DMC (2) config gen (0 ==> input/default value)'):
                        self.vmc_nonlocal_grid, self.dmc_nonlocal_grid = map(int, self.f.readline().split())
                    elif line.startswith('Number of grid points'):
                        grid_points = self.read_int()
                    elif line.startswith('R(i) in atomic units'):
                        self.ppotential = np.zeros(shape=(max_angular_momentum+2, grid_points), dtype=float)
                        for i in range(grid_points):
                            self.ppotential[0, i] = self.read_float()
                    elif line.startswith('r*potential'):
                        # take X from r*potential (L=X) in Ry
                        angular_momentum = int(line.split()[1][3])
                        for i in range(grid_points):
                            self.ppotential[angular_momentum+1, i] = self.read_float() * scale

    def read_ecp(self, base_path):
        """Read Pseudopotential from ORCA format file."""
        for file_name in os.listdir(base_path):
            if not file_name.endswith('_ecp.data'):
                continue
            file_path = os.path.join(base_path, file_name)
            if not os.path.isfile(file_path):
                return
            with open(file_path, 'r') as f:
                self.f = f
                for line in f:
                    line = line.strip()
                    if line.startswith('newecp'):
                        atom_symbol = line.split()[1]
                    elif line.startswith('N_core'):
                        pseudo_charge = self.atom_charge(atom_symbol) - int(line.split()[1])
                    elif line.startswith('lmax'):
                        l_max = line.split()[1]
                        l_max = dict(s=0, p=1, d=2)[l_max]
                        d = np.zeros(shape=(3, 8))
                        n = np.zeros(shape=(3, 8))
                        alpha = np.zeros(shape=(3, 8))
                        for i in range(l_max):
                            l, primitives = self.f.readline().split()
                            l = dict(s=0, p=1, d=2)[l]
                            for k in range(int(primitives)):
                                _, alpha[l, k], d[l, k], n[l, k] = map(float, self.f.readline().split())

        ppotential = np.empty_like(self.ppotential)
        ppotential[0] = self.ppotential[0]
        for l in range(1, self.ppotential.shape[0]):
            for i in range(1, self.ppotential.shape[1]):
                r = self.ppotential[0, i]
                ppotential[l, i] = np.sum(d[l-1] * r ** (n[l-1] - 1) * np.exp(-alpha[l-1] * r**2))
        ppotential[3] -= pseudo_charge
        ppotential[1] += ppotential[3]
        ppotential[2] += ppotential[3]
        for l in range(1, self.ppotential.shape[0]):
            for i in range(self.ppotential.shape[1]):
                if l == 1:
                    print(self.ppotential[l, i] / ppotential[l, i])
