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
