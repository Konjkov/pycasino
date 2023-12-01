import os
import numpy as np
import numba as nb

atomic_number_type = nb.int64
pseudo_charge_type = nb.float64
ppotential_type = nb.float64[:, :]


class PPotential:
    """Pseudopotential reader from file."""

    def read_str(self):
        return str(self.f.readline())

    def read_int(self):
        return int(self.f.readline())

    def read_float(self):
        return float(self.f.readline())

    def __init__(self):
        self.vmc_nonlocal_grid = 0
        self.dmc_nonlocal_grid = 0
        self.atomic_number = nb.typed.Dict.empty(nb.types.unicode_type, atomic_number_type)
        self.pseudo_charge = nb.typed.Dict.empty(nb.types.unicode_type, pseudo_charge_type)
        self.ppotential = nb.typed.Dict.empty(nb.types.unicode_type, ppotential_type)

    def read(self, base_path):
        """Read Pseudopotential from file."""
        for file_name in os.listdir(base_path):
            if not file_name.endswith('_pp.data'):
                continue
            pp_name = file_name.split('_')[0].capitalize()
            file_path = os.path.join(base_path, file_name)
            if not os.path.isfile(file_path):
                return
            with open(file_path, 'r') as f:
                self.f = f
                for line in f:
                    line = line.strip()
                    if line.startswith('Atomic number and pseudo-charge'):
                        atomic_number, pseudo_charge = self.f.readline().split()
                        self.atomic_number[pp_name] = int(atomic_number)
                        self.pseudo_charge[pp_name] = float(pseudo_charge)
                    elif line.startswith('Energy units (rydberg/hartree/ev)'):
                        units = self.read_str()[:-1]
                        if units == 'rydberg':
                            scale = 0.5
                        elif units == 'hartree':
                            scale = 1
                        elif units == 'ev':
                            scale = 27.2114079527
                    elif line.startswith('Angular momentum of local component (0=s,1=p,2=d..)'):
                        # FIXME: local channel not max angular momentum
                        max_angular_momentum = self.read_int()
                    elif line.startswith('NLRULE override (1) VMC/DMC (2) config gen (0 ==> input/default value)'):
                        self.vmc_nonlocal_grid, self.dmc_nonlocal_grid = map(int, self.f.readline().split())
                    elif line.startswith('Number of grid points'):
                        grid_points = self.read_int()
                    elif line.startswith('R(i) in atomic units'):
                        self.ppotential[pp_name] = np.zeros(shape=(max_angular_momentum+2, grid_points), dtype=float)
                        for i in range(grid_points):
                            self.ppotential[pp_name][0, i] = self.read_float()
                    elif line.startswith('r*potential'):
                        # take X from r*potential (L=X) in Ry
                        angular_momentum = int(line.split()[1][3])
                        for i in range(grid_points):
                            self.ppotential[pp_name][angular_momentum+1, i] = self.read_float() * scale
