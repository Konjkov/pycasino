import os
import numpy as np
import numba as nb

from math import factorial, pi, sqrt


ppotential_type = nb.float64[:, :]

SLATER_TYPE = 1
GAUSSIAN_TYPE = 0
periodic = ['', 'H', 'He']
periodic += ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
periodic += ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
periodic += ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']
periodic += ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe']
periodic += ['Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']
periodic += ['Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
periodic[58:58] = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
periodic[90:90] = ['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']


class FortranFile:
    """Reader of fortran formatted data from file."""

    def __init__(self):
        """Open file and read base types"""
        self.f = None

    def read_bool(self):
        return self.f.readline().strip() == '.true.'

    def read_str(self):
        return str(self.f.readline())

    def read_int(self):
        return int(self.f.readline())

    def read_float(self):
        return float(self.f.readline())

    def read_ints(self, n):
        result = list()
        while len(result) < n:
            line = self.f.readline()
            result += map(int, line.split())
        return result

    def read_floats(self, n):
        result = list()
        while len(result) < n:
            line = self.f.readline()
            result += map(float, [line[i * 20:(i + 1) * 20] for i in range(len(line) // 20)])
        return result


class Gwfn(FortranFile):
    """Gaussian wfn reader from gwfn.data file.
    CASINO manual: 7.10.1 gwfn.data file specification
    """
    # shell types (s/sp/p/d/f... 1/2/3/4/5...) -> l
    shell_map = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4}

    def read(self, base_path):
        """Open file and read gwfn.data"""
        file_path = os.path.join(base_path, 'gwfn.data')
        self.f = open(file_path, 'r')

        for line in self.f:
            if line.startswith('TITLE'):
                self.title = self.read_str()
            # BASIC_INFO
            # ----------
            elif line.startswith('Spin unrestricted'):
                self.unrestricted = self.read_bool()
            elif line.startswith('nuclear-nuclear repulsion energy'):
                self.repulsion = self.read_float()
            elif line.startswith('Number of electrons'):
                self.nelec = self.read_int()
            # GEOMETRY
            # --------
            elif line.startswith('Number of atoms'):
                self._natoms = self.read_int()
            elif line.startswith('Atomic positions'):
                pos = self.read_floats(self._natoms * 3)
                self.atom_positions = np.array(pos).reshape((self._natoms, 3))
            elif line.startswith('Atomic numbers for each atom'):
                self.atom_numbers = np.array(self.read_ints(self._natoms))
            elif line.startswith('Valence charges for each atom'):
                self.atom_charges = np.array(self.read_floats(self._natoms))
            # BASIS SET
            # ---------
            elif line.startswith('Number of Gaussian centres'):
                self._natoms = self.read_int()
            elif line.startswith('Number of shells per primitive cell'):
                self._nshell = self.read_int()
            elif line.startswith('Number of basis functions'):
                self.nbasis_functions = self.read_int()
            elif line.startswith('Number of Gaussian primitives'):
                self._nprimitives = self.read_int()
            elif line.startswith('Highest shell angular momentum'):
                self.highest_ang = self.read_int()
            elif line.startswith('Code for shell types'):
                shell_types = self.read_ints(self._nshell)
                # corrected shell_types
                self.shell_moments = np.array([self.shell_map[t] for t in shell_types])
            elif line.startswith('Number of primitive Gaussians in each shell'):
                self.primitives = np.array(self.read_ints(self._nshell))
                # self._max_primitives = np.max(self.primitives)
            elif line.startswith('Sequence number of first shell on each centre'):
                self.first_shells = np.array(self.read_ints(self._natoms + 1))
            elif line.startswith('Exponents of Gaussian primitives'):
                self.exponents = np.array(self.read_floats(self._nprimitives))
            elif line.startswith('Normalized contraction coefficients'):
                self.coefficients = np.array(self.read_floats(self._nprimitives))
            elif line.startswith('Position of each shell (au)'):
                pos = self.read_floats(3 * self._nshell)
                self._shell_positions = np.array(pos).reshape((self._nshell, 3))
            # ORBITAL COEFFICIENTS
            # --------------------
            elif line.startswith('ORBITAL COEFFICIENTS'):
                self.f.readline()  # skip line with -----------
                if self.unrestricted:
                    mo = self.read_floats(2 * self.nbasis_functions * self.nbasis_functions)
                    mo_up = mo[:self.nbasis_functions * self.nbasis_functions]
                    mo_down = mo[self.nbasis_functions * self.nbasis_functions:]
                    self.mo_up = np.array(mo_up).reshape((self.nbasis_functions, self.nbasis_functions))
                    self.mo_down = np.array(mo_down).reshape((self.nbasis_functions, self.nbasis_functions))
                else:
                    mo_up = self.read_floats(self.nbasis_functions * self.nbasis_functions)
                    self.mo_up = np.array(mo_up).reshape((self.nbasis_functions, self.nbasis_functions))
                    self.mo_down = np.copy(self.mo_up)

        self.orbital_types = np.full((self._nprimitives,), GAUSSIAN_TYPE, np.int64)
        self.slater_orders = np.zeros((self._nprimitives, ), np.int64)
        self.remove_premultiplied_factor()

        # Read pseudopotential from files
        self.is_pseudoatom = np.full_like(self.atom_numbers, False, dtype=np.bool_)
        self.vmc_nonlocal_grid = np.zeros_like(self.atom_numbers)
        self.dmc_nonlocal_grid = np.zeros_like(self.atom_numbers)
        self.local_angular_momentum = np.zeros_like(self.atom_numbers)
        ppotential_list = [np.zeros(shape=(0, 0))] * self.atom_numbers.size

        for file_name in os.listdir(base_path):
            if not file_name.endswith('_pp.data'):
                continue
            pp_name = file_name.split('_')[0].capitalize()
            file_path = os.path.join(base_path, file_name)
            if not os.path.isfile(file_path):
                continue
            # FIXME: pp_name ends with number
            try:
                atomic_number = periodic.index(pp_name.capitalize())
            except ValueError:
                continue
            ids = np.argwhere(self.atom_numbers == atomic_number)
            with open(file_path, 'r') as f:
                self.f = f
                for line in f:
                    line = line.strip()
                    if line.startswith('Atomic number and pseudo-charge'):
                        _atomic_number, pseudo_charge = self.f.readline().split()
                        if atomic_number != int(_atomic_number):
                            print(f'atomic_number from pp filename {atomic_number} don`t match content {_atomic_number}')
                        self.atom_charges[ids] = float(pseudo_charge)
                        self.is_pseudoatom[ids] = True
                    elif line.startswith('Energy units (rydberg/hartree/ev)'):
                        units = self.read_str()[:-1]
                        if units == 'rydberg':
                            scale = 0.5
                        elif units == 'hartree':
                            scale = 1
                        elif units == 'ev':
                            scale = 27.2114079527
                    elif line.startswith('Angular momentum of local component (0=s,1=p,2=d..)'):
                        local_angular_momentum = self.read_int()
                        self.local_angular_momentum[ids] = local_angular_momentum
                    elif line.startswith('NLRULE override (1) VMC/DMC (2) config gen (0 ==> input/default value)'):
                        self.vmc_nonlocal_grid[ids], self.dmc_nonlocal_grid[ids] = map(int, self.f.readline().split())
                    elif line.startswith('Number of grid points'):
                        grid_points = self.read_int()
                    elif line.startswith('R(i) in atomic units'):
                        # FIXME: local channel not max angular momentum
                        ppotential = np.zeros(shape=(local_angular_momentum+2, grid_points), dtype=float)
                        for i in range(grid_points):
                            ppotential[0, i] = self.read_float()
                    elif line.startswith('r*potential'):
                        # take X from r*potential (L=X) in Ry
                        angular_momentum = int(line.split()[1][3])
                        for i in range(grid_points):
                            ppotential[angular_momentum+1, i] = self.read_float() * scale
            for idx in ids:
                ppotential_list[idx[0]] = ppotential.copy()
        self.ppotential = nb.typed.List.empty_list(ppotential_type)
        [self.ppotential.append(pp) for pp in ppotential_list]

    def remove_premultiplied_factor(self):
        """
        One historical CASINO inconsistency which may be easily overlooked:
        Constant numerical factors in the real solid harmonics e.g. the '3' in the 3xy
        d function, or '15' in the (15x^3-45^xy2) f function, may be premultiplied into
        the orbital coefficients so that CASINO doesn't have to e.g. multiply by 3
        every time it evaluates that particular d function. In practice the CASINO
        orbital evaluators do this only for d functions, but *not for f and g* (this
        may or may not be changed in the future if it can be done in a.
        backwards-consistent way)
        """
        p = 0
        d_premultiplied_factor = np.array((0.5, 3.0, 3.0, 3.0, 6.0))
        for shell_moment in self.shell_moments:
            l = shell_moment
            if l == 2:
                self.mo_up[:, p:p+2*l+1] /= d_premultiplied_factor
                self.mo_down[:, p:p+2*l+1] /= d_premultiplied_factor
            p += 2*l+1


class Stowfn(FortranFile):
    """Slater wfn reader from stowfn.data file.
    CASINO manual: 7.10.6 stowfn.data file specification

      read CASINO distributive: /utils/wfn_converters/adf/stowfn.py for details
      polynorm[0] = sqrt(1./(4.*pi)); // 1
      polynorm[1] = sqrt(3./(4.*pi)); // x
      polynorm[2] = sqrt(3./(4.*pi)); // y
      polynorm[3] = sqrt(3./(4.*pi)); // z

      polynorm[4] = .5*sqrt(15./pi); // xy        -2
      polynorm[5] = .5*sqrt(15./pi); // yz        -1
      polynorm[6] = .5*sqrt(15./pi); // zx        +1
      polynorm[7] = .25*sqrt(5./pi); // 3*zz-r(2); 0
      polynorm[8] = .25*sqrt(15./pi); // xx-yy;   +2

      polynorm[ 9] = .25*sqrt(7./pi); // (2*zz-3*(xx+yy))*z;  0
      polynorm[10] = .25*sqrt(17.5/pi); // (4*zz-(xx+yy))*x; +1
      polynorm[11] = .25*sqrt(17.5/pi); // (4*zz-(xx+yy))*y; -1
      polynorm[12] = .25*sqrt(105./pi); // (xx-yy)*z;        +2
      polynorm[13] = .5*sqrt(105./pi); // xy*z;              -2
      polynorm[14] = .25*sqrt(10.5/pi); // (xx-3.0*yy)*x;    +3
      polynorm[15] = .25*sqrt(10.5/pi); // (3.0*xx-yy)*y;    -3

      polynorm[16] = .1875*sqrt(1./pi); // 35zzzz-30zzrr+3rrrr  0
      polynorm[17] = .75*sqrt(2.5/pi); // xz(7zz-3rr)          +1
      polynorm[18] = .75*sqrt(2.5/pi); // yz(7zz-3rr)          -1
      polynorm[19] = .375*sqrt(5./pi); // (xx-yy)(7zz-rr)      +2
      polynorm[20] = .75*sqrt(5./pi); // xy(7zz-rr)            -2
      polynorm[21] = .75*sqrt(17.5/pi); // xz(xx-3yy)          +3
      polynorm[22] = .75*sqrt(17.5/pi); // yz(3xx-yy)          -3
      polynorm[23] = .1875*sqrt(35./pi); // xxxx-6xxyy+yyyy    +4
      polynorm[24] = .75*sqrt(35./pi); // xxxy-xyyy            -4
    """
    # shell types (s/sp/p/d/f... 1/2/3/4/5...) -> l
    shell_map = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4}

    def read(self, base_path):
        """Open file and read stowfn.data"""
        file_path = os.path.join(base_path, 'stowfn.data')
        self.f = open(file_path, 'r')

        for line in self.f:
            # BASIC_INFO
            # ----------
            if line.startswith('Spin unrestricted'):
                self.unrestricted = self.read_bool()
            elif line.startswith('Nuclear repulsion energy'):
                self.repulsion = self.read_float()
            elif line.startswith('Number of electrons'):
                self.nelec = self.read_int()
            # GEOMETRY
            # --------
            elif line.startswith('Number of atoms'):
                self._natoms = self.read_int()
            elif line.startswith('Atomic positions'):
                pos = self.read_floats(self._natoms * 3)
                self.atom_positions = np.array(pos).reshape((self._natoms, 3))
            elif line.startswith('Atomic numbers for each atom'):
                self.atom_numbers = np.array(self.read_ints(self._natoms))
            elif line.startswith('Valence charges for each atom'):
                self.atom_charges = np.array(self.read_floats(self._natoms))
            # BASIS SET
            # ---------
            elif line.startswith('Number of STO centres'):
                self._natoms = self.read_int()
            elif line.startswith('Position of each centre (au)'):
                """same as Atomic positions ???"""
                # pos = self.read_floats(self._natoms * 3)
                # self._atomic_positions = np.array(pos).reshape((self._natoms, 3))
            elif line.startswith('Number of shells'):
                self._nshell = self.read_int()
            elif line.startswith('Sequence number of first shell on each centre'):
                self.first_shells = np.array(self.read_ints(self._natoms) + [self._nshell+1])
            elif line.startswith('Code for shell types'):
                shell_types = self.read_ints(self._nshell)
                # corrected shell_types
                self.shell_moments = np.array([self.shell_map[t] for t in shell_types])
            elif line.startswith('Order of radial prefactor r in each shell'):
                self.slater_orders = np.array(self.read_ints(self._nshell))
            elif line.startswith('Exponent in each STO shell'):
                self.exponents = np.array(self.read_floats(self._nshell))
            elif line.startswith('Number of basis functions'):
                self.nbasis_functions = self.read_int()
            elif line.startswith('Number of molecular orbitals (\'MO\')'):
                if self.unrestricted:
                    self.n_mo_up, self.n_mo_down = self.read_ints(2)
                else:
                    self.n_mo_up = self.n_mo_down = self.read_int()
            # ORBITAL COEFFICIENTS
            # --------------------
            elif line.startswith('ORBITAL COEFFICIENTS'):
                self.f.readline()  # skip line with -----------
                mo_up = self.read_floats(self.nbasis_functions * self.n_mo_up)
                self.mo_up = np.array(mo_up).reshape((self.n_mo_up, self.nbasis_functions))
                if self.unrestricted:
                    mo_down = self.read_floats(self.nbasis_functions * self.n_mo_down)
                    self.mo_down = np.array(mo_down).reshape((self.n_mo_down, self.nbasis_functions))
                else:
                    self.mo_down = np.copy(self.mo_up)

        self.orbital_types = np.full((self._nshell,), SLATER_TYPE, np.int64)
        self.primitives = np.ones((self._nshell,), np.int64)
        self.coefficients = np.ones((self._nshell,), np.float64)
        self.normalize_orbitals()

        # Read pseudopotential from files
        self.is_pseudoatom = np.full_like(self.atom_numbers, False, dtype=np.bool_)

    def normalize_orbitals(self):
        """
        Change order of d-orbitals: [-2, -1, +1, 0, +2] -> [0, +1, -1, +2, -2]
        add normalization`s factors to MO-coefficients.
        """
        p = 0
        d_exchange_order = np.array((3, 2, 1, 4, 0))
        for shell_moment, slater_order, exponent in zip(self.shell_moments, self.slater_orders, self.exponents):
            l = shell_moment
            n = slater_order + shell_moment + 1
            if shell_moment == 2:
                self.mo_up[:, p:p+2*l+1] = self.mo_up[:, p+d_exchange_order]
                self.mo_down[:, p:p+2*l+1] = self.mo_down[:, p+d_exchange_order]
            l_dependent_norm = sqrt(2*l+1)/sqrt(4*pi) * (2*exponent)**n * sqrt(2*exponent/factorial(2*n))
            # read CASINO distributive: examples/generic/gauss_dfg/README for details
            if shell_moment == 2:
                m_dependent_norm = 1 / np.array((1, sqrt(3), sqrt(3), 2*sqrt(3), 2*sqrt(3)))
            elif shell_moment == 3:
                m_dependent_norm = 1 / np.array((1, sqrt(6), sqrt(6), sqrt(60), sqrt(60), sqrt(360), sqrt(360)))
            elif shell_moment == 4:
                m_dependent_norm = 1 / np.array((1, sqrt(10), sqrt(10), sqrt(180), sqrt(180), sqrt(2520), sqrt(2520), sqrt(20160), sqrt(20160)))
            else:
                m_dependent_norm = np.ones((2*l+1, ))
            self.mo_up[:, p:p+2*l+1] *= l_dependent_norm * m_dependent_norm
            self.mo_down[:, p:p+2*l+1] *= l_dependent_norm * m_dependent_norm
            p += 2*l+1
