from math import factorial, pi, sqrt

import numpy as np
from cusp.slater import multiple_fits

GAUSSIAN_TYPE = 0
SLATER_TYPE = 1


class Base:

    # shell types (s/sp/p/d/f... 1/2/3/4/5...) -> l
    shell_map = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4}

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


class Gwfn(Base):
    """Gaussian wfn reader from gwfn.data file.
    CASINO manual: 7.10.1 gwfn.data file specification
    """

    def __init__(self, file_name):
        """Open file and read gwfn.data"""

        with open(file_name, 'r') as f:
            self.f = f
            for line in f:
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
                    self._atomic_positions = np.array(pos).reshape((self._natoms, 3))
                elif line.startswith('Atomic numbers for each atom'):
                    self._atom_numbers = self.read_ints(self._natoms)
                elif line.startswith('Valence charges for each atom'):
                    self._atom_charges = self.read_floats(self._natoms)
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
                    self._shell_types = self.read_ints(self._nshell)
                    # corrected shell_types
                    self._shell_types = np.array([self.shell_map[t] for t in self._shell_types])
                elif line.startswith('Number of primitive Gaussians in each shell'):
                    self._primitives = np.array(self.read_ints(self._nshell))
                    self._max_primitives = np.max(self._primitives)
                elif line.startswith('Sequence number of first shell on each centre'):
                    self._first_shells = self.read_ints(self._natoms + 1)
                elif line.startswith('Exponents of Gaussian primitives'):
                    self._exponents = self.read_floats(self._nprimitives)
                elif line.startswith('Normalized contraction coefficients'):
                    self._coefficients = self.read_floats(self._nprimitives)
                elif line.startswith('Position of each shell (au)'):
                    pos = self.read_floats(3 * self._nshell)
                    self._shell_positions = np.array(pos).reshape((self._nshell, 3))
                # ORBITAL COEFFICIENTS
                # --------------------
                elif line.startswith('ORBITAL COEFFICIENTS'):
                    f.readline()  # skip line with -----------
                    if self.unrestricted:
                        mo = self.read_floats(2 * self.nbasis_functions * self.nbasis_functions)
                        mo_up = mo[:self.nbasis_functions * self.nbasis_functions]
                        mo_down = mo[self.nbasis_functions * self.nbasis_functions:]
                        self.mo_up = np.array(mo_up).reshape((self.nbasis_functions, self.nbasis_functions))
                        self.mo_down = np.array(mo_down).reshape((self.nbasis_functions, self.nbasis_functions))
                    else:
                        mo_up = self.read_floats(self.nbasis_functions * self.nbasis_functions)
                        self.mo_up = self.mo_down = np.array(mo_up).reshape((self.nbasis_functions, self.nbasis_functions))

        self.atoms = self.set_atoms()
        self.shells = self.set_shells()
        # self.set_cusp()

    def set_atoms(self):
        _atoms = [(
            self._atom_numbers[natom],
            self._atom_charges[natom],
            self._atomic_positions[natom],
            [self._first_shells[natom]-1, self._first_shells[natom+1]-1],
        ) for natom in range(self._natoms)]
        return np.array(_atoms, dtype=[
            ('number', np.int),
            ('charge', np.int),
            ('position', np.float, 3),
            ('shells', np.int, 2),
        ])

    def set_shells(self):
        _shells = []
        p = 0
        for nshell in range(self._nshell):
            _shells.append((
                GAUSSIAN_TYPE,
                self._shell_types[nshell],
                0,
                self._primitives[nshell],
                self._coefficients[p:p+self._primitives[nshell]] + [0] * (self._max_primitives - self._primitives[nshell]),
                self._exponents[p:p + self._primitives[nshell]] + [0] * (self._max_primitives - self._primitives[nshell]),
            ))
            p += self._primitives[nshell]
        return np.array(_shells, dtype=[
            ('type', np.int),
            ('moment', np.int),
            ('order', np.int),
            ('primitives', np.int),
            ('coefficients', np.float, self._max_primitives),
            ('exponents', np.float, self._max_primitives)
        ])

    def set_cusp(self):
        """set cusped orbitals"""
        for atom in self.atoms:
            for shell in self.shells[slice(*atom['shells'])]:
                if shell['moment'] == 0 and shell['primitives'] >= 3:
                    primitives, coefficients, exponents = multiple_fits(shell, atom['charge'])
                    shell['type'] = SLATER_TYPE
                    shell['primitives'] = primitives
                    shell['coefficients'] = np.append(coefficients, np.zeros((self._max_primitives - primitives,)))
                    shell['exponents'] = np.append(exponents, np.zeros((self._max_primitives - primitives,)))


class Stowfn(Base):
    """Slater wfn reader from stowfn.data file.
    CASINO manual: 7.10.6 stowfn.data file specification
    """

    def __init__(self, file_name):
        """Open file and read stowfn.data"""
        with open(file_name, 'r') as f:
            self.f = f
            for line in f:
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
                    self._atomic_positions = np.array(pos).reshape((self._natoms, 3))
                elif line.startswith('Atomic numbers for each atom'):
                    self._atom_numbers = self.read_ints(self._natoms)
                elif line.startswith('Valence charges for each atom'):
                    self._atom_charges = self.read_floats(self._natoms)
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
                    self._first_shells = self.read_ints(self._natoms)
                elif line.startswith('Code for shell types'):
                    self._shell_types = self.read_ints(self._nshell)
                    # corrected shell_types
                    self._shell_types = np.array([self.shell_map[t] for t in self._shell_types])
                elif line.startswith('Order of radial prefactor r in each shell'):
                    self._radial_prefactor_order = self.read_ints(self._nshell)
                elif line.startswith('Exponent in each STO shell'):
                    self._exponents = self.read_floats(self._nshell)
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
                    f.readline()  # skip line with -----------
                    mo_up = self.read_floats(self.nbasis_functions * self.n_mo_up)
                    self.mo_up = np.array(mo_up).reshape((self.n_mo_up, self.nbasis_functions))
                    if self.unrestricted:
                        mo_down = self.read_floats(self.nbasis_functions * self.n_mo_down)
                        self.mo_down = np.array(mo_down).reshape((self.n_mo_down, self.nbasis_functions))
                    else:
                        self.mo_down = self.mo_up
        self.atoms = self.set_atoms()
        self.shells = self.set_shells()

    def set_atoms(self):
        _first_shells = self._first_shells + [self._nshell+1]
        _atoms = [(
            self._atom_numbers[natom],
            self._atom_charges[natom],
            self._atomic_positions[natom],
            [_first_shells[natom]-1, _first_shells[natom+1]-1],
        ) for natom in range(self._natoms)]
        return np.array(_atoms, dtype=[
            ('number', np.int),
            ('charge', np.int),
            ('position', np.float, 3),
            ('shells', np.int, 2),
        ])

    def set_shells(self):
        _shells = []
        # polynorm[0] = sqrt(1./(4.*pi)); // 1
        # polynorm[1] = sqrt(3./(4.*pi)); // x
        # polynorm[2] = sqrt(3./(4.*pi)); // y
        # polynorm[3] = sqrt(3./(4.*pi)); // z
        #
        # polynorm[4] = .5*sqrt(15./pi); // xy        -2
        # polynorm[5] = .5*sqrt(15./pi); // yz        -1
        # polynorm[6] = .5*sqrt(15./pi); // zx        +1
        # polynorm[7] = .25*sqrt(5./pi); // 3*zz-r(2); 0
        # polynorm[8] = .25*sqrt(15./pi); // xx-yy;   +2
        #
        # polynorm[ 9] = .25*sqrt(7./pi); // (2*zz-3*(xx+yy))*z;  0
        # polynorm[10] = .25*sqrt(17.5/pi); // (4*zz-(xx+yy))*x; +1
        # polynorm[11] = .25*sqrt(17.5/pi); // (4*zz-(xx+yy))*y; -1
        # polynorm[12] = .25*sqrt(105./pi); // (xx-yy)*z;        +2
        # polynorm[13] = .5*sqrt(105./pi); // xy*z;              -2
        # polynorm[14] = .25*sqrt(10.5/pi); // (xx-3.0*yy)*x;    +3
        # polynorm[15] = .25*sqrt(10.5/pi); // (3.0*xx-yy)*y;    -3
        #
        # polynorm[16] = .1875*sqrt(1./pi); // 35zzzz-30zzrr+3rrrr  0
        # polynorm[17] = .75*sqrt(2.5/pi); // xz(7zz-3rr)          +1
        # polynorm[18] = .75*sqrt(2.5/pi); // yz(7zz-3rr)          -1
        # polynorm[19] = .375*sqrt(5./pi); // (xx-yy)(7zz-rr)      +2
        # polynorm[20] = .75*sqrt(5./pi); // xy(7zz-rr)            -2
        # polynorm[21] = .75*sqrt(17.5/pi); // xz(xx-3yy)          +3
        # polynorm[22] = .75*sqrt(17.5/pi); // yz(3xx-yy)          -3
        # polynorm[23] = .1875*sqrt(35./pi); // xxxx-6xxyy+yyyy    +4
        # polynorm[24] = .75*sqrt(35./pi); // xxxy-xyyy            -4
        for nshell in range(self._nshell):
            n = self._shell_types[nshell]+1
            _shells.append((
                SLATER_TYPE,
                self._shell_types[nshell],
                self._radial_prefactor_order[nshell],
                1,
                [1/sqrt(4*pi) * (2*self._exponents[nshell])**n * sqrt(2*self._exponents[nshell]/factorial(2*n))],
                [self._exponents[nshell]],
            ))
        return np.array(_shells, dtype=[
            ('type', np.int),
            ('moment', np.int),
            ('order', np.int),
            ('primitives', np.int),
            ('coefficients', np.float, (1, )),
            ('exponents', np.float, (1, ))
        ])
