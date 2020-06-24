
import numpy as np
from cusp.slater import multiple_fits

GAUSSIAN_TYPE = 0
SLATER_TYPE = 1


class Casino:

    # shell types (s/sp/p/d/f... 1/2/3/4/5...) -> l
    shell_map = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4}

    def read_bool(self):
        return self.fp.readline().strip() == '.true.'

    def read_str(self):
        return str(self.fp.readline())

    def read_int(self):
        return int(self.fp.readline())

    def read_float(self):
        return float(self.fp.readline())

    def read_ints(self, n):
        result = list()
        while len(result) < n:
            line = self.fp.readline()
            result += map(int, line.split())
        return result

    def read_floats(self, n):
        result = list()
        while len(result) < n:
            line = self.fp.readline()
            result += map(float, [line[i * 20:(i + 1) * 20] for i in range(len(line) // 20)])
        return result


class Gwfn(Casino):
    """Gaussian wfn reader from file."""

    def __init__(self, file_name):
        """Open file and read gwfn data."""

        with open(file_name, 'r') as fp:
            self.fp = fp
            for line in fp:
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
                    fp.readline()  # skip line with -----------
                    mo = self.read_floats((self.unrestricted + 1) * self.nbasis_functions * self.nbasis_functions)
                    self.mo = np.array(mo).reshape((self.unrestricted + 1, self.nbasis_functions, self.nbasis_functions))

        self.atoms = self.set_atoms()
        self.shells = self.set_shells()
        # self.set_cusp()

    def set_shells(self):
        _shells = []
        p = 0
        for nshell in range(self._nshell):
            _shells.append((
                GAUSSIAN_TYPE,
                self._shell_types[nshell],
                self._shell_positions[nshell],
                self._primitives[nshell],
                self._coefficients[p:p+self._primitives[nshell]] + [0] * (self._max_primitives - self._primitives[nshell]),
                self._exponents[p:p + self._primitives[nshell]] + [0] * (self._max_primitives - self._primitives[nshell]),
            ))
            p += self._primitives[nshell]
        return np.array(_shells, dtype=[
            ('type', np.int),
            ('moment', np.int),
            ('position', np.float, 3),
            ('primitives', np.int),
            ('coefficients', np.float, self._max_primitives),
            ('exponents', np.float, self._max_primitives)
        ])

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


class Stowfn(Casino):
    """Slater wfn reader from file."""

    def __init__(self, file_name):
        """Open file and read stowfn data."""
        with open(file_name, 'r') as fp:
            self.fp = fp
            for line in fp:
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
                    self._first_shells = self.read_ints(self._natoms + 1)
                elif line.startswith('Code for shell types'):
                    self._shell_types = self.read_ints(self._nshell)
                    # corrected shell_types
                    self._shell_types = np.array([self.shell_map[t] for t in self._shell_types])
                elif line.startswith('Order of radial prefactor r in each shell'):
                    pass
                elif line.startswith('Exponents in each STO shell'):
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
                    fp.readline()  # skip line with -----------
                    mo_up = self.read_floats(self.nbasis_functions * self.n_mo_up)
                    self.mo_up = np.array(mo_up).reshape((self.nbasis_functions, self.n_mo_up))
                    if self.unrestricted:
                        mo_down = self.read_floats(self.nbasis_functions * self.n_mo_down)
                        self.mo_down = np.array(mo_down).reshape((self.nbasis_functions, self.n_mo_down))
                    else:
                        self.mo_down = self.mo_up
