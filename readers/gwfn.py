
import numpy as np
from cusp.slater import multiple_fits


class Gwfn:
    """Gaussian wfn reader from file."""

    # shell types (s/sp/p/d/f... 1/2/3/4/5...) -> l
    shell_map = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4}
    GAUSSIAN_TYPE = 0
    SLATER_TYPE = 1

    def __init__(self, file_name):
        """Open file and read gwfn data."""
        def read_bool():
            return fp.readline().strip() == '.true.'

        def read_str():
            return str(fp.readline())

        def read_int():
            return int(fp.readline())

        def read_float():
            return float(fp.readline())

        def read_ints(n):
            result = list()
            while len(result) < n:
                line = fp.readline()
                result += map(int, line.split())
            return result

        def read_floats(n):
            result = list()
            while len(result) < n:
                line = fp.readline()
                result += map(float, [line[i*20:(i+1)*20] for i in range(len(line)//20)])
            return result

        with open(file_name, 'r') as fp:
            for line in fp:
                if line.startswith('TITLE'):
                    self.title = read_str()
                # BASIC_INFO
                # ----------
                elif line.startswith('Spin unrestricted'):
                    self.unrestricted = read_bool()
                elif line.startswith('nuclear-nuclear repulsion energy'):
                    self.repulsion = read_float()
                elif line.startswith('Number of electrons'):
                    self.nelec = read_int()
                # GEOMETRY
                # --------
                elif line.startswith('Number of atoms'):
                    self._natoms = read_int()
                elif line.startswith('Atomic positions'):
                    pos = read_floats(self._natoms * 3)
                    self._atomic_positions = np.array(pos).reshape((self._natoms, 3))
                elif line.startswith('Atomic numbers for each atom'):
                    self._atom_numbers = read_ints(self._natoms)
                elif line.startswith('Valence charges for each atom'):
                    self._atom_charges = read_floats(self._natoms)
                # BASIS SET
                # ---------
                elif line.startswith('Number of Gaussian centres'):
                    self._natoms = read_int()
                elif line.startswith('Number of shells per primitive cell'):
                    self._nshell = read_int()
                elif line.startswith('Number of basis functions'):
                    self.nbasis_functions = read_int()
                elif line.startswith('Number of Gaussian primitives'):
                    self._nprimitives = read_int()
                elif line.startswith('Highest shell angular momentum'):
                    self.highest_ang = read_int()
                elif line.startswith('Code for shell types'):
                    self._shell_types = read_ints(self._nshell)
                    # corrected shell_types
                    self._shell_types = np.array([self.shell_map[t] for t in self._shell_types])
                elif line.startswith('Number of primitive Gaussians in each shell'):
                    self._primitives = np.array(read_ints(self._nshell))
                    self._max_primitives = np.max(self._primitives)
                elif line.startswith('Sequence number of first shell on each centre'):
                    self.first_shells = np.array(read_ints(self._natoms + 1))
                elif line.startswith('Exponents of Gaussian primitives'):
                    self._exponents = read_floats(self._nprimitives)
                elif line.startswith('Normalized contraction coefficients'):
                    self._coefficients = read_floats(self._nprimitives)
                elif line.startswith('Position of each shell (au)'):
                    pos = read_floats(3 * self._nshell)
                    self._shell_positions = np.array(pos).reshape((self._nshell, 3))
                # ORBITAL COEFFICIENTS
                # --------------------
                elif line.startswith('ORBITAL COEFFICIENTS'):
                    fp.readline()  # skip line
                    mo = read_floats((self.unrestricted + 1) * self.nbasis_functions * self.nbasis_functions)
                    self.mo = np.array(mo).reshape((self.unrestricted + 1, self.nbasis_functions, self.nbasis_functions))

        self.atoms = self.set_atoms()
        self.shells = self.set_shells()
        self.set_cusp()

    def set_shells(self):
        _shells = []
        p = 0
        for nshell in range(self._nshell):
            _shells.append((
                self.GAUSSIAN_TYPE,
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
            self._atomic_positions[natom]
        ) for natom in range(self._natoms)]
        return np.array(_atoms, dtype=[('number', np.int), ('charge', np.int), ('position', np.float, 3)])

    def set_cusp(self):
        """set cusped orbitals"""
        for shell in self.shells:
            if shell['moment'] == 0 and shell['primitives'] >= 3:
                popt, perr = multiple_fits(shell)
                shell['type'] = self.SLATER_TYPE
                shell['primitives'] = 1
                shell['coefficients'] = np.array([popt[0]] + [0] * (self._max_primitives - 1))
                shell['exponents'] = np.array([popt[1]] + [0] * (self._max_primitives - 1))
