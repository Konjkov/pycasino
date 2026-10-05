#!/usr/bin/env python3

import numpy as np


class Gwfn:
    """Gaussian wfn reader from file."""

    # shell types (s/sp/p/d/f... 1/2/3/4/5...) -> 2l+1
    shell_map = {1: 1, 2: 3, 3: 3, 4: 5, 5: 7, 6: 9}

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
                    self.natom = read_int()
                elif line.startswith('Atomic positions'):
                    self.atoms = read_floats(self.natom * 3)
                elif line.startswith('Atomic numbers for each atom'):
                    self.atom_numbers = read_ints(self.natom)
                elif line.startswith('Valence charges for each atom'):
                    self.atom_charges = read_floats(self.natom)
                # BASIS SET
                # ---------
                elif line.startswith('Number of Gaussian centres'):
                    self.natom = read_int()
                elif line.startswith('Number of shells per primitive cell'):
                    self.nshell = read_int()
                elif line.startswith('Number of basis functions'):
                    self.nbasis_functions = read_int()
                elif line.startswith('Number of Gaussian primitives'):
                    self.nprimitives = read_int()
                elif line.startswith('Highest shell angular momentum'):
                    self.highest_ang = read_int()
                elif line.startswith('Code for shell types'):
                    self.shell_types = read_ints(self.nshell)
                elif line.startswith('Number of primitive Gaussians in each shell'):
                    self.primitives = read_ints(self.nshell)
                elif line.startswith('Sequence number of first shell on each centre'):
                    self.first_shells = read_ints(self.natom)
                elif line.startswith('Exponents of Gaussian primitives'):
                    tail = read_floats(self.nprimitives)
                    res = []
                    for primitive, shell_type in zip(self.primitives, self.shell_types):
                        head, tail = tail[:primitive], tail[primitive:]
                        res.extend([head + [0] * (max(self.primitives) - primitive)] * self.shell_map[shell_type])
                    self.exponents = np.array(res)
                elif line.startswith('Normalized contraction coefficients'):
                    tail = read_floats(self.nprimitives)
                    res = []
                    for primitive, shell_type in zip(self.primitives, self.shell_types):
                        head, tail = tail[:primitive], tail[primitive:]
                        res.extend([head + [0] * (max(self.primitives) - primitive)] * self.shell_map[shell_type])
                    self.contraction_coefficients = np.array(res)
                elif line.startswith('Position of each shell (au)'):
                    shell_positions = read_floats(3 * self.nshell)
                    res = []
                    for shell, shell_type in enumerate(self.shell_types):
                        res.extend(shell_positions[shell*3:(shell+1)*3] * self.shell_map[shell_type])
                    self.basis_positions = np.array(res).reshape(3, self.nbasis_functions)
                # ORBITAL COEFFICIENTS
                # --------------------
                elif line.startswith('ORBITAL COEFFICIENTS'):
                    fp.readline()  # skip line
                    mo = read_floats((self.unrestricted + 1) * self.nbasis_functions * self.nbasis_functions)
                    self.mo = np.array(mo).reshape(self.unrestricted + 1, self.nbasis_functions, self.nbasis_functions)

    def angular_part(self, r):
        """
        :param r: electron coordinates with shape (3, n, m, l, ...)
        :return:
        """
        harmonic = []
        x, y, z = r
        r2 = np.sum(r**2, axis=0)
        for shell_type in self.shell_types:
            if shell_type == 1:
                harmonic.extend([np.ones(shape=x.shape)])
            if shell_type == 3:
                harmonic.extend([x, y, z])
            if shell_type == 4:
                harmonic.extend([
                           3 * z**2 - r2,
                           x * z,
                           y * z,
                           x**2 - y**2,
                           x * y
                       ])
            if shell_type == 5:
                harmonic.extend([
                           z * (5 * z**2 - 3 * r2) / 2,
                           3 * x * (5 * z**2 - r2) / 2,
                           3 * y * (5 * z**2 - r2) / 2,
                           15 * z * (x**2 - y**2),
                           30 * x * y * z,
                           15 * x * (x**2 - 3 * y**2),
                           15 * y * (3 * x**2 - y**2),
                       ])
            if shell_type == 6:
                harmonic.extend([
                           (35 * z*z*z*z - 30 * z*z*r2 + 3 * r2*r2) / 8,
                           5 * x * z * (7 * z * z - 3 * r2) / 2,
                           5 * y * z * (7 * z * z - 3 * r2) / 2,
                           15 * (x*x - y*y) * (7 * z * z - r2) / 2,
                           30 * x * y * (7 * z * z - r2) / 2,
                           105 * x * z * (x*x - 3 * y*y),
                           105 * y * z * (3 * x*x - y*y),
                           105 * (x*x*x*x - 6 * x*x*y*y + y*y*y*y),
                           420 * x * y * (x*x - y*y)
                       ])
        return np.moveaxis(np.array(harmonic), 0, -1)

    def mo_wfn(self, r, mo):
        """MO-wfn on grid.

        param r: coordinat grid - ndarray(dims=(3, a, b, c))
        param mo: MO-orbital

        ao - array[self.nbasis_functions]
        angular part of spherical harmonic - array[self.nbasis_functions]
        coeff - array[self.nprimitives]
        np.exp(-alpha*r^2) - array[self.nprimitives]

        r -> [3, a, b, c]
        mo -> array[self.nbasis_functions]
        coeff -> array[self.nbasis_functions, max_nprim_inbasis_wfn]
        alpha -> array[self.nbasis_functions, max_nprim_inbasis_wfn]
        coeff * np.exp(-alpha*r^2) -> array[self.nbasis_functions, max_nprim_inbasis_wfn, a, b, c]
        angular part of spherical harmonic > array[self.nbasis_functions, a, b, c]

        SUM along [self.nbasis_functions, max_nprim_inbasis_wfn]
        """
        r2 = np.sum(r**2, axis=0)[..., np.newaxis, np.newaxis]
        radial = np.sum(self.contraction_coefficients * np.exp(-self.exponents * r2), axis=-1)
        angular = mo * self.angular_part(r)
        return np.sum(angular * radial, axis=-1)

    def coulomb(self, r):
        """Coulomb attraction between the electron and nucleus on grid.

        param r: coordinat grid - ndarray(dims=(3, a, b, c))
        param mo: MO-number 0..n
        param spin: [ up | down ]
        """
        res = 0.0
        for position, charge in zip(self.atoms, self.atom_charges):
            rI = np.sqrt(np.sum((r - position)**2, axis=0))
            res -= charge / rI
        return res

    def Be_1s2s(self, r1, r2, r3, r4):
        """HF = |1s(r1)2s(r2)| * |1s(r3)2s(r4)|"""
        alpha = self.mo_wfn(r1, self.mo[0, 0]) * self.mo_wfn(r2, self.mo[0, 1]) - self.mo_wfn(r2, self.mo[0, 0]) * self.mo_wfn(r1, self.mo[0, 1])
        beta  = self.mo_wfn(r3, self.mo[0, 0]) * self.mo_wfn(r4, self.mo[0, 1]) - self.mo_wfn(r4, self.mo[0, 0]) * self.mo_wfn(r3, self.mo[0, 1])
        return alpha * beta

    def Be_1s2px(self, r1, r2, r3, r4):
        """HF = |1s(r1)2px(r2)| * |1s(r3)2px(r4)|"""
        alpha = self.mo_wfn(r1, self.mo[0, 0]) * self.mo_wfn(r2, self.mo[0, 2]) - self.mo_wfn(r2, self.mo[0, 0]) * self.mo_wfn(r1, self.mo[0, 2])
        beta  = self.mo_wfn(r3, self.mo[0, 0]) * self.mo_wfn(r4, self.mo[0, 2]) - self.mo_wfn(r4, self.mo[0, 0]) * self.mo_wfn(r3, self.mo[0, 2])
        return alpha * beta

    def Be_1s2py(self, r1, r2, r3, r4):
        """HF = |1s(r1)2py(r2)| * |1s(r3)2py(r4)|"""
        alpha = self.mo_wfn(r1, self.mo[0, 0]) * self.mo_wfn(r2, self.mo[0, 3]) - self.mo_wfn(r2, self.mo[0, 0]) * self.mo_wfn(r1, self.mo[0, 3])
        beta  = self.mo_wfn(r3, self.mo[0, 0]) * self.mo_wfn(r4, self.mo[0, 3]) - self.mo_wfn(r4, self.mo[0, 0]) * self.mo_wfn(r3, self.mo[0, 3])
        return alpha * beta

    def Be_1s2pz(self, r1, r2, r3, r4):
        """HF = |1s(r1)2pz(r2)| * |1s(r3)2pz(r4)|"""
        alpha = self.mo_wfn(r1, self.mo[0, 0]) * self.mo_wfn(r2, self.mo[0, 4]) - self.mo_wfn(r2, self.mo[0, 0]) * self.mo_wfn(r1, self.mo[0, 4])
        beta  = self.mo_wfn(r3, self.mo[0, 0]) * self.mo_wfn(r4, self.mo[0, 4]) - self.mo_wfn(r4, self.mo[0, 0]) * self.mo_wfn(r3, self.mo[0, 4])
        return alpha * beta

    def Be_4det(self, *args):
        """CASSCF(2,4) = phi(1s2, 2s2) + C * phi(1s2, 2p2)"""
        C = -0.18
        return self.Be_1s2s(*args) + C * self.Be_1s2px(*args) + C * self.Be_1s2py(*args) + C * self.Be_1s2pz(*args)
