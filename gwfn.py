#!/usr/bin/env python3

from math import exp

import numpy as np
import numba as nb
from numba.typed import List


@nb.jit(nopython=True, cache=True)
def angular_part(x, y, z, l, m, r2):
    """
    :return:
    """
    if l == 0:
        return 1
    elif l == 1:
        if m == 0:
            return x
        elif m == 1:
            return y
        elif m == 2:
            return z
    elif l == 2:
        if m == 0:
            return 3 * z ** 2 - r2
        elif m == 1:
            return x * z
        elif m == 2:
            return y * z
        elif m == 3:
            return x ** 2 + y ** 2
        elif m == 4:
            return x * y
    elif l == 3:
        if m == 0:
            return z * (5 * z ** 2 - 3 * r2) / 2
        if m == 1:
            return 3 * x * (5 * z ** 2 - 3 * r2) / 2
        if m == 2:
            return 3 * y * (5 * z ** 2 - 3 * r2) / 2
        if m == 3:
            return 15 * z * (x ** 2 - y ** 2)
        if m == 4:
            return 30 * x * y * z
        if m == 5:
            return 15 * x * (x ** 2 - 3 * y ** 2)
        if m == 6:
            return 15 * y * (3 * x ** 2 - y * 2)
    elif l == 4:
        if m == 0:
            return (35 * z * z * z * z - 30 * z * z * r2 + 3 * r2 * r2) / 8
        if m == 1:
            return 5 * x * z * (7 * z * z - 3 * r2) / 2
        if m == 2:
            return 5 * y * z * (7 * z * z - 3 * r2) / 2
        if m == 3:
            return 15 * (x * x - y * y) * (7 * z * z - r2) / 2
        if m == 4:
            return 30 * x * y * (7 * z * z - r2) / 2
        if m == 5:
            return 105 * x * z * (x * x - 3 * y * y)
        if m == 6:
            return 105 * y * z * (3 * x * x - y * y)
        if m == 7:
            return 105 * (x * x * x * x - 6 * x * x * y * y + y * y * y * y)
        if m == 8:
            return 420 * x * y * (x * x - y * y)
    return 0


@nb.jit(nopython=True, cache=True)
def wfn(r, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    """single electron wfn on the point.

    param r: coordinat
    param mo: MO
    """

    res = 0.0
    ao = 0
    p = 0
    for shell in range(nshell):
        I = shell_positions[shell]
        x = r[0] - I[0]
        y = r[1] - I[1]
        z = r[2] - I[2]
        # radial part
        r2 = x * x + y * y + z * z
        prim_sum = 0.0
        for primitive in range(p, p + primitives[shell]):
            prim_sum += contraction_coefficients[primitive] * exp(-exponents[primitive] * r2)
        p += primitives[shell]
        # angular part
        l = shell_types[shell]
        for m in range(2*l+1):
            angular = angular_part(x, y, z, l, m, r2)
            res += prim_sum * angular * mo[ao]
            ao += 1
    return res


def vmc(equlib, stat, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents):
    dX_max = 0.4
    X = np.random.uniform(-dX_max, dX_max, size=3)
    p = wfn(X, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
    for i in range(equlib):
        new_X = X + np.random.uniform(-dX_max, dX_max, size=3)
        new_p = wfn(new_X, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
        if new_p*new_p/p/p > np.random.random_sample(1)[0]:
            X, p = new_X, new_p

    j = 0
    sum = 0.0
    for dX in range(stat):
        new_X = X + np.random.uniform(-dX_max, dX_max, size=3)
        new_p = wfn(new_X, mo, nshell, shell_types, shell_positions, primitives, contraction_coefficients, exponents)
        if (new_p/p)**2 > np.random.random_sample(1)[0]:
            X, p = new_X, new_p
            j += 1
            sum += wfn.local_energy(X)
    return sum/j


class Gwfn:
    """Gaussian wfn reader from file."""

    # shell types (s/sp/p/d/f... 1/2/3/4/5...) -> l
    shell_map = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4}

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
                    pos = read_floats(self.natom * 3)
                    self.atomic_positions = np.array(pos).reshape(self.natom, 3)
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
                    self._shell_types = read_ints(self.nshell)
                    # corrected shell_types
                    self.shell_types = np.array([self.shell_map[t] for t in self._shell_types])
                elif line.startswith('Number of primitive Gaussians in each shell'):
                    self.primitives = np.array(read_ints(self.nshell))
                elif line.startswith('Sequence number of first shell on each centre'):
                    self.first_shells = np.array(read_ints(self.natom + 1))
                elif line.startswith('Exponents of Gaussian primitives'):
                    self.exponents = np.array(read_floats(self.nprimitives))
                elif line.startswith('Normalized contraction coefficients'):
                    self.contraction_coefficients = np.array(read_floats(self.nprimitives))
                elif line.startswith('Position of each shell (au)'):
                    pos = read_floats(3 * self.nshell)
                    self.shell_positions = np.array(pos).reshape(self.nshell, 3)
                # ORBITAL COEFFICIENTS
                # --------------------
                elif line.startswith('ORBITAL COEFFICIENTS'):
                    fp.readline()  # skip line
                    mo = read_floats((self.unrestricted + 1) * self.nbasis_functions * self.nbasis_functions)
                    self.mo = np.array(mo).reshape(self.unrestricted + 1, self.nbasis_functions, self.nbasis_functions)

    def integral(self):
        """single electron wfn on the point.

        param r: coordinat
        param mo: MO-orbital
        param spin: [ up | down ]
        """
        mo = self.mo[0, 0]
        steps = 140
        l = 10.0

        x_steps = y_steps = z_steps = steps
        x_min = y_min = z_min = -l
        x_max = y_max = z_max = l

        dV = 2 * l / (steps - 1) * 2 * l / (steps - 1) * 2 * l / (steps - 1)

        x = np.linspace(x_min, x_max, x_steps)
        y = np.linspace(y_min, y_max, y_steps)
        z = np.linspace(z_min, z_max, z_steps)

        grid = np.vstack(np.meshgrid(x, y, z)).reshape(3, -1).T

        return sum(wfn(r, mo, self.nshell, self.shell_types, self.shell_positions, self.primitives, self.contraction_coefficients, self.exponents) ** 2 for r in grid) * dV

    def vmc(self, equlib, stat):
        mo = self.mo[0, 0]
        return vmc(equlib, stat, mo, self.nshell, self.shell_types, self.shell_positions, self.primitives, self.contraction_coefficients, self.exponents)


if __name__ == '__main__':

    # gwfn = Gwfn('test/be/HF/cc-pVQZ/gwfn.data')
    gwfn = Gwfn('test/acetic/HF/cc-pVQZ/gwfn.data')
    # gwfn = Gwfn('test/acetaldehyde/HF/cc-pVQZ/gwfn.data')
    print(gwfn.integral())

    # print(gwfn.vmc(500, 500000))
