#!/usr/bin/env python3

import os

import numpy as np
import numba as nb
# import sympy as sp
from readers.numerical import rref

labels_type = nb.int64[:]
chi_mask_type = nb.boolean[:, :]
f_mask_type = nb.boolean[:, :, :, :]
chi_parameters_type = nb.float64[:, :]
f_parameters_type = nb.float64[:, :, :, :]


@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def construct_a_matrix(trunc, f_en_order, f_ee_order, f_cutoff, no_dup_u_term, no_dup_chi_term):
    """A-matrix has the following rows:
    (2 * f_en_order + 1) constraints imposed to satisfy electron–electron no-cusp condition.
    (f_en_order + f_ee_order + 1) constraints imposed to satisfy electron–nucleus no-cusp condition.
    (f_ee_order + 1) constraints imposed to prevent duplication of u-term
    (f_en_order + 1) constraints imposed to prevent duplication of chi-term
    copy-paste from /CASINO/src/pjastrow.f90 SUBROUTINE construct_A
    """
    ee_constrains = 2 * f_en_order + 1
    en_constrains = f_en_order + f_ee_order + 1
    no_dup_u_constrains = f_ee_order + 1
    no_dup_chi_constrains = f_en_order + 1

    n_constraints = ee_constrains + en_constrains
    if no_dup_u_term:
        n_constraints += no_dup_u_constrains
    if no_dup_chi_term:
        n_constraints += no_dup_chi_constrains

    parameters_size = (f_en_order + 1) * (f_en_order + 2) * (f_ee_order + 1) // 2
    a = np.zeros(shape=(n_constraints, parameters_size))
    p = 0
    for n in range(f_ee_order + 1):
        for m in range(f_en_order + 1):
            for l in range(m, f_en_order + 1):
                if n == 1:
                    if l == m:
                        a[l + m, p] = 1
                    else:
                        a[l + m, p] = 2
                if m == 1:
                    a[l + n + ee_constrains, p] = -f_cutoff
                elif m == 0:
                    a[l + n + ee_constrains, p] = trunc
                    if l == 1:
                        a[n + ee_constrains, p] = -f_cutoff
                    elif l == 0:
                        a[n + ee_constrains, p] = trunc
                    if no_dup_u_term:
                        if l == 0:
                            a[n + ee_constrains + en_constrains, p] = 1
                        if no_dup_chi_term and n == 0:
                            a[l + ee_constrains + en_constrains + no_dup_u_constrains, p] = 1
                    else:
                        if no_dup_chi_term and n == 0:
                            a[l + ee_constrains + en_constrains, p] = 1
                p += 1
    return a


class Jastrow:
    """Jastrow reader from file.
    CASINO manual: p. 7.4.2 Jastrow factor
                   p. 22.2 The u, χ and f terms in the Jastrow factor
    Jastrow correlation factor for atoms, molecules, and solids
    N. D. Drummond, M. D. Towler, and R. J. Needs
    Phys. Rev. B 70, 235119
    """

    def read_bool(self):
        return bool(int(self.f.readline()))

    def read_int(self):
        return int(self.f.readline())

    def read_parameter(self, index=None):
        if index:
            parameter, mask, _, comment = self.f.readline().split()
            casino_index = list(map(int, comment.split('_')[1].split(',')))
            if index != casino_index:
                print(index, casino_index)
        else:
            # https://www.python.org/dev/peps/pep-3132/
            parameter, mask, *_ = self.f.readline().split()
        return float(parameter), bool(int(mask))

    def read_ints(self):
        return list(map(int, self.f.readline().split()))

    def __init__(self, file):
        self.trunc = 0
        self.u_mask = np.zeros(shape=(0, 0), dtype=bool)
        self.chi_mask = nb.typed.List.empty_list(chi_mask_type)
        self.f_mask = nb.typed.List.empty_list(f_mask_type)
        self.u_parameters = np.zeros(shape=(0, 0), dtype=float)  # uu, ud, dd order
        self.chi_parameters = nb.typed.List.empty_list(chi_parameters_type)  # u, d order
        self.f_parameters = nb.typed.List.empty_list(f_parameters_type)  # uu, ud, dd order
        self.u_cutoff = 0
        self.chi_cutoff = np.zeros(0)
        self.f_cutoff = np.zeros(0)
        self.chi_cusp = np.zeros(0, bool)
        self.chi_labels = nb.typed.List.empty_list(labels_type)
        self.f_labels = nb.typed.List.empty_list(labels_type)
        self.no_dup_u_term = np.zeros(0, bool)
        self.no_dup_chi_term = np.zeros(0, bool)
        self.u_cusp_const = np.zeros(shape=(3, ))

        if not os.path.isfile(file):
            return
        with open(file, 'r') as f:
            u_term = chi_term = f_term = False
            self.f = f
            for line in f:
                line = line.strip()
                if line.startswith('START JASTROW'):
                    pass
                elif line.startswith('END JASTROW'):
                    break
                elif line.startswith('Truncation order'):
                    self.trunc = self.read_int()
                elif line.startswith('START U TERM'):
                    u_term = True
                elif line.startswith('START CHI TERM'):
                    chi_term = True
                elif line.startswith('START F TERM'):
                    f_term = True
                elif line.startswith('END U TERM'):
                    self.fix_u_parameters()
                    u_term = False
                elif line.startswith('END CHI TERM'):
                    self.fix_chi_parameters()
                    chi_term = False
                elif line.startswith('END F TERM'):
                    self.fix_f_parameters()
                    self.check_f_constrains()
                    f_term = False
                elif u_term:
                    if line.startswith('START SET'):
                        pass
                    elif line.startswith('Expansion order'):
                        u_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        u_spin_dep = self.read_int()
                    elif line.startswith('Cutoff'):
                        self.u_cutoff, _ = self.read_parameter()
                    elif line.startswith('Parameter'):
                        self.u_parameters = np.zeros(shape=(u_order+1, u_spin_dep+1), dtype=float)
                        self.u_mask = self.get_u_mask(self.u_parameters)
                        try:
                            for i in range(u_spin_dep + 1):
                                for l in range(u_order + 1):
                                    if self.u_mask[l, i]:
                                        self.u_parameters[l, i], _ = self.read_parameter()
                        except ValueError:
                            # set u_term[1] to zero
                            for i in range(u_spin_dep+1):
                                self.u_parameters[0, i] = -self.u_cutoff / np.array([4, 2, 4])[i] / (-self.u_cutoff) ** self.trunc / self.trunc
                    elif line.startswith('END SET'):
                        pass
                elif chi_term:
                    if line.startswith('Number of set'):
                        number_of_sets = self.read_ints()[0]
                        self.chi_cutoff = np.zeros(number_of_sets)
                        self.chi_cusp = np.zeros(number_of_sets, bool)
                    elif line.startswith('START SET'):
                        set_number = int(line.split()[2]) - 1
                    elif line.startswith('Label'):
                        chi_labels = np.array(self.read_ints()) - 1
                        self.chi_labels.append(chi_labels)
                    elif line.startswith('Impose electron-nucleus cusp'):
                        chi_cusp = self.read_bool()
                        self.chi_cusp[set_number] = chi_cusp
                    elif line.startswith('Expansion order'):
                        chi_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        chi_spin_dep = self.read_int()
                    elif line.startswith('Cutoff'):
                        chi_cutoff, _ = self.read_parameter()
                        self.chi_cutoff[set_number] = chi_cutoff
                    elif line.startswith('Parameter'):
                        chi_parameters = np.zeros(shape=(chi_order+1, chi_spin_dep+1), dtype=np.float)
                        chi_mask = self.get_chi_mask(chi_parameters)
                        try:
                            for i in range(chi_spin_dep + 1):
                                for m in range(chi_order + 1):
                                    if chi_mask[m, i]:
                                        chi_parameters[m, i], _ = self.read_parameter()
                        except ValueError:
                            pass
                        self.chi_mask.append(chi_mask)
                        self.chi_parameters.append(chi_parameters)
                    elif line.startswith('END SET'):
                        set_number = None
                elif f_term:
                    if line.startswith('Number of set'):
                        number_of_sets = self.read_ints()[0]
                        self.f_cutoff = np.zeros(number_of_sets)
                        self.no_dup_u_term = np.zeros(shape=number_of_sets, dtype=bool)
                        self.no_dup_chi_term = np.zeros(shape=number_of_sets, dtype=bool)
                    elif line.startswith('START SET'):
                        set_number = int(line.split()[2]) - 1
                    elif line.startswith('Label'):
                        f_labels = np.array(self.read_ints()) - 1
                        self.f_labels.append(f_labels)
                    elif line.startswith('Prevent duplication of u term'):
                        no_dup_u_term = self.read_bool()
                        self.no_dup_u_term[set_number] = no_dup_u_term
                    elif line.startswith('Prevent duplication of chi term'):
                        no_dup_chi_term = self.read_bool()
                        self.no_dup_chi_term[set_number] = no_dup_chi_term
                    elif line.startswith('Electron-nucleus expansion order'):
                        f_en_order = self.read_int()
                    elif line.startswith('Electron-electron expansion order'):
                        f_ee_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        f_spin_dep = self.read_int()
                    elif line.startswith('Cutoff'):
                        f_cutoff, _ = self.read_parameter()
                        self.f_cutoff[set_number] = f_cutoff
                    elif line.startswith('Parameter'):
                        f_parameters = np.zeros(shape=(f_en_order+1, f_en_order+1, f_ee_order+1, f_spin_dep+1), dtype=float)
                        f_mask = self.get_f_mask(f_parameters, f_cutoff, no_dup_u_term, no_dup_chi_term)
                        try:
                            for i in range(f_spin_dep + 1):
                                for n in range(f_ee_order + 1):
                                    for m in range(f_en_order + 1):
                                        for l in range(f_en_order + 1):
                                            if f_mask[l, m, n, i]:
                                                # γlmnI = γmlnI
                                                f_parameters[l, m, n, i], _ = f_parameters[m, l, n, i], _ = self.read_parameter([l, m, n, i+1, set_number+1])
                        except ValueError:
                            pass
                        self.f_mask.append(f_mask)
                        self.f_parameters.append(f_parameters)
                        # for i, parameters in enumerate(self.f_parameters):
                        #     # FIXME: put spin_dep to first place
                        #     self.f_parameters[i] = parameters.transpose((3, 0, 1, 2))
                    elif line.startswith('END SET'):
                        set_number = None

    @staticmethod
    def get_u_mask(parameters):
        """Mask dependent parameters in u-term"""
        mask = np.ones(parameters.shape, bool)
        mask[1] = False
        return mask

    @staticmethod
    def get_chi_mask(parameters):
        """Mask dependent parameters in chi-term"""
        mask = np.ones(parameters.shape, bool)
        mask[1] = False
        return mask

    def get_f_mask(self, f_parameters, f_cutoff, no_dup_u_term, no_dup_chi_term):
        """Mask dependent parameters in f-term."""
        f_en_order = f_parameters.shape[0] - 1
        f_ee_order = f_parameters.shape[2] - 1

        a = construct_a_matrix(self.trunc, f_en_order, f_ee_order, f_cutoff, no_dup_u_term, no_dup_chi_term)

        _, pivot_positions = rref(a)

        p = 0
        mask = np.zeros(shape=f_parameters.shape, dtype=bool)
        for n in range(f_parameters.shape[2]):
            for m in range(f_parameters.shape[1]):
                for l in range(m, f_parameters.shape[0]):
                    if p not in pivot_positions:
                        mask[l, m, n] = True
                    p += 1
        return mask

    def fix_u_parameters(self):
        """Fix u-term parameters"""
        C = self.trunc
        L = self.u_cutoff
        for i in range(3):
            self.u_cusp_const[i] = 1 / np.array([4, 2, 4])[i] / (-L) ** C + self.u_parameters[0, i % self.u_parameters.shape[1]] * C / L

    def fix_chi_parameters(self):
        """Fix chi-term parameters"""
        C = self.trunc
        for chi_parameters, L, chi_cusp in zip(self.chi_parameters, self.chi_cutoff, self.chi_cusp):
            chi_parameters[1] = chi_parameters[0] * C / L
            if chi_cusp:
                pass
                # FIXME: chi cusp not implemented
                # chi_parameters[1] -= charge / (-L) ** C

    def fix_f_parameters(self):
        """To find the dependent coefficients of f-term it is necessary to solve
        the system of linear equations:  A*x=b
        A-matrix has the following rows:
        (2 * f_en_order + 1) constraints imposed to satisfy electron–electron no-cusp condition.
        (f_en_order + f_ee_order + 1) constraints imposed to satisfy electron–nucleus no-cusp condition.
        (f_ee_order + 1) constraints imposed to prevent duplication of u-term
        (f_en_order + 1) constraints imposed to prevent duplication of chi-term
        b-column has the sum of independent coefficients for each condition.
        """
        for f_parameters, L, no_dup_u_term, no_dup_chi_term in zip(self.f_parameters, self.f_cutoff, self.no_dup_u_term, self.no_dup_chi_term):
            f_en_order = f_parameters.shape[0] - 1
            f_ee_order = f_parameters.shape[2] - 1
            f_spin_dep = f_parameters.shape[3] - 1

            a = construct_a_matrix(self.trunc, f_en_order, f_ee_order, L, no_dup_u_term, no_dup_chi_term)
            a, pivot_positions = rref(a)
            # remove zero-rows
            a = a[:pivot_positions.size, :]
            b = np.zeros(shape=(f_spin_dep + 1, pivot_positions.size))
            p = 0
            for n in range(f_ee_order + 1):
                for m in range(f_en_order + 1):
                    for l in range(m, f_en_order + 1):
                        if p not in pivot_positions:
                            for temp in range(pivot_positions.size):
                                b[:, temp] -= a[temp, p] * f_parameters[l, m, n, :]
                        p += 1

            x = np.empty(shape=(f_spin_dep + 1, a.shape[0]))
            for i in range(f_spin_dep + 1):
                x[i, :] = np.linalg.solve(a[:, pivot_positions], b[i])

            p = 0
            temp = 0
            for n in range(f_ee_order + 1):
                for m in range(f_en_order + 1):
                    for l in range(m, f_en_order + 1):
                        if temp in pivot_positions:
                            f_parameters[l, m, n, :] = f_parameters[m, l, n, :] = x[:, p]
                            p += 1
                        temp += 1

    def check_f_constrains(self):
        for f_parameters, L, no_dup_u_term, no_dup_chi_term in zip(self.f_parameters, self.f_cutoff, self.no_dup_u_term, self.no_dup_chi_term):
            f_en_order = f_parameters.shape[0] - 1
            f_ee_order = f_parameters.shape[2] - 1
            f_spin_dep = f_parameters.shape[3] - 1

            lm_sum = np.zeros(shape=(2 * f_en_order + 1, f_spin_dep + 1))
            for l in range(f_en_order + 1):
                for m in range(f_en_order + 1):
                    lm_sum[l + m] += f_parameters[l, m, 1, :]
            np.abs(lm_sum).max() > 1e-18 and print('lm_sum =', lm_sum)

            mn_sum = np.zeros(shape=(f_en_order + f_ee_order + 1, f_spin_dep + 1))
            for m in range(f_en_order + 1):
                for n in range(f_ee_order + 1):
                    mn_sum[m + n] += self.trunc * f_parameters[0, m, n, :] - L * f_parameters[1, m, n, :]
            np.abs(mn_sum).max() > 1e-18 and print('mn_sum =', mn_sum)

            if no_dup_u_term:
                print('should be equal to zero')
                print(f_parameters[1, 1, 0, :])
                print(f_parameters[0, 0, :, :])
            if no_dup_chi_term:
                print('should be equal to zero')
                print(f_parameters[:, 0, 0, :])


if __name__ == '__main__':
    """Read Jastrow terms
    """
    for f_term in (
        '11', '12', '13', '14', '15',
        '21', '22', '23', '24', '25',
        '31', '32', '33', '34', '35',
        '41', '42', '43', '44', '45',
        '51', '52', '53', '54', '55',
    ):
        print(f_term)
        path = f'test/jastrow/3_1/{f_term}/correlation.out.1'
        Jastrow(path)
