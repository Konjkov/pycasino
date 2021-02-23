#!/usr/bin/env python3

import os

import numpy as np
import numba as nb


labels_type = nb.int64[:]
chi_parameters_type = nb.float64[:, :]
f_parameters_type = nb.float64[:, :, :, :]


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

    def read_float(self):
        return float(self.f.readline().split()[0])

    def read_ints(self):
        return list(map(int, self.f.readline().split()))

    def __init__(self, file, atom_charges):
        self.trunc = 0
        self.u_parameters = np.zeros((0, 3), np.float)  # uu, ud, dd order
        self.chi_parameters = nb.typed.List.empty_list(chi_parameters_type)  # u, d order
        self.f_parameters = nb.typed.List.empty_list(f_parameters_type)  # uu, ud, dd order
        self.u_cutoff = 0
        self.chi_cutoff = np.zeros(0)
        self.chi_cusp = np.zeros(0, np.bool)
        self.chi_labels = nb.typed.List.empty_list(labels_type)
        self.f_cutoff = np.zeros(0)
        self.f_labels = nb.typed.List.empty_list(labels_type)
        self.no_dup_u_term = np.zeros(0, np.bool)
        self.no_dup_chi_term = np.zeros(0, np.bool)

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
                    u_term = False
                elif line.startswith('END CHI TERM'):
                    chi_term = False
                elif line.startswith('END F TERM'):
                    f_term = False
                elif u_term:
                    if line.startswith('START SET'):
                        pass
                    elif line.startswith('Expansion order'):
                        u_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        u_spin_dep = self.read_int()
                    elif line.startswith('Cutoff'):
                        self.u_cutoff = self.read_float()
                    elif line.startswith('Parameter'):
                        # uu, ud, dd order
                        self.u_parameters = np.zeros((u_order+1, u_spin_dep+1), np.float)
                        u_mask = self.get_u_mask(u_order)
                        for i in range(u_spin_dep + 1):
                            for l in range(u_order + 1):
                                if u_mask[l]:
                                    self.u_parameters[l, i] = self.read_float()
                    elif line.startswith('END SET'):
                        pass
                elif chi_term:
                    if line.startswith('Number of set'):
                        number_of_sets = self.read_ints()[0]
                        self.chi_cutoff = np.zeros(number_of_sets)
                        self.chi_cusp = np.zeros(number_of_sets, np.bool)
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
                        chi_cutoff = self.read_float()
                        self.chi_cutoff[set_number] = chi_cutoff
                    elif line.startswith('Parameter'):
                        # u, d
                        parameters = np.zeros((chi_order+1, chi_spin_dep+1), np.float)
                        chi_mask = self.get_chi_mask(chi_order)
                        for i in range(chi_spin_dep + 1):
                            for m in range(chi_order + 1):
                                if chi_mask[m]:
                                    parameters[m, i] = self.read_float()
                        self.chi_parameters.append(parameters)
                    elif line.startswith('END SET'):
                        set_number = None
                elif f_term:
                    if line.startswith('Number of set'):
                        number_of_sets = self.read_ints()[0]
                        self.f_cutoff = np.zeros(number_of_sets)
                        self.no_dup_u_term = np.zeros(number_of_sets, np.bool)
                        self.no_dup_chi_term = np.zeros(number_of_sets, np.bool)
                    elif line.startswith('START SET'):
                        set_number = int(line.split()[2]) - 1
                    elif line.startswith('Label'):
                        f_labels = np.array(self.read_ints()) - 1
                        self.f_labels.append(f_labels)
                    elif line.startswith('Prevent duplication of u term'):
                        no_dup_u_term = self.read_bool()
                    elif line.startswith('Prevent duplication of chi term'):
                        no_dup_chi_term = self.read_bool()
                    elif line.startswith('Electron-nucleus expansion order'):
                        f_en_order = self.read_int()
                    elif line.startswith('Electron-electron expansion order'):
                        f_ee_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        f_spin_dep = self.read_int()
                    elif line.startswith('Cutoff'):
                        f_cutoff = self.read_float()
                        self.f_cutoff[set_number] = f_cutoff
                    elif line.startswith('Parameter'):
                        parameters = np.zeros((f_en_order+1, f_en_order+1, f_ee_order+1, f_spin_dep+1), np.float)
                        f_mask = self.get_f_mask(f_en_order, f_ee_order, no_dup_u_term, no_dup_chi_term)
                        for i in range(f_spin_dep + 1):
                            for n in range(f_ee_order + 1):
                                for m in range(f_en_order + 1):
                                    for l in range(m, f_en_order + 1):
                                        if f_mask[l, m, n]:
                                            # γlmnI = γmlnI
                                            parameters[l, m, n, i] = parameters[m, l, n, i] = self.read_float()
                        self.no_dup_u_term[set_number] = no_dup_u_term
                        self.no_dup_chi_term[set_number] = no_dup_chi_term
                        self.f_parameters.append(parameters)
                        self.check_f_constrains(parameters, f_cutoff, no_dup_u_term, no_dup_chi_term)
                    elif line.startswith('END SET'):
                        set_number = None

    def get_u_mask(self, u_order):
        """u-term mask for all spin-deps"""
        mask = np.ones((u_order+1), np.bool)
        mask[1] = False
        return mask

    def get_chi_mask(self, chi_order):
        """chi-term mask for all spin-deps"""
        mask = np.ones((chi_order+1), np.bool)
        mask[1] = False
        return mask

    def get_f_mask(self, f_en_order, f_ee_order, no_dup_u_term, no_dup_chi_term):
        """f-term mask for all spin-deps"""
        mask = np.ones((f_en_order+1, f_en_order+1, f_ee_order+1), np.bool)
        for n in range(f_ee_order + 1):
            for m in range(f_en_order + 1):
                for l in range(m, f_en_order + 1):
                    if n == 0 and m == 0:
                        mask[l, m, n] = mask[m, l, n] = False
                    # sum(γlm1I) = 0
                    if n == 1 and (m == 0 or l == f_en_order or l == f_en_order - 1 and m == 1):
                        mask[l, m, n] = mask[m, l, n] = False
                    if l == f_en_order and m == 0:
                        mask[l, m, n] = mask[m, l, m] = False
                    if no_dup_u_term and (m == 0 and l == 0 or m == 1 and l == 1 and n == 0):
                        mask[l, m, n] = mask[m, l, n] = False
                    if no_dup_chi_term and m == 1 and n == 0:
                        mask[l, m, n] = mask[m, l, n] = False
        return mask

    def fix_f_not_implemented(self, f_parameters, f_cutoff, no_dup_u_term, no_dup_chi_term):
        """To find the dependent coefficients of f-term it is necessary to solve
        the system of linear equations:  a*x=b
        a-matrix has the following rows:
        (2 * f_en_order + 1) constraints imposed to satisfy electron–electron no-cusp condition.
        (f_en_order + f_ee_order + 1) constraints imposed to satisfy electron–nucleus no-cusp condition.
        (f_ee_order + 1) constraints imposed to prevent duplication of u-term
        (f_en_order + 1) constraints imposed to prevent duplication of chi-term
        b-column has the sum of independent coefficients for each condition.
        """
        f_en_order = f_parameters.shape[0] - 1
        f_ee_order = f_parameters.shape[2] - 1
        f_spin_dep = f_parameters.shape[3] - 1

        u_constrains = 2 * f_en_order + 1
        chi_constrains = f_en_order + f_ee_order + 1
        no_dup_u_constrains = f_ee_order + 1
        no_dup_chi_constrains = f_en_order + 1

        n_constraints = u_constrains + chi_constrains
        if no_dup_u_term:
            n_constraints += no_dup_u_constrains
        if no_dup_chi_term:
            n_constraints += no_dup_chi_constrains

        a = np.zeros((f_spin_dep+1, n_constraints, n_constraints))
        b = np.zeros((f_spin_dep+1, n_constraints))
        f_mask = self.get_f_mask(f_en_order, f_ee_order, no_dup_u_term, no_dup_chi_term)
        p = 0
        for n in range(f_ee_order + 1):
            for m in range(f_en_order + 1):
                for l in range(m, f_en_order + 1):
                    if f_mask[l, m, n]:
                        if n == 1:
                            if l == m:
                                b[:, l + m] -= f_parameters[l, m, n, :]
                            else:
                                b[:, l + m] -= 2 * f_parameters[l, m, n, :]
                        if m == 1:
                            b[:, l + n + u_constrains] += f_cutoff * f_parameters[l, m, n, :]
                        elif m == 0:
                            b[:, l + n + u_constrains] -= self.trunc * f_parameters[l, m, n, :]
                            if l == 1:
                                b[:, n + u_constrains] += f_cutoff * f_parameters[l, m, n, :]
                            # elif l == 0:
                            #     b[:, n + u_constrains] -= self.trunc * f_parameters[l, m, n, :]
                    else:
                        if n == 1:
                            if l == m:
                                a[:, l + m, p] = 1
                            else:
                                a[:, l + m, p] = 2
                        if m == 1:
                            a[:, l + n + u_constrains, p] = - f_cutoff
                        elif m == 0:
                            a[:, l + n + u_constrains, p] = self.trunc
                            if l == 1:
                                a[:, n + u_constrains, p] = - f_cutoff
                            # elif l == 0:
                            #     a[:, n + u_constrains, p] = self.trunc
                            if no_dup_u_term:
                                if l == 0:
                                    a[:, n + u_constrains + chi_constrains, p] = 1
                                if no_dup_chi_term and n == 0:
                                    a[:, l + u_constrains + chi_constrains + no_dup_u_constrains, p] = 1
                            else:
                                if no_dup_chi_term and n == 0:
                                    a[:, l + u_constrains + chi_constrains, p] = 1
                        p += 1

        x = np.linalg.solve(a, b)

        p = 0
        for n in range(f_ee_order + 1):
            for m in range(f_en_order + 1):
                for l in range(m, f_en_order + 1):
                    if not f_mask[l, m, n]:
                        f_parameters[l, m, n, :] = f_parameters[m, l, n, :] = x[:, p]
                        p += 1

        return f_parameters

    def check_f_constrains(self, f_parameters, f_cutoff, no_dup_u_term, no_dup_chi_term):
        """"""
        f_en_order = f_parameters.shape[0] - 1
        f_ee_order = f_parameters.shape[2] - 1
        f_spin_dep = f_parameters.shape[3] - 1
        for lm in range(2 * f_en_order + 1):
            lm_sum = np.zeros(f_spin_dep + 1)
            for l in range(f_en_order + 1):
                for m in range(f_en_order + 1):
                    if l + m == lm:
                        lm_sum += f_parameters[l, m, 1, :]
            print('lm=', lm, 'sum=', lm_sum)

        for mn in range(f_en_order + f_ee_order + 1):
            mn_sum = np.zeros(f_spin_dep + 1)
            for m in range(f_en_order + 1):
                for n in range(f_ee_order + 1):
                    if m + n == mn:
                        mn_sum += self.trunc * f_parameters[0, m, n, :] - f_cutoff * f_parameters[1, m, n, :]
            print('mn=', mn, 'sum=', mn_sum)
        if no_dup_u_term:
            print('should be equal to zero')
            print(f_parameters[1, 1, 0, :])
            print(f_parameters[0, 0, :, :])
        if no_dup_chi_term:
            print('should be equal to zero')
            print(f_parameters[:, 0, 0, :])
