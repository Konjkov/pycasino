#!/usr/bin/env python3

import os

import numpy as np
import numba as nb


labels_type = nb.int64[:]


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
        self.chi_parameters = nb.typed.List([np.zeros((0, 2), np.float)] * atom_charges.size)  # u, d order
        self.f_parameters = nb.typed.List([np.zeros((0, 0, 0, 3), np.float)] * atom_charges.size)  # uu, ud, dd order
        self.u_cutoff = 0
        self.chi_cutoff = np.zeros(atom_charges.size)
        self.f_cutoff = np.zeros(atom_charges.size)
        self.chi_cusp = np.zeros(atom_charges.size, dtype=np.bool)
        self.chi_labels = nb.typed.List.empty_list(labels_type)
        self.f_labels = nb.typed.List.empty_list(labels_type)
        self.u_spin_dep = 0
        self.chi_spin_dep = np.zeros((atom_charges.size, ), np.int64)
        self.f_spin_dep = np.zeros((atom_charges.size, ), np.int64)
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
                        self.u_spin_dep = self.read_int()
                    elif line.startswith('Cutoff'):
                        u_cutoff = self.read_float()
                        self.u_cutoff = u_cutoff
                    elif line.startswith('Parameter'):
                        # uu, ud, dd
                        parameters = np.zeros((u_order+1, 3), np.float)
                        u_mask = self.get_u_mask(u_order)
                        for i in range(self.u_spin_dep + 1):
                            for l in range(u_order + 1):
                                if not u_mask[l]:
                                    continue
                                parameters[l, i] = self.read_float()
                                if self.u_spin_dep == 0:
                                    parameters[l, 2] = parameters[l, 1] = parameters[l, 0]
                                elif self.u_spin_dep == 1:
                                    parameters[l, 2] = parameters[l, 0]
                        self.u_parameters = self.fix_u(parameters, u_cutoff)
                    elif line.startswith('END SET'):
                        pass
                elif chi_term:
                    if line.startswith('START SET'):
                        pass
                    elif line.startswith('Label'):
                        chi_labels = np.array(self.read_ints()) - 1
                        self.chi_labels.append(chi_labels)
                    elif line.startswith('Impose electron-nucleus cusp'):
                        chi_cusp = self.read_bool()
                        for label in chi_labels:
                            self.chi_cusp[label] = chi_cusp
                    elif line.startswith('Expansion order'):
                        chi_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        chi_spin_dep = self.read_int()
                    elif line.startswith('Cutoff'):
                        chi_cutoff = self.read_float()
                        for label in chi_labels:
                            self.chi_cutoff[label] = chi_cutoff
                    elif line.startswith('Parameter'):
                        # u, d
                        parameters = np.zeros((chi_order+1, chi_spin_dep+1), np.float)
                        chi_mask = self.get_chi_mask(chi_order)
                        for i in range(chi_spin_dep + 1):
                            for m in range(chi_order + 1):
                                if chi_mask[m]:
                                    parameters[m, i] = self.read_float()
                        for label in chi_labels:
                            self.chi_spin_dep[label] = chi_spin_dep
                            self.chi_parameters[label] = self.fix_chi(parameters, chi_cutoff, chi_cusp, atom_charges[label])
                    elif line.startswith('END SET'):
                        chi_labels = []
                elif f_term:
                    if line.startswith('START SET'):
                        pass
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
                        for label in f_labels:
                            self.f_cutoff[label] = f_cutoff
                    elif line.startswith('Parameter'):
                        parameters = np.zeros((f_en_order+1, f_en_order+1, f_ee_order+1, f_spin_dep+1), np.float)
                        f_mask = self.get_f_mask(f_en_order, f_ee_order, no_dup_u_term, no_dup_chi_term)
                        for i in range(f_spin_dep+1):
                            for n in range(f_ee_order + 1):
                                for m in range(f_en_order + 1):
                                    for l in range(m, f_en_order + 1):
                                        if f_mask[l, m, n]:
                                            # γlmnI = γmlnI
                                            parameters[l, m, n, i] = parameters[m, l, n, i] = self.read_float()
                        for label in f_labels:
                            self.f_spin_dep[label] = f_spin_dep
                            self.f_parameters[label] = self.fix_f(parameters, f_cutoff, no_dup_u_term, no_dup_chi_term)
                            self.check_f_constrains(self.f_parameters[label], f_cutoff, no_dup_u_term, no_dup_chi_term)
                    elif line.startswith('END SET'):
                        f_labels = []

    def get_u_mask(self, u_order):
        """u-term mask for all spin-deps"""
        mask = np.ones((u_order+1), dtype=np.bool)
        mask[1] = False
        return mask

    def get_chi_mask(self, chi_order):
        """chi-term mask for all spin-deps"""
        mask = np.ones((chi_order+1), dtype=np.bool)
        mask[1] = False
        return mask

    def get_f_mask(self, f_en_order, f_ee_order, no_dup_u_term, no_dup_chi_term):
        """f-term mask for all spin-deps"""
        mask = np.ones((f_en_order+1, f_en_order+1, f_ee_order+1), dtype=np.bool)
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

    def fix_u(self, u_parameters, u_cutoff):
        u_parameters[1] = 1 / np.array([4, 2, 4]) / (-u_cutoff) ** self.trunc + u_parameters[0] * self.trunc / u_cutoff
        return u_parameters

    def fix_chi(self, chi_parameters, chi_cutoff, chi_cusp, charge):
        chi_parameters[1] = chi_parameters[0] * self.trunc / chi_cutoff
        if chi_cusp:
            chi_parameters[1] -= charge / (-chi_cutoff) ** self.trunc
        return chi_parameters

    def fix_f(self, f_parameters, f_cutoff, no_dup_u_term, no_dup_chi_term):
        """
        :param f_parameters:
        :param f_cutoff:
        :param no_dup_u_term:
        :param no_dup_chi_term:
        :return:

        0 - zero value
        A - no electron–electron cusp constrains
        B - no electron–nucleus cusp constrains
        X - independent value

        n = 0            n = 1            n > 1
        -------------------------------------------------------
        B B B B B B B B  A A A A A A A B  ? X X X X X X B  <- m
        B X X X X X X X  A X X X X X A A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A A X X X X X A  X X X X X X X X
        B X X X X X X X  B A A A A A A A  B X X X X X X X
        ---------------- no_dup_u_term ------------------------
        0 B B B B B B B  0 A A A A A A B  0 X X X X X X B  <- m
        B B X X X X X X  A X X X X X A A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A X X X X X X A  X X X X X X X X
        B X X X X X X X  A A X X X X X A  X X X X X X X X
        B X X X X X X X  B A A A A A A A  B X X X X X X X
        ---------------- no_dup_chi_term ----------------------
        0 0 0 0 0 0 0 0  A A A A A A A B  X X X X X X X B  <- m
        0 B B B B B B B  A X X X X X A A  X X X X X X X X
        0 B X X X X X X  A X X X X X X A  X X X X X X X X
        0 B X X X X X X  A X X X X X X A  X X X X X X X X
        0 B X X X X X X  A X X X X X X A  X X X X X X X X
        0 B X X X X X X  A X X X X X X A  X X X X X X X X
        0 B X X X X X X  A A X X X X X A  X X X X X X X X
        0 B X X X X X X  B A A A A A A A  B X X X X X X X
        ^
        l
        """
        f_en_order = f_parameters.shape[0] - 1
        f_ee_order = f_parameters.shape[2] - 1
        f_spin_dep = f_parameters.shape[3] - 1
        """fix 2 * f_en_order e–e cusp constrains"""
        for lm in range(2 * f_en_order + 1):
            lm_sum = np.zeros(f_spin_dep + 1)
            for l in range(f_en_order + 1):
                for m in range(f_en_order + 1):
                    if l + m == lm:
                        lm_sum += f_parameters[l, m, 1, :]
            if lm < f_en_order:
                f_parameters[0, lm, 1, :] = -lm_sum / 2
                f_parameters[lm, 0, 1, :] = -lm_sum / 2
            elif lm == f_en_order:
                sum_1 = -lm_sum / 2
            elif lm > f_en_order:
                f_parameters[f_en_order, lm - f_en_order, 1, :] = -lm_sum / 2
                f_parameters[lm - f_en_order, f_en_order, 1, :] = -lm_sum / 2

        """fix f_en_order+f_ee_order e–n cusp constrains"""
        for mn in reversed(range(f_en_order + f_ee_order + 1)):
            mn_sum = np.zeros(f_spin_dep + 1)
            for m in range(f_en_order + 1):
                for n in range(f_ee_order + 1):
                    if m + n == mn:
                        mn_sum += self.trunc * f_parameters[0, m, n, :] - f_cutoff * f_parameters[1, m, n, :]
            if mn > f_en_order:
                f_parameters[0, f_en_order, mn - f_en_order, :] = -mn_sum / self.trunc
                f_parameters[f_en_order, 0, mn - f_en_order, :] = -mn_sum / self.trunc
            elif mn == f_en_order:
                sum_2 = -mn_sum
            elif mn < f_en_order:
                if no_dup_chi_term:
                    f_parameters[1, mn, 0, :] = mn_sum / f_cutoff
                    f_parameters[mn, 1, 0, :] = mn_sum / f_cutoff
                else:
                    f_parameters[0, mn, 0, :] = -mn_sum / self.trunc
                    f_parameters[mn, 0, 0, :] = -mn_sum / self.trunc

        """fix (l=en_order - 1, m=1, n=1) term"""
        f_parameters[f_en_order - 1, 1, 1, :] = sum_1 - f_parameters[f_en_order, 0, 1, :]
        f_parameters[1, f_en_order - 1, 1, :] = sum_1 - f_parameters[0, f_en_order, 1, :]

        sum_2 += f_cutoff * f_parameters[f_en_order - 1, 1, 1, :]

        """fix (l=en_order, m=0, n=0) term"""
        if no_dup_chi_term:
            f_parameters[f_en_order, 1, 0, :] = - sum_2 / f_cutoff
            f_parameters[1, f_en_order, 0, :] = - sum_2 / f_cutoff
        else:
            f_parameters[f_en_order, 0, 0, :] = sum_2 / self.trunc
            f_parameters[0, f_en_order, 0, :] = sum_2 / self.trunc

        return f_parameters

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
