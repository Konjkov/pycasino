#!/usr/bin/env python3

import os

import numpy as np


class Jastrow:
    """Jastrow reader from file.
    1. CASINO manual, 22.2 The u, χ and f terms in the Jastrow factor
    2. Jastrow correlation factor for atoms, molecules, and solids, N. D. Drummond, M. D. Towler, and R. J. Needs
    """

    def __init__(self, file, atoms):
        self.trunc = 0
        self.u_parameters = np.zeros((0, 3), np.float)
        self.chi_parameters = np.zeros((atoms.shape[0], 0, 2), np.float)
        self.f_parameters = np.zeros((atoms.shape[0], 0, 0, 0, 3), np.float)
        self.u_cutoff = 0.0
        self.chi_cutoff = np.zeros(atoms.shape[0])
        self.f_cutoff = np.zeros(atoms.shape[0])
        self.chi_cusp = np.zeros(atoms.shape[0])
        self.jastrow = u_term = chi_term = f_term = False
        if not os.path.isfile(file):
            return
        with open(file, 'r') as f:
            line = True
            while line:
                line = f.readline()
                if line.strip().startswith('START JASTROW'):
                    self.jastrow = True
                elif line.strip().startswith('END JASTROW'):
                    self.jastrow = False
                elif line.strip().startswith('Truncation order'):
                    self.trunc = float(f.readline().split()[0])
                elif line.strip().startswith('START U TERM'):
                    u_term = True
                elif line.strip().startswith('START CHI TERM'):
                    chi_term = True
                    chi_parameters_list = [0] * atoms.shape[0]
                    chi_spin_dep_list = [0] * atoms.shape[0]
                    chi_max_order = 0
                elif line.strip().startswith('START F TERM'):
                    f_term = True
                    f_parameters_list = [0] * atoms.shape[0]
                    f_spin_dep_list = [0] * atoms.shape[0]
                    f_en_max_order = 0
                    f_ee_max_order = 0
                elif line.strip().startswith('END U TERM'):
                    u_term = False
                elif line.strip().startswith('END CHI TERM'):
                    chi_term = False
                elif line.strip().startswith('END F TERM'):
                    f_term = False
                elif u_term:
                    if line.strip().startswith('Expansion order'):
                        u_order = int(f.readline())
                    elif line.strip().startswith('Spin dep'):
                        u_spin_dep = int(f.readline())
                    elif line.strip().startswith('Cutoff'):
                        self.u_cutoff = float(f.readline().split()[0])
                    elif line.strip().startswith('Parameter'):
                        # uu, ud, dd
                        self.u_parameters = np.zeros((u_order+1, 3), np.float)
                        u_mask = self.get_u_mask(u_order)
                        for i in range(u_spin_dep + 1):
                            for l in range(u_order + 1):
                                if not u_mask[l]:
                                    continue
                                self.u_parameters[l][i] = float(f.readline().split()[0])
                elif chi_term:
                    if line.strip().startswith('START SET'):
                        pass
                    elif line.strip().startswith('Label'):
                        atom_labels = list(map(int, f.readline().split()))
                    elif line.strip().startswith('Impose electron-nucleus cusp'):
                        param = bool(int(f.readline()))
                        for atom in atom_labels:
                            self.chi_cusp[atom-1] = param
                    elif line.strip().startswith('Expansion order'):
                        chi_order = int(f.readline())
                        chi_max_order = max(chi_max_order, chi_order)
                    elif line.strip().startswith('Spin dep'):
                        chi_spin_dep = int(f.readline())
                        for atom in atom_labels:
                            chi_spin_dep_list[atom-1] = chi_spin_dep
                    elif line.strip().startswith('Cutoff'):
                        param = float(f.readline().split()[0])
                        for atom in atom_labels:
                            self.chi_cutoff[atom-1] = param
                    elif line.strip().startswith('Parameter'):
                        # u, d
                        chi_parameters = np.zeros((chi_order+1, chi_spin_dep + 1), np.float)
                        chi_mask = self.get_chi_mask(chi_order)
                        for i in range(chi_spin_dep + 1):
                            for m in range(chi_order + 1):
                                if not chi_mask[m]:
                                    continue
                                chi_parameters[m, i] = float(f.readline().split()[0])
                        for atom in atom_labels:
                            chi_parameters_list[atom-1] = chi_parameters
                    elif line.strip().startswith('END SET'):
                        atom_labels = []
                elif f_term:
                    if line.strip().startswith('START SET'):
                        pass
                    elif line.strip().startswith('Label'):
                        atom_labels = list(map(int, f.readline().split()))
                    elif line.strip().startswith('Prevent duplication of u term'):
                        no_dup_u_term = bool(int(f.readline()))
                    elif line.strip().startswith('Prevent duplication of chi term'):
                        no_dup_chi_term = bool(int(f.readline()))
                    elif line.strip().startswith('Electron-nucleus expansion order'):
                        f_en_order = int(f.readline())
                        f_en_max_order = max(f_en_max_order, f_en_order)
                    elif line.strip().startswith('Electron-electron expansion order'):
                        f_ee_order = int(f.readline())
                        f_ee_max_order = max(f_ee_max_order, f_ee_order)
                    elif line.strip().startswith('Spin dep'):
                        f_spin_dep = int(f.readline())
                        for atom in atom_labels:
                            f_spin_dep_list[atom-1] = f_spin_dep
                    elif line.strip().startswith('Cutoff'):
                        param = float(f.readline().split()[0])
                        for atom in atom_labels:
                            self.f_cutoff[atom-1] = param
                    elif line.strip().startswith('Parameter'):
                        f_parameters = np.zeros((f_en_order+1, f_en_order+1, f_ee_order+1, f_spin_dep+1), np.float)
                        f_mask = self.get_f_mask(f_en_order, f_ee_order, no_dup_u_term, no_dup_chi_term)
                        for i in range(f_spin_dep+1):
                            for n in range(f_ee_order + 1):
                                for m in range(f_en_order + 1):
                                    for l in range(m, f_en_order + 1):
                                        if not f_mask[l, m, n]:
                                            continue
                                        line = f.readline()
                                        # print(line[:-1], l, m, n, i)
                                        # γlmnI = γmlnI
                                        f_parameters[l, m, n, i] = f_parameters[m, l, n, i] = float(line.split()[0])
                        for atom in atom_labels:
                            f_parameters_list[atom-1] = f_parameters
                    elif line.strip().startswith('END SET'):
                        atom_labels = []

        if self.u_cutoff:
            if u_spin_dep == 0:
                self.u_parameters[:, 2] = self.u_parameters[:, 1] = self.u_parameters[:, 0]
            elif u_spin_dep == 1:
                self.u_parameters[:, 2] = self.u_parameters[:, 0]
            self.u_parameters[1] = np.array([1/4, 1/2, 1/4])/(-self.u_cutoff)**self.trunc + self.u_parameters[0]*self.trunc/self.u_cutoff

        if self.chi_cutoff.any():
            self.chi_parameters = np.zeros((atoms.shape[0], chi_max_order+1, 2), np.float)

            for atom in range(atoms.shape[0]):
                chi_parameters = self.fix_chi(chi_parameters_list[atom], self.chi_cutoff[atom], self.chi_cusp[atom], atoms[atom]['charge'])
                self.chi_parameters[atom, :chi_parameters.shape[0], :chi_parameters.shape[1]] = chi_parameters
                if chi_spin_dep_list[atom] == 0:
                    self.chi_parameters[atom, :, 1] = self.chi_parameters[atom, :, 0]

        if self.f_cutoff.any():
            self.f_parameters = np.zeros((atoms.shape[0], f_en_max_order+1, f_en_max_order+1, f_ee_max_order+1, 3), np.float)

            for atom in range(atoms.shape[0]):
                f_parameters = self.fix_f(f_parameters_list[atom], self.f_cutoff[atom])
                self.f_parameters[atom, :f_parameters.shape[0], :f_parameters.shape[1], :f_parameters.shape[2], :f_parameters.shape[3]] = f_parameters
                if f_spin_dep_list[atom] == 0:
                    self.f_parameters[atom, :, :, :, 2] = self.f_parameters[atom, :, :, :, 1] = self.f_parameters[atom, :, :, :, 0]
                elif f_spin_dep_list[atom] == 1:
                    self.f_parameters[atom, :, :, :, 2] = self.f_parameters[atom, :, :, :, 0]

            # self.check_f_constrains(atoms, f_en_order, f_ee_order, no_dup_u_term, no_dup_chi_term)

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

    def fix_chi(self, chi_parameters, chi_cutoff, chi_cusp, charge):
        chi_parameters[1] = chi_parameters[0] * self.trunc / chi_cutoff
        if chi_cusp:
            chi_parameters[1] -= charge / (-chi_cutoff) ** self.trunc
        return chi_parameters

    def fix_f(self, f_parameters, f_cutoff):
        f_en_order = f_parameters.shape[0] - 1
        f_ee_order = f_parameters.shape[2] - 1
        f_spin_dep = f_parameters.shape[3] - 1
        """fix 2 * f_en_order constrains"""
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

        """fix f_en_order+f_ee_order constrains"""
        for mn in reversed(range(f_en_order + f_ee_order + 1)):
            mn_sum = np.zeros(f_spin_dep + 1)
            for m in range(f_en_order + 1):
                for n in range(f_ee_order + 1):
                    if m + n == mn:
                        mn_sum += self.trunc * f_parameters[0, m, n, :] - f_cutoff * f_parameters[1, m, n, :]
            if mn < f_en_order:
                f_parameters[0, mn, 0, :] = -mn_sum / self.trunc
                f_parameters[mn, 0, 0, :] = -mn_sum / self.trunc
            elif mn == f_en_order:
                sum_2 = -mn_sum / self.trunc
            elif mn > f_en_order:
                f_parameters[0, f_en_order, mn - f_en_order, :] = -mn_sum / self.trunc
                f_parameters[f_en_order, 0, mn - f_en_order, :] = -mn_sum / self.trunc

        """fix (l=en_order - 1, m=1, n=1) term"""
        f_parameters[f_en_order - 1, 1, 1, :] = sum_1 - f_parameters[f_en_order, 0, 1, :]
        f_parameters[1, f_en_order - 1, 1, :] = sum_1 - f_parameters[0, f_en_order, 1, :]

        """fix (l=en_order, m=0, n=0) term"""
        f_parameters[f_en_order, 0, 0, :] = sum_2 + f_cutoff * f_parameters[f_en_order - 1, 1, 1, :] / self.trunc
        f_parameters[0, f_en_order, 0, :] = sum_2 + f_cutoff * f_parameters[1, f_en_order - 1, 1, :] / self.trunc

        return f_parameters

    def check_f_constrains(self, atoms, f_en_order, f_ee_order, no_dup_u_term, no_dup_chi_term):
        """"""
        for atom in range(atoms.shape[0]):
            for lm in range(2 * f_en_order + 1):
                lm_sum = np.zeros(3)
                for l in range(f_en_order + 1):
                    for m in range(f_en_order + 1):
                        if l + m == lm:
                            lm_sum += self.f_parameters[atom, l, m, 1, :]
                print('lm=', lm, 'sum=', lm_sum)

            for mn in range(f_en_order + f_ee_order + 1):
                mn_sum = np.zeros(3)
                for m in range(f_en_order + 1):
                    for n in range(f_ee_order + 1):
                        if m + n == mn:
                            mn_sum += self.trunc * self.f_parameters[atom, 0, m, n, :] - self.f_cutoff[atom] * self.f_parameters[atom, 1, m, n, :]
                print('mn=', mn, 'sum=', mn_sum)
            if no_dup_u_term:
                print(self.f_parameters[atom, 1, 1, 0, :])  # should be equal to zero
                print(self.f_parameters[atom, 0, 0, :, :])  # should be equal to zero
            if no_dup_chi_term:
                print(self.f_parameters[atom, :, 0, 0, :])  # should be equal to zero
