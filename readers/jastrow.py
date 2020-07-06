#!/usr/bin/env python3

import numpy as np


class Jastrow:
    """Jastrow reader from file.
    CASINO manual
    22.2 The u, χ and f terms in the Jastrow factor
    """

    def __init__(self, file):
        jastrow = False
        self.u_parameters = self.chi_parameters = self.f_parameters = None
        u_term = chi_term = f_term = False
        with open(file, 'r') as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.strip().startswith('START JASTROW'):
                    jastrow = True
                elif line.strip().startswith('Truncation order'):
                    self.trunc = float(f.readline().split()[0])
                elif line.strip().startswith('START U TERM'):
                    u_term = True
                elif line.strip().startswith('START CHI TERM'):
                    chi_term = True
                elif line.strip().startswith('START F TERM'):
                    f_term = True
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
                        for i in range(u_spin_dep+1):
                            for l in range(u_order+1):
                                if l == 1:
                                    continue
                                self.u_parameters[l][i] = float(f.readline().split()[0])
                elif chi_term:
                    if line.strip().startswith('Expansion order'):
                        chi_order = int(f.readline())
                    elif line.strip().startswith('Spin dep'):
                        chi_spin_dep = int(f.readline())
                    elif line.strip().startswith('Cutoff'):
                        self.chi_cutoff = float(f.readline().split()[0])
                    elif line.strip().startswith('Parameter'):
                        # u, d
                        self.chi_parameters = np.zeros((chi_order+1, 2), np.float)
                        for i in range(chi_spin_dep+1):
                            for m in range(chi_order):
                                if m == 1:
                                    continue
                                self.chi_parameters[m][i] = float(f.readline().split()[0])
                elif f_term:
                    if line.strip().startswith('Electron-nucleus expansion order'):
                        f_en_order = int(f.readline())
                    if line.strip().startswith('Electron-electron expansion order'):
                        f_ee_order = int(f.readline())
                    elif line.strip().startswith('Spin dep'):
                        f_spin_dep = int(f.readline())
                    elif line.strip().startswith('Parameter'):
                        self.f_parameters = np.zeros((f_en_order+1, f_en_order+1, f_ee_order+1, 3), np.float)
                        for i in range(f_spin_dep+1):
                            for n in range(f_ee_order + 1):
                                for m in range(f_en_order + 1):
                                    for l in range(m, f_en_order + 1):
                                        if n == 0 and (l == 0 or m == 0):
                                            continue
                                        # sum(γlm1I) = 0
                                        if n == 1 and (l - m >= 2 or l == 0 or m == 0 or l == f_en_order or m == f_en_order):
                                            continue
                                        # ???
                                        if l == f_en_order and m == 0:
                                            continue
                                        line = f.readline()
                                        print(line[:-1], l, m, n, i)
                                        self.f_parameters[l][m][n][i] = float(line.split()[0])
                                        if m != l:
                                            #  γlmnI = γmlnI
                                            self.f_parameters[m][l][n][i] = self.f_parameters[l][m][n][i]

            if not jastrow:
                print('No JASTROW section found')
                exit(0)
        if self.u_parameters is not None:
            if u_spin_dep == 0:
                self.u_parameters[:, 2] = self.u_parameters[:, 1] = self.u_parameters[:, 0]
            elif u_spin_dep == 1:
                self.u_parameters[:, 2] = self.u_parameters[:, 0]
            self.u_parameters[1] = np.array([1/4, 1/2, 1/4])/(-self.u_cutoff)**self.trunc + self.u_parameters[0]*self.trunc/self.u_cutoff
        if self.chi_parameters is not None:
            Z = 1
            self.chi_parameters[1] = Z/(-self.chi_cutoff)**self.trunc + self.u_parameters[0]*self.trunc/self.chi_cutoff
        if self.f_parameters is not None:
            pass
