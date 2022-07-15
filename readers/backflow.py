#!/usr/bin/env python3

import os

import numpy as np
import numba as nb
from readers.numerical import rref

labels_type = nb.int64[:]
mu_mask_type = nb.boolean[:, :]
phi_mask_type = nb.boolean[:, :, :, :]
theta_mask_type = nb.boolean[:, :, :, :]
mu_parameters_type = nb.float64[:, :]
phi_parameters_type = nb.float64[:, :, :, :]
theta_parameters_type = nb.float64[:, :, :, :]


class Backflow:
    """Backflow reader from file.
    Inhomogeneous backflow transformations in quantum Monte Carlo.
    P. Lopez Rıos, A. Ma, N. D. Drummond, M. D. Towler, and R. J. Needs
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

    def check_parameter(self):
        """check parameter index against Casino"""
        _, _, _, comment = self.f.readline().split()
        return list(map(int, comment.split('_')[1].split(',')))

    def read_ints(self):
        return list(map(int, self.f.readline().split()))

    def __init__(self, file, atoms):
        self.trunc = 0
        self.eta_parameters = np.zeros((0, 0), float)  # uu, ud, dd order
        self.mu_parameters = nb.typed.List.empty_list(mu_parameters_type)  # u, d order
        self.phi_parameters = nb.typed.List.empty_list(phi_parameters_type)  # uu, ud, dd order
        self.theta_parameters = nb.typed.List.empty_list(theta_parameters_type)  # uu, ud, dd order
        self.eta_cutoff = np.zeros(0)
        self.mu_cutoff = np.zeros(0)
        self.phi_cutoff = np.zeros(0)
        self.mu_labels = nb.typed.List.empty_list(labels_type)
        self.phi_labels = nb.typed.List.empty_list(labels_type)
        self.eta_cutoff = np.zeros((2,), float)
        self.ae_cutoff = np.zeros(atoms.shape[0])
        self.phi_irrotational = np.zeros(0, bool)

        if not os.path.isfile(file):
            return
        with open(file, 'r') as f:
            eta_term = mu_term = phi_term = ae_term = False
            self.f = f
            for line in f:
                line = line.strip()
                if line.startswith('START BACKFLOW'):
                    pass
                elif line.startswith('END BACKFLOW'):
                    break
                elif line.startswith('Truncation order'):
                    self.trunc = self.read_int()
                elif line.startswith('START ETA TERM'):
                    eta_term = True
                elif line.startswith('END ETA TERM'):
                    eta_term = False
                elif line.startswith('START MU TERM'):
                    mu_term = True
                elif line.startswith('END MU TERM'):
                    mu_term = False
                elif line.startswith('START PHI TERM'):
                    phi_term = True
                elif line.startswith('END PHI TERM'):
                    phi_term = False
                elif line.startswith('START AE CUTOFFS'):
                    ae_term = True
                elif line.startswith('END AE CUTOFFS'):
                    ae_term = False
                elif eta_term:
                    if line.startswith('Expansion order'):
                        eta_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        eta_spin_dep = self.read_int()
                    elif line.startswith('Cut-off radii'):
                        line = f.readline().split()
                        # Optimizable (0=NO; 1=YES; 2=YES BUT NO SPIN-DEP)
                        if line[1] == '2':
                            self.eta_cutoff = np.zeros((1, ))
                        else:
                            self.eta_cutoff = np.zeros((eta_spin_dep+1,))
                        self.eta_cutoff[0] = float(line[0])
                        for i in range(1, self.eta_cutoff.shape[0]):
                            self.eta_cutoff[i], _ = self.read_parameter()
                    elif line.startswith('Parameter'):
                        self.eta_parameters = np.zeros((eta_order+1, eta_spin_dep+1), np.float)
                        self.eta_mask = self.get_eta_mask(self.eta_parameters)
                        try:
                            for i in range(eta_spin_dep + 1):
                                for j in range(eta_order + 1):
                                    if self.eta_mask[j, i]:
                                        self.eta_parameters[j, i], _ = self.read_parameter()
                        except ValueError:
                            pass
                        self.fix_eta_parameters()
                elif mu_term:
                    if line.startswith('Number of sets'):
                        number_of_sets = self.read_ints()[0]
                        self.mu_cutoff = np.zeros(number_of_sets)
                    elif line.startswith('START SET'):
                        set_number = int(line.split()[2]) - 1
                    elif line.startswith('Label'):
                        mu_labels = np.array(self.read_ints()) - 1
                        self.mu_labels.append(mu_labels)
                    elif line.startswith('Expansion order'):
                        mu_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        mu_spin_dep = self.read_int()
                    elif line.startswith('Cutoff (a.u.)'):
                        mu_cutoff, _ = self.read_parameter()
                        self.mu_cutoff[set_number] = mu_cutoff
                    elif line.startswith('Parameter values'):
                        mu_parameters = np.zeros((mu_order+1, mu_spin_dep+1), float)
                        mu_mask = self.get_mu_mask(mu_parameters)
                        try:
                            for i in range(mu_spin_dep + 1):
                                for j in range(mu_order + 1):
                                    if mu_mask[j, i]:
                                        mu_parameters[j, i], _ = self.read_parameter()
                            self.mu_parameters.append(mu_parameters)
                        except ValueError:
                            pass
                    elif line.startswith('END SET'):
                        pass
                elif phi_term:
                    if line.startswith('Number of sets'):
                        number_of_sets = self.read_ints()[0]
                        self.phi_cutoff = np.zeros(number_of_sets)
                        self.phi_irrotational = np.zeros(number_of_sets, bool)
                    elif line.startswith('START SET'):
                        set_number = int(line.split()[2]) - 1
                    elif line.startswith('Label'):
                        phi_labels = np.array(self.read_ints()) - 1
                        self.phi_labels.append(phi_labels)
                    elif line.startswith('Type of e-N cusp conditions'):
                        phi_cusp = self.read_bool()
                    elif line.startswith('Irrotational Phi'):
                        phi_irrotational = self.read_bool()
                    elif line.startswith('Electron-nucleus expansion order'):
                        phi_en_order = self.read_int()
                    elif line.startswith('Electron-electron expansion order'):
                        phi_ee_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        phi_spin_dep = self.read_int()
                    elif line.startswith('Cutoff (a.u.)'):
                        phi_cutoff, _ = self.read_parameter()
                        self.phi_cutoff[set_number] = phi_cutoff
                    elif line.startswith('Parameter values'):
                        phi_parameters = np.zeros((phi_en_order+1, phi_en_order+1, phi_ee_order+1, phi_spin_dep+1), float)
                        theta_parameters = np.zeros((phi_en_order+1, phi_en_order+1, phi_ee_order+1, phi_spin_dep+1), float)
                        for i in range(phi_spin_dep + 1):
                            phi_mask, theta_mask = self.get_phi_theta_mask(phi_parameters, phi_cutoff, i, phi_cusp, phi_irrotational)
                            for m in range(phi_ee_order + 1):
                                for l in range(phi_en_order + 1):
                                    for k in range(phi_en_order + 1):
                                        if phi_mask[k, l, m]:
                                            phi_parameters[k, l, m, i], _ = self.read_parameter([k, l, m, i + 1])
                            for m in range(phi_ee_order + 1):
                                for l in range(phi_en_order + 1):
                                    for k in range(phi_en_order + 1):
                                        if theta_mask[k, l, m]:
                                            theta_parameters[k, l, m, i], _ = self.read_parameter([k, l, m, i + 1])
                            self.fix_phi_parameters(phi_parameters, theta_parameters, phi_cutoff, i, phi_cusp, phi_irrotational)
                            self.check_phi_constrains(phi_parameters, theta_parameters, phi_cutoff, i, phi_irrotational)
                        self.phi_parameters.append(phi_parameters)
                        self.theta_parameters.append(theta_parameters)
                        self.phi_irrotational[set_number] = phi_irrotational
                    elif line.startswith('END SET'):
                        pass
                elif ae_term:
                    if line.startswith('Nucleus'):
                        for atom in range(self.ae_cutoff.shape[0]):
                            line = f.readline().split()
                            self.ae_cutoff[int(line[1])-1] = float(line[2])

    @staticmethod
    def get_eta_mask(parameters):
        """To obey the cusp conditions,
        we constrain the parallel-spin η(rij) function to have zero derivative at rij = 0,
        while the antiparallel-spin η function may have a nonzero derivative"""
        mask = np.ones(parameters.shape, np.bool)
        mask[1, 0] = False
        if parameters.shape[1] == 3:
            mask[1, 2] = False
        return mask

    @staticmethod
    def get_mu_mask(parameters):
        mask = np.ones(parameters.shape, np.bool)
        mask[0] = mask[1] = False
        return mask

    def construct_c_matrix(self, phi_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational):
        """C-matrix has the following rows:
        ...
        copy-paste from /CASINO/src/pbackflow.f90 SUBROUTINE construct_C
        """
        phi_en_order = phi_parameters.shape[0] - 1
        phi_ee_order = phi_parameters.shape[2] - 1

        ee_constrains = 2 * phi_en_order + 1
        en_constrains = phi_en_order + phi_ee_order + 1

        offset = 0
        phi_constraints = 6 * en_constrains - 2
        if phi_cusp and spin_dep in (0, 2):
            phi_constraints += ee_constrains
            offset += ee_constrains

        theta_constraints = 5 * en_constrains + ee_constrains - 2
        n_constraints = phi_constraints + theta_constraints
        if phi_irrotational:
            n_constraints += ((phi_en_order + 3) * (phi_ee_order + 2) - 4) * (phi_en_order + 1)
            if self.trunc == 0:
                n_constraints -= (phi_en_order + 1) * (phi_ee_order + 1)

        parameters_size = 2 * (phi_parameters.shape[0] * phi_parameters.shape[1] * phi_parameters.shape[2])
        c = np.zeros((n_constraints, parameters_size))
        p = 0
        for m in range(phi_parameters.shape[2]):
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0]):
                    if phi_cusp and spin_dep in (0, 2):  # e-e cusp
                        if m == 1:
                            c[k+l, p] = 1
                    if l == 0:
                        c[k+m + offset + en_constrains, p] = 1
                        if m > 0:
                            c[k+m-1 + offset + 5 * en_constrains - 1, p] = m
                    elif l == 1:
                        c[k+m + offset + 3 * en_constrains, p] = 1
                    if k == 0:
                        c[l+m + offset, p] = 1
                        if m > 0:
                            c[l+m-1 + offset + 4 * en_constrains, p] = m
                    elif k == 1:
                        c[l+m + offset + 2 * en_constrains, p] = 1
                    p += 1

        offset = phi_constraints
        for m in range(phi_parameters.shape[2]):
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0]):
                    if m == 1:
                        c[k+l + offset, p] = 1
                    if l == 0:
                        c[k+m + offset + ee_constrains + 2 * en_constrains, p] = -self.trunc/phi_cutoff
                        if m > 0:
                            c[k+m-1 + offset + ee_constrains + 4 * en_constrains - 1, p] = m
                    elif l == 1:
                        c[k+m + offset + ee_constrains + 2 * en_constrains, p] = 1
                    if k == 0:
                        c[l+m + offset + ee_constrains, p] = 1
                        if m > 0:
                            c[l+m-1 + offset + ee_constrains + 3 * en_constrains, p] = m
                    elif k == 1:
                        c[l+m + offset + ee_constrains + en_constrains, p] = 1
                    p += 1

        n = phi_constraints + theta_constraints
        if phi_irrotational:
            p = 0
            inc_k = 1
            inc_l = inc_k * (phi_en_order+1)
            inc_m = inc_l * (phi_en_order+1)
            nphi = inc_m * (phi_ee_order+1)
            for m in range(phi_parameters.shape[2]):
                for l in range(phi_parameters.shape[1]):
                    for k in range(phi_parameters.shape[0]):
                        if self.trunc > 0:
                            if m > 0:
                                c[n, p - inc_m] = self.trunc + k
                                if k < phi_en_order:
                                    c[n, p + inc_k - inc_m] = -phi_cutoff * (k+1)
                            if m < phi_ee_order:
                                if k > 1:
                                    c[n, p + nphi - 2*inc_k + inc_m] = -(m+1)
                                if k > 0:
                                    c[n, p + nphi - inc_k + inc_m] = phi_cutoff * (m+1)
                        else:
                            if m > 0 and k < phi_en_order:
                                c[n, p + inc_k - inc_m] = k+1
                            if k > 0 and m < phi_ee_order:
                                c[n, p + nphi - inc_k + inc_m] = -(m+1)
                        p += 1
                        n += 1
            if self.trunc > 0:
                # Same as above, for m=N_ee+1...
                p = phi_ee_order * (phi_en_order+1)**2
                for l in range(phi_parameters.shape[1]):
                    for k in range(phi_parameters.shape[0]):
                        c[n, p] = self.trunc + k
                        if k < phi_en_order:
                            c[n, p+inc_k] = -phi_cutoff * (k+1)
                        p += 1
                        n += 1
                # ...for k=N_eN+1...
                p = phi_en_order-1
                for m in range(phi_parameters.shape[2]-1):
                    for l in range(phi_parameters.shape[1]):
                        c[n, p+nphi+inc_m] = -(m+1)
                        c[n, p+nphi+inc_k+inc_m] = phi_cutoff * (m+1)
                        p += inc_l
                        n += 1
                # ...and for k=N_eN+2.
                p = phi_en_order
                for m in range(phi_parameters.shape[2]-1):
                    for l in range(phi_parameters.shape[1]):
                        c[n, p+nphi+inc_m] = -(m+1)
                        p += inc_l
                        n += 1
            else:
                # Same as above, for m=N_ee+1...
                p = phi_ee_order * (phi_en_order+1)**2
                for l in range(phi_parameters.shape[1]):
                    for k in range(phi_parameters.shape[0]-1):
                        c[n, p+inc_k] = 1  # just zeroes the corresponding param
                        p += 1
                        n += 1
                # ...and for k=N_eN+1.
                p = phi_en_order-1
                for m in range(phi_parameters.shape[2]-1):
                    for l in range(phi_parameters.shape[1]):
                        c[n, p+nphi+inc_m] = 1  # just zeroes the corresponding param
                        p += inc_l
                        n += 1

        assert n == n_constraints
        return c

    def get_phi_theta_mask(self, phi_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational):
        """Mask dependent parameters in phi-term.
        """
        c = self.construct_c_matrix(phi_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational)
        _, pivot_positions = rref(c)

        p = 0
        phi_mask = np.zeros(shape=phi_parameters.shape[:3], dtype=bool)
        for m in range(phi_parameters.shape[2]):
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0]):
                    if p not in pivot_positions:
                        phi_mask[k, l, m] = True
                    p += 1

        theta_mask = np.zeros(shape=phi_parameters.shape[:3], dtype=bool)
        for m in range(phi_parameters.shape[2]):
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0]):
                    if p not in pivot_positions:
                        theta_mask[k, l, m] = True
                    p += 1
        return phi_mask, theta_mask

    def fix_eta_parameters(self):
        """Fix eta-term parameters"""
        C = self.trunc
        L = self.eta_cutoff[0]
        self.eta_parameters[1, 0] = C * self.eta_parameters[0, 0] / L
        if self.eta_parameters.shape[1] == 3:
            L = self.eta_cutoff[2] or self.eta_cutoff[0]
            self.eta_parameters[1, 2] = C * self.eta_parameters[0, 2] / L

    def fix_mu_parameters(self):
        """Fix mu-term parameters"""

    def fix_phi_parameters(self, phi_parameters, theta_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational):
        """Fix phi-term parameters"""
        c = self.construct_c_matrix(phi_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational)
        c, pivot_positions = rref(c)
        c = c[:pivot_positions.size, :]
        mask = np.zeros(shape=phi_parameters.size, dtype=bool)
        mask[pivot_positions] = True

        b = np.zeros((c.shape[0], ))
        p = 0
        for m in range(phi_parameters.shape[2]):
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0]):
                    if p not in pivot_positions:
                        for temp in range(c.shape[0]):
                            b[temp] -= c[temp, p] * phi_parameters[k, l, m, spin_dep]
                    p += 1

        for m in range(phi_parameters.shape[2]):
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0]):
                    if p not in pivot_positions:
                        for temp in range(c.shape[0]):
                            b[temp] -= c[temp, p] * theta_parameters[k, l, m, spin_dep]
                    p += 1

        x = np.linalg.solve(c[:, mask], b)

        p = 0
        temp = 0
        for m in range(phi_parameters.shape[2]):
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0]):
                    if temp in pivot_positions:
                        phi_parameters[k, l, m, spin_dep] = x[p]
                        p += 1
                    temp += 1

        for m in range(phi_parameters.shape[2]):
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0]):
                    if temp in pivot_positions:
                        theta_parameters[k, l, m, spin_dep] = x[p]
                        p += 1
                    temp += 1

    def check_phi_constrains(self, phi_parameters, theta_parameters, f_cutoff, i, phi_irrotational):
        """"""
        phi_en_order = phi_parameters.shape[0] - 1
        phi_ee_order = phi_parameters.shape[2] - 1

        lm_phi_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
        lm_phi_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
        lm_phi_m_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
        lm_theta_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
        lm_theta_m_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))

        for l in range(phi_parameters.shape[1]):
            for m in range(phi_parameters.shape[2]):
                lm_phi_sum[l + m] += self.trunc * phi_parameters[0, l, m, i] - f_cutoff * phi_parameters[1, l, m, i]
                lm_phi_ae_sum[l + m] += phi_parameters[0, l, m, i]
                lm_phi_m_ae_sum[l + m] += m * phi_parameters[0, l, m, i]
                lm_theta_ae_sum[l + m] += theta_parameters[0, l, m, i]
                lm_theta_m_ae_sum[l + m] += m * theta_parameters[0, l, m, i]

        np.abs(lm_phi_sum).max() > 1e-14 and print(f'lm_phi_sum = {lm_phi_sum}')
        np.abs(lm_phi_ae_sum).max() > 1e-14 and print(f'lm_phi_ae_sum = {lm_phi_ae_sum}'),
        np.abs(lm_phi_m_ae_sum).max() > 1e-14 and print(f'lm_phi_m_ae_sum = {lm_phi_m_ae_sum}')
        np.abs(lm_theta_ae_sum).max() > 1e-14 and print(f'lm_theta_ae_sum = {lm_theta_ae_sum}')
        np.abs(lm_theta_m_ae_sum).max() > 1e-14 and print(f'lm_theta_m_ae_sum = {lm_theta_m_ae_sum}')

        km_phi_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
        km_theta_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
        km_phi_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
        km_phi_m_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
        km_theta_m_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))

        for k in range(phi_parameters.shape[0]):
            for m in range(phi_parameters.shape[2]):
                km_phi_sum[k + m] += self.trunc * phi_parameters[k, 0, m, i] - f_cutoff * phi_parameters[k, 1, m, i]
                km_theta_sum[k + m] += self.trunc * theta_parameters[k, 0, m, i] - f_cutoff * theta_parameters[k, 1, m, i]
                km_phi_ae_sum[k + m] += phi_parameters[k, 0, m, i]
                km_phi_m_ae_sum[k + m] += m * phi_parameters[k, 0, m, i]
                km_theta_m_ae_sum[k + m] += m * theta_parameters[k, 0, m, i]

        np.abs(km_phi_sum).max() > 1e-14 and print(f'km_phi_sum = {km_phi_sum}')
        np.abs(km_theta_sum).max() > 1e-14 and print(f'km_theta_sum = {km_theta_sum}')
        np.abs(km_phi_ae_sum).max() > 1e-14 and print(f'km_phi_ae_sum = {km_phi_ae_sum}')
        np.abs(km_phi_m_ae_sum).max() > 1e-14 and print(f'km_phi_m_ae_sum = {km_phi_m_ae_sum}')
        np.abs(km_theta_m_ae_sum).max() > 1e-14 and print(f'km_theta_m_ae_sum = {km_theta_m_ae_sum}')

        if i in (0, 2):
            kl_phi_sum = np.zeros((2 * phi_en_order + + 1,))
            kl_theta_sum = np.zeros((2 * phi_en_order + + 1,))
            for k in range(phi_parameters.shape[0]):
                for l in range(phi_parameters.shape[1]):
                    kl_phi_sum[k + l] += phi_parameters[k, l, 1, i]
                    kl_theta_sum[k + l] += theta_parameters[k, l, 1, i]

            np.abs(kl_phi_sum).max() > 1e-14 and print(f'kl_phi_sum = {kl_phi_sum}')
            np.abs(kl_theta_sum).max() > 1e-14 and print(f'kl_theta_sum = {kl_theta_sum}')

        if phi_irrotational:
            irrot_sum = np.zeros((phi_parameters.shape[0] + 2, phi_parameters.shape[1], phi_parameters.shape[2] + 1))
            for k in range(phi_parameters.shape[0] + 2):
                for l in range(phi_parameters.shape[1]):
                    for m in range(phi_parameters.shape[2] + 1):
                        if self.trunc > 0:
                            if m - 1 >= 0:
                                if phi_parameters.shape[0] > k:
                                    irrot_sum[k, l, m] += (self.trunc + k) * phi_parameters[k, l, m - 1, i]
                                if phi_parameters.shape[0] > k + 1:
                                    irrot_sum[k, l, m] -= f_cutoff * (k + 1) * phi_parameters[k + 1, l, m - 1, i]
                            if phi_parameters.shape[2] > m + 1:
                                if k - 2 >= 0:
                                    irrot_sum[k, l, m] -= (m + 1) * theta_parameters[k - 2, l, m + 1, i]
                                if phi_parameters.shape[0] > k - 1 >= 0:
                                    irrot_sum[k, l, m] += f_cutoff * (m + 1) * theta_parameters[k - 1, l, m + 1, i]
                        else:
                            if phi_parameters.shape[0] > k + 1 and phi_parameters.shape[2] > m - 1 >= 0:
                                irrot_sum[k, l, m] += (k + 1) * phi_parameters[k + 1, l, m - 1, i]
                            if phi_parameters.shape[0] > k - 1 >= 0 and phi_parameters.shape[2] > m + 1:
                                irrot_sum[k, l, m] -= (m + 1) * theta_parameters[k - 1, l, m + 1, i]

            np.abs(irrot_sum).max() > 1e-14 and print('irrot_sum', irrot_sum)


if __name__ == '__main__':
    """Read Backflow terms
    """
    debug = False
    atom_positions = np.array([[0, 0, 0]])

    for phi_term in (
        '21', '22', '23', '24', '25',
        '31', '32', '33', '34', '35',
        '41', '42', '43', '44', '45',
        '51', '52', '53', '54', '55',
    ):
        print(phi_term)
        # path = f'test/backflow/0_1_0/{phi_term}/correlation.out.1'
        # path = f'test/backflow/3_1_0/{phi_term}/correlation.out.1'
        # path = f'test/backflow/0_1_1/{phi_term}/correlation.out.1'
        path = f'test/backflow/3_1_1/{phi_term}/correlation.out.1'
        Backflow(path, atom_positions)
