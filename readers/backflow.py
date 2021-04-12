import os

import numpy as np
import numba as nb
import sympy as sp


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

    def read_parameter(self):
        # https://www.python.org/dev/peps/pep-3132/
        parameter, mask, *_ = self.f.readline().split()
        return float(parameter), bool(int(mask))

    def read_ints(self):
        return list(map(int, self.f.readline().split()))

    def __init__(self, file, atoms):
        self.trunc = 0
        self.eta_parameters = np.zeros((0, 0), np.float)  # uu, ud, dd order
        self.mu_parameters = nb.typed.List.empty_list(mu_parameters_type)  # u, d order
        self.phi_parameters = nb.typed.List.empty_list(phi_parameters_type)  # uu, ud, dd order
        self.theta_parameters = nb.typed.List.empty_list(theta_parameters_type)  # uu, ud, dd order
        self.eta_cutoff = np.zeros(0)
        self.mu_cutoff = np.zeros(0)
        self.phi_cutoff = np.zeros(0)
        self.mu_labels = nb.typed.List.empty_list(labels_type)
        self.phi_labels = nb.typed.List.empty_list(labels_type)
        self.eta_cutoff = np.zeros((2,), np.float)
        self.ae_cutoff = np.zeros(atoms.shape[0])
        self.phi_irrotational = np.zeros(0, np.bool)

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
                        mu_parameters = np.zeros((mu_order+1, mu_spin_dep+1), np.float)
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
                        self.phi_irrotational = np.zeros(number_of_sets, np.bool)
                    elif line.startswith('START SET'):
                        set_number = int(line.split()[2]) - 1
                    elif line.startswith('Label'):
                        phi_labels = np.array(self.read_ints()) - 1
                        self.phi_labels.append(phi_labels)
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
                        phi_parameters = np.zeros((phi_en_order+1, phi_en_order+1, phi_ee_order+1, phi_spin_dep+1), np.float)
                        phi_mask = self.get_phi_mask(phi_parameters, phi_irrotational)
                        theta_parameters = np.zeros((phi_en_order+1, phi_en_order+1, phi_ee_order+1, phi_spin_dep+1), np.float)
                        theta_mask = self.get_theta_mask(phi_parameters, phi_irrotational)
                        for i in range(phi_spin_dep + 1):
                            phi_mask_new, theta_mask_new = self.get_phi_theta_mask(phi_parameters, phi_cutoff, phi_spin_dep, phi_irrotational)
                            for j in range(phi_ee_order + 1):
                                for k in range(phi_en_order + 1):
                                    for l in range(phi_en_order + 1):
                                        if phi_mask[l, k, j, i] != phi_mask_new[l, k, j]:
                                            print(l, k, j, i+1, phi_mask[l, k, j, i], phi_mask_new[l, k, j])
                                        if phi_mask[l, k, j, i]:
                                            phi_parameters[l, k, j, i], _ = self.read_parameter()
                            for j in range(phi_ee_order + 1):
                                for k in range(phi_en_order + 1):
                                    for l in range(phi_en_order + 1):
                                        if theta_mask[l, k, j, i] != theta_mask_new[l, k, j]:
                                            print(l, k, j, i+1, theta_mask[l, k, j, i], theta_mask_new[l, k, j])
                                        if theta_mask[l, k, j, i]:
                                            theta_parameters[l, k, j, i], _ = self.read_parameter()
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

    @staticmethod
    def get_phi_mask(parameters, phi_irrotational):
        mask = np.ones(parameters.shape, np.bool)
        if phi_irrotational:
            return np.zeros(parameters.shape, np.bool)
        phi_en_order = parameters.shape[0] - 1
        for m in range(parameters.shape[2]):
            for l in range(parameters.shape[1]):
                for k in range(parameters.shape[0]):
                    if m == 0 and (k < 2 or l < 2):
                        mask[k, l, m] = False
                    # sum(φkl1) = 0
                    elif m == 1 and (k == 0 or l < 2 or k == phi_en_order or l == 2 and k == phi_en_order - 1):
                        mask[k, l, m] = False
                    elif m == 2 and (l == 0 and k == 0):
                        mask[k, l, m] = False
                    elif l == phi_en_order and k == 0 or l == 0 and k == phi_en_order:
                        mask[k, l, m] = False
                    elif l == phi_en_order - 1 and k == 0 or l == 0 and k == phi_en_order - 1:
                        mask[k, l, m] = False
                    elif l == 1 and k == phi_en_order or l == phi_en_order and k == 1:
                        mask[k, l, m] = False
        return mask

    @staticmethod
    def get_theta_mask(parameters, phi_irrotational):
        """Mask dependent parameters in theta-term.
        my own version
        """
        mask = np.ones(parameters.shape, np.bool)
        phi_en_order = parameters.shape[0] - 1
        for m in range(parameters.shape[2]):
            for l in range(parameters.shape[1]):
                for k in range(parameters.shape[0]):
                    if m > 0 and phi_irrotational:
                        mask[k, l, m] = False
                    elif m == 0 and (k < 2 or l == 0):
                        mask[k, l, m] = False
                    # sum(θkl1) = 0
                    elif m == 1 and (k == 0 or l < 2 or k == phi_en_order or l == 2 and k == phi_en_order - 1):
                        mask[k, l, m] = False
                    elif m == 2 and (l == 0 and k == 0):
                        mask[k, l, m] = False
                    elif m > 0 and (l < 2 and k == phi_en_order):
                        mask[k, l, m] = False
                    elif l == phi_en_order and k == 0 or l == phi_en_order - 1 and k == 0 or l == phi_en_order and k == 1:
                        mask[k, l, m] = False
        return mask

    def get_phi_theta_mask(self, parameters, phi_cutoff, phi_spin_dep, phi_irrotational):
        """Mask dependent parameters in phi-term.
        copy-paste from /CASINO/src/pbackflow.f90 SUBROUTINE construct_C
        """
        phi_en_order = parameters.shape[0] - 1
        phi_ee_order = parameters.shape[1] - 1

        Cee = phi_spin_dep != 0  # if spin-like electrons
        ee_constrains = 2 * phi_en_order + 1
        en_constrains = phi_en_order + phi_ee_order + 1

        offset = 0
        phi_constraints = 6 * en_constrains - 1
        if Cee:
            phi_constraints += ee_constrains
            offset += ee_constrains

        theta_constraints = 5 * en_constrains + ee_constrains - 1
        n_constraints = phi_constraints + theta_constraints

        parameter_size = 2 * (parameters.shape[0] * parameters.shape[1] * parameters.shape[2])
        c = np.zeros((n_constraints, parameter_size))
        p = 0
        for m in range(parameters.shape[2]):
            for l in range(parameters.shape[1]):
                for k in range(parameters.shape[0]):
                    if Cee:  # e-e cusp
                        if m == 1:
                            c[k+l+1, p] = 1
                    if l == 0:
                        c[k+m+1 + offset + en_constrains, p] = 1
                        if m > 0:
                            c[k+m + offset + 5 * en_constrains - 1, p] = m
                    elif l == 1:
                        c[k+m+1 + offset + 3 * en_constrains, p] = 1
                    if k == 0:
                        c[l+m+1 + offset, p] = 1
                        if m > 0:
                            c[l+m + offset + 4 * en_constrains, p] = m
                    elif k == 1:
                        c[l+m+1 + offset + 2 * en_constrains, p] = 1
                    p += 1

        offset = phi_constraints
        for m in range(parameters.shape[2]):
            for l in range(parameters.shape[1]):
                for k in range(parameters.shape[0]):
                    if m == 1:
                        c[k+l+1 + offset, p] = 1
                    if l == 0:
                        c[k+m+1 + offset + ee_constrains + 2 * en_constrains, p] = -self.trunc/phi_cutoff
                        if m > 0:
                            c[k+m + offset + ee_constrains + 4 * en_constrains - 1, p] = m
                    elif l == 1:
                        c[k+m+1 + offset + ee_constrains + 2 * en_constrains, p] = 1
                    if k == 0:
                        c[l+m+1 + offset + ee_constrains, p] = 1
                        if m > 0:
                            c[l+m + offset + ee_constrains + 3 * en_constrains, p] = m
                    elif k == 1:
                        c[l+m+1 + offset + ee_constrains + en_constrains, p] = 1
                    p += 1

        if phi_irrotational:
            offset_irrot = offset + ee_constrains + 5 * en_constrains - 1
            p = 0
            for m in range(parameters.shape[2]):
                for l in range(parameters.shape[1]):
                    for k in range(parameters.shape[0]):
                        p += 1

        _, pivot = sp.Matrix(c).rref(iszerofunc=lambda x: abs(x) < 1e-10)

        p = 0
        phi_mask = np.zeros(parameters.shape[:3], np.bool)
        for n in range(parameters.shape[2]):
            for m in range(parameters.shape[1]):
                for l in range(parameters.shape[0]):
                    if p not in pivot:
                        phi_mask[l, m, n] = True
                    p += 1

        theta_mask = np.zeros(parameters.shape[:3], np.bool)
        for n in range(parameters.shape[2]):
            for m in range(parameters.shape[1]):
                for l in range(parameters.shape[0]):
                    if p not in pivot:
                        theta_mask[l, m, n] = True
                    p += 1
        return phi_mask, theta_mask
