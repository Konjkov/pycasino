import os

import numpy as np
import numba as nb


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
        """Init."""
        self.trunc = 0
        self.eta_parameters = np.zeros((0, 0), np.float)  # uu, ud, dd order
        self.mu_parameters = nb.typed.List.empty_list(mu_parameters_type)  # u, d order
        self.phi_parameters = nb.typed.List.empty_list(phi_parameters_type)  # uu, ud, dd order
        self.theta_parameters = nb.typed.List.empty_list(theta_parameters_type)  # uu, ud, dd order
        self.mu_labels = nb.typed.List.empty_list(labels_type)
        self.phi_labels = nb.typed.List.empty_list(labels_type)
        self.eta_cutoff = np.zeros((2,), np.float)
        self.ae_cutoff = np.zeros(atoms.shape[0])

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
                        self.eta_cutoff[0] = float(line[0])
                        for i in range(1, eta_spin_dep):
                            # Optimizable (0=NO; 1=YES; 2=YES BUT NO SPIN-DEP)
                            if line[1] == '2':
                                self.eta_cutoff[i], _ = self.read_parameter()
                            else:
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
                    if line.startswith('START SET'):
                        pass
                    elif line.startswith('Label'):
                        mu_labels = np.array(self.read_ints()) - 1
                        self.mu_labels.append(mu_labels)
                    elif line.startswith('Expansion order'):
                        mu_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        mu_spin_dep = self.read_int()
                    elif line.startswith('Cutoff (a.u.)'):
                        mu_cutoff = float(f.readline().split()[0])
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
                    if line.startswith('START SET'):
                        pass
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
                        phi_cutoff = float(f.readline().split()[0])
                    elif line.startswith('Parameter values'):
                        phi_parameters = np.zeros((phi_en_order+1, phi_en_order+1, phi_ee_order+1, phi_spin_dep+1), np.float)
                        phi_mask = self.get_phi_mask(phi_parameters)
                        if not phi_irrotational:
                            self.theta_parameters = np.zeros((phi_en_order+1, phi_en_order+1, phi_ee_order+1, phi_spin_dep+1), np.float)
                        for i in range(phi_spin_dep + 1):
                            for j in range(phi_ee_order + 1):
                                for k in range(phi_en_order + 1):
                                    for l in range(phi_en_order + 1):
                                        if phi_mask[l, k, j, i]:
                                            phi_parameters[l, k, j, i], _ = self.read_parameter()
                        self.phi_parameters.append(phi_parameters)
                    elif line.startswith('END SET'):
                        pass
                elif ae_term:
                    if line.startswith('Nucleus'):
                        for atom in range(self.ae_cutoff.shape[0]):
                            line = f.readline().split()
                            self.ae_cutoff[int(line[1])-1] = float(line[2])

    @staticmethod
    def get_eta_mask(parameters):
        mask = np.ones(parameters.shape, np.bool)
        mask[1, 0] = False
        return mask

    @staticmethod
    def get_mu_mask(parameters):
        mask = np.ones(parameters.shape, np.bool)
        mask[0] = mask[1] = False
        return mask

    @staticmethod
    def get_phi_mask(parameters):
        mask = np.ones(parameters.shape, np.bool)
        for i in range(parameters.shape[3]):
            for j in range(parameters.shape[2]):
                for k in range(parameters.shape[1]):
                    for l in range(parameters.shape[0]):
                        # if (l < 2 or k < 2) and j == 0 and i == 1:
                        #     continue
                        # if (l < 1 or l > phi_en_order-2 or k < 2) and j == 1 and i == 1:
                        #     continue
                        # if abs(k-l-1) > phi_en_order-4 and j == 1 and i == 1:
                        #     continue
                        mask[l, k, j, i] = False
        return mask

    @staticmethod
    def get_theta_mask(parameters):
        mask = np.ones(parameters.shape, np.bool)
        for i in range(parameters.shape[3]):
            for j in range(parameters.shape[2]):
                for k in range(parameters.shape[1]):
                    for l in range(parameters.shape[0]):
                        mask[l, k, j, i] = False
        return mask
