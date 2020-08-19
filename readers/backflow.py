
import numpy as np
import numba as nb


class Backflow:
    """Backflow reader from file.
    Inhomogeneous backflow transformations in quantum Monte Carlo.
    P. Lopez Rıos, A. Ma, N. D. Drummond, M. D. Towler, and R. J. Needs
    """

    def __init__(self, file, atoms):
        """Init."""
        self.trunc = 0
        self.eta_parameters = np.zeros((0, 3), np.float)
        self.mu_parameters = nb.typed.List([np.zeros((0, 2), np.float)] * atoms.shape[0])
        self.phi_parameters = nb.typed.List([np.zeros((0, 0, 0, 3), np.float)] * atoms.shape[0])
        self.theta_parameters = nb.typed.List([np.zeros((0, 0, 0, 3), np.float)] * atoms.shape[0])
        self.eta_cutoff = np.zeros((2,), np.float)
        self.ae_cutoff = np.zeros(atoms.shape[0])
        self.read(file)

    def get_eta_mask(self, eta_order):
        mask = np.ones((eta_order+1, 3), dtype=np.bool)
        mask[1, 0] = False
        return mask

    def get_mu_mask(self, mu_order):
        mask = np.ones((mu_order+1, 2), dtype=np.bool)
        mask[0] = mask[1] = False
        return mask

    def get_phi_mask(self, phi_spin_dep, phi_ee_order, phi_en_order):
        mask = np.ones((phi_ee_order+1, 3), dtype=np.bool)
        for i in range(phi_spin_dep):
            for j in range(phi_ee_order):
                for k in range(phi_en_order):
                    for l in range(phi_en_order):
                        if (l < 2 or k < 2) and j == 0 and i == 1:
                            continue
                        if (l < 1 or l > phi_en_order-2 or k < 2) and j == 1 and i == 1:
                            continue
                        if abs(k-l-1) > phi_en_order-4 and j == 1 and i == 1:
                            continue
                        mask[l, k, j, i] = False
        return mask

    def get_thets_mask(self, phi_spin_dep, phi_ee_order, phi_en_order):
        mask = np.ones((phi_ee_order+1, 3), dtype=np.bool)
        for i in range(phi_spin_dep+1):
            for j in range(phi_ee_order):
                for k in range(phi_en_order):
                    for l in range(phi_en_order):
                        mask[l, k, j, i] = False
        return mask

    def read(self, file):
        """Open file and read backflow data."""
        with open(file, 'r') as f:
            eta_term = mu_term = phi_term = ae_term = False
            for line in f:
                line = line.strip()
                if line.startswith('START BACKFLOW'):
                    pass
                elif line.startswith('END BACKFLOW'):
                    break
                elif line.startswith('Truncation order'):
                    self.trunc = float(f.readline().split()[0])
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
                        eta_order = int(f.readline().split()[0])
                    elif line.startswith('Spin dep'):
                        eta_spin_dep = int(f.readline().split()[0])
                    elif line.startswith('Cut-off radii'):
                        line = f.readline().split()
                        self.eta_cutoff[0] = float(line[0])
                        for i in range(1, eta_spin_dep):
                            # Optimizable (0=NO; 1=YES; 2=YES BUT NO SPIN-DEP)
                            if line[1] == '2':
                                self.eta_cutoff[i] = float(line[0])
                            else:
                                self.eta_cutoff[i] = float(f.readline().split()[0])
                    elif line.startswith('Parameter'):
                        self.eta_parameters = np.zeros((eta_order + 1, 3), np.float)
                        eta_mask = self.get_eta_mask(eta_order)
                        for i in range(eta_spin_dep + 1):
                            for j in range(eta_order + 1):
                                if not eta_mask[j, i]:
                                    continue
                                self.eta_parameters[j, i] = float(f.readline().split()[0])
                elif mu_term:
                    if line.startswith('START SET'):
                        pass
                    elif line.startswith('Label'):
                        mu_labels = list(map(int, f.readline().split()))
                    elif line.startswith('Expansion order'):
                        mu_order = int(f.readline().split()[0])
                    elif line.startswith('Spin dep'):
                        mu_spin_dep = int(f.readline().split()[0])
                    elif line.startswith('Cutoff (a.u.)'):
                        mu_cutoff = float(f.readline().split()[0])
                    elif line.startswith('Parameter values'):
                        self.mu_parameters = np.zeros((mu_order, 2), np.float)
                        mu_mask = self.get_mu_mask(eta_order)
                        for i in range(mu_spin_dep + 1):
                            for j in range(mu_order + 1):
                                if not mu_mask[j, i]:
                                    continue
                                self.mu_parameters[j, i] = float(f.readline().split()[0])
                    elif line.startswith('END SET'):
                        mu_labels = []
                elif phi_term:
                    if line.startswith('START SET'):
                        pass
                    elif line.startswith('Label'):
                        phi_labels = list(map(int, f.readline().split()))
                    elif line.startswith('Irrotational Phi'):
                        phi_irrotational = bool(int(f.readline().split()[0]))
                    elif line.startswith('Electron-nucleus expansion order'):
                        phi_en_order = int(f.readline().split()[0])
                    elif line.startswith('Electron-electron expansion order'):
                        phi_ee_order = int(f.readline().split()[0])
                    elif line.startswith('Spin dep'):
                        phi_spin_dep = int(f.readline().split()[0])
                    elif line.startswith('Cutoff (a.u.)'):
                        phi_cutoff = float(f.readline().split()[0])
                    elif line.startswith('Parameter values'):
                        self.phi_parameters = np.zeros((phi_en_order+1, phi_en_order+1, phi_ee_order+1, phi_spin_dep+1), np.float)
                        if not phi_irrotational:
                            self.theta_parameters = np.zeros((phi_en_order+1, phi_en_order+1, phi_ee_order+1, phi_spin_dep+1), np.float)
                        for i in range(phi_spin_dep + 1):
                            for j in range(phi_en_order + 1):
                                if not mu_mask[j, i]:
                                    continue
                                self.phi_parameters[j, i] = float(f.readline().split()[0])
                    elif line.startswith('END SET'):
                        phi_labels = []
                elif ae_term:
                    if line.startswith('Nucleus'):
                        for atom in range(self.ae_cutoff.shape[0]):
                            line = f.readline().split()
                            self.ae_cutoff[int(line[2])] = float(line[2])
