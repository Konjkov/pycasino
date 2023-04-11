#!/usr/bin/env python3

import os

import numpy as np
import numba as nb
from readers.numerical import rref

labels_type = nb.int64[:]
mu_parameters_type = nb.float64[:, :]
mu_parameters_optimizable_type = nb.boolean[:, :]
phi_parameters_type = nb.float64[:, :, :, :]
phi_parameters_optimizable_type = nb.boolean[:, :, :, :]
theta_parameters_type = nb.float64[:, :, :, :]
theta_parameters_optimizable_type = nb.boolean[:, :, :, :]

backflow_template = """\
 START BACKFLOW
 Title
  {title}
 Truncation order
   {trunc}
 {terms}\
 START AE CUTOFFS
 Nucleus ; Set ; Cutoff length     ;  Optimizable (0=NO; 1=YES)
 {ae_cutoffs}
 END AE CUTOFFS
 END BACKFLOW

"""

eta_term_template = """\
 START ETA TERM
 {eta_set}
 END ETA TERM
"""

eta_set_template = """\
Expansion order
   {eta_order}
 Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
   {eta_spin_dep}
 Cut-off radii ;      Optimizable (0=NO; 1=YES; 2=YES BUT NO SPIN-DEP)
   {eta_cutoff:.16f}                {eta_cutoff_optimizable}
 Parameter values  ;  Optimizable (0=NO; 1=YES)
  {eta_parameters}"""

mu_term_template = """\
 START MU TERM
 Number of sets ; labelling (1->atom in s. cell; 2->atom in p. cell; 3->species)
  {n_mu_sets} 1
 {mu_sets}
 END MU TERM
"""

mu_set_template = """\
START SET {n_set}
 Number of atoms in set
   {n_atoms}
 Labels of the atoms in this set
   {mu_labels}
 Type of e-N cusp conditions (0->PP/cuspless AE; 1->AE with cusp)
   {mu_cusp}
 Expansion order
   {mu_order}
 Spin dep (0->u=d; 1->u/=d)
   {mu_spin_dep}
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   {mu_cutoff:.16f}                {mu_cutoff_optimizable}
 Parameter values  ;  Optimizable (0=NO; 1=YES)
  {mu_parameters}
 END SET {n_set}"""

phi_term_template = """\
 START PHI TERM
 Number of sets ; labelling (1->atom in s. cell; 2->atom in p. cell; 3->species)
  {n_phi_sets} 1
 {phi_sets}
 END PHI TERM
"""

phi_set_template = """\
START SET {n_set}
 Number of atoms in set
   {n_atoms}
 Label of the atom in this set
   {phi_labels}
 Type of e-N cusp conditions (0=PP; 1=AE)
   {phi_cusp}
 Irrotational Phi term (0=NO; 1=YES)
   {phi_irrotational}
 Electron-nucleus expansion order N_eN
   {phi_en_order}
 Electron-electron expansion order N_ee
   {phi_ee_order}
 Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
   {phi_spin_dep}
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   {phi_cutoff:.16f}                {phi_cutoff_optimizable}
 Parameter values  ;  Optimizable (0=NO; 1=YES)
  {phi_parameters}
 END SET {n_set}"""


@nb.njit(nogil=True, parallel=False, cache=True)
def construct_c_matrix(trunc, phi_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational):
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
        if trunc == 0:
            n_constraints -= (phi_en_order + 1) * (phi_ee_order + 1)

    parameters_size = 2 * (phi_parameters.shape[0] * phi_parameters.shape[1] * phi_parameters.shape[2])
    c = np.zeros((n_constraints, parameters_size))
    p = 0
    # Do Phi bit of the constraint matrix.
    for m in range(phi_parameters.shape[2]):
        for l in range(phi_parameters.shape[1]):
            for k in range(phi_parameters.shape[0]):
                if phi_cusp and spin_dep in (0, 2):  # e-e cusp
                    if m == 1:
                        c[k + l, p] = 1
                if l == 0:
                    c[k + m + offset + en_constrains, p] = 1
                    if m > 0:
                        c[k + m - 1 + offset + 5 * en_constrains - 1, p] = m
                elif l == 1:
                    c[k + m + offset + 3 * en_constrains, p] = 1
                if k == 0:
                    c[l + m + offset, p] = 1
                    if m > 0:
                        c[l + m - 1 + offset + 4 * en_constrains, p] = m
                elif k == 1:
                    c[l + m + offset + 2 * en_constrains, p] = 1
                p += 1
    # Do Theta bit of the constraint matrix.
    offset = phi_constraints
    for m in range(phi_parameters.shape[2]):
        for l in range(phi_parameters.shape[1]):
            for k in range(phi_parameters.shape[0]):
                if m == 1:
                    c[k + l + offset, p] = 1
                if l == 0:
                    c[k + m + offset + ee_constrains + 2 * en_constrains, p] = -trunc / phi_cutoff
                    if m > 0:
                        c[k + m - 1 + offset + ee_constrains + 4 * en_constrains - 1, p] = m
                elif l == 1:
                    c[k + m + offset + ee_constrains + 2 * en_constrains, p] = 1
                if k == 0:
                    c[l + m + offset + ee_constrains, p] = 1
                    if m > 0:
                        c[l + m - 1 + offset + ee_constrains + 3 * en_constrains, p] = m
                elif k == 1:
                    c[l + m + offset + ee_constrains + en_constrains, p] = 1
                p += 1
    # Do irrotational bit of the constraint matrix.
    n = phi_constraints + theta_constraints
    if phi_irrotational:
        p = 0
        inc_k = 1
        inc_l = inc_k * (phi_en_order + 1)
        inc_m = inc_l * (phi_en_order + 1)
        nphi = inc_m * (phi_ee_order + 1)
        for m in range(phi_parameters.shape[2]):
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0]):
                    if trunc > 0:
                        if m > 0:
                            c[n, p - inc_m] = trunc + k
                            if k < phi_en_order:
                                c[n, p + inc_k - inc_m] = -phi_cutoff * (k + 1)
                        if m < phi_ee_order:
                            if k > 1:
                                c[n, p + nphi - 2 * inc_k + inc_m] = -(m + 1)
                            if k > 0:
                                c[n, p + nphi - inc_k + inc_m] = phi_cutoff * (m + 1)
                    else:
                        if m > 0 and k < phi_en_order:
                            c[n, p + inc_k - inc_m] = k + 1
                        if k > 0 and m < phi_ee_order:
                            c[n, p + nphi - inc_k + inc_m] = -(m + 1)
                    p += 1
                    n += 1
        if trunc > 0:
            # Same as above, for m=N_ee+1...
            p = phi_ee_order * (phi_en_order + 1) ** 2
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0]):
                    c[n, p] = trunc + k
                    if k < phi_en_order:
                        c[n, p + inc_k] = -phi_cutoff * (k + 1)
                    p += 1
                    n += 1
            # ...for k=N_eN+1...
            p = phi_en_order - 1
            for m in range(phi_parameters.shape[2] - 1):
                for l in range(phi_parameters.shape[1]):
                    c[n, p + nphi + inc_m] = -(m + 1)
                    c[n, p + nphi + inc_k + inc_m] = phi_cutoff * (m + 1)
                    p += inc_l
                    n += 1
            # ...and for k=N_eN+2.
            p = phi_en_order
            for m in range(phi_parameters.shape[2] - 1):
                for l in range(phi_parameters.shape[1]):
                    c[n, p + nphi + inc_m] = -(m + 1)
                    p += inc_l
                    n += 1
        else:
            # Same as above, for m=N_ee+1...
            p = phi_ee_order * (phi_en_order + 1) ** 2
            for l in range(phi_parameters.shape[1]):
                for k in range(phi_parameters.shape[0] - 1):
                    c[n, p + inc_k] = 1  # just zeroes the corresponding param
                    p += 1
                    n += 1
            # ...and for k=N_eN+1.
            p = phi_en_order - 1
            for m in range(phi_parameters.shape[2] - 1):
                for l in range(phi_parameters.shape[1]):
                    c[n, p + nphi + inc_m] = 1  # just zeroes the corresponding param
                    p += inc_l
                    n += 1

    assert n == n_constraints
    return c


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
        return float(parameter), int(mask)

    def check_parameter(self):
        """check parameter index against Casino"""
        _, _, _, comment = self.f.readline().split()
        return list(map(int, comment.split('_')[1].split(',')))

    def read_ints(self):
        return list(map(int, self.f.readline().split()))

    def __init__(self):
        self.trunc = 0
        self.eta_parameters = np.zeros((0, 0), dtype=float)  # uu, ud, dd order
        self.eta_parameters_optimizable = np.zeros(shape=(0, 0), dtype=bool)  # uu, ud, dd order
        self.mu_parameters = nb.typed.List.empty_list(mu_parameters_type)  # u, d order
        self.mu_parameters_optimizable = nb.typed.List.empty_list(mu_parameters_optimizable_type)  # u, d order
        self.phi_parameters = nb.typed.List.empty_list(phi_parameters_type)  # uu, ud, dd order
        self.phi_parameters_optimizable = nb.typed.List.empty_list(phi_parameters_optimizable_type)  # uu, ud, dd order
        self.theta_parameters = nb.typed.List.empty_list(theta_parameters_type)  # uu, ud, dd order
        self.theta_parameters_optimizable = nb.typed.List.empty_list(theta_parameters_optimizable_type)  # uu, ud, dd order
        self.eta_cutoff = np.zeros(shape=0, dtype=[('value', float), ('optimizable', bool)])
        self.mu_cutoff = np.zeros(shape=0, dtype=[('value', float), ('optimizable', bool)])
        self.phi_cutoff = np.zeros(shape=0, dtype=[('value', float), ('optimizable', bool)])
        self.phi_cutoff_optimizable = np.zeros(0)
        self.mu_labels = nb.typed.List.empty_list(labels_type)
        self.phi_labels = nb.typed.List.empty_list(labels_type)
        self.mu_cusp = np.zeros(0, dtype=bool)
        self.phi_cusp = np.zeros(0, dtype=bool)
        self.ae_cutoff = np.zeros(0)
        self.ae_cutoff_optimizable = np.zeros(0)
        self.phi_irrotational = np.zeros(0, dtype=bool)

    def read(self, base_path):
        file_path = os.path.join(base_path, 'correlation.data')
        if not os.path.isfile(file_path):
            return
        with open(file_path, 'r') as f:
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
                    self.fix_eta_parameters()
                    eta_term = False
                elif line.startswith('START MU TERM'):
                    mu_term = True
                elif line.startswith('END MU TERM'):
                    self.fix_mu_parameters()
                    mu_term = False
                elif line.startswith('START PHI TERM'):
                    phi_term = True
                elif line.startswith('END PHI TERM'):
                    self.fix_phi_parameters()
                    # self.check_phi_constrains()
                    phi_term = False
                elif line.startswith('START AE CUTOFFS'):
                    ae_term = True
                    ae_cutoff = []
                    ae_cutoff_optimizable = []
                elif line.startswith('END AE CUTOFFS'):
                    ae_term = False
                    self.ae_cutoff = np.array(ae_cutoff)
                    self.ae_cutoff_optimizable = np.array(ae_cutoff_optimizable)
                elif eta_term:
                    if line.startswith('Expansion order'):
                        eta_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        eta_spin_dep = self.read_int()
                    elif line.startswith('Cut-off radii'):
                        eta_cutoff, eta_cutoff_optimizable = self.read_parameter()
                        # Optimizable (0=NO; 1=YES; 2=YES BUT NO SPIN-DEP)
                        if eta_cutoff_optimizable == 2:
                            self.eta_cutoff = np.zeros(shape=1, dtype=[('value', float), ('optimizable', bool)])
                        else:
                            self.eta_cutoff = np.zeros(shape=eta_spin_dep+1, dtype=[('value', float), ('optimizable', bool)])
                        self.eta_cutoff[0] = eta_cutoff, eta_cutoff_optimizable
                        for i in range(1, self.eta_cutoff.shape[0]):
                            self.eta_cutoff[i] = self.read_parameter()
                    elif line.startswith('Parameter'):
                        self.eta_parameters = np.zeros((eta_order+1, eta_spin_dep+1), dtype=float)
                        self.eta_parameters_optimizable = np.zeros((eta_order + 1, eta_spin_dep + 1), dtype=bool)
                        eta_parameters_independent = self.eta_parameters_independent(self.eta_parameters)
                        try:
                            for i in range(eta_spin_dep + 1):
                                for j in range(eta_order + 1):
                                    if eta_parameters_independent[j, i]:
                                        self.eta_parameters[j, i], self.eta_parameters_optimizable[j, i] = self.read_parameter()
                        except ValueError:
                            eta_term = False
                            self.eta_parameters_optimizable = eta_parameters_independent
                elif mu_term:
                    if line.startswith('Number of sets'):
                        number_of_sets = self.read_ints()[0]
                        self.mu_cusp = np.zeros(number_of_sets, dtype=bool)
                        self.mu_cutoff = np.zeros(number_of_sets, dtype=[('value', float), ('optimizable', bool)])
                    elif line.startswith('START SET'):
                        set_number = int(line.split()[2]) - 1
                    elif line.startswith('Label'):
                        mu_labels = np.array(self.read_ints()) - 1
                        self.mu_labels.append(mu_labels)
                    elif line.startswith('Type of e-N cusp conditions'):
                        mu_cusp = self.read_bool()
                        self.mu_cusp[set_number] = mu_cusp
                    elif line.startswith('Expansion order'):
                        mu_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        mu_spin_dep = self.read_int()
                    elif line.startswith('Cutoff (a.u.)'):
                        self.mu_cutoff[set_number] = self.read_parameter()
                    elif line.startswith('Parameter values'):
                        mu_parameters = np.zeros((mu_order+1, mu_spin_dep+1), dtype=float)
                        mu_parameters_optimizable = np.zeros((mu_order + 1, mu_spin_dep + 1), dtype=bool)
                        mu_parameters_independent = self.mu_parameters_independent(mu_parameters)
                        try:
                            for i in range(mu_spin_dep + 1):
                                for j in range(mu_order + 1):
                                    if mu_parameters_independent[j, i]:
                                        mu_parameters[j, i], mu_parameters_optimizable[j, i] = self.read_parameter()
                        except ValueError:
                            mu_parameters_optimizable = mu_parameters_independent
                        # self.mu_mask.append(mu_parameters_independent)
                        self.mu_parameters.append(mu_parameters)
                        self.mu_parameters_optimizable.append(mu_parameters_optimizable)
                    elif line.startswith('END SET'):
                        pass
                elif phi_term:
                    if line.startswith('Number of sets'):
                        number_of_sets = self.read_ints()[0]
                        self.phi_cusp = np.zeros(number_of_sets, dtype=bool)
                        self.phi_cutoff = np.zeros(number_of_sets, dtype=[('value', float), ('optimizable', bool)])
                        self.phi_irrotational = np.zeros(number_of_sets, dtype=bool)
                    elif line.startswith('START SET'):
                        set_number = int(line.split()[2]) - 1
                    elif line.startswith('Label'):
                        phi_labels = np.array(self.read_ints()) - 1
                        self.phi_labels.append(phi_labels)
                    elif line.startswith('Type of e-N cusp conditions'):
                        phi_cusp = self.read_bool()
                        self.phi_cusp[set_number] = phi_cusp
                    elif line.startswith('Irrotational Phi'):
                        phi_irrotational = self.read_bool()
                    elif line.startswith('Electron-nucleus expansion order'):
                        phi_en_order = self.read_int()
                    elif line.startswith('Electron-electron expansion order'):
                        phi_ee_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        phi_spin_dep = self.read_int()
                    elif line.startswith('Cutoff (a.u.)'):
                        phi_cutoff, phi_cutoff_optimizable = self.read_parameter()
                        self.phi_cutoff[set_number]['value'] = phi_cutoff
                        self.phi_cutoff[set_number]['optimizable'] = phi_cutoff_optimizable
                    elif line.startswith('Parameter values'):
                        phi_parameters = np.zeros((phi_en_order+1, phi_en_order+1, phi_ee_order+1, phi_spin_dep+1), float)
                        phi_parameters_optimizable = np.zeros((phi_en_order + 1, phi_en_order + 1, phi_ee_order + 1, phi_spin_dep + 1), bool)
                        theta_parameters = np.zeros((phi_en_order+1, phi_en_order+1, phi_ee_order+1, phi_spin_dep+1), float)
                        theta_parameters_optimizable = np.zeros((phi_en_order + 1, phi_en_order + 1, phi_ee_order + 1, phi_spin_dep + 1), bool)
                        phi_parameters_independent, theta_parameters_independent = self.phi_theta_parameters_independent(phi_parameters, phi_cutoff, phi_cusp, phi_irrotational)
                        try:
                            for i in range(phi_spin_dep + 1):
                                for m in range(phi_ee_order + 1):
                                    for l in range(phi_en_order + 1):
                                        for k in range(phi_en_order + 1):
                                            if phi_parameters_independent[k, l, m, i]:
                                                phi_parameters[k, l, m, i], phi_parameters_optimizable[k, l, m, i] = self.read_parameter([k, l, m, i + 1])
                                for m in range(phi_ee_order + 1):
                                    for l in range(phi_en_order + 1):
                                        for k in range(phi_en_order + 1):
                                            if theta_parameters_independent[k, l, m, i]:
                                                theta_parameters[k, l, m, i], theta_parameters_optimizable[k, l, m, i] = self.read_parameter([k, l, m, i + 1])
                        except ValueError:
                            phi_parameters_optimizable = phi_parameters_independent
                            theta_parameters_optimizable = theta_parameters_independent
                        self.phi_parameters.append(phi_parameters)
                        self.theta_parameters.append(theta_parameters)
                        self.phi_parameters_optimizable.append(phi_parameters_optimizable)
                        self.theta_parameters_optimizable.append(theta_parameters_optimizable)
                        self.phi_irrotational[set_number] = phi_irrotational
                    elif line.startswith('END SET'):
                        pass
                elif ae_term:
                    if line.startswith('Nucleus'):
                        # Nucleus ; Set ; Cutoff length     ;  Optimizable (0=NO; 1=YES)
                        pass
                    else:
                        _, _, cutoff_length, cutoff_length_optimizable = line.split()
                        ae_cutoff.append(float(cutoff_length))
                        ae_cutoff_optimizable.append(float(cutoff_length_optimizable))

    def write(self):
        eta_term = ""
        if self.eta_cutoff['value'].any():
            eta_parameters_list = []
            eta_parameters_independent = self.eta_parameters_independent(self.eta_parameters)
            for i in range(self.eta_parameters.shape[1]):
                for j in range(self.eta_parameters.shape[0]):
                    if eta_parameters_independent[j, i]:
                        eta_parameters_list.append(f'{self.eta_parameters[j, i]: .16e}            {int(self.eta_parameters_optimizable[j, i])}       ! c_{j},{i + 1}')
            eta_set = eta_set_template.format(
                eta_order=self.eta_parameters.shape[0] - 1,
                eta_spin_dep=self.eta_parameters.shape[1] - 1,
                # FIXME: Optimizable (0=NO; 1=YES; 2=YES BUT NO SPIN-DEP)
                eta_cutoff=self.eta_cutoff[0]['value'],
                eta_cutoff_optimizable=int(self.eta_cutoff[0]['optimizable']),
                eta_parameters='\n  '.join(eta_parameters_list),
            )
            eta_term = eta_term_template.format(eta_set=eta_set)

        n_mu_set = 0
        mu_term = mu_sets = ''
        for n_mu_set, (mu_labels, mu_parameters, mu_parameters_optimizable, mu_cutoff, mu_cusp) in enumerate(zip(self.mu_labels, self.mu_parameters, self.mu_parameters_optimizable, self.mu_cutoff, self.mu_cusp)):
            mu_parameters_list = []
            mu_parameters_independent = self.mu_parameters_independent(mu_parameters)
            for i in range(mu_parameters.shape[1]):
                for j in range(mu_parameters.shape[0]):
                    if mu_parameters_independent[j, i]:
                        mu_parameters_list.append(f'{mu_parameters[j, i]: .16e}            {int(mu_parameters_optimizable[j, i])}       ! mu_{j},{i + 1}')
            mu_sets += mu_set_template.format(
                n_set=n_mu_set + 1,
                n_atoms=len(mu_labels),
                mu_cusp=int(mu_cusp),
                mu_labels=' '.join(['{}'.format(i + 1) for i in mu_labels]),
                mu_order=mu_parameters.shape[0] - 1,
                mu_spin_dep=mu_parameters.shape[1] - 1,
                mu_cutoff=mu_cutoff['value'],
                mu_cutoff_optimizable=int(mu_cutoff['optimizable']),
                mu_parameters='\n  '.join(mu_parameters_list),
            )
        if mu_sets:
            mu_term = mu_term_template.format(n_mu_sets=n_mu_set + 1, mu_sets=mu_sets)

        n_phi_set = 0
        phi_term = phi_sets = ''
        for n_phi_set, (phi_labels, phi_parameters, phi_parameters_optimizable, theta_parameters, theta_parameters_optimizable, phi_cutoff, phi_cusp, phi_irrotational) in enumerate(zip(self.phi_labels, self.phi_parameters, self.phi_parameters_optimizable, self.theta_parameters, self.theta_parameters_optimizable, self.phi_cutoff, self.phi_cusp, self.phi_irrotational)):
            phi_theta_parameters_list = []
            phi_parameters_independent, theta_parameters_independent = self.phi_theta_parameters_independent(phi_parameters, phi_cutoff['value'], phi_cusp, phi_irrotational)
            for i in range(phi_parameters.shape[3]):
                for m in range(phi_parameters.shape[2]):
                    for l in range(phi_parameters.shape[1]):
                        for k in range(phi_parameters.shape[0]):
                            if phi_parameters_independent[k, l, m, i]:
                                phi_theta_parameters_list.append(f'{phi_parameters[k, l, m, i]: .16e}            {int(phi_parameters_optimizable[k, l, m, i])}       ! phi_{k},{l},{m},{i + 1}')
                for m in range(phi_parameters.shape[2]):
                    for l in range(phi_parameters.shape[1]):
                        for k in range(phi_parameters.shape[0]):
                            if theta_parameters_independent[k, l, m, i]:
                                phi_theta_parameters_list.append(f'{theta_parameters[k, l, m, i]: .16e}            {int(theta_parameters_optimizable[k, l, m, i])}       ! theta_{k},{l},{m},{i + 1}')
            phi_sets += phi_set_template.format(
                n_set=n_phi_set + 1,
                n_atoms=len(phi_labels),
                phi_cusp=int(phi_cusp),
                phi_labels=' '.join(['{}'.format(i + 1) for i in phi_labels]),
                phi_en_order=phi_parameters.shape[0] - 1,
                phi_ee_order=phi_parameters.shape[2] - 1,
                phi_spin_dep=phi_parameters.shape[3] - 1,
                phi_cutoff=phi_cutoff['value'],
                phi_cutoff_optimizable=int(phi_cutoff['optimizable']),
                phi_irrotational=int(phi_irrotational),
                phi_parameters='\n  '.join(phi_theta_parameters_list),
            )
        if phi_sets:
            phi_term = phi_term_template.format(n_phi_sets=n_phi_set + 1, phi_sets=phi_sets)

        ae_cutoff_list = []
        for i, (ae_cutoff, ae_cutoff_optimizable) in enumerate(zip(self.ae_cutoff, self.ae_cutoff_optimizable)):
            ae_cutoff_list.append(f' {i + 1}         1      {ae_cutoff}                               {int(ae_cutoff_optimizable)}')
        backflow = backflow_template.format(
            title='no title given',
            trunc=self.trunc,
            terms=eta_term + mu_term + phi_term,
            ae_cutoffs='\n  '.join(ae_cutoff_list),
        )
        return backflow

    @staticmethod
    def eta_parameters_independent(parameters):
        """To obey the cusp conditions,
        we constrain the parallel-spin η(rij) function to have zero derivative at rij = 0,
        while the antiparallel-spin η function may have a nonzero derivative"""
        mask = np.ones(parameters.shape, np.bool)
        mask[1, 0] = False
        if parameters.shape[1] == 3:
            mask[1, 2] = False
        return mask

    @staticmethod
    def mu_parameters_independent(parameters):
        mask = np.ones(parameters.shape, np.bool)
        mask[0:2] = False
        return mask

    def phi_theta_parameters_independent(self, phi_parameters, phi_cutoff, phi_cusp, phi_irrotational):
        """Mask dependent parameters in phi-term.
        """
        phi_mask = np.zeros(shape=phi_parameters.shape, dtype=bool)
        theta_mask = np.zeros(shape=phi_parameters.shape, dtype=bool)
        for spin_dep in range(phi_parameters.shape[3]):
            c = construct_c_matrix(self.trunc, phi_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational)
            _, pivot_positions = rref(c)

            p = 0
            for m in range(phi_parameters.shape[2]):
                for l in range(phi_parameters.shape[1]):
                    for k in range(phi_parameters.shape[0]):
                        if p not in pivot_positions:
                            phi_mask[k, l, m, spin_dep] = True
                        p += 1

            for m in range(phi_parameters.shape[2]):
                for l in range(phi_parameters.shape[1]):
                    for k in range(phi_parameters.shape[0]):
                        if p not in pivot_positions:
                            theta_mask[k, l, m, spin_dep] = True
                        p += 1
        return phi_mask, theta_mask

    def fix_eta_parameters(self):
        """Fix eta-term parameters"""
        C = self.trunc
        L = self.eta_cutoff[0]['value']
        self.eta_parameters[1, 0] = C * self.eta_parameters[0, 0] / L
        if self.eta_parameters.shape[1] == 3:
            L = self.eta_cutoff[2]['value'] or self.eta_cutoff[0]['value']
            self.eta_parameters[1, 2] = C * self.eta_parameters[0, 2] / L

    def fix_mu_parameters(self):
        """Fix mu-term parameters"""
        for mu_parameters in self.mu_parameters:
            # for AE atoms
            mu_parameters[0:2] = 0

    def fix_phi_parameters(self):
        """Fix phi-term parameters"""
        for phi_parameters, theta_parameters, phi_cutoff, phi_cusp, phi_irrotational in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff['value'], self.phi_cusp, self.phi_irrotational):
            if not phi_parameters.any():
                continue
            for spin_dep in range(phi_parameters.shape[3]):
                c = construct_c_matrix(self.trunc, phi_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational)
                c, pivot_positions = rref(c)
                c = c[:pivot_positions.size, :]

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

                x = np.linalg.solve(c[:, pivot_positions], b)

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

    def check_phi_constrains(self):
        """"""
        for phi_parameters, theta_parameters, phi_cutoff, phi_cusp, phi_irrotational in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff['value'], self.phi_cusp, self.phi_irrotational):
            phi_en_order = phi_parameters.shape[0] - 1
            phi_ee_order = phi_parameters.shape[2] - 1

            for spin_dep in range(phi_parameters.shape[3]):
                lm_phi_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
                lm_phi_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
                lm_phi_m_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
                lm_theta_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))
                lm_theta_m_ae_sum = np.zeros((phi_en_order + phi_ee_order + 1,))

                for l in range(phi_parameters.shape[1]):
                    for m in range(phi_parameters.shape[2]):
                        lm_phi_sum[l + m] += self.trunc * phi_parameters[0, l, m, spin_dep] - phi_cutoff * phi_parameters[1, l, m, spin_dep]
                        lm_phi_ae_sum[l + m] += phi_parameters[0, l, m, spin_dep]
                        lm_phi_m_ae_sum[l + m] += m * phi_parameters[0, l, m, spin_dep]
                        lm_theta_ae_sum[l + m] += theta_parameters[0, l, m, spin_dep]
                        lm_theta_m_ae_sum[l + m] += m * theta_parameters[0, l, m, spin_dep]

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
                        km_phi_sum[k + m] += self.trunc * phi_parameters[k, 0, m, spin_dep] - phi_cutoff * phi_parameters[k, 1, m, spin_dep]
                        km_theta_sum[k + m] += self.trunc * theta_parameters[k, 0, m, spin_dep] - phi_cutoff * theta_parameters[k, 1, m, spin_dep]
                        km_phi_ae_sum[k + m] += phi_parameters[k, 0, m, spin_dep]
                        km_phi_m_ae_sum[k + m] += m * phi_parameters[k, 0, m, spin_dep]
                        km_theta_m_ae_sum[k + m] += m * theta_parameters[k, 0, m, spin_dep]

                np.abs(km_phi_sum).max() > 1e-14 and print(f'km_phi_sum = {km_phi_sum}')
                np.abs(km_theta_sum).max() > 1e-13 and print(f'km_theta_sum = {km_theta_sum}')
                np.abs(km_phi_ae_sum).max() > 1e-14 and print(f'km_phi_ae_sum = {km_phi_ae_sum}')
                np.abs(km_phi_m_ae_sum).max() > 1e-14 and print(f'km_phi_m_ae_sum = {km_phi_m_ae_sum}')
                np.abs(km_theta_m_ae_sum).max() > 1e-14 and print(f'km_theta_m_ae_sum = {km_theta_m_ae_sum}')

                if spin_dep in (0, 2):
                    kl_phi_sum = np.zeros((2 * phi_en_order + + 1,))
                    kl_theta_sum = np.zeros((2 * phi_en_order + + 1,))
                    for k in range(phi_parameters.shape[0]):
                        for l in range(phi_parameters.shape[1]):
                            kl_phi_sum[k + l] += phi_parameters[k, l, 1, spin_dep]
                            kl_theta_sum[k + l] += theta_parameters[k, l, 1, spin_dep]

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
                                            irrot_sum[k, l, m] += (self.trunc + k) * phi_parameters[k, l, m - 1, spin_dep]
                                        if phi_parameters.shape[0] > k + 1:
                                            irrot_sum[k, l, m] -= phi_cutoff * (k + 1) * phi_parameters[k + 1, l, m - 1, spin_dep]
                                    if phi_parameters.shape[2] > m + 1:
                                        if k - 2 >= 0:
                                            irrot_sum[k, l, m] -= (m + 1) * theta_parameters[k - 2, l, m + 1, spin_dep]
                                        if phi_parameters.shape[0] > k - 1 >= 0:
                                            irrot_sum[k, l, m] += phi_cutoff * (m + 1) * theta_parameters[k - 1, l, m + 1, spin_dep]
                                else:
                                    if phi_parameters.shape[0] > k + 1 and phi_parameters.shape[2] > m - 1 >= 0:
                                        irrot_sum[k, l, m] += (k + 1) * phi_parameters[k + 1, l, m - 1, spin_dep]
                                    if phi_parameters.shape[0] > k - 1 >= 0 and phi_parameters.shape[2] > m + 1:
                                        irrot_sum[k, l, m] -= (m + 1) * theta_parameters[k - 1, l, m + 1, spin_dep]

                    np.abs(irrot_sum).max() > 1e-13 and print('irrot_sum', irrot_sum)


if __name__ == '__main__':
    """Read Backflow terms
    """
    for phi_term in (
        '21', '22', '23', '24', '25',
        '31', '32', '33', '34', '35',
        '41', '42', '43', '44', '45',
        '51', '52', '53', '54', '55',
    ):
        print(phi_term)
        # path = f'test/backflow/0_1_0/{phi_term}/correlation.out.1'
        # path = f'test/backflow/3_1_0/{phi_term}/correlation.out.1'
        path = f'test/backflow/0_1_1/{phi_term}/correlation.out.1'
        # path = f'test/backflow/3_1_1/{phi_term}/correlation.out.1'
        Backflow().read(path)
