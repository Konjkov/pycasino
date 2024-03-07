#!/usr/bin/env python3

import os

import numpy as np
import numba as nb
from casino.jastrow import construct_a_matrix
from casino.overload import rref

labels_type = nb.int64[:]
chi_parameters_type = nb.float64[:, :]
chi_parameters_optimizable_type = nb.boolean[:, :]
f_parameters_type = nb.float64[:, :, :, :]
f_parameters_optimizable_type = nb.boolean[:, :, :, :]

jastrow_template = """\
 START JASTROW
 Title
  {title}
 Truncation order C
   {trunc}
 {terms}\
 END JASTROW
"""

u_term_template = """\
 START U TERM
 Number of sets
   1
 {u_set}
 END U TERM
"""

u_set_template = """\
START SET 1
 Spherical harmonic l,m
   0 0
 Expansion order N_u
   {u_order}
 Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
   {u_spin_dep}
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   {u_cutoff:.16f}                {u_cutoff_optimizable}
 Parameter values  ;  Optimizable (0=NO; 1=YES)
  {u_parameters}
 END SET 1"""

chi_term_template = """\
 START CHI TERM
 Number of sets ; labelling (1->atom in s. cell; 2->atom in p. cell; 3->species)
   {n_chi_sets} 1
 {chi_sets}
 END CHI TERM
"""

chi_set_template = """\
START SET {n_set}
 Spherical harmonic l,m
   0 0
 Number of atoms in set
   {n_atoms}
 Label of the atom in this set
   {chi_labels}
 Impose electron-nucleus cusp (0=NO; 1=YES)
   {chi_cusp}
 Expansion order N_chi
   {chi_order}
 Spin dep (0->u=d; 1->u/=d)
   {chi_spin_dep}
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   {chi_cutoff:.16f}                {chi_cutoff_optimizable}
 Parameter values  ;  Optimizable (0=NO; 1=YES)
  {chi_parameters}
 END SET {n_set}"""

f_term_template = """\
 START F TERM
 Number of sets ; labelling (1->atom in s. cell; 2->atom in p. cell; 3->species)
   {n_f_sets} 1
 {f_sets}
 END F TERM
"""

f_set_template = """\
START SET {n_set}
 Number of atoms in set
   {n_atoms}
 Label of the atom in this set
   {f_labels}
 Prevent duplication of u term (0=NO; 1=YES)
   {no_dup_u_term}
 Prevent duplication of chi term (0=NO; 1=YES)
   {no_dup_chi_term}
 Electron-nucleus expansion order N_f_eN
   {f_en_order}
 Electron-electron expansion order N_f_ee
   {f_ee_order}
 Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
   {f_spin_dep}
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   {f_cutoff:.16f}                {f_cutoff_optimizable}
 Parameter values  ;  Optimizable (0=NO; 1=YES)
  {f_parameters}
 END SET {n_set}"""


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

    def __init__(self):
        self.trunc = 0
        self.u_parameters = np.zeros(shape=(0, 0), dtype=float)  # uu, ud, dd order
        self.u_parameters_optimizable = np.zeros(shape=(0, 0), dtype=bool)
        self.chi_parameters = nb.typed.List.empty_list(chi_parameters_type)  # u, d order
        self.chi_parameters_optimizable = nb.typed.List.empty_list(chi_parameters_optimizable_type)  # u, d order
        self.f_parameters = nb.typed.List.empty_list(f_parameters_type)  # uu, ud, dd order
        self.f_parameters_optimizable = nb.typed.List.empty_list(f_parameters_optimizable_type)  # uu, ud, dd order
        self.u_cutoff = np.zeros(shape=1, dtype=[('value', float), ('optimizable', bool)])
        self.chi_cutoff = np.zeros(shape=0, dtype=[('value', float), ('optimizable', bool)])
        self.f_cutoff = np.zeros(shape=0, dtype=[('value', float), ('optimizable', bool)])
        self.chi_cusp = np.zeros(0, bool)
        self.chi_labels = nb.typed.List.empty_list(labels_type)
        self.f_labels = nb.typed.List.empty_list(labels_type)
        self.no_dup_u_term = np.zeros(0, bool)
        self.no_dup_chi_term = np.zeros(0, bool)

    def read(self, base_path):
        """Read Jastrow config from file."""
        file_path = os.path.join(base_path, 'correlation.data')
        if not os.path.isfile(file_path):
            return
        with open(file_path, 'r') as f:
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
                    # self.check_f_constrains()
                    f_term = False
                elif u_term:
                    if line.startswith('START SET'):
                        pass
                    elif line.startswith('Expansion order'):
                        u_order = self.read_int()
                    elif line.startswith('Spin dep'):
                        u_spin_dep = self.read_int()
                    elif line.startswith('Cutoff'):
                        self.u_cutoff[0] = self.read_parameter()
                    elif line.startswith('Parameter'):
                        self.u_parameters = np.zeros(shape=(u_spin_dep+1, u_order+1), dtype=float)
                        self.u_parameters_optimizable = np.zeros(shape=(u_spin_dep+1, u_order+1), dtype=bool)
                        u_parameters_independent = self.u_parameters_independent(self.u_parameters)
                        try:
                            for i in range(u_spin_dep + 1):
                                for l in range(u_order + 1):
                                    if u_parameters_independent[i, l]:
                                        self.u_parameters[i, l], self.u_parameters_optimizable[i, l] = self.read_parameter()
                        except ValueError:
                            u_term = False
                            self.u_parameters_optimizable = u_parameters_independent
                    elif line.startswith('END SET'):
                        pass
                elif chi_term:
                    if line.startswith('Number of set'):
                        number_of_sets = self.read_ints()[0]
                        self.chi_cutoff = np.zeros(number_of_sets, dtype=[('value', float), ('optimizable', bool)])
                        self.chi_cusp = np.zeros(number_of_sets, dtype=bool)
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
                        self.chi_cutoff[set_number] = self.read_parameter()
                    elif line.startswith('Parameter'):
                        chi_parameters = np.zeros(shape=(chi_spin_dep+1, chi_order+1), dtype=float)
                        chi_parameters_optimizable = np.zeros(shape=(chi_spin_dep + 1, chi_order + 1), dtype=bool)
                        chi_parameters_independent = self.chi_parameters_independent(chi_parameters)
                        try:
                            for i in range(chi_spin_dep + 1):
                                for m in range(chi_order + 1):
                                    if chi_parameters_independent[i, m]:
                                        chi_parameters[i, m], chi_parameters_optimizable[i, m] = self.read_parameter()
                        except ValueError:
                            chi_parameters_optimizable = chi_parameters_independent
                        self.chi_parameters.append(chi_parameters)
                        self.chi_parameters_optimizable.append(chi_parameters_optimizable)
                    elif line.startswith('END SET'):
                        set_number = None
                elif f_term:
                    if line.startswith('Number of set'):
                        number_of_sets = self.read_ints()[0]
                        self.f_cutoff = np.zeros(number_of_sets, dtype=[('value', float), ('optimizable', bool)])
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
                        f_cutoff, f_cutoff_optimizable = self.read_parameter()
                        self.f_cutoff[set_number]['value'] = f_cutoff
                        self.f_cutoff[set_number]['optimizable'] = f_cutoff_optimizable
                    elif line.startswith('Parameter'):
                        f_parameters = np.zeros(shape=(f_spin_dep+1, f_ee_order+1, f_en_order+1, f_en_order+1), dtype=float)
                        f_parameters_optimizable = np.zeros(shape=(f_spin_dep+1, f_ee_order+1, f_en_order+1, f_en_order+1), dtype=bool)
                        f_parameters_independent = self.f_parameters_independent(f_parameters, f_cutoff, no_dup_u_term, no_dup_chi_term)
                        try:
                            for i in range(f_spin_dep + 1):
                                for n in range(f_ee_order + 1):
                                    for m in range(f_en_order + 1):
                                        for l in range(f_en_order + 1):
                                            if f_parameters_independent[i, n, m, l]:
                                                # γlmnI = γmlnI
                                                p = self.read_parameter([l, m, n, i+1, set_number+1])
                                                f_parameters[i, n, m, l], f_parameters_optimizable[i, n, m, l] = p
                                                f_parameters[i, n, l, m], f_parameters_optimizable[i, n, l, m] = p
                        except ValueError:
                            f_parameters_optimizable = f_parameters_independent
                        # reverses the order of the axes
                        self.f_parameters.append(f_parameters.T)
                        self.f_parameters_optimizable.append(f_parameters_optimizable)
                    elif line.startswith('END SET'):
                        set_number = None

    def write(self, title='no title given'):
        """Write Jastrow config to file"""
        u_term = ""
        if self.u_cutoff:
            u_parameters_list = []
            u_parameters_independent = self.u_parameters_independent(self.u_parameters)
            for i in range(self.u_parameters.shape[0]):
                for l in range(self.u_parameters.shape[1]):
                    if u_parameters_independent[i, l]:
                        u_parameters_list.append(f'{self.u_parameters[i, l]: .16e}            {int(self.u_parameters_optimizable[i, l])}       ! alpha_{l},{i + 1}')
            u_set = u_set_template.format(
                u_spin_dep=self.u_parameters.shape[0] - 1,
                u_order=self.u_parameters.shape[1] - 1,
                u_cutoff=self.u_cutoff[0]['value'],
                u_cutoff_optimizable=int(self.u_cutoff[0]['optimizable']),
                u_parameters='\n  '.join(u_parameters_list),
            )
            u_term = u_term_template.format(u_set=u_set)

        n_chi_set = 0
        chi_term = ''
        chi_sets = []
        for n_chi_set, (chi_labels, chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_cusp) in enumerate(zip(self.chi_labels, self.chi_parameters, self.chi_parameters_optimizable, self.chi_cutoff, self.chi_cusp)):
            chi_parameters_list = []
            chi_parameters_independent = self.chi_parameters_independent(chi_parameters)
            for i in range(chi_parameters.shape[0]):
                for m in range(chi_parameters.shape[1]):
                    if chi_parameters_independent[i, m]:
                        chi_parameters_list.append(f'{chi_parameters[i, m]: .16e}            {int(chi_parameters_optimizable[i, m])}       ! beta_{m},{i + 1},{n_chi_set + 1}')
            chi_sets.append(
                chi_set_template.format(
                    n_set=n_chi_set + 1,
                    n_atoms=len(chi_labels),
                    chi_cusp=int(chi_cusp),
                    chi_labels=' '.join(['{}'.format(i + 1) for i in chi_labels]),
                    chi_spin_dep=chi_parameters.shape[0] - 1,
                    chi_order=chi_parameters.shape[1] - 1,
                    chi_cutoff=chi_cutoff['value'],
                    chi_cutoff_optimizable=int(chi_cutoff['optimizable']),
                    chi_parameters='\n  '.join(chi_parameters_list),
                ))
        if chi_sets:
            chi_term = chi_term_template.format(n_chi_sets=n_chi_set + 1, chi_sets='\n '.join(chi_sets))

        n_f_set = 0
        f_term = ''
        f_sets = []
        for n_f_set, (f_labels, f_parameters, f_parameters_optimizable, f_cutoff, no_dup_u_term, no_dup_chi_term) in enumerate(zip(self.f_labels, self.f_parameters, self.f_parameters_optimizable, self.f_cutoff, self.no_dup_u_term, self.no_dup_chi_term)):
            f_parameters = f_parameters.T
            f_parameters_list = []
            f_parameters_independent = self.f_parameters_independent(f_parameters, f_cutoff['value'], no_dup_u_term, no_dup_chi_term)
            for i in range(f_parameters.shape[0]):
                for n in range(f_parameters.shape[1]):
                    for m in range(f_parameters.shape[2]):
                        for l in range(f_parameters.shape[3]):
                            if f_parameters_independent[i, n, m, l]:
                                f_parameters_list.append(f'{f_parameters[i, n, m, l]: .16e}            {int(f_parameters_optimizable[i, n, m, l])}       ! gamma_{l},{m},{n},{i + 1},{n_f_set + 1}')
            f_sets.append(
                f_set_template.format(
                    n_set=n_f_set + 1,
                    n_atoms=len(f_labels),
                    f_labels=' '.join(['{}'.format(i + 1) for i in f_labels]),
                    no_dup_u_term=int(no_dup_u_term),
                    no_dup_chi_term=int(no_dup_chi_term),
                    f_spin_dep=f_parameters.shape[0] - 1,
                    f_ee_order=f_parameters.shape[1] - 1,
                    f_en_order=f_parameters.shape[2] - 1,
                    f_cutoff=f_cutoff['value'],
                    f_cutoff_optimizable=int(f_cutoff['optimizable']),
                    f_parameters='\n  '.join(f_parameters_list),
                ))
        if f_sets:
            f_term = f_term_template.format(n_f_sets=n_f_set + 1, f_sets='\n '.join(f_sets))

        jastrow = jastrow_template.format(
            title=title,
            trunc=self.trunc,
            terms=u_term + chi_term + f_term,
        )
        return jastrow

    @staticmethod
    def u_parameters_independent(parameters):
        """Mask dependent parameters in u-term"""
        mask = np.ones(parameters.shape, bool)
        mask[:, 1] = False
        return mask

    @staticmethod
    def chi_parameters_independent(parameters):
        """Mask dependent parameters in chi-term"""
        mask = np.ones(parameters.shape, bool)
        mask[:, 1] = False
        return mask

    def f_parameters_independent(self, f_parameters, f_cutoff, no_dup_u_term, no_dup_chi_term):
        """Mask dependent parameters in f-term."""
        a, _ = construct_a_matrix(self.trunc, f_parameters.T, f_cutoff, 0, no_dup_u_term, no_dup_chi_term)

        _, pivot_positions = rref(a)

        p = 0
        mask = np.zeros(shape=f_parameters.shape, dtype=bool)
        for n in range(f_parameters.shape[1]):
            for m in range(f_parameters.shape[2]):
                for l in range(m, f_parameters.shape[3]):
                    if p not in pivot_positions:
                        mask[:, n, m, l] = True
                    p += 1
        return mask

    def fix_u_parameters(self):
        """Fix u-term parameters"""
        # impose e-e cusp condition only if it's not initial Jastrow
        if not self.u_parameters.any():
            return
        C = self.trunc
        L = self.u_cutoff[0]['value']
        Gamma = 1 / np.array([4, 2, 4][:self.u_parameters.shape[0]])
        self.u_parameters[:, 1] = Gamma / (-L) ** C + self.u_parameters[:, 0] * C / L

    def fix_chi_parameters(self):
        """Fix chi-term parameters"""
        C = self.trunc
        for chi_parameters, chi_cutoff, chi_cusp in zip(self.chi_parameters, self.chi_cutoff, self.chi_cusp):
            if not chi_parameters.any():
                continue
            L = chi_cutoff['value']
            chi_parameters[:, 1] = chi_parameters[:, 0] * C / L
            if chi_cusp:
                pass
                # FIXME: chi cusp not implemented
                # chi_parameters[:, 1] -= charge / (-L) ** C

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
        for f_parameters, f_cutoff, no_dup_u_term, no_dup_chi_term in zip(self.f_parameters, self.f_cutoff, self.no_dup_u_term, self.no_dup_chi_term):
            if not f_parameters.any():
                continue
            L = f_cutoff['value']
            f_en_order = f_parameters.shape[0] - 1
            f_ee_order = f_parameters.shape[2] - 1
            f_spin_dep = f_parameters.shape[3] - 1

            a, _ = construct_a_matrix(self.trunc, f_parameters, L, 0, no_dup_u_term, no_dup_chi_term)
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
        for f_parameters, f_cutoff, no_dup_u_term, no_dup_chi_term in zip(self.f_parameters, self.f_cutoff, self.no_dup_u_term, self.no_dup_chi_term):
            L = f_cutoff['value']
            f_spin_dep = f_parameters.shape[0] - 1
            f_ee_order = f_parameters.shape[1] - 1
            f_en_order = f_parameters.shape[2] - 1

            lm_sum = np.zeros(shape=(2 * f_en_order + 1, f_spin_dep + 1))
            for l in range(f_en_order + 1):
                for m in range(f_en_order + 1):
                    lm_sum[l + m] += f_parameters[:, 1, m, l]
            np.abs(lm_sum).max() > 1e-18 and print('lm_sum =', lm_sum)

            mn_sum = np.zeros(shape=(f_en_order + f_ee_order + 1, f_spin_dep + 1))
            for m in range(f_en_order + 1):
                for n in range(f_ee_order + 1):
                    mn_sum[m + n] += self.trunc * f_parameters[:, n, m, 0] - L * f_parameters[:, n, m, 1]
            np.abs(mn_sum).max() > 1e-18 and print('mn_sum =', mn_sum)

            if no_dup_u_term:
                print('should be equal to zero')
                print(f_parameters[:, 0, 1, 1])
                print(f_parameters[:, :, 0, 0])
            if no_dup_chi_term:
                print('should be equal to zero')
                print(f_parameters[:, 0, 0, :])


if __name__ == '__main__':
    """Read Jastrow terms
    """
    for f_term_order in (
        '11', '12', '13', '14', '15',
        '21', '22', '23', '24', '25',
        '31', '32', '33', '34', '35',
        '41', '42', '43', '44', '45',
        '51', '52', '53', '54', '55',
    ):
        print(f_term_order)
        path = f'{os.path.dirname(__file__)}/../../tests/jastrow/3_1/{f_term_order}/correlation.out.1'
        Jastrow().read(path)
