#!/usr/bin/env python3
"""Write Jastrow_emin_uncut inputs with approximate starting values for the systems not computed yet.

    u    -gamma b exp(-r/b)       (Functional form 2), b = 1.0 (parallel), 0.8 (antiparallel), typical of the
                                  optimized stowfn Jastrow_emin_uncut runs
    chi  A exp(-(r/a)^2)          (Functional form 2) per species set, (A, a) of the optimized N and O3 runs
                                  for all-electron N and O, A = 1, a = 1.5 otherwise
    f    empty polynomial F TERM, to be optimized

Systems with a polynomial Jastrow_emin keep its chi and f sets, its F TERM and its input without vm_filter.
Systems with a Slater directory only get one chi and one f set per species, f cutoff 5 bohr, and the input
of the gwfn Be Jastrow_emin with their numbers of electrons (100000 VMC steps above 20 electrons).
"""

import os

import numba as nb
import numpy as np

from collect import ROOT

from casino.jastrow import UNCUT
from casino.readers.jastrow import Jastrow, labels_type

FROM_POLYNOMIAL = ['examples/gwfn/B2H6/HF/cc-pVQZ/CBCS'] + [
    f'examples/ppotential_HF/{system}/HF/aug-cc-pVQZ-CDF/CBCS' for system in ['B', 'B2H6', 'C', 'F', 'H', 'N', 'Ne', 'O']
]
FROM_SLATER = [f'examples/gwfn/{system}/HF/cc-pVQZ/CBCS' for system in ['C2H2', 'C4H4', 'C6H6', 'CH4', 'H2O', 'HF', 'NH3', 'Li⁺', 'Be²⁺']]
INPUT_TEMPLATE = 'examples/gwfn/Be/HF/cc-pVQZ/CBCS/Jastrow_emin/input'
U_START = [1.0, 0.8]
# all-electron (A, a) of the optimized N atom and O3 runs, averaged over spins and sets
SPECIES_CHI = {7: [2.3, 2.1], 8: [1.3, 1.35]}
CHI_DEFAULT = [1.0, 1.5]
F_CUTOFF = 5.0

header = """\
 START HEADER
 No title given.
 END HEADER

 START VERSION
   1
 END VERSION

"""

f_set_template = """\
 START SET {n_set}
 Number of atoms in set
   {n_atoms}
 Label of the atom in this set
   {labels}
 Prevent duplication of u term (0=NO; 1=YES)
   0
 Prevent duplication of chi term (0=NO; 1=YES)
   0
 Electron-nucleus expansion order N_f_eN
   3
 Electron-electron expansion order N_f_ee
   3
 Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
   1
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   {cutoff:.1f}                               1
 Parameter values  ;  Optimizable (0=NO; 1=YES)
 END SET {n_set}
"""


def atomic_numbers(path):
    with open(os.path.join(ROOT, path, '..', 'gwfn.data')) as f:
        lines = f.read().splitlines()
    start = lines.index('Atomic numbers for each atom:') + 1
    end = lines.index('Valence charges for each atom:')
    return np.array(' '.join(lines[start:end]).split(), dtype=int)


def uncut_jastrow(title, chi_labels, chi_spins, charges, all_electron):
    jastrow = Jastrow()
    jastrow.title = title
    jastrow.trunc = 3
    jastrow.u_form = UNCUT
    jastrow.u_parameters = np.array(U_START).reshape(2, 1)
    jastrow.u_parameters_optimizable = np.ones_like(jastrow.u_parameters, dtype=bool)
    jastrow.u_cutoff = np.array([(np.inf, False)], dtype=[('value', float), ('optimizable', bool)])
    jastrow.chi_cutoff = np.zeros(len(chi_labels), dtype=[('value', float), ('optimizable', bool)])
    jastrow.chi_cutoff['value'] = np.inf
    jastrow.chi_cusp = np.zeros(len(chi_labels), dtype=bool)
    jastrow.chi_form = np.full(len(chi_labels), UNCUT, dtype=np.int64)
    jastrow.chi_labels = nb.typed.List.empty_list(labels_type)
    for labels, spins in zip(chi_labels, chi_spins):
        start = CHI_DEFAULT
        if all_electron:
            start = SPECIES_CHI.get(charges[labels[0]], CHI_DEFAULT)
        parameters = np.array([start] * spins, dtype=float)
        jastrow.chi_labels.append(np.array(labels, dtype=np.int64))
        jastrow.chi_parameters.append(parameters)
        jastrow.chi_parameters_optimizable.append(np.ones_like(parameters, dtype=bool))
    text = jastrow.write()
    return text[: text.index(' END JASTROW')]


def from_polynomial(path):
    source = os.path.join(ROOT, path, 'Jastrow_emin')
    polynomial = Jastrow()
    polynomial.read(source)
    with open(os.path.join(source, 'correlation.data')) as f:
        text = f.read()
    f_term = text[text.index(' START F TERM') : text.index(' END F TERM') + len(' END F TERM\n')]
    all_electron = not [name for name in os.listdir(source) if name.endswith('_pp.data')]
    chi_labels = [labels.tolist() for labels in polynomial.chi_labels]
    chi_spins = [parameters.shape[0] for parameters in polynomial.chi_parameters]
    title = f'{path.split("/")[2]}, exponential u-term and Gaussian chi-term without cutoff, empty f-term'
    jastrow = uncut_jastrow(title, chi_labels, chi_spins, atomic_numbers(path), all_electron)
    with open(os.path.join(source, 'input')) as f:
        lines = [line for line in f if not line.startswith('vm_filter')]
    return source, header + jastrow + f_term + ' END JASTROW\n', lines


def from_slater(path):
    source = os.path.join(ROOT, path, 'Slater')
    charges = atomic_numbers(path)
    species = [np.flatnonzero(charges == z).tolist() for z in sorted(set(charges), reverse=True)]
    system = path.split('/')[2]
    title = f'{system}, exponential u-term and Gaussian chi-term without cutoff, empty f-term'
    jastrow = uncut_jastrow(title, species, [1] * len(species), charges, True)
    f_sets = [
        f_set_template.format(n_set=n + 1, n_atoms=len(labels), labels=' '.join(str(i + 1) for i in labels), cutoff=F_CUTOFF)
        for n, labels in enumerate(species)
    ]
    f_term = f' START F TERM\n Number of sets ; labelling (1->atom in s. cell; 2->atom in p. cell; 3->species)\n   {len(species)} 1\n'
    f_term += ''.join(f_sets) + ' END F TERM\n'

    with open(os.path.join(source, 'input')) as f:
        slater = f.read().splitlines(keepends=True)
    electrons = {line.split()[0]: line for line in slater if line.startswith(('neu ', 'ned '))}
    n_electrons = sum(int(line.split()[2]) for line in electrons.values())
    with open(os.path.join(ROOT, INPUT_TEMPLATE)) as f:
        lines = []
        for line in f:
            if line.startswith('# Be molecule'):
                line = slater[4]
            elif line.startswith(('neu ', 'ned ')):
                line = electrons[line.split()[0]]
            elif line.startswith(('vmc_nstep ', 'vmc_nconfig_write ')) and n_electrons > 20:
                line = line.replace('1000000 ', '100000  ')
            lines.append(line)
    return source, header + jastrow + f_term + ' END JASTROW\n', lines


def write(path, source, text, lines):
    target = os.path.join(ROOT, path, 'Jastrow_emin_uncut')
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, 'correlation.data'), 'w') as f:
        f.write(text)
    with open(os.path.join(target, 'input'), 'w') as f:
        f.writelines(lines)
    for name in os.listdir(source):
        link = os.path.join(target, name)
        if os.path.islink(os.path.join(source, name)) and not os.path.lexists(link):
            os.symlink(os.readlink(os.path.join(source, name)), link)
    print(os.path.relpath(target, ROOT))


def main():
    for path in FROM_POLYNOMIAL:
        write(path, *from_polynomial(path))
    for path in FROM_SLATER:
        write(path, *from_slater(path))


if __name__ == '__main__':
    main()
