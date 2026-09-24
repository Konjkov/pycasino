#!/usr/bin/env python3
"""Write examples/stowfn/<system>/HF/QZ4P/CBCS/Jastrow_emin_analytic with the exponential u-term and bell chi-term.

Starting values come from the fits of the CASINO profiles (results/radial_fits.json, last emin stage):
    u    hole radii b of the 'exp' form: parallel channel -> b_1, antiparallel -> b_2 (spin dep 1);
         the u cutoff is the fitted one, limited to U_CUTOFF_MAX
    chi  amplitude A and cutoff L of the 'window' form, per set, with the spin dependence of the CASINO file
The input is copied from the neighbouring Jastrow_emin directory, stowfn.data is linked as there.
"""

import json
import os
import shutil

import numba as nb
import numpy as np

from collect import ROOT
from terms import HERE, load

from casino.jastrow import ANALYTIC
from casino.readers.jastrow import Jastrow, labels_type

SYSTEMS = ['N', 'Ne', 'Ar', 'Kr', 'O3']
U_CUTOFF_MAX = 8.0

header = """\
 START HEADER
 No title given.
 END HEADER

 START VERSION
   1
 END VERSION

"""


def fits(term, system):
    with open(os.path.join(HERE, 'results', 'radial_fits.json')) as f:
        rows = json.load(f)[term]
    return [row for row in rows if row['system'] == f'stowfn:{system}']


def build(system):
    entry = [e for e in load(['Jastrow_emin']) if e['basis'] == 'stowfn' and e['system'] == system][0]
    u_fit = fits('u', system)[0]['forms']['exp']
    b = [p[0] for p in u_fit['parameters']]

    jastrow = Jastrow()
    jastrow.title = f'{system}, exponential u-term and bell chi-term'
    jastrow.trunc = entry['jastrow']['trunc']
    jastrow.u_form = ANALYTIC
    # parallel and antiparallel hole radii; a third (dd) channel of spin dep 2 is not kept
    jastrow.u_parameters = np.array([[b[0]], [b[1]]])
    jastrow.u_parameters_optimizable = np.ones_like(jastrow.u_parameters, dtype=bool)
    jastrow.u_cutoff = np.array([(min(u_fit['L'], U_CUTOFF_MAX), True)], dtype=[('value', float), ('optimizable', bool)])

    chi_sets = entry['jastrow']['chi']
    chi_fits = fits('chi', system)
    jastrow.chi_cutoff = np.zeros(len(chi_sets), dtype=[('value', float), ('optimizable', bool)])
    jastrow.chi_cusp = np.zeros(len(chi_sets), dtype=bool)
    jastrow.chi_form = np.full(len(chi_sets), ANALYTIC, dtype=np.int64)
    jastrow.chi_labels = nb.typed.List.empty_list(labels_type)
    for i, (chi, chi_fit) in enumerate(zip(chi_sets, chi_fits)):
        window = chi_fit['forms']['window']
        parameters = np.array([[p[0]] for p in window['parameters']])
        jastrow.chi_labels.append(np.array(chi['labels'], dtype=np.int64))
        jastrow.chi_parameters.append(parameters)
        jastrow.chi_parameters_optimizable.append(np.ones_like(parameters, dtype=bool))
        jastrow.chi_cutoff[i] = (window['L'], True)
    return entry, header + jastrow.write()


def main():
    for system in SYSTEMS:
        entry, text = build(system)
        source = os.path.join(ROOT, entry['path'])
        target = os.path.join(os.path.dirname(source), 'Jastrow_emin_analytic')
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, 'correlation.data'), 'w') as f:
            f.write(text)
        shutil.copy(os.path.join(source, 'input'), os.path.join(target, 'input'))
        link = os.path.join(target, 'stowfn.data')
        if not os.path.lexists(link):
            os.symlink(os.readlink(os.path.join(source, 'stowfn.data')), link)
        print(os.path.relpath(target, ROOT))


if __name__ == '__main__':
    main()
