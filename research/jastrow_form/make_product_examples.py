#!/usr/bin/env python3
"""Write examples/stowfn/<system>/HF/QZ4P/CBCS/Jastrow_emin_product: u and chi without cutoff and the product f-term.

    u    -gamma b exp(-r/b)                        (Functional form 2), parallel -> b_1, antiparallel -> b_2
    chi  A exp(-(r/a)^2)                           (Functional form 2), A and a per spin set as in the CASINO file
    f    (r1-L)^C (r2-L)^C g(r1) g(r2) h(r12)      (Functional form 1), g and h cubic, per f set and spin set,
                                                   L the CASINO f cutoff, kept fixed

The starting values are the joint fit (A) of product_f.py to the full CASINO Jastrow on VMC configurations
(results/product_f.json). There the product is written in x = r/L with the cutoff factor (1 - x)^C:
    (1 - x1)^C (1 - x2)^C = (r1 - L)^C (r2 - L)^C / L^(2C),  g_k -> g_k / L^k,  h_k -> h_k / L^k / L^(2C)
"""

import json
import os
import shutil

import numba as nb
import numpy as np

from collect import ROOT
from make_uncut_examples import B_DEFAULT, header
from terms import HERE, load

from casino.jastrow import PRODUCT, UNCUT
from casino.readers.jastrow import Jastrow, labels_type

SYSTEMS = ['He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3']


def build(system, uncut, product):
    entry = [e for e in load(['Jastrow_emin']) if e['basis'] == 'stowfn' and e['system'] == system][0]
    trunc = entry['jastrow']['trunc']
    b = np.exp(uncut[:2])
    if entry['neu'] < 2 and entry['ned'] < 2:
        # no parallel pairs
        b[0] = B_DEFAULT

    jastrow = Jastrow()
    jastrow.title = f'{system}, exponential u-term and Gaussian chi-term without cutoff, product f-term'
    jastrow.trunc = trunc
    jastrow.u_form = UNCUT
    jastrow.u_parameters = b.reshape(2, 1)
    jastrow.u_parameters_optimizable = np.ones_like(jastrow.u_parameters, dtype=bool)
    jastrow.u_cutoff = np.array([(np.inf, False)], dtype=[('value', float), ('optimizable', bool)])

    chi_sets = entry['jastrow']['chi']
    jastrow.chi_cutoff = np.zeros(len(chi_sets), dtype=[('value', float), ('optimizable', bool)])
    jastrow.chi_cutoff['value'] = np.inf
    jastrow.chi_cusp = np.zeros(len(chi_sets), dtype=bool)
    jastrow.chi_form = np.full(len(chi_sets), UNCUT, dtype=np.int64)
    jastrow.chi_labels = nb.typed.List.empty_list(labels_type)
    n = 2
    for chi in chi_sets:
        spins = len(chi['parameters'])
        parameters = np.array([[uncut[n + 2 * s], np.exp(uncut[n + 2 * s + 1])] for s in range(spins)])
        jastrow.chi_labels.append(np.array(chi['labels'], dtype=np.int64))
        jastrow.chi_parameters.append(parameters)
        jastrow.chi_parameters_optimizable.append(np.ones_like(parameters, dtype=bool))
        n += 2 * spins

    f_sets = entry['jastrow']['f']
    jastrow.f_cutoff = np.zeros(len(f_sets), dtype=[('value', float), ('optimizable', bool)])
    jastrow.no_dup_u_term = np.zeros(len(f_sets), dtype=bool)
    jastrow.no_dup_chi_term = np.zeros(len(f_sets), dtype=bool)
    jastrow.f_form = np.full(len(f_sets), PRODUCT, dtype=np.int64)
    n = 0
    for i, f in enumerate(f_sets):
        L = f['cutoff']
        spins = len(f['parameters'])
        f_product = np.zeros(shape=(spins, 8))
        for s in range(spins):
            g2, g3, h0, h2, h3 = product[n : n + 5]
            f_product[s] = [1, trunc / L, g2 / L**2, g3 / L**3, h0, 0, h2 / L**2, h3 / L**3]
            f_product[s, 4:] /= L ** (2 * trunc)
            n += 5
        jastrow.f_labels.append(np.array(f['labels'], dtype=np.int64))
        jastrow.f_cutoff[i] = (L, False)
        jastrow.f_parameters.append(np.zeros(shape=(spins, 4, 4, 4)))
        jastrow.f_parameters_optimizable.append(np.zeros(shape=(spins, 4, 4, 4), dtype=bool))
        jastrow.f_product.append(f_product)
        jastrow.f_product_optimizable.append(jastrow.f_product_independent(f_product, 4))
    jastrow.fix_f_parameters()
    return entry, header + jastrow.write()


def main():
    with open(os.path.join(HERE, 'results', 'product_f.json')) as f:
        fits = json.load(f)
    for system in SYSTEMS:
        entry, text = build(system, np.array(fits[system]['uncut']), np.array(fits[system]['product']))
        source = os.path.join(ROOT, entry['path'])
        target = os.path.join(os.path.dirname(source), 'Jastrow_emin_product')
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
