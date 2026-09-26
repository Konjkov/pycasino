#!/usr/bin/env python3
"""Write examples/stowfn/<system>/HF/QZ4P/CBCS/Jastrow_emin_product: u and chi without cutoff and the product f-term.

    u, chi  the optimized u and chi (Functional form 2) of the last stage of Jastrow_emin_uncut (correlation.out.4)
    f       (r1-L)^C (r2-L)^C g(r1) g(r2) h(r12)   (Functional form 1), g and h cubic, per f set and spin set,
            fitted to the optimized polynomial f of the same stage, L its cutoff, kept fixed

The product is fitted on configurations sampled by VMC from the CASINO Slater-Jastrow (product_f.py). There it
is written in x = r/L with the cutoff factor (1 - x)^C:
    (1 - x1)^C (1 - x2)^C = (r1 - L)^C (r2 - L)^C / L^(2C),  g_k -> g_k / L^k,  h_k -> h_k / L^k / L^(2C)
The input is that of Jastrow_emin_uncut.
"""

import os
import shutil
import tempfile

import numpy as np
from scipy.optimize import least_squares

from collect import ROOT
from make_uncut_examples import geometry, header
from product_f import casino_f, configurations, f_blocks, product_f, product_offsets
from terms import load

from casino.jastrow import PRODUCT
from casino.readers.jastrow import Jastrow

SYSTEMS = ['He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3']
# every SUBSAMPLE-th cached configuration is enough for a starting point
SUBSAMPLE = 10


def read_uncut(path):
    """Jastrow of the last stage of an optimization."""
    jastrow = Jastrow()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(path, 'correlation.out.4'), os.path.join(tmp, 'correlation.data'))
        jastrow.read(tmp)
    return jastrow


def fit(entry, jastrow):
    """Product f fitted to the polynomial f of the Jastrow, parameters of product_f.product_f()."""
    f_sets = [
        dict(labels=labels.tolist(), cutoff=cutoff['value'], parameters=parameters)
        for labels, cutoff, parameters in zip(jastrow.f_labels, jastrow.f_cutoff, jastrow.f_parameters)
    ]
    entry = dict(entry, jastrow=dict(entry['jastrow'], trunc=jastrow.trunc, f=f_sets))
    position, atoms = configurations(entry)
    r_ee, _, r_en, _ = geometry(position[::SUBSAMPLE], atoms, entry['neu'])
    blocks = f_blocks(entry, r_ee, r_en)
    offsets, n_product = product_offsets(entry)
    target = casino_f(entry, blocks)
    sol = least_squares(lambda q: product_f(q, blocks, offsets) + q[-1] - target, np.zeros(n_product + 1), method='lm')
    rms = np.sqrt(np.mean(sol.fun**2))
    print(f'{entry["system"]}: rms(f) {np.std(target):.3f}, rms(fit - f) {rms:.3f}')
    return sol.x[:-1]


def build(system):
    entry = [e for e in load(['Jastrow_emin']) if e['basis'] == 'stowfn' and e['system'] == system][0]
    uncut = os.path.join(os.path.dirname(os.path.join(ROOT, entry['path'])), 'Jastrow_emin_uncut')
    jastrow = read_uncut(uncut)
    product = fit(entry, jastrow)

    trunc = jastrow.trunc
    jastrow.title = f'{system}, exponential u-term and Gaussian chi-term without cutoff, product f-term'
    jastrow.f_form = np.full(len(jastrow.f_parameters), PRODUCT, dtype=np.int64)
    n = 0
    for i, f_parameters in enumerate(jastrow.f_parameters):
        L = jastrow.f_cutoff[i]['value']
        spins = f_parameters.shape[0]
        f_product = np.zeros(shape=(spins, 8))
        for s in range(spins):
            g2, g3, h0, h2, h3 = product[n : n + 5]
            f_product[s] = [1, trunc / L, g2 / L**2, g3 / L**3, h0, 0, h2 / L**2, h3 / L**3]
            f_product[s, 4:] /= L ** (2 * trunc)
            n += 5
        jastrow.f_cutoff[i]['optimizable'] = False
        jastrow.f_parameters[i] = np.zeros(shape=(spins, 4, 4, 4))
        jastrow.f_parameters_optimizable[i] = np.zeros(shape=(spins, 4, 4, 4), dtype=bool)
        jastrow.f_product[i] = f_product
        jastrow.f_product_optimizable[i] = jastrow.f_product_independent(f_product, 4)
    jastrow.fix_f_parameters()
    return uncut, header + jastrow.write()


def main():
    for system in SYSTEMS:
        uncut, text = build(system)
        target = os.path.join(os.path.dirname(uncut), 'Jastrow_emin_product')
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, 'correlation.data'), 'w') as f:
            f.write(text)
        shutil.copy(os.path.join(uncut, 'input'), os.path.join(target, 'input'))
        link = os.path.join(target, 'stowfn.data')
        if not os.path.lexists(link):
            os.symlink(os.readlink(os.path.join(uncut, 'stowfn.data')), link)
        print(os.path.relpath(target, ROOT))


if __name__ == '__main__':
    main()
