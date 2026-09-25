#!/usr/bin/env python3
"""Write examples/stowfn/<system>/HF/QZ4P/CBCS/Jastrow_emin_uncut: u and chi without cutoff and the CASINO f-term.

    u    -gamma b exp(-r/b)          (Functional form 2), b per spin channel: parallel -> b_1, antiparallel -> b_2
    chi  A exp(-(r/a)^2)             (Functional form 2), A and a per spin set as in the CASINO file
    f    the F TERM block of the last CASINO emin stage, as it is

The f-term is kept, so the new Jastrow reproduces the CASINO one when
    sum_pairs u(r_ij) + sum_iI chi(r_iI)  =  sum_pairs u_CASINO(r_ij) + sum_iI chi_CASINO(r_iI) + const
on the configurations that matter. The starting values are the least-squares fit of this equation on
configurations sampled by VMC from the CASINO Slater-Jastrow wave function. Fitting the radial profiles
one by one is not enough: for Be the Jastrow built from profile fits gives -14.467 instead of -14.650.
"""

import os
import shutil
import tempfile

import numba as nb
import numpy as np
from scipy.optimize import least_squares

from collect import ROOT
from terms import CHI_SPIN_LABELS, U_CUSP, U_SPIN_LABELS, chi_profile, load, u_profile

from casino.jastrow import UNCUT
from casino.pycasino import Casino
from casino.readers.jastrow import Jastrow, labels_type

SYSTEMS = ['He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3']
# hole radius of a channel without pairs (the parallel channel of He)
B_DEFAULT = 1.0
# range of the hole radius accepted from the fit (bohr)
B_MIN = 0.1
B_MAX = 20.0
N_CONFIGS = 5000

header = """\
 START HEADER
 No title given.
 END HEADER

 START VERSION
   1
 END VERSION

"""


def sample(entry):
    """Configurations of the CASINO Slater-Jastrow wave function (last emin stage)."""
    source = os.path.join(ROOT, entry['path'])
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(source, 'input'), tmp)
        shutil.copy(os.path.join(source, 'stowfn.data'), tmp)
        shutil.copy(os.path.join(source, entry['file']), os.path.join(tmp, 'correlation.data'))
        casino = Casino(tmp)
        casino.equilibrate(5000)
        casino.optimize_vmc_step(3000)
        position = casino.vmc.random_walk(N_CONFIGS * casino.decorr_period, casino.decorr_period)
    position = position[~np.isnan(position[:, 0, 0])]
    return position, np.array(entry['atom_positions'])


def geometry(position, atoms, neu):
    """Pair distances with their spin channel (0 parallel, 1 antiparallel) and e-n distances with the electron spin."""
    ne = position.shape[1]
    i, j = np.triu_indices(ne, 1)
    r_ee = np.linalg.norm(position[:, i] - position[:, j], axis=-1)
    antiparallel = (i < neu) != (j < neu)
    r_en = np.linalg.norm(position[:, :, None, :] - atoms[None, None, :, :], axis=-1)
    down = np.arange(ne) >= neu
    return r_ee, antiparallel, r_en, down


def casino_u_chi(entry, r_ee, r_en, down):
    """sum over pairs of u and over electrons and nuclei of chi of the CASINO Jastrow."""
    jastrow = entry['jastrow']
    u = jastrow['u']
    labels = U_SPIN_LABELS[len(u['parameters']) - 1]
    neu = entry['neu']
    ne = r_en.shape[1]
    i, j = np.triu_indices(ne, 1)
    res = np.zeros(r_ee.shape[0])
    for k in range(r_ee.shape[1]):
        pair = 'ud'
        if (i[k] < neu) == (j[k] < neu):
            pair = 'uu'
            if i[k] >= neu and 'dd' in labels:
                pair = 'dd'
        res += u_profile(u, jastrow['trunc'], labels.index(pair), r_ee[:, k])
    for chi in jastrow['chi']:
        spins = CHI_SPIN_LABELS[len(chi['parameters']) - 1]
        for atom in chi['labels']:
            for e in range(ne):
                spin = 0
                if len(spins) == 2 and down[e]:
                    spin = 1
                res += chi_profile(chi, jastrow['trunc'], spin, r_en[:, e, atom])
    return res


def model(p, r_ee, antiparallel, r_en, down, chi_sets, chi_spins):
    """sum of the uncut u and chi; p = [ln b_par, ln b_anti, (A, ln a) per chi set and spin, const]."""
    res = np.full(r_ee.shape[0], p[-1])
    for k in range(r_ee.shape[1]):
        channel = int(antiparallel[k])
        gamma = U_CUSP[['uu', 'ud'][channel]]
        b = np.exp(p[channel])
        res += -gamma * b * np.exp(-r_ee[:, k] / b)
    n = 2
    for atoms, spins in zip(chi_sets, chi_spins):
        for atom in atoms:
            for e in range(r_en.shape[1]):
                m = n
                if spins == 2 and down[e]:
                    m = n + 2
                res += p[m] * np.exp(-((r_en[:, e, atom] / np.exp(p[m + 1])) ** 2))
        n += 2 * spins
    return res


def fit(entry):
    position, atoms = sample(entry)
    r_ee, antiparallel, r_en, down = geometry(position, atoms, entry['neu'])
    target = casino_u_chi(entry, r_ee, r_en, down)
    chi_sets = [chi['labels'] for chi in entry['jastrow']['chi']]
    chi_spins = [len(chi['parameters']) for chi in entry['jastrow']['chi']]
    p0 = [0.0, 0.0]
    for chi, spins in zip(entry['jastrow']['chi'], chi_spins):
        p0 += [chi_profile(chi, entry['jastrow']['trunc'], 0, np.zeros(1))[0], np.log(chi['cutoff'] / 2)] * spins
    p0.append(0.0)
    p0 = np.array(p0, dtype=float)

    def residuals(p):
        return model(p, r_ee, antiparallel, r_en, down, chi_sets, chi_spins) - target

    # Levenberg-Marquardt: the trust-region default stops at the start on these badly scaled parameters
    sol = least_squares(residuals, p0, method='lm')
    lower = np.full(p0.size, -np.inf)
    upper = np.full(p0.size, np.inf)
    lower[:2], upper[:2] = np.log(B_MIN), np.log(B_MAX)
    if np.any(sol.x[:2] < lower[:2]) or np.any(sol.x[:2] > upper[:2]):
        # a hole radius ran away (Kr parallel pairs): refit inside the bounds from the clipped solution
        x0 = np.clip(sol.x, lower + 1e-6, upper - 1e-6)
        sol = least_squares(residuals, x0, bounds=(lower, upper), x_scale='jac')
    rms = np.sqrt(np.mean(sol.fun**2))
    print(f'{entry["system"]}: {len(position)} configurations, rms(target) {np.std(target):.3f}, rms(fit - target) {rms:.3f}')
    return sol.x, chi_spins


def f_term(entry):
    """F TERM block of the optimized CASINO Jastrow."""
    with open(os.path.join(ROOT, entry['path'], entry['file'])) as f:
        text = f.read()
    return text[text.index(' START F TERM') : text.index(' END F TERM') + len(' END F TERM\n')]


def build(system):
    entry = [e for e in load(['Jastrow_emin']) if e['basis'] == 'stowfn' and e['system'] == system][0]
    p, chi_spins = fit(entry)
    b = np.exp(p[:2])
    if entry['neu'] < 2 and entry['ned'] < 2:
        # no parallel pairs
        b[0] = B_DEFAULT

    jastrow = Jastrow()
    jastrow.title = f'{system}, exponential u-term and Gaussian chi-term without cutoff, CASINO f-term'
    jastrow.trunc = entry['jastrow']['trunc']
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
    for chi, spins in zip(chi_sets, chi_spins):
        parameters = np.array([[p[n + 2 * s], np.exp(p[n + 2 * s + 1])] for s in range(spins)])
        jastrow.chi_labels.append(np.array(chi['labels'], dtype=np.int64))
        jastrow.chi_parameters.append(parameters)
        jastrow.chi_parameters_optimizable.append(np.ones_like(parameters, dtype=bool))
        n += 2 * spins
    text = jastrow.write()
    end = text.index(' END JASTROW')
    return entry, header + text[:end] + f_term(entry) + text[end:]


def main():
    for system in SYSTEMS:
        entry, text = build(system)
        source = os.path.join(ROOT, entry['path'])
        target = os.path.join(os.path.dirname(source), 'Jastrow_emin_uncut')
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
