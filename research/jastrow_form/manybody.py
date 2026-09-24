#!/usr/bin/env python3
"""Separability of the e-e-n terms f (Jastrow) and Phi, Theta (backflow).

Triplets (r1, r2, r12) around a nucleus are drawn from the independent-electron distribution:
r1, r2 from 4 pi r^2 rho_I(r) (HF, spherically averaged) and a uniform cos(angle) between them.
This is the weight with which f_I enters the energy.  Each reduced model below is a linear
least-squares fit (same weight) multiplied by the natural cutoff (1 - r1/L)^C (1 - r2/L)^C;
R^2 is the fraction of the weighted variance of the CASINO term reproduced by the model.

Models (P_k are polynomials, the number of coefficients is given in the table):
    additive   A(r1) + A(r2) + H(r12)          - pure duplication of chi and u inside the cutoff
    s,r12      F(r1 + r2, r12)                 - depends on the mean distance only
    d,r12      F(|r1 - r2|, r12)
    s,d        F(r1 + r2, |r1 - r2|)            - no explicit e-e dependence
    s,d^2,r12  F(r1 + r2, (r1 - r2)^2, r12)    - full symmetric polynomial of lower degree
    product    g(r1) g(r2) h(r12)              - rank-1 separable (alternating least squares)
"""

import os

import numpy as np

from forms import TRUNC
from radial import load_densities
from terms import HERE, eta_profile, f_value, load, phi_value, species_label, u_profile

N_SAMPLES = 20000
DEGREE = 3


def sample_triplets(r_grid, rho, L, seed=0):
    rng = np.random.default_rng(seed)
    radial = 4 * np.pi * r_grid**2 * rho
    radial = np.where(r_grid < L, radial, 0.0)
    cdf = np.concatenate([[0], np.cumsum((radial[1:] + radial[:-1]) / 2 * np.diff(r_grid))])
    cdf /= cdf[-1]
    r1 = np.interp(rng.uniform(size=N_SAMPLES), cdf, r_grid)
    r2 = np.interp(rng.uniform(size=N_SAMPLES), cdf, r_grid)
    cos = rng.uniform(-1, 1, size=N_SAMPLES)
    r12 = np.sqrt(np.maximum(r1**2 + r2**2 - 2 * r1 * r2 * cos, 0))
    return r1, r2, r12


def cutoff_factor(r1, r2, L):
    return (1 - r1 / L) ** TRUNC * (1 - r2 / L) ** TRUNC


def poly_columns(variables, degree):
    """All monomials of total degree <= degree in the given variables (scaled to O(1))."""
    columns = [np.ones_like(variables[0])]
    if len(variables) == 1:
        for k in range(1, degree + 1):
            columns.append(variables[0] ** k)
        return columns
    for k in range(1, degree + 1):
        for power in powers(len(variables), k):
            column = np.ones_like(variables[0])
            for v, p in zip(variables, power):
                column = column * v**p
            columns.append(column)
    return columns


def powers(n, total):
    if n == 1:
        return [[total]]
    res = []
    for k in range(total + 1):
        for rest in powers(n - 1, total - k):
            res.append([k] + rest)
    return res


def linear_r2(y, columns, cut):
    a = np.array(columns).T * cut[:, None]
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    fit = a @ coef
    return 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2), a.shape[1]


def product_r2(y, x1, x2, x12, cut, iterations=50):
    """Rank-1 separable g1(r1) g2(r2) h(r12) with cubic factors, alternating linear least squares.
    The samples are symmetrized (r1 <-> r2), so g1 = g2 at convergence."""
    y = np.concatenate([y, y])
    x1, x2 = np.concatenate([x1, x2]), np.concatenate([x2, x1])
    x12, cut = np.concatenate([x12, x12]), np.concatenate([cut, cut])
    bases = [np.array(poly_columns([x], 3)).T for x in (x1, x2, x12)]
    coefs = [np.array([1.0, 0, 0, 0]) for _ in bases]
    for _ in range(iterations):
        for k in range(3):
            other = cut.copy()
            for j in range(3):
                if j != k:
                    other = other * (bases[j] @ coefs[j])
            coefs[k], *_ = np.linalg.lstsq(bases[k] * other[:, None], y, rcond=None)
    fit = cut * (bases[0] @ coefs[0]) * (bases[1] @ coefs[1]) * (bases[2] @ coefs[2])
    return 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2), 4 + 3 + 4 - 1


def models(y, r1, r2, r12, L):
    x1, x2, x12 = r1 / L, r2 / L, r12 / L
    s, d = (x1 + x2) / 2, np.abs(x1 - x2)
    cut = cutoff_factor(r1, r2, L)
    res = {}
    additive = poly_columns([x1], DEGREE + 1)
    additive = [c + cc for c, cc in zip(additive, poly_columns([x2], DEGREE + 1))] + poly_columns([x12], DEGREE + 1)[1:]
    res['additive'] = linear_r2(y, additive, cut)
    res['s,r12'] = linear_r2(y, poly_columns([s, x12], DEGREE + 1), cut)
    res['d,r12'] = linear_r2(y, poly_columns([d, x12], DEGREE + 1), cut)
    res['s,d'] = linear_r2(y, poly_columns([s, d], DEGREE + 1), cut)
    res['s,d^2,r12'] = linear_r2(y, poly_columns([s, d**2, x12], DEGREE), cut)
    res['product'] = product_r2(y, x1, x2, x12, cut)
    return res


MODEL_NAMES = ['additive', 's,r12', 'd,r12', 's,d', 's,d^2,r12', 'product']


def analyze(entry, term, density, with_ratio=True):
    """R^2 of every reduced model for every set and spin channel of one term; rows of table cells."""
    r_grid = np.array(density['grid'])
    if term == 'f':
        sets = entry['jastrow']['f']
        trunc = entry['jastrow']['trunc']
    else:
        sets = entry['backflow']['phi']
        trunc = entry['backflow']['trunc']
    rows = []
    for term_set in sets:
        rho = np.mean([density['rho'][i] for i in term_set['labels']], axis=0)
        L = term_set['cutoff']
        r1, r2, r12 = sample_triplets(r_grid, rho, L)
        if term == 'f':
            parameters = term_set['parameters']
        else:
            parameters = term_set[term]
        for spin in range(len(parameters)):
            if term == 'f':
                y = f_value(term_set, trunc, spin, r1, r2, r12)
            else:
                y = phi_value(term_set, trunc, spin, r1, r2, r12, term)
            if not np.any(y):
                continue
            ratio = np.nan
            if with_ratio and term == 'f':
                u = entry['jastrow']['u']
                pair = u_profile(u, trunc, min(spin, len(u['parameters']) - 1), r12)
                ratio = np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(pair**2))
            elif with_ratio:
                eta = entry['backflow']['eta']
                pair = eta_profile(eta, trunc, min(spin, len(eta['parameters']) - 1), r12)
                ratio = np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(pair**2))
            res = models(y, r1, r2, r12, L)
            rows.append({'species': species_label(entry, term_set['labels']), 'spin': spin, 'ratio': ratio, 'r2': {k: v[0] for k, v in res.items()}})
    return rows


def main():
    densities = load_densities()
    counts = models(*([np.linspace(0.1, 1, 50)] * 4), 2.0)
    lines = ['| system | term | species | spin | rms(term)/rms(u or eta) | ' + ' | '.join(MODEL_NAMES) + ' |']
    lines.append('|' + '---|' * 11)
    lines.append('| (coefficients per spin channel) | | | | | ' + ' | '.join(str(counts[name][1]) for name in MODEL_NAMES) + ' |')
    medians = {}
    for kind, term in (('Jastrow_emin', 'f'), ('Backflow_emin', 'phi'), ('Backflow_emin', 'theta')):
        medians[term] = []
        for entry in load([kind]):
            key = f'{entry["basis"]}:{entry["system"]}'
            if key not in densities or entry['basis'] not in ('gwfn', 'stowfn', 'pp'):
                continue
            for row in analyze(entry, term, densities[key]):
                medians[term].append([row['r2'][name] for name in MODEL_NAMES])
                cells = [key, term, row['species'], str(row['spin']), f'{row["ratio"]:.3f}'] + [f'{row["r2"][name]:.3f}' for name in MODEL_NAMES]
                lines.append('| ' + ' | '.join(cells) + ' |')
                print(lines[-1])
    summary = ['| term | channels | ' + ' | '.join(MODEL_NAMES) + ' |', '|' + '---|' * 8]
    for term, values in medians.items():
        values = np.array(values)
        summary.append(f'| {term} (median R^2) | {len(values)} | ' + ' | '.join(f'{v:.3f}' for v in np.median(values, axis=0)) + ' |')
        summary.append(f'| {term} (fraction R^2 > 0.95) | {len(values)} | ' + ' | '.join(f'{v:.2f}' for v in np.mean(values > 0.95, axis=0)) + ' |')
    scan = ['| scan (Be, STO) | term | N_eN | N_ee | ' + ' | '.join(MODEL_NAMES) + ' |', '|' + '---|' * 10]
    for entry in load():
        if entry['basis'] != 'stowfn-scan':
            continue
        term = 'f'
        parameters = entry.get('jastrow', {}).get('f')
        if not parameters:
            term = 'phi'
            parameters = entry['backflow']['phi']
        shape = np.array(parameters[0].get('parameters', parameters[0].get('phi'))).shape
        for row in analyze(entry, term, densities['stowfn:Be'], with_ratio=False):
            if row['spin'] != 1:
                continue
            scan.append(
                f'| {entry["kind"]} | {term} | {shape[2] - 1} | {shape[1] - 1} | '
                + ' | '.join(f'{row["r2"][name]:.3f}' for name in MODEL_NAMES)
                + ' |'
            )
    with open(os.path.join(HERE, 'results', 'manybody_separability.md'), 'w') as f:
        f.write('## Summary\n\n' + '\n'.join(summary) + '\n\n## Be expansion-order scans (antiparallel channel)\n\n' + '\n'.join(scan))
        f.write('\n\n## Per system\n\n' + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
