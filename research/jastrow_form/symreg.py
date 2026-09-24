#!/usr/bin/env python3
"""Symbolic regression (gplearn; PySR needs Julia, which cannot be downloaded here) on pooled profiles.

Every profile is put on a common scale before pooling, with scales read off the profile itself
(no fitted model is assumed):
    u (antiparallel)   x = r / b0, b0 = -u(0) / gamma      y = u / |u(0)|    ; exact exp(-x) if u = -gamma*b*exp(-r/b)
    chi                x = r / a0, chi(a0) = chi(0) / 2    y = chi / chi(0)
    eta (antiparallel) x = r / b0, eta(b0) = eta(0) / 2    y = eta / eta(0)
The points are weighted with the density weights and restricted to r < 0.7 L (away from the cutoff).
The parsimony coefficient penalizes the expression length (the number of parameters).
"""

import os

import numpy as np
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

from radial import load_densities, targets
from terms import HERE, load


def _exp(x):
    with np.errstate(over='ignore'):
        return np.where(x < 50, np.exp(np.minimum(x, 50)), 0.0)


def _weighted_mse(y, y_pred, w):
    return np.average((y - y_pred) ** 2, weights=w)


EXP = make_function(function=_exp, name='exp', arity=1)
FITNESS = make_fitness(function=_weighted_mse, greater_is_better=False)


def half_point(r, y):
    below = np.nonzero(np.abs(y) < np.abs(y[0]) / 2)[0]
    return r[below[0]]


def pool(term, channel_label):
    densities = load_densities()
    kind = 'Jastrow_emin'
    if term == 'eta':
        kind = 'Backflow_emin'
    entries = [e for e in load([kind]) if e['basis'] in ('gwfn', 'stowfn', 'pp')]
    xs, ys, ws, names = [], [], [], []
    for target in targets(entries, densities, term):
        for channel in target['channels']:
            if channel['label'] != channel_label:
                continue
            r, y, w = target['r'], channel['y'], target['w']
            if term == 'u':
                scale = -y[0] / channel['channel']
            else:
                scale = half_point(r, y)
            if scale <= 0:
                # u(0) > 0: parallel-spin shell correlation (Be, N), not a hole
                continue
            inside = (r < 0.7 * target['L']) & (w > 0.02)
            xs.append(r[inside] / scale)
            ys.append(y[inside] / abs(y[0]))
            ws.append(w[inside])
            names.append(f'{target["key"]}:{target["species"]}')
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), names


def run(term, channel_label, parsimony):
    x, y, w, names = pool(term, channel_label)
    model = SymbolicRegressor(
        population_size=3000,
        generations=30,
        function_set=('add', 'sub', 'mul', 'div', EXP),
        metric=FITNESS,
        parsimony_coefficient=parsimony,
        const_range=(-2.0, 2.0),
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        random_state=0,
        n_jobs=1,
    )
    model.fit(x[:, None], y, sample_weight=w)
    y_fit = model.predict(x[:, None])
    rms = np.sqrt(np.average((y_fit - y) ** 2, weights=w) / np.average(y**2, weights=w))
    return model._program, model._program.length_, rms, len(names)


def reference(term, channel_label):
    """The same metric for simple closed forms on the same pooled data."""
    x, y, w, _ = pool(term, channel_label)
    candidates = {
        'u': {'exp(-x)': np.exp(-x), '1/(1+x) - pade': 1 / (1 + x)},
        'chi': {'gauss exp(-ln2 x^2)': np.exp(-np.log(2) * x**2), 'lorentz 1/(1+x^2)': 1 / (1 + x**2), 'sech': 1 / np.cosh(1.3170 * x)},
        'eta': {'exp(-ln2 x)': np.exp(-np.log(2) * x), 'gauss': np.exp(-np.log(2) * x**2)},
    }[term]
    res = {}
    for name, y_fit in candidates.items():
        res[name] = np.sqrt(np.average((y_fit - y) ** 2, weights=w) / np.average(y**2, weights=w))
    return res


def main():
    lines = [
        '| term | channel | parsimony | expression (X0 = scaled r) | length | rel. weighted RMS | profiles pooled |',
        '|---|---|---|---|---|---|---|',
    ]
    ref_lines = ['| term | channel | closed form (no free parameters on the scaled axis) | rel. weighted RMS |', '|---|---|---|---|']
    for term, channel_label in (('u', 'ud'), ('u', 'uu'), ('chi', 'ud'), ('eta', 'ud')):
        for name, value in reference(term, channel_label).items():
            ref_lines.append(f'| {term} | {channel_label} | {name} | {value:.3f} |')
        for parsimony in (0.01, 0.001, 0.0001):
            program, length, rms, count = run(term, channel_label, parsimony)
            lines.append(f'| {term} | {channel_label} | {parsimony} | `{program}` | {length} | {rms:.3f} | {count} |')
            print(lines[-1])
    with open(os.path.join(HERE, 'results', 'symreg.md'), 'w') as f:
        f.write('\n'.join(ref_lines) + '\n\n' + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
