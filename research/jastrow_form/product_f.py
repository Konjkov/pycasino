#!/usr/bin/env python3
"""Step 1 (A) of the jastrow-forms plan: can a product f replace the polynomial CASINO f?

On configurations sampled by VMC from the CASINO Slater-Jastrow wave function (last emin stage) the full
CASINO Jastrow J = sum u + sum chi + sum f is fitted with

    (C) uncut u and chi, the CASINO f kept as it is (the Jastrow_emin_uncut examples)
    (A) uncut u and chi, and the product f per f set, nucleus and spin channel

        f(r1, r2, r12) = (1 - x1)^C (1 - x2)^C g(x1) g(x2) h(x12),  x = r / L,  r1, r2 < L
        g(x) = 1 + C x + g2 x^2 + g3 x^3     (no e-n cusp: d/dx [(1 - x)^C g(x)] = 0 at x = 0)
        h(x) = h0 + h2 x^2 + h3 x^3          (no e-e cusp: h'(0) = 0)

    with L the CASINO f cutoff, 5 parameters per spin channel (g(0) = 1 fixes the scale of the product).
Every model has a free constant. The criterion is rms(fit - J) against the spread of J.
The product f is also fitted alone to the CASINO f on the same configurations.
"""

import json
import os
import tempfile

import numpy as np
from scipy.optimize import least_squares

from make_uncut_examples import B_MAX, B_MIN, casino_u_chi, geometry, model, sample
from terms import f_value, load

SYSTEMS = ['He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3']
CACHE = os.path.join(tempfile.gettempdir(), 'jastrow_form_configs')
N_PRODUCT = 5


def configurations(entry):
    os.makedirs(CACHE, exist_ok=True)
    file_name = os.path.join(CACHE, f'{entry["system"]}.npz')
    if not os.path.exists(file_name):
        position, atoms = sample(entry)
        np.savez(file_name, position=position, atoms=atoms)
    data = np.load(file_name)
    return data['position'], data['atoms']


def pair_channels(entry, n_labels):
    """Spin channel of each pair i < j in the order of np.triu_indices: 0 uu, 1 ud, 2 dd (uu if absent)."""
    neu = entry['neu']
    ne = neu + entry['ned']
    i, j = np.triu_indices(ne, 1)
    channel = np.where((i < neu) == (j < neu), 0, 1)
    if n_labels == 3:
        channel = np.where((i >= neu) & (j >= neu), 2, channel)
    return channel


def f_blocks(entry, r_ee, r_en):
    """Per f set, atom and spin channel: scaled distances and the cutoff factor of the pairs of the channel."""
    ne = r_en.shape[1]
    i, j = np.triu_indices(ne, 1)
    trunc = entry['jastrow']['trunc']
    blocks = []
    for n, f in enumerate(entry['jastrow']['f']):
        L = f['cutoff']
        n_labels = len(f['parameters'])
        channel = pair_channels(entry, n_labels)
        for atom in f['labels']:
            for spin in range(n_labels):
                k = channel == spin
                if not np.any(k):
                    continue
                x1 = r_en[:, i[k], atom] / L
                x2 = r_en[:, j[k], atom] / L
                x12 = r_ee[:, k] / L
                cut = np.where((x1 < 1) & (x2 < 1), (1 - x1) ** trunc * (1 - x2) ** trunc, 0.0)
                blocks.append(dict(set=n, spin=spin, x1=x1, x2=x2, x12=x12, cut=cut, trunc=trunc))
    return blocks


def casino_f(entry, blocks):
    res = 0.0
    for block in blocks:
        f = entry['jastrow']['f'][block['set']]
        L = f['cutoff']
        value = f_value(f, block['trunc'], block['spin'], block['x1'] * L, block['x2'] * L, block['x12'] * L)
        res = res + value.sum(axis=1)
    return res


def product_offsets(entry):
    """Index of the first product parameter of every (f set, spin channel)."""
    offsets = {}
    n = 0
    for s, f in enumerate(entry['jastrow']['f']):
        for spin in range(len(f['parameters'])):
            offsets[(s, spin)] = n
            n += N_PRODUCT
    return offsets, n


def product_f(q, blocks, offsets):
    res = 0.0
    for block in blocks:
        g2, g3, h0, h2, h3 = q[offsets[(block['set'], block['spin'])] :][:N_PRODUCT]
        C = block['trunc']
        x1, x2, x12 = block['x1'], block['x2'], block['x12']
        g_1 = 1 + C * x1 + g2 * x1**2 + g3 * x1**3
        g_2 = 1 + C * x2 + g2 * x2**2 + g3 * x2**3
        h = h0 + h2 * x12**2 + h3 * x12**3
        res = res + (block['cut'] * g_1 * g_2 * h).sum(axis=1)
    return res


def uncut_start(entry, n_chi):
    p = [0.0, 0.0]
    for chi, spins in zip(entry['jastrow']['chi'], n_chi):
        p += [0.0, np.log(chi['cutoff'] / 2)] * spins
    p.append(0.0)
    return np.array(p)


def rms(x):
    return np.sqrt(np.mean(x**2))


def analyze(system):
    entry = [e for e in load(['Jastrow_emin']) if e['basis'] == 'stowfn' and e['system'] == system][0]
    position, atoms = configurations(entry)
    r_ee, antiparallel, r_en, down = geometry(position, atoms, entry['neu'])
    chi_sets = [chi['labels'] for chi in entry['jastrow']['chi']]
    n_chi = [len(chi['parameters']) for chi in entry['jastrow']['chi']]
    blocks = f_blocks(entry, r_ee, r_en)
    offsets, n_product = product_offsets(entry)

    f_target = casino_f(entry, blocks)
    target = casino_u_chi(entry, r_ee, r_en, down) + f_target

    def uncut(p):
        return model(p, r_ee, antiparallel, r_en, down, chi_sets, chi_spins=n_chi)

    # (C) uncut u and chi with the CASINO f
    c = least_squares(lambda p: uncut(p) + f_target - target, uncut_start(entry, n_chi), method='lm')

    # the product f alone against the CASINO f (plus a constant)
    q0 = np.zeros(n_product + 1)
    alone = least_squares(lambda q: product_f(q, blocks, offsets) + q[-1] - f_target, q0, method='lm')

    # (A) uncut u and chi and the product f together
    n_uncut = c.x.size
    x0 = np.concatenate([c.x, alone.x[:-1]])

    def residuals(x):
        return uncut(x[:n_uncut]) + product_f(x[n_uncut:], blocks, offsets) - target

    a = least_squares(residuals, x0, method='lm')
    lower = np.full(x0.size, -np.inf)
    upper = np.full(x0.size, np.inf)
    lower[:2], upper[:2] = np.log(B_MIN), np.log(B_MAX)
    if np.any(a.x[:2] < lower[:2]) or np.any(a.x[:2] > upper[:2]):
        # a hole radius ran away (Kr parallel pairs): refit inside the bounds from the clipped solution
        a = least_squares(residuals, np.clip(a.x, lower + 1e-6, upper - 1e-6), bounds=(lower, upper), x_scale='jac')

    row = dict(
        system=system,
        configs=len(position),
        spread=np.std(target),
        f_spread=np.std(f_target),
        C=rms(c.fun),
        f_alone=rms(alone.fun),
        A=rms(a.fun),
        n_f_casino=int(sum(np.count_nonzero(np.array(f['parameters'])) for f in entry['jastrow']['f'])),
        n_f_product=n_product,
        b=np.exp(a.x[:2]),
        uncut=a.x[:n_uncut].tolist(),
        product=a.x[n_uncut:].tolist(),
    )
    print(
        f'{system:3s} spread {row["spread"]:.3f}  f spread {row["f_spread"]:.3f}  '
        f'(C) {row["C"]:.3f}  f alone {row["f_alone"]:.3f}  (A) {row["A"]:.3f}',
        flush=True,
    )
    return row


def main():
    rows = [analyze(system) for system in SYSTEMS]
    lines = [
        '# Product f against the full CASINO Jastrow (plan step 1 A)',
        '',
        'rms over VMC configurations of the CASINO Slater-Jastrow (last emin stage), a.u.',
        'spread: std of the full CASINO J; f spread: std of the CASINO sum f;',
        '(C): uncut u, chi + CASINO f; f alone: product f fitted to the CASINO f; (A): uncut u, chi + product f.',
        'f params: stored nonzero CASINO f coefficients / product f parameters.',
        '',
        '| system | configs | spread | f spread | (C) | f alone | (A) | (A)/(C) | f params | b_par, b_anti |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for r in rows:
        lines.append(
            f'| {r["system"]} | {r["configs"]} | {r["spread"]:.3f} | {r["f_spread"]:.3f} | {r["C"]:.3f} | '
            f'{r["f_alone"]:.3f} | {r["A"]:.3f} | {r["A"] / r["C"]:.2f} | {r["n_f_casino"]} / {r["n_f_product"]} | '
            f'{r["b"][0]:.2f}, {r["b"][1]:.2f} |'
        )
    text = '\n'.join(lines) + '\n'
    results = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    with open(os.path.join(results, 'product_f.md'), 'w') as f:
        f.write(text)
    # starting values of the Jastrow_emin_product examples, in the parameters of model() and product_f()
    with open(os.path.join(results, 'product_f.json'), 'w') as f:
        json.dump({r['system']: dict(uncut=r['uncut'], product=r['product']) for r in rows}, f, indent=1)
    print(text)


if __name__ == '__main__':
    main()
