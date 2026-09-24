#!/usr/bin/env python3
"""Universal laws and atom <-> molecule transferability.

Model-free descriptors are read directly off the CASINO profiles (last emin stage):
    u      b0 = -u(0)/gamma (hole radius of an exponential with the same depth and cusp), r_half
    chi    chi(0), r_half (chi(r_half) = chi(0)/2)
    eta    eta_ud(0), r_half
    mu     position and value of the extremum of mu
    L      cutoff lengths of the CASINO terms
and compared with the size of the atom: r_out = position of the outermost maximum of 4 pi r^2 rho.
Power laws  descriptor = c * Z^alpha  are fitted over the all-electron atoms (gwfn + stowfn)
and, separately, over the pseudopotential atoms.
"""

import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from forms import CHI_FORMS, value  # noqa: E402
from radial import derivative_error, load_densities, targets, weighted_error  # noqa: E402
from terms import HERE, PLOTS, load  # noqa: E402

BASES = ('gwfn', 'stowfn', 'pp')
MOLECULES = ('B2H6', 'O3')


def half_point(r, y):
    below = np.nonzero(np.abs(y) < np.abs(y[0]) / 2)[0]
    if below.size == 0 or y[0] == 0:
        return np.nan
    return r[below[0]]


def outer_radius(r, rho):
    radial = 4 * np.pi * r**2 * rho
    peaks = np.nonzero((radial[1:-1] > radial[:-2]) & (radial[1:-1] >= radial[2:]) & (radial[1:-1] > 0.05 * radial.max()))[0] + 1
    return r[peaks[-1]]


def descriptors():
    densities = load_densities()
    rows = []
    for term, kind in (('u', 'Jastrow_emin'), ('chi', 'Jastrow_emin'), ('eta', 'Backflow_emin'), ('mu', 'Backflow_emin')):
        entries = [e for e in load([kind]) if e['basis'] in BASES]
        for target in targets(entries, densities, term):
            r = target['r']
            density = densities[target['key']]
            for channel in target['channels']:
                y = channel['y']
                row = {
                    'term': term,
                    'basis': target['basis'],
                    'system': target['system'],
                    'species': target['species'],
                    'Z': target['Z'],
                    'pp': bool(target.get('pp', target['basis'] == 'pp')),
                    'molecule': target['system'] in MOLECULES,
                    'channel': channel['label'],
                    'L': target['L'],
                    'r_half': half_point(r, y),
                    'y0': y[0],
                }
                if term == 'u':
                    row['b0'] = -y[0] / channel['channel']
                if 'labels' in target:
                    rho = np.mean([density['rho'][i] for i in target['labels']], axis=0)
                    row['r_out'] = outer_radius(r, rho)
                if term == 'mu':
                    inside = r < target['L']
                    k = np.argmax(np.abs(y[inside]) * target['w'][inside])
                    row['r_ext'] = r[inside][k]
                    row['y_ext'] = y[inside][k]
                rows.append(row)
    return rows


def power_law(z, v):
    good = np.isfinite(v) & (v > 0) & (z > 0)
    if good.sum() < 3:
        return np.nan, np.nan, np.nan
    x, y = np.log(z[good]), np.log(v[good])
    alpha, logc = np.polyfit(x, y, 1)
    fit = logc + alpha * x
    r2 = 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2)
    return np.exp(logc), alpha, r2


def shell_table(rows):
    """chi and mu length scales against the radius of the outermost shell r_out (proportional fit)."""
    lines = ['| term | descriptor | atoms | k in descriptor = k r_out | R^2 |', '|---|---|---|---|---|']
    for term, name in (('chi', 'r_half'), ('chi', 'L'), ('mu', 'r_ext')):
        for group, pp in (('AE', False), ('PP', True), ('all', None)):
            x, y = [], []
            for row in rows:
                if row['term'] != term or row['molecule'] or row['channel'] == 'd' or row['Z'] < 2:
                    continue
                if pp is not None and row['pp'] != pp:
                    continue
                if np.isfinite(row.get(name, np.nan)):
                    x.append(row['r_out'])
                    y.append(row[name])
            x, y = np.array(x), np.array(y)
            k = np.sum(x * y) / np.sum(x * x)
            r2 = 1 - np.sum((y - k * x) ** 2) / np.sum((y - y.mean()) ** 2)
            lines.append(f'| {term} | {name} | {group} ({len(x)}) | {k:.3f} | {r2:.3f} |')
    return lines


def law_table(rows):
    lines = ['| term | channel | descriptor | atoms | c | alpha | R^2 (log-log) |', '|---|---|---|---|---|---|---|']
    laws = {}
    specs = [
        ('u', 'ud', 'b0'),
        ('u', 'uu', 'b0'),
        ('u', 'ud', 'L'),
        ('chi', None, 'y0'),
        ('chi', None, 'r_half'),
        ('chi', None, 'L'),
        ('chi', None, 'r_out'),
        ('eta', 'ud', 'y0'),
        ('eta', 'ud', 'r_half'),
        ('mu', None, 'r_ext'),
    ]
    for term, channel, name in specs:
        for group, pp in (('AE', False), ('PP', True)):
            selected = []
            for row in rows:
                if row['term'] != term or row['molecule'] or row['pp'] != pp:
                    continue
                if channel is not None and row['channel'] != channel:
                    continue
                if channel is None and row['channel'] == 'd':
                    continue
                if term in ('u', 'eta'):
                    selected.append(row)
                elif row['Z'] > 1:
                    selected.append(row)
            if term in ('u', 'eta'):
                z = np.array([atomic_z(row) for row in selected], dtype=float)
            else:
                z = np.array([row['Z'] for row in selected], dtype=float)
            v = np.array([abs(row.get(name, np.nan)) for row in selected], dtype=float)
            c, alpha, r2 = power_law(z, v)
            laws[(term, channel, name, group)] = (c, alpha)
            lines.append(f'| {term} | {channel or "all"} | {name} | {group} ({len(selected)}) | {c:.3g} | {alpha:.3f} | {r2:.3f} |')
    return lines, laws


ATOM_Z = {'H': 1, 'He': 2, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Ar': 18, 'Kr': 36}


def atomic_z(row):
    return ATOM_Z.get(row['system'], np.nan)


def plot_laws(rows):
    specs = [('u', 'ud', 'b0'), ('chi', 'ud', 'y0'), ('chi', 'ud', 'r_half'), ('chi', 'ud', 'L'), ('eta', 'ud', 'y0'), ('mu', 'ud', 'r_ext')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (term, channel, name) in zip(axes.flat, specs):
        for pp, molecule, marker, color, label in (
            (False, False, 'o', 'C0', 'AE atom'),
            (True, False, 's', 'C1', 'PP atom'),
            (False, True, '*', 'C2', 'AE nucleus in molecule'),
            (True, True, 'x', 'C3', 'PP nucleus in molecule'),
        ):
            z, v, names = [], [], []
            for row in rows:
                if row['term'] != term or row['pp'] != pp or row['molecule'] != molecule or row['channel'] not in (channel, 'u'):
                    continue
                if term in ('u', 'eta'):
                    zz = atomic_z(row)
                else:
                    zz = row['Z']
                z.append(zz)
                v.append(abs(row.get(name, np.nan)))
                names.append(f'{row["system"]}:{row["species"]}')
            ax.plot(z, v, marker, color=color, ls='', label=label)
            for zz, vv, nn in zip(z, v, names):
                if molecule:
                    ax.annotate(nn, (zz, vv), fontsize=6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Z')
        ax.set_title(f'{term} {name}')
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'scaling_laws.png'), dpi=80)
    plt.close(fig)


def chi_transfer(laws):
    """chi of the nuclei in B2H6 and O3 predicted by the window form chi = A w(r/L) with A(Z), L(Z) from the atoms."""
    with open(os.path.join(HERE, 'results', 'radial_fits.json')) as f:
        fits = json.load(f)['chi']
    densities = load_densities()
    entries = [e for e in load(['Jastrow_emin']) if e['basis'] in BASES]
    form = CHI_FORMS['window']
    atoms = {}
    for row in fits:
        system = row['system'].split(':')[1]
        if system not in MOLECULES and not row['pp'] and row['Z'] > 1:
            params = row['forms']['window']
            atoms[row['Z']] = atoms.get(row['Z'], []) + [(params['parameters'][0][0], params['L'])]
    z = np.array(sorted(atoms), dtype=float)
    a = np.array([np.mean([p[0] for p in atoms[k]]) for k in sorted(atoms)])
    ll = np.array([np.mean([p[1] for p in atoms[k]]) for k in sorted(atoms)])
    fit_a = np.polyfit(np.log(z), np.log(a), 1)
    fit_l = np.polyfit(np.log(z), np.log(ll), 1)
    lines = [
        f'Window-form Z-law from AE atoms: A = {np.exp(fit_a[1]):.3f} Z^{fit_a[0]:.3f}, L = {np.exp(fit_l[1]):.3f} Z^{fit_l[0]:.3f}',
        '',
        '| molecule | species | eps (eps_d) own window fit | eps (eps_d) Z-law (A, L) | eps (eps_d) nearest AE atom | noise emin/varmin | noise CASINO/PyCasino emin |',
        '|---|---|---|---|---|---|---|',
    ]
    for target in targets(entries, densities, 'chi'):
        if target['system'] not in MOLECULES:
            continue
        row = [x for x in fits if x['system'] == target['key'] and x['species'] == target['species'] and abs(x['L_casino'] - target['L']) < 1e-9][0]
        Z = target['Z']
        A_law = np.exp(np.polyval(fit_a, np.log(Z)))
        L_law = np.exp(np.polyval(fit_l, np.log(Z)))
        predicted = [value(form, target['r'], [A_law], None, L_law) for _ in target['channels']]
        nearest = min(atoms, key=lambda k: abs(k - Z))
        A_near, L_near = atoms[nearest][0]
        near = [value(form, target['r'], [A_near], None, L_near) for _ in target['channels']]
        own = row['forms']['window']
        lines.append(
            f'| {target["key"]} | {target["species"]} | {own["error"]:.3f} ({own["d_error"]:.2f}) | '
            f'{weighted_error(target, predicted):.3f} ({derivative_error(target, predicted):.2f}) | '
            f'{weighted_error(target, near):.3f} ({derivative_error(target, near):.2f}) Z={nearest} | '
            f'{row["noise"]:.3f} ({row["noise_d"]:.2f}) | {row["noise_same"]:.3f} ({row["noise_same_d"]:.2f}) |'
        )
    return lines


def main():
    rows = descriptors()
    lines = ['# Scaling laws and transferability (generated by scaling.py)', '', '## Power laws over atoms', '']
    law_lines, laws = law_table(rows)
    lines += law_lines
    lines += ['', '## Length scales vs the outermost shell radius', '']
    lines += shell_table(rows)
    lines += [
        '',
        '## Descriptors per system',
        '',
        '| term | system | species | channel | Z | PP | L | y(0) | r_half | b0 / r_ext | r_out |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for row in rows:
        extra = row.get('b0', row.get('r_ext', np.nan))
        lines.append(
            f'| {row["term"]} | {row["basis"]}:{row["system"]} | {row["species"]} | {row["channel"]} | {row["Z"]} | {int(row["pp"])} | {row["L"]:.2f} | '
            f'{row["y0"]:.4f} | {row["r_half"]:.2f} | {extra:.2f} | {row.get("r_out", np.nan):.2f} |'
        )
    lines += ['', '## chi transfer atom -> molecule', '']
    lines += chi_transfer(laws)
    plot_laws(rows)
    with open(os.path.join(HERE, 'results', 'scaling.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:40]))


if __name__ == '__main__':
    main()
