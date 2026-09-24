#!/usr/bin/env python3
"""Fit candidate forms to the radial profiles u, chi, eta, mu and compare with the CASINO polynomial.

Error metric (all tables): density-weighted relative RMS
    eps = sqrt(sum_w (y_fit - y)^2 / sum_w y^2)
with w = pair-distance distribution (u, eta) or 4 pi r^2 rho_I(r) (chi, mu).
The noise floor eps_noise is the same metric between the emin and varmin optimizations
of the same system: a form with eps < eps_noise reproduces the profile to within the
reproducibility of the optimization itself.
"""

import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from forms import CHI_FORMS, ETA_FORMS, MU_FORMS, U_FORMS  # noqa: E402
from radial import derivative_error, fit_form, fit_polynomial, load_densities, match, n_form_params, noise_error, targets  # noqa: E402
from terms import HERE, PLOTS, load  # noqa: E402

TERMS = {
    'u': (U_FORMS, 'Jastrow_emin', 'Jastrow_varmin'),
    'chi': (CHI_FORMS, 'Jastrow_emin', 'Jastrow_varmin'),
    'eta': (ETA_FORMS, 'Backflow_emin', 'Backflow_varmin'),
    'mu': (MU_FORMS, 'Backflow_emin', 'Backflow_varmin'),
}
POLY_ORDERS = (2, 3, 4)
BASES = ('gwfn', 'stowfn', 'pp')
RESULTS = os.path.join(HERE, 'results')


def run_term(term, densities):
    forms, kind, noise_kind = TERMS[term]
    entries = [e for e in load([kind]) if e['basis'] in BASES]
    noise_entries = [e for e in load([noise_kind]) if e['basis'] in BASES]
    same_entries = [e for e in load([kind + '/linear']) if e['basis'] in BASES]
    main_targets = targets(entries, densities, term)
    noise_targets = targets(noise_entries, densities, term)
    same_targets = targets(same_entries, densities, term)
    rows = []
    for target in main_targets:
        row = {
            'system': target['key'],
            'species': target['species'],
            'Z': target['Z'],
            'pp': bool(target.get('pp', False)),
            'L_casino': target['L'],
            'n_casino': target['n_casino'],
            'noise': None,
            'forms': {},
            'poly': {},
        }
        row['noise'], row['noise_d'] = noise_error(target, match(noise_targets, target))
        row['noise_same'], row['noise_same_d'] = noise_error(target, match(same_targets, target))
        curves = {}
        for name, form in forms.items():
            error, parameters, L, y_fit = fit_form(target, form)
            d_error = derivative_error(target, y_fit)
            row['forms'][name] = {'error': error, 'd_error': d_error, 'parameters': parameters, 'L': L, 'n': n_form_params(form, target)}
            curves[name] = y_fit
        for order in POLY_ORDERS:
            error, n_params, L, y_fit = fit_polynomial(target, order)
            row['poly'][order] = {'error': error, 'd_error': derivative_error(target, y_fit), 'L': L, 'n': n_params}
            curves[f'poly{order}'] = y_fit
        rows.append(row)
        plot_target(term, target, curves, row)
        print(
            term,
            target['key'],
            target['species'],
            {k: round(v['error'], 3) for k, v in row['forms'].items()},
            'noise',
            np.round(row['noise'], 3),
            np.round(row['noise_d'], 3),
        )
    return rows


def plot_target(term, target, curves, row):
    os.makedirs(os.path.join(PLOTS, term), exist_ok=True)
    fig, axes = plt.subplots(1, len(target['channels']), figsize=(6 * len(target['channels']), 4), squeeze=False)
    r = target['r']
    r_max = max(target['L'], 1.0) * 1.1
    for i, channel in enumerate(target['channels']):
        ax = axes[0, i]
        ax.plot(r, channel['y'], 'k', lw=2.5, label=f'CASINO ({row["n_casino"]} p)')
        for name, y_fit in curves.items():
            style = '-'
            if name.startswith('poly'):
                style = ':'
            ax.plot(r, y_fit[i], style, label=name)
        scale = np.max(np.abs(channel['y'])) or 1.0
        ax.fill_between(r, 0, target['w'] * scale, color='grey', alpha=0.2, label='weight')
        ax.set_xlim(0, r_max)
        ax.set_title(f'{term} {target["key"]} {target["species"]} {channel["label"]}')
        ax.set_xlabel('r (bohr)')
        ax.legend(fontsize=6)
    fig.tight_layout()
    name = f'{target["key"]}_{target["species"]}{target.get("labels", [""])[0]}'.replace(':', '_').replace('+', '')
    fig.savefig(os.path.join(PLOTS, term, f'{name}.png'), dpi=70)
    plt.close(fig)


def summary_line(label, n, results, rows, meaning):
    errors = np.array([x['error'] for x in results])
    d_errors = np.array([x['d_error'] for x in results])
    noise = np.array([row['noise'] for row in rows])
    noise_d = np.array([row['noise_d'] for row in rows])
    same_d = np.array([row['noise_same_d'] for row in rows])
    good = np.sum(errors <= noise)
    good_d = np.sum(d_errors <= noise_d)
    good_same = np.sum(d_errors <= same_d)
    total = np.sum(np.isfinite(noise))
    total_same = np.sum(np.isfinite(same_d))
    return (
        f'| {label} | {n} | {np.median(errors):.3f} | {errors.max():.3f} | {np.median(d_errors):.3f} | {good}/{total} | {good_d}/{total} | '
        f'{good_same}/{total_same} | {meaning} |'
    )


def summary(term, rows):
    forms = TERMS[term][0]
    lines = [f'### {term}', '']
    lines.append(
        '| form | params (first system) | median eps | max eps | median eps_d | eps <= noise | eps_d <= noise_d | eps_d <= noise_same_d | interpretation |'
    )
    lines.append('|---|---|---|---|---|---|---|---|---|')
    lines.append(f'| CASINO full | {rows[0]["n_casino"]} | 0 | 0 | 0 | - | - | - | polynomial with (r-L)^C cutoff |')
    for order in POLY_ORDERS:
        results = [row['poly'][order] for row in rows]
        lines.append(summary_line(f'CASINO N={order} refit', rows[0]['poly'][order]['n'], results, rows, 'polynomial'))
    for name, form in forms.items():
        results = [row['forms'][name] for row in rows]
        lines.append(summary_line(name, rows[0]['forms'][name]['n'], results, rows, form['meaning']))
    noise = np.array([row['noise'] for row in rows])
    noise_d = np.array([row['noise_d'] for row in rows])
    lines.append(f'| noise: emin vs varmin | - | {np.nanmedian(noise):.3f} | {np.nanmax(noise):.3f} | {np.nanmedian(noise_d):.3f} | - | - | - | - |')
    same = np.array([row['noise_same'] for row in rows])
    same_d = np.array([row['noise_same_d'] for row in rows])
    if np.any(np.isfinite(same)):
        count = np.sum(np.isfinite(same))
        lines.append(
            f'| noise: CASINO emin vs PyCasino emin ({count} systems) | - | {np.nanmedian(same):.3f} | {np.nanmax(same):.3f} | {np.nanmedian(same_d):.3f} | - | - | - | - |'
        )
    lines.append('')
    header = ['system', 'species', 'noise', 'noise_d', 'noise_same'] + [f'N={order}' for order in POLY_ORDERS] + list(forms)
    lines.append('eps (eps_d) per system:')
    lines.append('')
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '---|' * len(header))
    for row in rows:
        cells = [row['system'], row['species'], f'{row["noise"]:.3f}', f'{row["noise_d"]:.3f}', f'{row["noise_same"]:.3f}']
        cells += [f'{row["poly"][order]["error"]:.3f} ({row["poly"][order]["d_error"]:.2f})' for order in POLY_ORDERS]
        cells += [f'{row["forms"][name]["error"]:.3f} ({row["forms"][name]["d_error"]:.2f})' for name in forms]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def main():
    os.makedirs(RESULTS, exist_ok=True)
    densities = load_densities()
    results = {}
    lines = ['# Radial fits (generated by fit_radial.py)', '']
    for term in TERMS:
        rows = run_term(term, densities)
        results[term] = rows
        lines += summary(term, rows)
    with open(os.path.join(RESULTS, 'radial_fits.json'), 'w') as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(RESULTS, 'radial_fits.md'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
