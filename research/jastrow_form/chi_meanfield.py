#!/usr/bin/env python3
"""Is chi slaved to u?  Density-preserving mean-field estimate of the e-n term.

With |Psi|^2 = |D|^2 exp(2J) the one-electron density is, to first order in J,
    rho(r) ~ rho_HF(r) exp(2 chi(r) + 2 V(r)),   V_s(r) = sum_s' (N_s' - delta_ss') / N  int rho(r') u_ss'(|r - r'|) dr'
so the e-n term that keeps the HF density unchanged is chi(r) = -V(r) + const.
The script regresses the optimized chi on V: chi ~ alpha + beta V, weighted by 4 pi r^2 rho.
beta = -1 means chi only undoes the density distortion caused by u; the residual is the part
of chi that actually improves the density.
"""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from radial import channel_used, en_weight, load_densities  # noqa: E402
from terms import CHI_SPIN_LABELS, HERE, PLOTS, U_SPIN_LABELS, chi_profile, load, species_label, u_profile  # noqa: E402

T_NODES, T_WEIGHTS = np.polynomial.legendre.leggauss(48)


def spherical_convolution(r, rho, u):
    """int rho(r') u(|r - r'|) d^3r' for a spherical density rho on grid r and a function u."""
    rp = r[None, :, None]
    distance = np.sqrt(np.maximum(r[:, None, None] ** 2 + rp**2 - 2 * r[:, None, None] * rp * T_NODES[None, None, :], 0))
    angular = np.sum(u(distance) * T_WEIGHTS, axis=2)
    return 2 * np.pi * np.trapezoid(angular * r[None, :] ** 2 * rho[None, :], r, axis=1)


def mean_field(entry, r, rho, spin):
    """V_s(r) for electron spin s ('u' or 'd'), spin densities approximated by rho * N_s / N."""
    jastrow = entry['jastrow']
    u = jastrow['u']
    labels = U_SPIN_LABELS[len(u['parameters']) - 1]
    n = {'u': entry['neu'], 'd': entry['ned']}
    total = entry['neu'] + entry['ned']
    res = np.zeros_like(r)
    for other in ('u', 'd'):
        pair = 'ud'
        if other == spin:
            pair = spin + spin
        if pair not in labels:
            pair = 'uu'
        count = n[other] - int(other == spin)
        if count <= 0:
            continue
        index = labels.index(pair)

        def u_pair(x):
            return u_profile(u, jastrow['trunc'], index, x)

        res += count / total * spherical_convolution(r, rho, u_pair)
    return res


def main():
    densities = load_densities()
    lines = [
        '| system | species | spin | beta | weighted R^2 | eps (beta=-1, alpha fitted) | eps (beta, alpha fitted) |',
        '|---|---|---|---|---|---|---|',
    ]
    os.makedirs(os.path.join(PLOTS, 'chi_meanfield'), exist_ok=True)
    for entry in load(['Jastrow_emin']):
        key = f'{entry["basis"]}:{entry["system"]}'
        if key not in densities or entry['basis'] not in ('gwfn', 'stowfn', 'pp'):
            continue
        density = densities[key]
        r = np.array(density['grid'])
        jastrow = entry['jastrow']
        for chi in jastrow['chi']:
            rho = np.mean([density['rho'][i] for i in chi['labels']], axis=0)
            w = en_weight(density, chi['labels'])
            spin_labels = CHI_SPIN_LABELS[len(chi['parameters']) - 1]
            fig, ax = plt.subplots(figsize=(6, 4))
            for index, label in enumerate(spin_labels):
                y = chi_profile(chi, jastrow['trunc'], index, r)
                if not np.any(y):
                    continue
                spins = [s for s in ('u', 'd') if channel_used(entry, s) and (label == 'ud' or label == s)]
                weights = np.array([{'u': entry['neu'], 'd': entry['ned']}[s] for s in spins], dtype=float)
                v = sum(wt * mean_field(entry, r, rho, s) for wt, s in zip(weights, spins)) / weights.sum()
                inside = r < chi['cutoff']
                sw = w * inside
                a = np.stack([np.ones_like(r), v], axis=1)
                coef, *_ = np.linalg.lstsq(np.sqrt(sw)[:, None] * a, np.sqrt(sw) * y, rcond=None)
                fit = a @ coef
                mean = np.sum(sw * y) / np.sum(sw)
                r2 = 1 - np.sum(sw * (y - fit) ** 2) / np.sum(sw * (y - mean) ** 2)
                alpha_fixed = np.sum(sw * (y + v)) / np.sum(sw)
                fixed = alpha_fixed - v
                eps_fixed = np.sqrt(np.sum(sw * (fixed - y) ** 2) / np.sum(sw * y**2))
                eps_fit = np.sqrt(np.sum(sw * (fit - y) ** 2) / np.sum(sw * y**2))
                species = species_label(entry, chi['labels'])
                lines.append(f'| {key} | {species} | {label} | {coef[1]:.2f} | {r2:.3f} | {eps_fixed:.3f} | {eps_fit:.3f} |')
                ax.plot(r[inside], y[inside], 'k', lw=2, label=f'chi {label} (CASINO)')
                ax.plot(r[inside], fixed[inside], '--', label='-V + const')
                ax.plot(r[inside], fit[inside], ':', label=f'{coef[1]:.2f} V + const')
            ax.fill_between(r, 0, w * max(1e-3, np.max(np.abs(y))), color='grey', alpha=0.2)
            ax.set_xlim(0, chi['cutoff'])
            ax.set_title(f'{key} {species_label(entry, chi["labels"])}')
            ax.legend(fontsize=7)
            fig.tight_layout()
            name = f'{key}_{species_label(entry, chi["labels"])}_{chi["labels"][0]}'.replace(':', '_')
            fig.savefig(os.path.join(PLOTS, 'chi_meanfield', f'{name}.png'), dpi=70)
            plt.close(fig)
    with open(os.path.join(HERE, 'results', 'chi_meanfield.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
