#!/usr/bin/env python3
"""Slices of f and a local-density interpretation of the e-e-n Jastrow term.

For two antiparallel electrons at the same distance R from the nucleus (r1 = r2 = R) the
pair function seen by the electrons is
    u_eff(r12; R) = u(r12) + f(R, R, r12),  0 <= r12 <= 2R.
f obeys the e-e no-cusp condition, so u_eff keeps the Kato slope 1/2 and can be fitted by
    u_eff = c(R) - gamma b(R) exp(-r12 / b(R)),
b(R) being the local hole radius.  It is compared with the local Wigner-Seitz radius
r_s(R) = (3 / (4 pi rho(R)))^(1/3): a law b = k r_s^beta turns f into a one- or two-parameter
'local-density hole' u(r12; rho(r1), rho(r2)).
"""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from radial import load_densities  # noqa: E402
from terms import HERE, PLOTS, f_value, load, u_profile  # noqa: E402

GAMMA = 0.5


def hole_radius(r12, y):
    def residuals(p):
        c, b = p
        return c - GAMMA * b * np.exp(-r12 / b) - y

    sol = least_squares(residuals, [y[-1], 1.0], bounds=([-np.inf, 0.05], [np.inf, 20.0]))
    fit = residuals(sol.x) + y
    error = np.sqrt(np.mean((fit - y) ** 2)) / max(np.ptp(y), 1e-12)
    return sol.x[1], error


def universal(r_s, b):
    """Pooled fit b = B r_s / (r_s + c) in log space."""

    def residuals(p):
        return np.log(p[0] * r_s / (r_s + p[1])) - np.log(b)

    sol = least_squares(residuals, [2.0, 1.0], bounds=([0.01, 0.01], [100, 100]))
    res = residuals(sol.x)
    r2 = 1 - np.sum(res**2) / np.sum((np.log(b) - np.log(b).mean()) ** 2)
    return sol.x, r2


def main():
    densities = load_densities()
    pooled_rs, pooled_b = [], []
    lines = ['| system | R range (bohr) | k | beta (b = k r_s^beta) | R^2 | median slice misfit |', '|---|---|---|---|---|---|']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for entry in load(['Jastrow_emin']):
        if entry['basis'] not in ('gwfn', 'stowfn', 'pp') or len(entry['atom_numbers']) > 1:
            continue
        if entry['neu'] == 0 or entry['ned'] == 0 or not entry['jastrow']['f']:
            continue
        key = f'{entry["basis"]}:{entry["system"]}'
        density = densities[key]
        grid = np.array(density['grid'])
        rho = np.array(density['rho'][0])
        jastrow = entry['jastrow']
        f_set = jastrow['f'][0]
        u = jastrow['u']
        L = f_set['cutoff']
        radii, holes, r_s, misfits = [], [], [], []
        for R in np.linspace(0.15, 0.6 * L, 25):
            r12 = np.linspace(0, 2 * R, 60)
            y = u_profile(u, jastrow['trunc'], 1, r12) + f_value(f_set, jastrow['trunc'], 1, R, R, r12)
            b, misfit = hole_radius(r12, y)
            radii.append(R)
            holes.append(b)
            misfits.append(misfit)
            r_s.append((3 / (4 * np.pi * np.interp(R, grid, rho))) ** (1 / 3))
        radii, holes, r_s = np.array(radii), np.array(holes), np.array(r_s)
        # the hole is resolved only where it fits in the slice (b < R) and the density is not negligible
        good = (holes < radii) & (holes > 0.06) & (holes < 19)
        if good.sum() >= 4:
            beta, logk = np.polyfit(np.log(r_s[good]), np.log(holes[good]), 1)
            fit = logk + beta * np.log(r_s[good])
            r2 = 1 - np.sum((np.log(holes[good]) - fit) ** 2) / np.sum((np.log(holes[good]) - np.log(holes[good]).mean()) ** 2)
            lines.append(
                f'| {key} | {radii[good].min():.2f}-{radii[good].max():.2f} | {np.exp(logk):.3f} | {beta:.3f} | {r2:.3f} | {np.median(misfits):.3f} |'
            )
        else:
            lines.append(f'| {key} | - | - | - | - | {np.median(misfits):.3f} |')
        pooled_rs += list(r_s[good])
        pooled_b += list(holes[good])
        axes[0].plot(radii, holes, 'o-', ms=3, label=key)
        axes[1].loglog(r_s[good], holes[good], 'o', ms=3, label=key)
        if entry['system'] == 'Ne' and entry['basis'] == 'gwfn':
            for R in (0.3, 0.6, 1.0, 1.5, 2.0):
                r12 = np.linspace(0, 2 * R, 60)
                axes[2].plot(r12, f_value(f_set, jastrow['trunc'], 1, R, R, r12), label=f'R = {R}')
    (B, c), r2 = universal(np.array(pooled_rs), np.array(pooled_b))
    lines += ['', f'Universal law over all atoms ({len(pooled_b)} slices): b = {B:.3f} r_s / (r_s + {c:.3f}),  R^2 (log) = {r2:.3f}']
    x = np.logspace(-1.1, 1.1, 50)
    axes[1].loglog(x, B * x / (x + c), 'k-', lw=2, label=f'b = {B:.2f} r_s/(r_s+{c:.2f})')
    axes[0].set_xlabel('R (bohr)')
    axes[0].set_ylabel('local hole radius b(R)')
    axes[1].set_xlabel('r_s(R) (bohr)')
    axes[1].set_ylabel('b(R)')
    axes[2].set_title('gwfn:Ne  f(R, R, r12), antiparallel')
    axes[2].set_xlabel('r12 (bohr)')
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'f_local_hole.png'), dpi=80)
    plt.close(fig)
    with open(os.path.join(HERE, 'results', 'f_local_hole.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
