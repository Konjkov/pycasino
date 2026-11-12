#!/usr/bin/env python3

"""Acceptance ratio against the VMC time step normalized to its 50% value, measured in CASINO.

    examples/acceptance_curve/acceptance_curve.py [SYSTEM ...]

One CASINO VMC run per point, all of them prepared beside this file with opt_dtvmc off so the
step is exactly the one asked for. Each `a<A>` is one point of the grid Casino.vmc_step_graph
lays out equally in acceptance, dtvmc = dtvmc50 * [erfinv(1 - A)/erfinv(1/2)]^2 over the
nineteen targets 0.95 ... 0.05. `opt` is the run that lets CASINO set the step itself. dtvmc50
is not taken from that run -- CASINO only optimizes the step under vmc_method 1 -- but
interpolated from the grid, and the optimized value is printed beside it as a check.

Two curves are drawn beside the measurement. The Gaussian one is what the step would do if
X = ln(Psi'^2/Psi^2) were normal: the proposal is Gaussian with variance dtvmc, so sd(X)
grows as sqrt(dtvmc) and the acceptance is 2*Phi(-sd/2), pinned to 1/2 at x = 1. The other is
the two-parameter form p = C / (C + exp(u) - 1) of the acceptance curve, fitted here in
u = a * sqrt(x), the variable the Gaussian model is linear in.
"""

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc, erfinv

# categorical slots 1-2 for the systems, 3 for the model
COLORS = ('#2a78d6', '#eb6834', '#1baf7a')
INK = '#0b0b0b'
MUTED = '#8a8a85'

systems = sys.argv[1:] or ['He/CBCS', 'He/EBES']
here = os.path.dirname(os.path.abspath(__file__))

# 2*Phi(-sd/2) = 1/2 at x = 1
sd50 = 2 * np.sqrt(2) * erfinv(1 / 2)


def read_run(path):
    with open(os.path.join(path, 'out')) as f:
        out = f.read()
    optimized = re.search(r'Optimized DTVMC: *(\S+)', out)
    if optimized:
        dtvmc = float(optimized.group(1))
    else:
        dtvmc = float(re.search(r'DTVMC \(VMC time step\) *: *(\S+)', out).group(1))
    acceptance = float(re.search(r'Acceptance ratio *\(%\) *= *(\S+)', out).group(1)) / 100
    nstep = int(re.search(r'VMC_NSTEP.*: *(\d+)', out).group(1))
    period = int(re.search(r'VMC_DECORR_PERIOD.*: *(\d+)', out).group(1))
    method = int(re.search(r'VMC_METHOD.*: *(\d+)', out).group(1))
    efficiency = re.search(r'Efficiency *\(au\^-2 s\^-1\) *= *(\S+)', out)
    return dtvmc, acceptance, nstep * period, method, float(efficiency.group(1)) if efficiency else np.nan


def gaussian_model(x):
    return erfc(sd50 * np.sqrt(x) / (2 * np.sqrt(2)))


def curve(x, c, a):
    return c / (c + np.exp(a * np.sqrt(x)) - 1)


fig, ax = plt.subplots(figsize=(7.5, 5))
for color, system in zip(COLORS, systems):
    root = os.path.join(here, system)
    name = system.replace('/', '-')
    points = []
    for entry in sorted(os.listdir(root)):
        if entry.startswith('a'):
            points.append(read_run(os.path.join(root, entry)))
    points.sort()
    dtvmc, acceptance, moves, method, efficiency = np.array(points).T
    error = np.sqrt(acceptance * (1 - acceptance) / moves)

    dtvmc50 = np.exp(np.interp(0.5, acceptance[::-1], np.log(dtvmc)[::-1]))
    optimized = read_run(os.path.join(root, 'opt'))[0]
    x = dtvmc / dtvmc50
    # fitted where the acceptance is measured to a per-cent, the window the form was made for
    window = (acceptance >= 0.05) & (acceptance <= 0.95)
    (c, a), _ = curve_fit(curve, x[window], acceptance[window], p0=(1.0, sd50))
    fitted = curve(x, c, a)
    rms = np.sqrt(np.mean((acceptance[window] - fitted[window]) ** 2))

    with open(os.path.join(here, f'{name}.dat'), 'w') as f:
        f.write(f'# system = {system}\n')
        f.write(f'# path = examples/acceptance_curve/{system}\n')
        f.write(f'# code = CASINO, runtype vmc, vmc_method {method[0]:.0f}\n')
        f.write(f'# moves per point = {moves[0]:.0f}\n')
        f.write(f'# dtvmc50 = {dtvmc50:.5e}, optimized DTVMC = {optimized:.5e}\n')
        f.write(f'# fit p = C / (C + exp(a*sqrt(dtvmc/dtvmc50)) - 1) over 0.05 < p < 0.95, C = {c:.5f}, a = {a:.5f}, rms = {rms:.5f}\n')
        f.write('#       dtvmc   dtvmc/dtvmc50  acceptance      error        fit   gaussian  efficiency\n')
        f.writelines(
            '{:12.5e} {:9.4f} {:11.5f} {:10.5f} {:10.5f} {:10.5f} {:11.4e}\n'.format(*row)
            for row in zip(dtvmc, x, acceptance, error, fitted, gaussian_model(x), efficiency)
        )

    ax.plot(
        100 * acceptance,
        efficiency,
        'o-',
        ms=7,
        lw=1.4,
        color=color,
        zorder=3,
        label=f'{name}, vmc_method {method[0]:.0f}',
    )
    best = np.nanargmax(efficiency)
    print(
        f'{name}: dtvmc50 = {dtvmc50:.5e}, optimized = {optimized:.5e}, C = {c:.5f}, a = {a:.5f}, rms = {rms:.5f}, '
        f'best efficiency {efficiency[best]:.4e} at dtvmc/dtvmc50 = {x[best]:.2f}, acceptance {acceptance[best]:.3f}'
    )

ax.axvline(50, color=MUTED, lw=0.8, zorder=1)
ax.set_xlabel('acceptance ratio, %', color=INK)
ax.set_ylabel('efficiency, au$^{-2}$ s$^{-1}$', color=INK)
ax.set_title('VMC efficiency against the acceptance ratio, He', color=INK)
ax.set_xlim(0, 100)
ax.set_ylim(0, None)
ax.legend(frameon=False, labelcolor=INK)
ax.grid(True, color=MUTED, alpha=0.25, lw=0.6)
ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(here, 'He.png'), dpi=160)
