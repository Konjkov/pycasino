#!/usr/bin/env python3

"""Read an acceptance.py campaign and print where the efficiency optimum sits.

    examples/step_profile/acceptance_report.py [OUTDIR]

At fixed wall time the error bar is sqrt(var(E_L) * ms_indep / T), and var(E_L) belongs to the
wave function rather than to the step, so the whole comparison is a comparison of ms_indep and
the gain in the error bar is its square root. The variance column is still worth a glance: where
it falls below the value a long production run reports, the walk is not reaching the tail of the
local energy at the nuclei and the error bar quoted there is optimistic rather than good.
"""

import os
import sys

import numpy as np

COLUMNS = 'dtvmc target acc diffusion corr_E corr_err corr_r2 corr_TD variance decorr ms_indep us_move us_energy move_frac'.split()  # noqa: SIM905

outdir = sys.argv[1] if len(sys.argv) > 1 else 'examples/step_profile/acceptance'
data = {}
for name in sorted(os.listdir(outdir)):
    if not name.endswith('.dat'):
        continue
    system, method = name[:-4].rsplit('.m', 1)
    with open(os.path.join(outdir, name)) as f:
        rows = np.array([[float(v) for v in line.split()] for line in f if not line.startswith('#')])
    data[system, int(method)] = dict(zip(COLUMNS, rows.T))


def target(row, value):
    return int(np.argmin(np.abs(row['target'] - value)))


def best(row):
    return int(np.argmin(row['ms_indep']))


# cheapest first, which is the order the campaign runs them in and roughly the order in electrons
systems = sorted({system for system, _ in data}, key=lambda s: min(data[s, m]['ms_indep'].min() for m in (1, 4) if (s, m) in data))
print(
    f'{"system":10s} {"m":>2s} {"target":>7s} {"acc":>6s} {"dtvmc":>9s} {"corr_E":>7s} {"var":>9s} {"ms_indep":>9s} {"vs 50%":>7s} {"err":>6s} {"diff_max":>9s}'
)
for system in systems:
    for method in (1, 4):
        row = data.get((system, method))
        if row is None:
            continue
        gain = row['ms_indep'][target(row, 0.5)] / row['ms_indep'][best(row)]
        print(
            f'{system:10s} {method:2d} {row["target"][best(row)]:7.2f} {row["acc"][best(row)]:6.3f} {row["dtvmc"][best(row)]:9.4f} '
            f'{row["corr_E"][best(row)]:7.2f} {row["variance"][best(row)]:9.3f} {row["ms_indep"][best(row)]:9.4f} '
            f'{gain:6.2f}x {-100 * (1 - 1 / np.sqrt(gain)):5.0f}% {row["acc"][int(np.argmax(row["diffusion"]))]:9.3f}'
        )  # fmt: skip

print('\nerror bar at fixed wall time, relative to vmc_method 1 at the 50% target')
for system in systems:
    if (system, 1) not in data or (system, 4) not in data:
        continue
    base = data[system, 1]['ms_indep'][target(data[system, 1], 0.5)]
    parts = []
    for method in (1, 4):
        row = data[system, method]
        for label, index in (('50%', target(row, 0.5)), ('opt', best(row))):
            parts.append(f'm{method} {label} {np.sqrt(row["ms_indep"][index] / base):5.2f}')
    print(f'{system:10s} ' + '   '.join(parts))
