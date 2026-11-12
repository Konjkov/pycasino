#!/usr/bin/env python3

"""How much of vmc_method 4 is lost to the crude shape of vmc.step_profile.

    examples/step_profile/tabulated.py [OUTDIR [STEPS [SYSTEM ...]]]

The profile is max(1, min(Z**2, Z / r)) inverted, three branches with no constant to fit, and the
question this answers is what the best conceivable function of position would give instead. That
function is <|grad_i ln psi|**2 | r>, measured here on a radial grid before the campaign starts:
it is exact in the mean by construction, so whatever it fails to buy cannot be bought by any
sharper formula either, only by a step that depends on more than the position of the electron.

Each system is run twice at the same acceptance targets, once with the formula and once with the
table, and the tabulated run pays for np.interp on every move, so ms_indep compares them as they
stand. The table is baked into a patched copy of the package as a literal rather than passed in,
which keeps numba's source hash a function of the numbers in it and the cache honest.

Read the output with acceptance_report.py: the two runs are written as separate systems, SYSTEM
and SYSTEM_table, and what matters is the ratio of their ms_indep at their own optima.
"""

import os
import shutil
import subprocess
import sys
from timeit import default_timer

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEMS = {
    'Ne-gto': 'examples/gwfn/Ne/HF/cc-pVQZ/CBCS/Slater',
    'Ar-gto': 'examples/gwfn/Ar/HF/cc-pVQZ/CBCS/Slater',
    'CH4-gto': 'examples/gwfn/CH4/HF/cc-pVQZ/CBCS/Slater',
}

# the profile can be built out of either mean of the two, and they are not the same thing: the
# step goes as 1 / g, so the harmonic mean is what averages the step itself, while the arithmetic
# one averages what the acceptance responds to. Both are written to the table file
COLUMN = 'mean'
# the grid the table is measured on. It has to reach inside the K shell, where the cusp pins the
# gradient to Z, and out past the last electron, where nothing is sampled and the last measured
# value is carried forward
GRID = np.geomspace(0.01, 8.0, 33)
# configurations behind one table. The bins in the valence hold tens of thousands of electrons at
# this count and the innermost ones a few hundred, which is enough where the cusp makes the
# gradient nearly constant anyway
TABLE_STEPS = 20000

IMPORT = 'from casino.wfn import Wfn_t\n'

FORMULA = """    def impl(self, r_e, e):
        gradient = 0.0
        for atom in range(self.wfn.atom_positions.shape[0]):
            charge = self.wfn.atom_charges[atom]
            r = np.sqrt(((r_e[e] - self.wfn.atom_positions[atom]) ** 2).sum())
            screened = 0.815 * charge ** (1 / 3) / r
            gradient = max(gradient, min(charge * charge, screened * screened))
        return 1 / (gradient + 1.577)
"""

TABULATED = """    def impl(self, r_e, e):
        r = np.inf
        for atom in range(self.wfn.atom_positions.shape[0]):
            r = min(r, np.sqrt(((r_e[e] - self.wfn.atom_positions[atom]) ** 2).sum()))
        return 1 / np.exp(np.interp(np.log(r), LOG_GRID, LOG_TABLE))
"""


def measure(path, out):
    """Bin <|grad_i ln psi|**2> by the distance to the nearest nucleus."""
    import logging

    logging.basicConfig(level=logging.INFO, stream=open(os.devnull, 'w'), format='%(message)s')  # noqa: SIM115

    import numba as nb
    from casino.pycasino import Casino

    sys.stdout = sys.__stdout__

    @nb.njit(nogil=True, parallel=False, cache=True)
    def gradient_squared(wfn, r_e):
        v = wfn.drift_velocity(r_e).reshape(-1, 3)
        res = np.empty(v.shape[0])
        for i in range(v.shape[0]):
            res[i] = v[i] @ v[i]
        return res

    casino = Casino(path)
    casino.vmc.method = 4
    positions = casino.config.wfn.atom_positions
    casino.vmc.random_walk(TABLE_STEPS // 4, 3)
    r = casino.vmc.random_walk(TABLE_STEPS, 3)
    r = r[~np.isnan(r[:, 0, 0])]
    d = np.linalg.norm(r[:, :, None, :] - positions[None, None, :, :], axis=3).min(axis=2)
    g = np.empty(d.shape)
    for i, r_e in enumerate(r):
        g[i] = gradient_squared(casino.wfn, r_e)

    edges = np.concatenate(([0.0], np.sqrt(GRID[1:] * GRID[:-1]), [np.inf]))
    table = np.empty((GRID.size, 3))
    for k in range(GRID.size):
        m = (d >= edges[k]) & (d < edges[k + 1])
        if m.sum():
            table[k] = m.sum(), g[m].mean(), 1 / (1 / g[m]).mean()
        else:
            # a bin outside the density is never asked for, and carrying the last value forward
            # keeps the interpolation monotone there instead of dropping it to zero
            table[k] = 0, table[k - 1, 1] if k else 1.0, table[k - 1, 2] if k else 1.0
    with open(out, 'w') as f:
        f.write(f'# path = {path}\n# configurations = {r.shape[0]}\n# electrons = {r.shape[1]}\n')
        f.write(f'#{"r":>11} {"count":>8} {"mean":>12} {"harmonic":>12}\n')
        for k in range(GRID.size):
            f.write(f'{GRID[k]:12.6f} {int(table[k, 0]):8d} {table[k, 1]:12.6f} {table[k, 2]:12.6f}\n')
    return table[:, 1 if COLUMN == 'mean' else 2]


def patch(pkg, table):
    """Copy the package next to the results with the table in place of the formula."""
    shutil.rmtree(pkg, ignore_errors=True)
    shutil.copytree(os.path.join(root, 'casino'), os.path.join(pkg, 'casino'), ignore=shutil.ignore_patterns('__pycache__'))
    name = os.path.join(pkg, 'casino', 'vmc.py')
    with open(name) as f:
        source = f.read()
    assert FORMULA in source, 'vmc.step_profile is not the function this script knows how to replace'
    assert IMPORT in source, 'nowhere to put the table'
    literal = (
        f'\nGRID = np.array({np.array2string(GRID, precision=6, separator=", ", max_line_width=110, threshold=GRID.size)})\n'
        f'TABLE = np.array({np.array2string(table, precision=6, separator=", ", max_line_width=110, threshold=table.size)})\n'
        'LOG_GRID = np.log(GRID)\nLOG_TABLE = np.log(TABLE)\n'
    )
    source = source.replace(FORMULA, TABULATED)
    source = source.replace(IMPORT, IMPORT + literal, 1)
    with open(name, 'w') as f:
        f.write(source)


def sweep(pkg, path, out, steps):
    """vmc_corr_graph in a process of its own, since the package is chosen by sys.path."""
    script = (
        'import logging, os, sys\n'
        f'sys.path.insert(0, {pkg!r})\n'
        f'os.chdir({root!r})\n'
        "logging.basicConfig(level=logging.INFO, stream=open(os.devnull, 'w'), format='%(message)s')\n"
        'from casino.readers import CasinoConfig\n'
        '_read = CasinoConfig.read\n'
        'def read(self, *args, **kwargs):\n'
        '    _read(self, *args, **kwargs)\n'
        '    self.input.vmc_method = 4\n'
        'CasinoConfig.read = read\n'
        'from casino.pycasino import Casino\n'
        'import casino\n'
        f'assert os.path.abspath(casino.__file__).startswith({pkg!r}), casino.__file__\n'
        f'handler = logging.FileHandler({out + ".tmp"!r}, mode="w")\n'
        "handler.setFormatter(logging.Formatter('%(message)s'))\n"
        "logging.getLogger('casino.pycasino').addHandler(handler)\n"
        f'Casino({path!r}).vmc_corr_graph({steps})\n'
    )
    subprocess.run([sys.executable, '-c', script], check=True, cwd=root)
    with open(out + '.tmp') as raw, open(out, 'w') as f:
        f.write(f'# path = {path}\n# package = {pkg}\n# steps per point = {steps}\n')
        for line in raw:
            line = line.rstrip()
            if not line.strip():
                continue
            try:
                float(line.split()[0])
            except ValueError:
                f.write('#' + line + '\n')
            else:
                f.write(line + '\n')
    os.remove(out + '.tmp')


if __name__ == '__main__':
    sys.path.insert(0, root)
    os.chdir(root)

    outdir = sys.argv[1] if len(sys.argv) > 1 else 'examples/step_profile/tabulated'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000
    wanted = sys.argv[3:] or list(SYSTEMS)
    os.makedirs(outdir, exist_ok=True)

    for name in wanted:
        path = SYSTEMS[name]
        start = default_timer()
        table_file = os.path.join(outdir, f'{name}.table')
        if os.path.exists(table_file):
            table = np.loadtxt(table_file)[:, 2 if COLUMN == 'mean' else 3]
            print(f'{name:10s} table  skipped, {table_file} exists', flush=True)
        else:
            table = measure(path, table_file)
            print(f'{name:10s} table  done in {default_timer() - start:8.1f} s -> {table_file}', flush=True)

        # the formula runs from the working tree, the table from a patched copy, and the copy
        # compiles from scratch: that cost is not in ms_indep, which times the walk alone
        pkg = os.path.abspath(os.path.join(outdir, f'{name}.pkg'))
        for label, package in ((f'{name}_table', pkg), (name, root)):
            out = os.path.join(outdir, f'{label}.m4.dat')
            if os.path.exists(out):
                print(f'{label:16s} skipped, {out} exists', flush=True)
                continue
            if package == pkg:
                patch(pkg, table)
            start = default_timer()
            sweep(package, path, out, steps)
            print(f'{label:16s} done in {default_timer() - start:8.1f} s -> {out}', flush=True)
