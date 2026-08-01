#!/usr/bin/env python3

"""Measure the CBCS acceptance curve for every calibration system in one format.

Runnable from anywhere: it puts the working tree it lives in ahead of the environment on the path
and insists on importing that casino rather than whatever is installed.

    examples/time_step/time_step.py [OUTDIR [STEPS [SYSTEM ...]]]

Each system is written as soon as it is done and skipped if its file already exists, so a run can
be interrupted and resumed. A million steps per point takes about half a day for the whole set; a
hundred thousand costs a tenth of that and still resolves the shape of the curve, but leaves the
header <T> of the pseudoatoms uncertain to 1.5%.

Conventions, none of them free to differ between systems, which is the whole point of collecting
them here rather than in a shell loop: Slater only (the curve agrees to better than 1% across
Slater, Jastrow and backflow), cusp_correction : F (only |Psi|**2 is needed, it is smooth for a
gaussian basis, and the cusp correction is broken on carbon), CBCS, one grid laid out equally in
acceptance around the measured 50% point, step sizes written in atomic units so the data stay
unnormalized by the law they test, and <T> of that same wave function in the header.
"""

import logging
import os
import sys
from timeit import default_timer

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)
os.chdir(root)

SYSTEMS = {
    'He-gto': 'examples/gwfn/He/HF/cc-pVQZ/CBCS/Slater',
    'Li+-gto': 'examples/gwfn/Li⁺/HF/cc-pVQZ/CBCS/Slater',
    'Be-gto': 'examples/gwfn/Be/HF/cc-pVQZ/CBCS/Slater',
    'Be2+-gto': 'examples/gwfn/Be²⁺/HF/cc-pVQZ/CBCS/Slater',
    'N-gto': 'examples/gwfn/N/HF/cc-pVQZ/CBCS/Slater',
    'Ne-gto': 'examples/gwfn/Ne/HF/cc-pVQZ/CBCS/Slater',
    'Ar-gto': 'examples/gwfn/Ar/HF/cc-pVQZ/CBCS/Slater',
    'Kr-gto': 'examples/gwfn/Kr/HF/cc-pVQZ/CBCS/Slater',
    'CH4-gto': 'examples/gwfn/CH4/HF/cc-pVQZ/CBCS/Slater',
    'NH3-gto': 'examples/gwfn/NH3/HF/cc-pVQZ/CBCS/Slater',
    'H2O-gto': 'examples/gwfn/H2O/HF/cc-pVQZ/CBCS/Slater',
    'HF-gto': 'examples/gwfn/HF/HF/cc-pVQZ/CBCS/Slater',
    'O3-gto': 'examples/gwfn/O3/HF/cc-pVQZ/CBCS/Slater',
    'B2H6-gto': 'examples/gwfn/B2H6/HF/cc-pVQZ/CBCS/Slater',
    'C2H2-gto': 'examples/gwfn/C2H2/HF/cc-pVQZ/CBCS/Slater',
    'C4H4-gto': 'examples/gwfn/C4H4/HF/cc-pVQZ/CBCS/Slater',
    'C6H6-gto': 'examples/gwfn/C6H6/HF/cc-pVQZ/CBCS/Slater',
    # the same systems in a Slater basis, so that each pairs with its gaussian twin and says whether
    # anything in the law is an artefact of how the orbitals are expanded
    'He-sto': 'examples/stowfn/He/HF/QZ4P/CBCS/Slater',
    'Be-sto': 'examples/stowfn/Be/HF/QZ4P/CBCS/Slater',
    'N-sto': 'examples/stowfn/N/HF/QZ4P/CBCS/Slater',
    'Ne-sto': 'examples/stowfn/Ne/HF/QZ4P/CBCS/Slater',
    'Ar-sto': 'examples/stowfn/Ar/HF/QZ4P/CBCS/Slater',
    'Kr-sto': 'examples/stowfn/Kr/HF/QZ4P/CBCS/Slater',
    'O3-sto': 'examples/stowfn/O3/HF/QZ4P/CBCS/Slater',
    'H-pp': 'examples/ppotential_HF/H/HF/aug-cc-pVQZ-CDF/CBCS/Slater',
    'B-pp': 'examples/ppotential_HF/B/HF/aug-cc-pVQZ-CDF/CBCS/Slater',
    'C-pp': 'examples/ppotential_HF/C/HF/aug-cc-pVQZ-CDF/CBCS/Slater',
    'N-pp': 'examples/ppotential_HF/N/HF/aug-cc-pVQZ-CDF/CBCS/Slater',
    'O-pp': 'examples/ppotential_HF/O/HF/aug-cc-pVQZ-CDF/CBCS/Slater',
    'F-pp': 'examples/ppotential_HF/F/HF/aug-cc-pVQZ-CDF/CBCS/Slater',
    'Ne-pp': 'examples/ppotential_HF/Ne/HF/aug-cc-pVQZ-CDF/CBCS/Slater',
}

outdir = sys.argv[1] if len(sys.argv) > 1 else 'examples/time_step/CBCS'
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000
wanted = sys.argv[3:] or list(SYSTEMS)
os.makedirs(outdir, exist_ok=True)

devnull = open(os.devnull, 'w')  # noqa: SIM115
logging.basicConfig(level=logging.INFO, stream=devnull, format='%(message)s')

from casino.readers import CasinoConfig

_read = CasinoConfig.read


def read(self, *args, **kwargs):
    _read(self, *args, **kwargs)
    # the acceptance curve needs only |Psi|**2, which is smooth for a gaussian basis, and the
    # cusp correction is broken on carbon
    self.input.cusp_correction = False
    self.input.vmc_method = 3


CasinoConfig.read = read

from casino.pycasino import Casino

sys.stdout = sys.__stdout__
import casino

assert os.path.abspath(casino.__file__).startswith(os.getcwd()), f'imported {casino.__file__}, not the working tree'


for name in wanted:
    path = SYSTEMS[name]
    out = os.path.join(outdir, f'{name}.dat')
    if os.path.exists(out):
        # resuming is what makes a day long run survivable, but only a file this script wrote may
        # be taken as done: the same names hold measurements from the older grid, and skipping
        # those would turn the whole run into a silent no-op
        with open(out) as f:
            written_here = any(line.startswith('# steps per point') for line in f)
        if not written_here:
            raise SystemExit(f'{out} predates this script, move it aside before rerunning')
        print(f'{name:10s} skipped, {out} exists', flush=True)
        continue
    handler = logging.FileHandler(out + '.tmp', mode='w')
    handler.setFormatter(logging.Formatter('%(message)s'))
    log = logging.getLogger('casino.pycasino')
    log.addHandler(handler)
    start = default_timer()
    try:
        Casino(path).vmc_step_graph(steps)
    finally:
        log.removeHandler(handler)
        handler.close()
    with open(out + '.tmp') as raw, open(out, 'w') as f:
        f.write(f'# system = {name}\n# path = {path}\n# steps per point = {steps}\n')
        for line in raw:
            line = line.rstrip()
            if not line.strip():
                continue
            try:
                float(line.split()[0])
            except ValueError:
                f.write('# ' + line.strip() + '\n')
            else:
                f.write(line + '\n')
    os.remove(out + '.tmp')
    print(f'{name:10s} done in {default_timer() - start:8.1f} s -> {out}', flush=True)
