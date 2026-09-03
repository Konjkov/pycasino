#!/usr/bin/env python3

"""Measure the correlation time, the diffusion constant and the efficiency vs step size.

The companion of time_step.py: that one measures where the step size is, this one measures what
it costs. Same systems, same conventions, same resumable one file per system layout.

    examples/time_step/corr_time.py [OUTDIR [STEPS [SYSTEM ...]]]

The sampling method is taken from the name of the output directory, CBCS or EBES, rather than from
the input files, which all say CBCS. A step is one configuration in the first and one sweep over the
electrons in the second, so the two campaigns compare row for row; what does not compare is
acc_ratio, which counts one proposal and hence one electron in EBES.

Cost. Unlike the acceptance curve, every stored configuration needs the local energy and the drift
kinetic energy, so a point is some six times a point of time_step.py on the same nineteen point
grid. A hundred thousand steps per point is the default: the correlation time is wanted to 20%
rather than to a percent, and n grows only linearly with it, so the extra decade the acceptance
curve needs buys nothing here. Memory is the other limit — the walk is kept in full and carried
forward once to recover the rejected steps, two arrays of steps by electrons by three.

Conventions are those of time_step.py except for the cusp correction, which is on here and off
there for the reason given below: Slater only, one grid laid out equally in acceptance around the
measured 50% point, and the step sizes in atomic units. The system list is a copy of the one there
rather than an import, because that script runs its campaign on import.
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

outdir = sys.argv[1] if len(sys.argv) > 1 else 'examples/time_step/corr/CBCS'
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
wanted = sys.argv[3:] or list(SYSTEMS)
mode = os.path.basename(os.path.normpath(outdir))
if mode not in ('CBCS', 'EBES'):
    raise SystemExit(f'name the output directory CBCS or EBES, not {mode!r}')
os.makedirs(outdir, exist_ok=True)

devnull = open(os.devnull, 'w')  # noqa: SIM115
logging.basicConfig(level=logging.INFO, stream=devnull, format='%(message)s')

from casino.readers import CasinoConfig

_read = CasinoConfig.read


def read(self, *args, **kwargs):
    _read(self, *args, **kwargs)
    # unlike the acceptance curve, which needs only |Psi|**2, this one needs the local energy, and
    # an uncorrected gaussian orbital leaves E_L going as -Z/r at the nucleus. The variance of that
    # is finite but its fourth moment is not, so the sample variance never settles: measured
    # cuspless, it swings by a factor of 400 between rows of one system. A slater basis carries the
    # cusp from the converter and a pseudoatom has no singularity to cancel, and Casino skips both
    self.input.cusp_correction = True
    self.input.vmc_method = 3 if mode == 'CBCS' else 1


CasinoConfig.read = read

from casino.pycasino import Casino

sys.stdout = sys.__stdout__
import casino

assert os.path.abspath(casino.__file__).startswith(os.getcwd()), f'imported {casino.__file__}, not the working tree'


for name in wanted:
    path = SYSTEMS[name]
    out = os.path.join(outdir, f'{name}.dat')
    if os.path.exists(out):
        print(f'{name:10s} skipped, {out} exists', flush=True)
        continue
    handler = logging.FileHandler(out + '.tmp', mode='w')
    handler.setFormatter(logging.Formatter('%(message)s'))
    log = logging.getLogger('casino.pycasino')
    log.addHandler(handler)
    start = default_timer()
    try:
        Casino(path).vmc_corr_graph(steps)
    finally:
        log.removeHandler(handler)
        handler.close()
    with open(out + '.tmp') as raw, open(out, 'w') as f:
        f.write(f'# system = {name}\n# path = {path}\n# method = {mode}\n# steps per point = {steps}\n')
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
