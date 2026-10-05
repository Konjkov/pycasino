#!/usr/bin/env python3

"""Find the acceptance target that maximizes efficiency, with and without the step profile.

    examples/step_profile/acceptance.py [OUTDIR [STEPS [SYSTEM ...]]]

The 50% rule sets the step so that the average electron is accepted half the time, and the
diffusion-optimal target of Roberts, Gelman and Gilks is 0.234, a step 3.12 times larger. The
argument against taking it in EBES was that the two targets describe different electrons: at a
step where the valence is near 0.234 a core electron is rejected always, so the walk buys
displacement in the coordinates that carry least of Var(E_L). The profile removes exactly that
objection by putting every electron at the same acceptance, which makes the question worth
measuring again and only under vmc_method 4.

One file per system per method, holding the nineteen rows of Casino.vmc_corr_graph. The column
to read is 1 / (variance * ms_indep): ms_indep is the wall time of one independent sample with
the decorrelation period already minimized out of it, so the maximum over target is the answer
and corr_E alone is not, a longer step buying correlation time back with a worse variance.

Method 1 is run beside 4 as the control: it should reproduce the known result that 50% is close
to right there, and any shift of the optimum under 4 is then the profile's doing rather than the
system's.
"""

import logging
import os
import sys
from timeit import default_timer

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)
os.chdir(root)

SYSTEMS = {
    # one shell, no core to equalize against: the profile is nearly inactive and the optimum
    # measured here is the acceptance target on its own, free of any mixture
    'He-gto': 'examples/gwfn/He/HF/cc-pVQZ/CBCS/Slater',
    'Be-gto': 'examples/gwfn/Be/HF/cc-pVQZ/CBCS/Slater',
    'N-gto': 'examples/gwfn/N/HF/cc-pVQZ/CBCS/Slater',
    'Ne-gto': 'examples/gwfn/Ne/HF/cc-pVQZ/CBCS/Slater',
    # the largest core in the set that still costs under an hour a point
    'Ar-gto': 'examples/gwfn/Ar/HF/cc-pVQZ/CBCS/Slater',
    # mixed charges, where a single step cannot serve both species whatever the target: the step
    # the carbon core wants freezes the protons, the step the protons want is rejected on carbon
    'CH4-gto': 'examples/gwfn/CH4/HF/cc-pVQZ/CBCS/Slater',
    'C2H2-gto': 'examples/gwfn/C2H2/HF/cc-pVQZ/CBCS/Slater',
    'O3-gto': 'examples/gwfn/O3/HF/cc-pVQZ/CBCS/Slater',
}

outdir = sys.argv[1] if len(sys.argv) > 1 else 'examples/step_profile/acceptance'
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
wanted = sys.argv[3:] or list(SYSTEMS)
os.makedirs(outdir, exist_ok=True)

devnull = open(os.devnull, 'w')  # noqa: SIM115
logging.basicConfig(level=logging.INFO, stream=devnull, format='%(message)s')

from casino.readers import CasinoConfig

method = 1

_read = CasinoConfig.read


def read(self, *args, **kwargs):
    _read(self, *args, **kwargs)
    self.input.vmc_method = method


CasinoConfig.read = read

from casino.pycasino import Casino

sys.stdout = sys.__stdout__
import casino

assert os.path.abspath(casino.__file__).startswith(os.getcwd()), f'imported {casino.__file__}, not the working tree'


for name in wanted:
    path = SYSTEMS[name]
    for method in (1, 4):
        out = os.path.join(outdir, f'{name}.m{method}.dat')
        if os.path.exists(out):
            print(f'{name:10s} m{method} skipped, {out} exists', flush=True)
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
            f.write(f'# system = {name}\n# path = {path}\n# vmc_method = {method}\n# steps per point = {steps}\n')
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
        print(f'{name:10s} m{method} done in {default_timer() - start:8.1f} s -> {out}', flush=True)
