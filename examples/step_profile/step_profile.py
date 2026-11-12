#!/usr/bin/env python3

"""Measure what the position dependent step of vmc.step_profile buys, one file per system.

Runnable from anywhere: it puts the working tree it lives in ahead of the environment on the path
and insists on importing that casino rather than whatever is installed.

    examples/step_profile/step_profile.py [OUTDIR [STEPS [SYSTEM ...]]]

Each file holds the two rows of Casino.vmc_profile_graph, vmc_method 1 and 4, each at its own 50%
acceptance. The question the campaign answers is whether corr_E falls, since the acceptance is
equalized between the electrons by construction and equalizing it is not the point.

EBES only, and the reason is measured rather than assumed. In EBES a proposal is accepted for one
electron, so the profile equalizes acceptance and hands the core more accepted moves, and corr_E
falls by 2 to 4.5 on every all-electron system. In CBCS the whole configuration is accepted at
once, so all a profile can share out is how much of Var(ln(psi'**2/psi**2)) each electron takes;
measured over the same systems it moves the displacement of every electron by 3 to 5 and leaves
corr_E where it was, which is why vmc_method 4 is EBES and there is no CBCS counterpart. The
earlier three-profile data, including that CBCS set, are kept in EBES/ and CBCS/ as written.

Unlike time_step.py the cusp correction is left alone: the columns here are the local energy and
its correlation time, and an uncorrected gaussian cusp makes both of them a property of the
divergence rather than of the sampling. That is also why the set is gaussian and pseudopotential
systems only, PyCasino having no cusp correction for slater orbitals.

Ordered cheap first, and Kr last, the cost of a sweep being what it is in production plus the
local energy of every stored configuration. Start with a hundred thousand steps and read the
corr_err column before paying for more.

Each system is written as soon as it is done and skipped if its file already exists, so a run can
be interrupted and resumed.
"""

import logging
import os
import sys
from timeit import default_timer

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)
os.chdir(root)

SYSTEMS = {
    # one electron: EBES is CBCS and no profile can change anything, so every column must repeat
    'H-pp': 'examples/ppotential_HF/H/HF/aug-cc-pVQZ-CDF/CBCS/Slater',
    # one occupied shell: the electrons are already equivalent, so the profile has nothing to fix
    # and the measured EBES correction is 1.13
    'He-gto': 'examples/gwfn/He/HF/cc-pVQZ/CBCS/Slater',
    # a core and a valence shell at the same nucleus, the smallest system where the mixture exists
    'Be-gto': 'examples/gwfn/Be/HF/cc-pVQZ/CBCS/Slater',
    'N-gto': 'examples/gwfn/N/HF/cc-pVQZ/CBCS/Slater',
    'Ne-gto': 'examples/gwfn/Ne/HF/cc-pVQZ/CBCS/Slater',
    # the valence shell on its own: correction 1.21 to 1.25, so this is the control for what is
    # left once the core is removed by hand rather than by the step
    'C-pp': 'examples/ppotential_HF/C/HF/aug-cc-pVQZ-CDF/CBCS/Slater',
    'Ne-pp': 'examples/ppotential_HF/Ne/HF/aug-cc-pVQZ-CDF/CBCS/Slater',
    # several nuclei of unequal charge, where the profile has to place hydrogen and carbon at once
    'CH4-gto': 'examples/gwfn/CH4/HF/cc-pVQZ/CBCS/Slater',
    'C2H2-gto': 'examples/gwfn/C2H2/HF/cc-pVQZ/CBCS/Slater',
    # three occupied shells, correction 2.19, the largest effect to remove and the largest bill
    'Ar-gto': 'examples/gwfn/Ar/HF/cc-pVQZ/CBCS/Slater',
    'Kr-gto': 'examples/gwfn/Kr/HF/cc-pVQZ/CBCS/Slater',
}

outdir = sys.argv[1] if len(sys.argv) > 1 else 'examples/step_profile/EBES'
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
wanted = sys.argv[3:] or list(SYSTEMS)
os.makedirs(outdir, exist_ok=True)

devnull = open(os.devnull, 'w')  # noqa: SIM115
logging.basicConfig(level=logging.INFO, stream=devnull, format='%(message)s')

from casino.readers import CasinoConfig

_read = CasinoConfig.read


def read(self, *args, **kwargs):
    _read(self, *args, **kwargs)
    # the graph walks both methods itself, so what the input says is only where it starts
    self.input.vmc_method = 1


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
        Casino(path).vmc_profile_graph(steps)
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
                f.write('#' + line + '\n')
            else:
                f.write(line + '\n')
    os.remove(out + '.tmp')
    print(f'{name:10s} done in {default_timer() - start:8.1f} s -> {out}', flush=True)
