#!/usr/bin/env python3

"""Run directories for the nodal descriptor over the Be c_2 scan - the benchmark stage 1 lacked.

Nine wave functions of one Hamiltonian in one basis, differing only in the fixed CSF coefficient
c_2 of 2s^2 -> 2p^2, each with its own Jastrow and Phi-backflow optimized by emin from scratch.
Their fixed-node energies are known to 20-50 uHa and span 2.4 mHa with a minimum at C = 0.164:

    C          0.00      0.01      0.02      0.05      0.10      0.12      0.15      0.20      0.25
    dE_FN     2.379     2.068     1.781     1.087     0.235     0.077    -0.022     0.065     0.537

so a descriptor that ranks nodes has to reproduce that parabola. The cross-system amplitude
confound that killed stage 1 cannot arise here - all nine share a Hamiltonian, a basis and, for a
fixed weight, literally the same Phi.

The wave functions themselves are linked from the DMC runs, the last emin cycle of each point. At
C = 0 there is no MDET block at all, that point is the bare HF node with four nodal domains.

    python examples/noda_surface/be_c2/make.py

Then, from the project root, cheapest first:

    mpiexec -n 4 python casino/nodal_descriptor.py -n 10000000 -d 10 examples/noda_surface/be_c2/*
    mpiexec -n 4 python casino/nodal_descriptor.py -n 100000000 -d 10 examples/noda_surface/be_c2/*

The first is a shape test, ~2.3% per point; the second is 0.72% and costs hours per point. Before
either, check that the readers take the combination gwfn + MDET + Phi-backflow + casl together,
which they have never been asked to do:

    python -m casino.pycasino --check examples/noda_surface/be_c2/0.15

The links are absolute because the source tree is on another filesystem; if that tree moves they
all break at once and this script rebuilds them.
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))
SCAN = '/mnt/sdb1/quantum_chemistry/!PROJECT/ORCA/MP2-CASSCF(2.4)/ano-pVDZ/Be/VMC_DMC_BF/emin'
POINTS = ('0.00', '0.01', '0.02', '0.05', '0.10', '0.12', '0.15', '0.20', '0.25')
RUN = 'tmax_1_1024_1'
LINKED = ('gwfn.data', 'correlation.data', 'parameters.casl')

INPUT = """#-------------------#
# CASINO input file #
#-------------------#

# Be atom, CAS(2,4) node at a fixed c_2, Jastrow and backflow from the emin of that point

# SYSTEM
neu               : 2              #*! Number of up electrons (Integer)
ned               : 2              #*! Number of down electrons (Integer)
periodic          : F              #*! Periodic boundary conditions (Boolean)
atom_basis_type   : gaussian       #*! Basis set type (text)

# RUN
runtype           : vmc            #*! Type of calculation (Text)
testrun           : F              #*! Test run flag (Boolean)

# VMC
vmc_method        : 3              #*! Configuration-by-configuration algorithm
opt_dtvmc         : 1              #*! VMC time-step optimization (Integer)
vmc_equil_nstep   : 5000           #*! Number of equilibration steps (Integer)
vmc_nstep         : 100000000      #*! Number of steps (Integer)
vmc_nblock        : 10             #*! Number of checkpoints (Integer)
vmc_decorr_period : 10             #*! VMC decorrelation period (0 - auto)

# GENERAL PARAMETERS
use_gjastrow      : T              #*! Use a Jastrow function (Boolean)
backflow          : T              #*! Use backflow corrections (Boolean)
"""

for point in POINTS:
    source = os.path.join(SCAN, point, RUN)
    out = os.path.join(BASE, point)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'input'), 'w') as f:
        f.write(INPUT)
    for name in LINKED:
        link = os.path.join(out, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.realpath(os.path.join(source, name)), link)
    print(f'{out}  <- {os.path.realpath(os.path.join(source, "correlation.data"))}')
