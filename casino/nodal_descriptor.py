#!/usr/bin/env python3

"""The nodal surface descriptor of any system, and a comparison of several of them.

    E_kin^nda[D] = int_dOmega |grad D| dS / int |D| dR

taken over the determinant part D of the trial function alone.

**At zeta = 0 it does not rank nodes. Measured over Be, N and Ne against their fixed-node DMC
energies, it gets N wrong by 8.7 standard errors.** With Phi = 1/J the estimator is exactly
P(sigma < eps)/eps^2 under the measure |D|dR - the fraction of the density lying within eps of the
node - so it answers "how diffuse is D" at least as loudly as "where is its node". A determinant
stripped of the Jastrow it was optimized with is a diffuse object: over the same three systems the
accompanying E_pot^nda moves by 0.28, 0.72 and 11.88 au between the HF and the backflow run, and
the descriptor follows that rather than the DMC ordering.

**With -z it can.** A weight that does not depend on the wave function being measured puts them on
a common measure, which is the one thing the comparison lacked. On the Be c_2 scan - nine nodes of
one Hamiltonian in one basis, fixed-node energies known to 20-50 uHa - the correlation with them
goes from -0.09 at zeta = 0 to +0.98 at zeta = 0.25, and the minimum lands next to the true one.
It ranks; it does not measure, spanning 0.4 au where the fixed-node energies span 2.4 mHa.

Two things this script does for you, both of them lessons paid for:

  * It runs every directory with the JASTROW SWITCHED OFF, whatever the input says. A Jastrow does
    not move the node, yet it changes the nda numbers by a factor of 14 (the gap E^nda - <H>) or
    by 21% (E_kin^nda itself), because with Psi = J*D the surface integral picks up <J> on the node
    against <J> in the bulk, and the node sits where the electrons are spread out. Weighting by
    Phi = 1/J removes it exactly, and Phi|Psi| = |D| means that weighting is nothing but running
    without the Jastrow. Backflow and multideterminant coefficients belong to D and stay.

  * It picks the tube thicknesses from the wave function instead of from a fixed grid. The scan is
    useful well below the median of sigma = 1/|grad ln Psi|, which is the typical log-gradient of
    the system, of order 1/Z: once epsilon reaches it the tube swallows the whole sample and the
    table explodes. The grid spans median/100 to median/8.

Read the table, do not trust a single row: the answer is the value over the flat region. The count
inside the tube falls as epsilon^2, so the small end is limited by statistics and the large end by
the O(epsilon) bias of the estimator. The summary line reports the widest tube and one 1.9 times
narrower - if those two agree within their error bars, the plateau is real.

All-electron systems only: a nonlocal pseudopotential has no place in the identity these averages
come from.

Usage, from anywhere:

    nodal_descriptor.py [-n <steps>] [-d <period>] [-b] [-z <ζ,ζ,...>] <run dir> [<run dir> ...]
    mpiexec nodal_descriptor.py -n 100000000 -d 10 <run dir> [<run dir> ...]

-z is the grid of exponents of the one-particle weight Φ = Π exp(-ζ r_iI), taken from one walk
because the weight is applied to the sample rather than sampled from. Eq. (18) is exact for every
ζ, so what a scan over it shows is variance and not bias, and what it costs is the effective
sample size the run reports next to every ζ. ζ = 0 is the constant weight and the default.

A run dir is an ordinary CASINO one: input, gwfn.data or stowfn.data, and correlation.data or
parameters.casl if the wave function has them. A jastrowless copy of the input is written to
./nodal_<name>/ next to pycasino.log.

-b drops the backflow as well, leaving the bare determinant. It is a diagnostic, not a measure to
rank with: backflow moves the node, so without it a different node is being measured and the
fixed-node energies of the runs no longer belong to it.

What it diagnosed on the Be c_2 scan, at 10^7 steps: with the backflow kept, the potential
component runs from -14.35 to -1.55 au over the last three points of the scan, whose fixed-node
energies differ by half a millihartree; with the backflow dropped it sits at -14.42 +- 0.01 over
the same three, 1625 times tighter. **A backflow optimized together with the Jastrow and evaluated
without it is what destroys the measure, not the determinant at a large c_2.** Since the backflow
cannot be dropped without changing the node, this cannot be repaired by taking pieces out of D -
it is the argument for a weight that is not built from the wave function being measured.

**Pass -n and -d whenever the runs being compared come from different kinds of calculation.** A
`*_dmc` directory carries vmc_nstep 1024 and vmc_decorr_period 1, because its VMC only had to
generate configurations for DMC, while a `Slater` directory carries 10^8 and 10; comparing the two
as they stand measures one of them on a thousand steps and gives the other error bars that assume
an independence it does not have. The estimator counts configurations in the tube and reports a
Poisson error, so a decorrelation period long enough to make them independent is what makes that
error honest.

10^8 steps gives around 0.7% on the widest tube for Be and costs half an hour on four processes;
the relative error is 72/sqrt(vmc_nstep) as long as the count in the flat region stays above 10^4.
"""

import argparse
import os

import numpy as np

from casino.pycasino import Casino, configure_logging, logger, mpi_comm

PILOT_STEPS = 10000
LINKED = ('gwfn.data', 'stowfn.data', 'correlation.data', 'parameters.casl')


def jastrowless(path, nstep, decorr, no_backflow):
    """the same run with the Jastrow switched off, the rest of it linked rather than copied. The
    directory is named after the leaf of the path and so is reused by two runs called the same -
    harmless, since it is consumed before the next one is made, as long as stale links go first"""
    out = os.path.join(os.getcwd(), 'nodal_' + os.path.basename(os.path.abspath(path)))
    override = ['use_jastrow       : F\n', 'use_gjastrow      : F\n']
    dropped = ['use_jastrow', 'use_gjastrow']
    if no_backflow:
        override.append('backflow          : F\n')
        dropped.append('backflow')
    if nstep is not None:
        override.append(f'vmc_nstep         : {nstep}\n')
        dropped.append('vmc_nstep')
    if decorr is not None:
        override.append(f'vmc_decorr_period : {decorr}\n')
        dropped.append('vmc_decorr_period')
    if mpi_comm.rank == 0:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(path, 'input')) as f:
            lines = [line for line in f if not line.startswith(tuple(dropped))]
        with open(os.path.join(out, 'input'), 'w') as f:
            f.writelines(lines + override)
        for name in LINKED:
            source = os.path.join(os.path.abspath(path), name)
            link = os.path.join(out, name)
            if os.path.lexists(link):
                os.remove(link)
            if os.path.exists(source):
                os.symlink(source, link)
    mpi_comm.barrier()
    return out


def descriptor(path, nstep, decorr, no_backflow, zeta):
    """E_kin^nda of the determinant part over a grid of tube thicknesses
    :param path: a CASINO run directory
    :param nstep: vmc_nstep to run at, the input's own if None
    :param decorr: vmc_decorr_period to run at, the input's own if None
    :param no_backflow: measure the bare determinant, dropping the backflow with the Jastrow
    :param zeta: exponents of the one-particle weight, in inverse bohr
    :return: epsilon, the configurations inside the tube, E_kin^nda and its standard error of every
        row of every table, on the root process and None elsewhere - array(zeta.size, 9, 4)
    """
    casino = Casino(jastrowless(path, nstep, decorr, no_backflow))
    casino.vmc.power = 1.0
    casino.equilibrate(casino.config.input.vmc_equil_nstep)
    if casino.config.input.opt_dtvmc == 1:
        casino.optimize_vmc_step(3000)
    position = casino.vmc.random_walk(PILOT_STEPS, casino.decorr_period)
    integrand = casino.vmc.observable(casino.wfn.nodal_surface_integrand, position)
    scale = mpi_comm.bcast(np.median(1 / np.sqrt(integrand[:, 1])))
    logger.info(
        f' Median sigma = 1/|grad ln psi| : {scale:.5e} bohr\n'
    )  # fmt: skip
    return casino.nodal_domain_accumulation(np.geomspace(scale / 100, scale / 8, 9), zeta)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('path', type=str, nargs='+', help='CASINO run directories to compare')
parser.add_argument('-n', '--nstep', type=int, help='vmc_nstep to run at, overriding every input')
parser.add_argument('-d', '--decorr', type=int, help='vmc_decorr_period to run at, overriding every input')
parser.add_argument('-b', '--no-backflow', action='store_true', help='drop the backflow too, leaving the bare determinant')
parser.add_argument('-z', '--zeta', type=str, default='0', help='comma separated exponents of the one-particle weight, 0 for a constant one')
args = parser.parse_args()

configure_logging()
zeta = np.array([float(z) for z in args.zeta.split(',')])
table = {path: descriptor(path, args.nstep, args.decorr, args.no_backflow, zeta) for path in args.path}
if mpi_comm.rank == 0:
    for i, z in enumerate(zeta):
        logger.info(
            f' =========================================================================\n'
            f' NODAL SURFACE DESCRIPTOR, zeta = {z:.4f}\n\n'
            f' {"run":<40} {"widest tube":>22} {"1.9x narrower":>22}\n'
        )
        for path, rows in table.items():
            logger.info(
                f' {path:<40} {rows[i, 8, 2]:12.6f} +/- {rows[i, 8, 3]:.6f} {rows[i, 6, 2]:12.6f} +/- {rows[i, 6, 3]:.6f}'
            )  # fmt: skip
        logger.info('')
