# CASINO input keywords

Full dump of `casinohelp <keyword>` for all 304 keywords of CASINO v3.1.0 (August 2026), plus the default
value read from the `esdf_*()` call site in `src/` (file given in brackets; a symbolic
default such as `no_default` or a variable name means it is computed or mandatory).

Regenerate with `.claude/skills/casino-run/references/gen_keywords.py`.
Types: `Logical` = T/F, `Physical` = number + optional unit, `Block` = `%block name ... %endblock name`.

### neu

*Number of up electrons* — Integer, Basic, default `-1` (monte_carlo.f90)

For real systems containing atoms, NEU is the total number of spin-up electrons referenced
by the many-body wave function (for periodic systems, this is the number of spin-up
electrons in the simulation cell, rather than the underlying primitive cell). The number of
spin-down electrons is given by the keyword NED. Note that in the presence of addition or
subtraction excitations, NEU refers to the state of the system AFTER the required number of
electrons have been added or removed. For model electron(-hole) phases such as the HEG, set
NEU to zero and use the FREE_PARTICLES block to define the number of spin-up electrons.

### ned

*Number of down electrons* — Integer, Basic, default `-1` (monte_carlo.f90)

For real systems containing atoms, NED is the total number of spin-down electrons referenced
by the many-body wave function (for periodic systems, this is the number of spin-down
electrons in the simulation cell, rather than the underlying primitive cell). The number of
spin-up electrons is given by the keyword NEU. Note that in the presence of addition or
subtraction excitations, NED refers to the state of the system AFTER the required number of
electrons have been added or removed. For model electron(-hole) phases such as the HEG, set
NED to zero and use the FREE_PARTICLES block to define the number of spin-down electrons.

### nhu

*Number of spin-up fermions other than electrons in real s* — Integer, Intermediate, default `0` (monte_carlo.f90)

NHU is the number of spin-up fermions other than electrons in real systems. For example, if
you are interested in positronic molecules then NHU should be set to 1, and the up-spin
positron ("spin" 3) should be defined appropriately in the PARTICLES block. Likewise for
muonic systems. At present only the Gaussian routines can be used to return orbitals for
species other than electrons. (There exists a modified version of the GAUSSIAN code that can
be used to generate orbitals for positronic molecules.) If you want to use a different basis
or study muons or something then you will need to make the appropriate changes first. For
model systems such as electron-hole gases, etc., please use the FREE_PARTICLES block to
define the number of spin-up holes.

### nhd

*Number of spin-down fermions other than electrons in real* — Integer, Intermediate, default `0` (monte_carlo.f90)

Like NHU, but for spin-down particles.

### runtype

*Type of calculation* — String, Basic, default `'default'` (monte_carlo.f90)

This keyword specifies the type of QMC run to be carried out. It can take the following
values: 'vmc' (perform a single VMC simulation); 'dmc_equil' (perform DMC equilibration);
'dmc_stats' (perform DMC statistics accumulation); 'dmc_dmc' (perform DMC equilibration,
then statistics accumulation); 'vmc_dmc' (perform VMC, then DMC equilibration, then DMC
statistics accumulation); 'vmc_dmc_equil' (perform VMC, then DMC equilibration); 'opt'
(perform a single optimization run); 'vmc_opt' (perform OPT_CYCLES cycles of VMC and
optimization, alternately); 'opt_vmc' (perform OPT_CYCLES cycles of optimization and VMC,
alternately); 'gen_mdet_casl' (generate an mdet.casl file for post-processing by
det_compress); 'gen_mpc' (generate an mpc.data file enabling the use of the MPC interaction;
generation of mpc.data requires COMPLEX_WF=T); 'gen_blip' (generate a bwfn.data (or
bwfn.data.bin) file from plane-wave orbitals in pwfn.data); 'gen_gpcc' (generate a gpcc.casl
file with a Jastrow factor term reproducing the e-n cusp correction from the GPCC facility);
'gen_gpcc_single' and 'gen_gpcc_simple' (like gen_gpcc, but make assumptions to simplify the
term); 'rmc' (perform a single reptation simulation: EITHER equilibration OR statistics
accumulation -- EXPERIMENTAL); 'rmc_rmc' (perform reptation equilibration, then reptation
statistics accumulation -- EXPERIMENTAL); 'plot' (perform plot specified by either block
PLOT or block QMC_PLOT).; NOTE: in earlier versions of the code, we used 'runtype : dmc'
with the now-redundant keyword 'iaccumulate : T or F' to indicate whether stats accumulation
was activated or not. This usage is now deprecated and - unless iaccumulate is specifically
defined in input then 'runtype : dmc' is just a synonym for 'runtype : dmc_dmc'.

### atom_basis_type

*Basis set type* — String, Basic, default `'default'` (monte_carlo.f90)

ATOM_BASIS_TYPE selects the basis set in which the atom-centred orbitals are expanded (thus
choosing which file to read the orbitals from), or more generally, the 'type of orbital' to
be used. Possible values are: * 'none' : [default] no atoms are present, and therefore no
atomic orbitals are read in; * 'plane-wave' : use a plane-wave basis set; the orbitals are
read in from pwfn.data; * 'gaussian' : use a Gaussian basis set; the orbitals are read in
from gwfn.data; * 'slater-type': use a Slater-type orbital basis set; the orbitals are read
in from stowfn.data; * 'numerical' : use orbitals tabulated on a grid (atomic systems only);
the orbitals are read in from awfn.data; * 'dimer ' : use orbitals tabulated on a grid
(molecular dimers only); the orbitals are read in from dwfn.data; * 'blip' : use a blip
basis set; the orbitals are read in from bwfn.data; Some special wave function types are
also available: * 'nonint_he' : use exact orbitals for a non-interacting Helium atom. * 'h2'
: wave function for the H2 molecule where each orbital is the sum over hydrogen nuclei of a
parameter-less exponential centred at each nucleus. * 'h3plus' : wave function for the H3+
molecular ion where each orbital is the sum over hydrogen nuclei of a parameter-less
exponential centred at each nucleus. For free-particle and external-potential-related
orbitals, set ATOM_BASIS_TYPE to 'none' and use the input block FREE_PARTICLES. For GEN_BLIP
calculations, ATOM_BASIS_TYPE should be set to 'plane-wave', since the geometry and orbitals
will be read from the pwfn.data file.

### dmc_method

*DMC method* — Integer, Intermediate, default `1` (monte_carlo.f90)

DMC_METHOD selects which version of DMC to use: 1 = electron-by-electron algorithm; 2 =
configuration-by-configuration algorithm. Method 1 is the default.

### vmc_equil_nstep

*Number of VMC equilibration steps* — Integer, Basic, default `5000` (monte_carlo.f90)

Total number of equilibration steps in VMC; should normally be at least a few thousand.
Equilibration is only performed if NEWRUN=T, thus the value of VMC_EQUIL_NSTEP is ignored on
restarts. This is a single-processor quantity, that is, all MPI processes run
VMC_EQUIL_NSTEP equilibration steps. The value of VMC_DECORR_PERIOD is ignored during
equilibration.

### vmc_nstep

*Number of (main) VMC steps* — Integer, Basic, default `-1` (monte_carlo.f90)

Total number of VMC steps summed over all MPI processes; this correspondsto the total number
of particle configurations for which the energy andother quantities to be averaged are
calculated (required only if STOP_METHOD = 'nstep'). Note that because adjacent moves are
likely to be serially correlated, there is also an inner decorrelation loop of length
VMC_DECORR_PERIOD, so the total number of configuration moves attempted in a VMC run
following equilibration is VMC_NSTEP*VMC_DECORR_PERIOD. On parallel machines, each MPI
process will do the same number of steps and for each step the energy is averaged over the
processes and written to the vmc.hist file (which will ultimately contain VMC_NSTEP/NPROCS
lines - though VMC_AVE_PERIOD adjacent lines may be averaged over to reduce the file size).
This means that if VMC_NSTEP is not divisible by the number of MPI processes then it will
internally be rounded up to the nearest multiple of the number of MPI processes (example: on
a 12-core machine, given VMC_NSTEP=20 in input, CASINO will round up VMC_NSTEP to 24; each
core will then do two steps and a total of two records will be written to vmc.hist, each of
which is an average of 12 energies). On a single-core machine with VMC_NSTEP=20, CASINO will
move the single config 20 times, and 20 records will be written to vmc.hist. Note the
VMC_NBLOCK or BLOCK_TIME keywords may be used to vary the frequency with which checkpointing
is done i.e. how often we write the data to disk; this does not affect the total number of
VMC steps and expectation values such as average energy should be independent ofit. Instead
of executing a fixed number of VMC steps, CASINO may be requested to do as many steps as are
required to attain a target error bar or to make the error bar as small as possible - see
keyword STOP_METHOD.

### vmc_nblock

*Number of blocks in VMC* — Integer, Intermediate, default `1` (monte_carlo.f90)

Setting VMC_NBLOCK is one of two ways of specifying the number of blocks into which the
total VMC run is divided post-equilibration (the other way being to specify BLOCK_TIME). The
number of blocks determines how often the output, history and configuration/checkpoint files
are written to disk. More specifically, at the end of each block: (1) the process- and
block-averaged energies and a short 'report' are written to out. (2) the process-averaged
energies for each step in the current block are appended to vmc.hist (and other quantities
to expval.data). (3) the current VMC state plus any accumulated configs are written to the
config.out file (this latter only if the CHECKPOINT input keyword is increased to 2 from its
default value of 1 - otherwise config.out is only written after the end of the final block).
Note that the total energy and error bar should be effectively independent of VMC_NBLOCK
(provided it is ensured that the random number sequence is independent of the number of
blocks - which it has not been at various periods in CASINO's history, though it should
benow). Note that the value of VMC_NBLOCK is ignored if VMC_NTWIST>0, or if BLOCK_TIME>0.0.
The default value of VMC_NBLOCK is 1.

### periodic

*Periodic boundary conditions* — Boolean, Basic, default `.true.` (monte_carlo.f90)

PERIODIC should be T if the system is periodic in 1, 2 or 3 dimensions. It should be F if
system is finite without periodic boundary conditions.

### testrun

*Test run flag* — Boolean, Basic, default `.false.` (monte_carlo.f90)

If TESTRUN is T, CASINO will just read in all input files, print out the input information,
then stop without performing any QMC calculation.

### newrun

*New run or continue old* — Boolean, Basic, default `.true.` (monte_carlo.f90)

NEWRUN determines whether this is a new run or a continuation of a previous one. * VMC:
NEWRUN=T: VMC_EQUIL_NSTEP Metropolis steps are performed on a set of randomly generated
configs before accumulation of statistics begins. NEWRUN=F: a set of old (and presumably
equilibrated) electron positions are read from a config.in file, and no Metropolis
equilibration steps are performed before accumulation of statistics. * DMC: NEWRUN=T: a set
of VMC configs is read from a config.in file and the initial best estimate of the energy
EBEST is calculated as the mean energy of these configs. In the special case of
DMC_REWEIGHT_CONFIGS=T, config.in may also contain DMC configs which are reweighted to a new
wave function and EBEST shifted by the current energy diff between the wave functions.
NEWRUN=F: a set of DMC configs is read from a config.in file and EBEST is not recomputed but
taken to have the value written on the end of the config.in file (presumably by a previous
DMC run - either equilibration or accumulation).

### density

*Accumulate density* — Boolean, Basic, default `.false.` (monte_carlo.f90)

IF DENSITY is set to T the charge density is accumulated and written to the expval.data
file.

### dtvmc

*VMC time step* — Double Precision, Basic, default `0.01` (monte_carlo.f90)

DTVMC is the time step for VMC runs in atomic units. In systems with more than one 'family'
of particles, the time step used is the value of DTVMC divided by the particle mass. If
OPT_DTVMC is set to 1, the time step is optimized to give acceptance ratios of about 50%
(this is done for each 'family' individually). Both DTVMC and OPT_DTVMC are ignored if input
block DTVMCS is present.

### dtdmc

*DMC time step* — Double Precision, Basic, default `0.01` (monte_carlo.f90)

DTDMC is the time step for DMC runs in atomic units.

### tpdmc

*DMC number of factors in Pi-wts* — Integer, Expert, default `0` (monte_carlo.f90)

TPDMC (T_p) is the number of time steps for which the effects of changes in the
(theoretically constant) reference energy should be undone in order to estimate the DMC
energy at a given point. It is assumed that the best estimates of the DMC energy separated
by an amount greater than this are not correlated by fluctuations in the reference energy.
Thus T_p should exceed the timescale of fluctuations in the reference energy. Umrigar
suggests using T_p=10/tau where tau is the time step. If you set it to 9999 in the input,
then the code will automatically use this value; if you set it to 0 then the reweighting
scheme for population control biasing will not be used. The latter is the default. Note that
this procedure is not really all that useful. If you suffer from population control biasing,
then the scheme will help to correct it, but you have to run for longer because the Pi-
weights fluctuate, increasing the variance of the energy estimate. In fact, it is not clear
that you gain anything, because the extra run time turns out to be about the same as you
would need if you simply used a sufficiently large population. So use more configs to avoid
population control biasing, basically.

### vmc_decorr_period

*VMC decorrelation period* — Integer, Intermediate, default `corper_default_vmc` (monte_carlo.f90)

Length of inner decorrelation loop in VMC. The code will do VMC_DECORR_PERIOD configuration
moves between successive evaluations of the local energy (and other quantities to be
averaged) in order to ensure the configurations used are not significantly correlated.
Setting VMC_DECORR_PERIOD to a value greater than 1 should reduce serial correlation, but
the length of the run will be increased. It is normally stated that typical values might be
3 or 4 for a pure VMC calculation, and >= 10 during a config-generation run for optimization
or DMC (though this depends on the system, and clearly setting the decorrelation period to a
higher value in DMC config generation is less important than in wave function optimization,
since the correlations will disappear as the DMC calculation evolves). The defaults are 3
for VMC calculations and 15 for config generation runs. A slight complication is that if
VMC_NSTEP is greater than VMC_NCONFIG_WRITE (as it might be if you need more moves than the
number of desired configs to calculate a VMC energy with a small enough error bar), then
CASINO is able to exploit the extra moves to space the config writes further apart, and it
is no longer necessary for the inner decorrelation loop to be so long in config generation
runs. In such a case, VMC_DECORR_PERIOD should be taken to represent the *minimum* number of
steps separating config writes; internally the length of the decorrelation loop will be
reduced as far as practical without going below the VMC default of 3. Note that VMC_NSTEP
refers to the number of moves at which energies are evaluated or configurations are
(sometimes) written out; this is not affected by the value of VMC_DECORR_PERIOD. The value
of VMC_DECORR_PERIOD is ignored during equilibration. Note that CASINO can automatically
determine VMC_DECORR_PERIOD to maximize run efficiency. This is done by adding an extra set
of moves after equilibration to compute an estimation of the correlation time of the local
energies. The feature is enabled by setting VMC_DECORR_PERIOD=0.

### vmc_nconfig_write

*Number of configs to write in VMC* — Integer, Basic, default `0` (monte_carlo.f90)

Total number of configurations to be written out in VMC for later use (wave-function
optimization or DMC). This number must be <= VMC_NSTEP (though you may want to set VMC_NSTEP
to be significantly greater than VMC_NCONFIG_WRITE to get an acceptable error bar on the
energy; this is useful for e.g. judging the success of an optimization after each stage).
Since each MPI process always does the same number of steps, then VMC_NCONFIG_WRITE (and
VMC_NSTEP) will be rounded up to the nearest multiple of the number of MPI processes (e.g.
VMC_NCONFIG_WRITE=20 will be rounded up internally to 24 on 12 MPI processes, and 24 configs
will be written to config.out - 2 from each process). Note that the config.out file will
still be written even if VMC_NCONFIG_WRITE is zero, since this file is used to store the
current state of the system at the end of every VMC block (equivalent to writing one config,
though of course multiple MPI processes write multiple configs to save the state). Writing
of config.out may be suppressed completely with an appropriate value for the CHECKPOINT
keyword, and the data will be held in memory between different stages of the calculation.

### con_loc

*Configs directory* — String, Basic, default `'.'` (monte_carlo.f90)

Directory in which config files are kept to be written/read. Default: './'.

### dmc_target_weight

*Total target weight in DMC* — Double Precision, Basic, default `0.` (monte_carlo.f90)

Total target weight in DMC, summed over MPI processes. This is synonymous with the "target
population" of configs, except that DMC_TARGET_WEIGHT is allowed to be non-integer.
Typically DMC_TARGET_WEIGHT will be the same as VMC_NCONFIG_WRITE when RUNTYPE=vmc_dmc,
though it does not have to be. It may seem bizarre to allow non-integer total target
weights, but a possible use for this is in increasing the parallel efficiency when you have
a very small population per process. Suppose your target weight is 1 config per process and
you are running on 100000 MPI processes. Half the time your total population will be a bit
higher than 100000, and when this happens nearly all of your MPI processes will spend half
their time twiddling their thumbs waiting for the small number of MPI processes that have
two configurations to finish the iteration. So around 25% of the computer time is wasted. If
instead you set your target weight to 0.98 configs per process then it is very unlikely that
any MPI processes will have two configurations. Instead you have on average 2% of your MPI
processes sitting idle, which is sad, but still more efficient than having large numbers of
MPI processes wait for a small number of over-burdened MPI processes.

### growth_estimator

*Calculate DMC growth estimator* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If this flag is set to T then the growth estimator of the DMC energy is evaluated in
addition to the usual mixed estimator. A statistically significant difference between the
mixed estimator and the growth estimator for the energy normally implies the presence of
time-step bias. Other than that, the growth estimator is not generally useful because it has
a significantly greater statistical error than the mixed estimator.

### gautol

*Gaussian tolerance* — Double Precision, Intermediate, default `7.` (monte_carlo.f90)

GAUTOL controls the accuracy of evaluating orbitals expanded in Gaussian basis sets. Roughly
speaking, any contribution from a Gaussian-type function to an orbital at a point in space
is neglected if exp(-a * r^2) is less than 10^-GAUTOL. Typical values in solids are around
6-7. This parameter can have a large effect on the cost of a calculation in periodic systems
- you can wind it up much higher in molecules.

### dbarrc

*DBAR recalculation period* — Integer, Intermediate, default `100000` (monte_carlo.f90)

DBARRC is the number of moves between full recalculation of the cofactor (='DBAR') matrices.
Basically every time an electron move is accepted in equilibration/VMC/DMC, the update_dbar
routine is called which updates these matrices using the efficient Eq. (26) of Fahy et al.,
PRB 42, 3503 (1990). As a numerical precaution that the (unstable!) update procedure is
working, every DBARRC accepted moves the DBAR matrices and determinant are recomputed from
scratch from the orbitals in the Slater matrix. If the new DBAR differs by too much from the
old updated DBAR, then the program ought to be stopped (but in fact it is not, since it
happens at least once per simulation and is very irritating). In principle one can boost the
value of DBARRC up to a fairly large number before this happens, and this is a good idea
since reevaluation of the matrix costs quite a lot. The default value is 100,000. Tests show
that for systems with 1024 particles per spin channel it may be safe to do up to 1,000,000
updates with accuracy better than single precision. NOTE: the DBARRC keyword had a different
meaning very early in the life of CASINO where it was appropriate to set it to a value of
e.g. 10. Re-using an old input file with such a setting now can cause the code to slow down
by 1-2 orders of magnitude without the user necessarily understanding why (real world
examples have been observed of people doing this). It is therefore now forbidden to set
DBARRC to a value lower than the default; if the user has a genuine reason for wishing to do
this he/she may search for the error trap in the source code and in the runqmc script and
comment it out.

### interaction

*Interaction type* — String, Intermediate, default `'default'` (monte_carlo.f90)

INTERACTION determines the type of interaction between the simulated particles. INTERACTION
can take the following values: 'none' : non-interacting particles; 'coulomb' : Coulomb
interaction; 'ewald' : periodic Coulomb interaction computed using Ewald summation; 'mpc' :
periodic Coulomb interaction computed using the model periodic Coulomb (MPC) method;
'ewald_mpc' : compute and report both Ewald and MPC results, but use Ewald in DMC
propagation; 'mpc_ewald' : compute and report both Ewald and MPC results, but use MPC in DMC
propagation; 'manual' : compute a user-defined interaction (see the MANUAL_INTERACTION block
input keyword); 'ewaldpp' 'ewaldpp_mpc' 'mpc_ewaldpp': as their above counterparts, but
using an electron-electron pseudopotential for the Ewald interaction, whose parameters (see
Eq. (4) of https://doi.org/10.1103/PhysRevB.92.075106) must be specified in the
MANUAL_INTERACTION block. 'ewald_kel' : periodic Keldysh interaction computed using Ewald
summation, whose parameters are given in a MANUAL_INTERACTION block. The values 'coulomb'
and 'ewald' can be used interchangeably, although 'coulomb' should strictly refer to
aperiodic systems and 'ewald' to periodic systems. The MPC interaction is generally
significantly faster than the Ewald interaction and should give smaller finite size effects.
Note that currently the MPC interaction is not implemented for 1D systems.

### manual_interaction

*Manual particle interactions* — Block, Intermediate

This block defines the interaction form and parameters when INTERACTION = 'manual'. The
first line specifies the interaction type, and subsequent lines define the parameters, to be
given as 'name : value' (real and int) or 'name' (boolean). The possible 'manual'
interactions and their corresponding parameters (real-valued [a.u.] and mandatory unless
otherwise stated; m and q are taken from PARTICLES block) are: * 'square_well' | V = height
for r<width, 0 otherwise | params: height (<0 for well), width * 'poschl_teller' | V = 2 v_0
mu^2 / [m cosh^2(mu*r)] | params: mu, v_0 (<0 for well) * 'hard_sphere' | V = Infinity for
r<=D, lambda/r^3 otherwise | params: D or R (=D/2), lambda (opt, default=0), | op_spins (opt
bool, restrict to opp-spin) * 'polynomial' | V = sum_{k=0}^{order-1} c_k r^k for r<cutoff,
else 0 | params: order (integer), cutoff, c_0..c_{order-1} (opt, default=0) * 'logarithmic'
| V = qi*qj * [log(2*rstar/r) - Euler] | params: rstar (length scale) * 'keldysh' | V =
qi*qj * F(r,rstar) | params: rstar (length scale) | note: F -> 1/r for r>>rstar and F ->
log(2*rstar/r)-Euler for r<<rstar * 'keldysh_bilayer' | V = qi*qj * FB(r, rstar1, rstar2) |
params : rstar1, rstar2, | bilayer_grid_size (opt, default=10000), | bilayer_grid_mult (opt,
default=10.). | FB is Layer-dependent, see derivation in manual * 'dipole' | V = d^2/r^3 |
params: d^2 * 'pseudodipole' | (like polynomial, with extra param d^2) * 'tilted_dipole' |
(like dipole, with extra params theta,phi,c_6,c_12) * 'tilted_pseudodipole' | (like
pseudodipole, with extra param theta) * 'softened_dipole' | (like dipole, with extra param
r0^3) * 'clifford' | (Coulomb interaction on a Clifford torus).

### npcell

*# primcells/axis* — Block, Basic

NPCELL is a vector of length three giving the number of primitive cells in each dimension
that make up the simulation cell. NB, for the polymer case, npcell(2) and npcell(3) must be
1; for the 2D slab case npcell(3) must be 1. If you wish to use a more general 3x3x3
supercell matrix to define the simulation cell, then you can define this with the
SCELL_MATRIX keyword.

### scell_matrix

*# Supercell matrix array* — Block, Expert

SCELL_MATRIX is a 3x3 integer matrix giving the supercell lattice vectors in terms of the
primitive-cell lattice vectors. This is a generalization of NPCELL.

### input_example

*Print example input* — Defined, Basic

If INPUT_EXAMPLE is T, then an example of a QMC input file will be written out with all
currently known keywords and their default values. A modified version of this can be used as
an input file in future runs.

### ibran

*DMC branching* — Boolean, Expert, default `.true.` (monte_carlo.f90)

IBRAN enables weighting and branching in DMC. Normally IBRAN = T for a DMC run; a value of F
is allowed for DMC checking purposes.

### limdmc

*Green function mods* — Integer, Expert, default `4` (monte_carlo.f90)

LIMDMC is used to set the type of modifications to the Green function (see manual). Possible
values are: 0 = no modifications applied; 1 = Depasquale limits to drift velocity and
energy; 2 = Umrigar mods (need ALIMIT); 3 = Rothstein-Vrbik mods (need ALIMIT). 4 = Umrigar
mods to drift velocity (need ALIMIT); Zen-Sorella-Alfe` mods to energy; 5 = New Zen-Alfe
scheme; 6 = Electron-by-electron UNR scheme. We strongly recommend you use LIMDMC=4.

### alimit

*DMC limit parameter* — Double Precision, Expert, default `0.5` (monte_carlo.f90)

ALIMIT is a parameter required when LIMDMC is 2, 3, 4, 5. A value of 0.25 was suggested by
Umrigar et al. for all-electron calculations, but a value of 0.5 may be more appropriate for
pseudopotential calculations. The answers are essentially insensitive to the precise value.
ALIMIT is not required if NUCLEUS_GF_MODS is set to true.

### alphalimit

*DMC limit parameter for branching factor in ZSGMA* — Double Precision, Expert, default `0.2` (monte_carlo.f90)

ALPHALIMIT is a parameter required when LIMDMC is 2, 3, 4, 5. A value of 0.2 was suggested
by Zen et al. PRB 93, 241118(R) (2016) RAPID but a different value may be more appropriate
for systems with different variance.

### nucleus_gf_mods

*Green fn mods for bare nuclei* — Boolean, Expert, default `.true.` (monte_carlo.f90)

NUCLEUS_GF_MODS is the switch for enabling the use of the modifications to the DMC Green's
function in the presence of bare nuclei, suggested in J. Chem Phys. 99, 2865 (1993).

### dmc_dteff_method

*DMC effective time step method* — Integer, Expert, default `1` (monte_carlo.f90)

DMC_DTEFF_METHOD is used to select the method used to evaluate the DMC effective time step
(see manual). Possible values are: 0 : effective time step is same as actual time step; 1 :
<diff^2>_acc is av of proposed diff^2, weighted by acceptance prob; 2 : <diff^2>_acc is av
of diff^2 over all accepted moves. We recommend you use DMC_DTEFF_METHOD=1.

### dmc_eref_method

*DMC reference-energy method* — Integer, Expert, default `1` (monte_carlo.f90)

DMC_EREF_METHOD selects the method used to evaluate the DMC reference energy E_T (see
manual). Possible values are: 1 : the algorithm in J. Chem. Phys. 99, 2865 (1993) (default);
2 : like 1, but with the mixed estimate of the energy replaced by the growth estimator; 3 :
like 1, but with the mixed estimate of the energy replaced by the mean limited local energy.
We recommend you use the default value DMC_EREF_METHOD=1.

### mpc_cutoff

*G vector cutoff for MPC* — Physical, Expert, default `30.,'hartree'` (monte_carlo.f90)

MPC_CUTOFF is the energy cutoff for G vectors used in (a) the FFT of the MPC interaction,
and (b) the FFT of the one-particle density required when generating the mpc.data file. We
use the convention that we include the set of G vectors such that (1/2)|G|^2<mpc_cutoff. The
program will suggest a value for MPC_CUTOFF if the existing value is unsuitable, or if the
user inputs a value of zero. This is a keyword of type 'Physical' hence you need to supply
units, such as 'ev', 'ry', 'hartree', 'kcal/mol' etc. The default is 30.0 hartree.

### non_local_grid

*Non-local integration rule* — Integer, Intermediate, default `-1` (monte_carlo.f90)

NON_LOCAL_GRID selects the grid for non-local integration, ranging from coarse (low
NON_LOCAL_GRID value) to fine (high NON_LOCAL_GRID value) to finer grids. The value is
assumed to be the same for all atoms if it is controlled through this keyword; you can
provide an override value of NON_LOCAL_GRID for particular atoms at the top of the
corresponding pseudopotential file, where it is called NLRULE1. The following table gives
the grid details: +---------------------------------------------------------+ |
NON_LOCAL_GRID Exactly integrates l=... No. points |
+---------------------------------------------------------+ | 1 0 1 | | 2 2 4 | | 3 3 6 | |
4 5 12 | | 5 5 18 | | 6 7 26 | | 7 11 50 |
+---------------------------------------------------------+ Notice that NON_LOCAL_GRID=5
offers no theoretical advantage over NON_LOCAL_GRID=4, and is significantly more expensive
(+50% points). We recommend that NON_LOCAL_GRID=5 not be used. The default value is
NON_LOCAL_GRID=4 (this is also adopted if NON_LOCAL_GRID is given a negative value).

### opt_info

*Optimizer information level* — Integer, Expert, default `2` (monte_carlo.f90)

Controls amount of information displayed and/or written out during optimization. In variance
minimization: 1 = display no information; 2 = display variance and energies at each
iteration; 3 = also print parameters, derivatives and intermediate evaluations; 4 =
calculate and print weights as well; 5 = write out configs and their energies etc as they
are read in (lots of data). In energy minimization: 1 = very little information, no extra
output files; 2 = basic information, no extra output files; 3 = full information, write
matrix algebra log file; 4 = also write full B~ and BH~ matrix files; 5 = also write SVD
components files (if SVD used).

### vm_reweight

*Varmin reweighting* — Boolean, Expert, default `.false.` (monte_carlo.f90)

In variance and MAD minimization, if VM_REWEIGHT is F (default) then all weights are set to
unity. When using weights in correlated sampling, the procedure may become numerically
unstable, particularly for large system sizes, typically because a few configurations (often
only one) acquire a very large weight, resulting in spuriously low variances for parameter
sets which then give extremely poor results in VMC. Strategies to address this include
increasing the number of configurations (see VMC_NCONFIG_WRITE), setting weight limits (see
VM_W_MAX), and setting the weights to 1 with VM_REWEIGHT=F, which we recommend and use by
default. The effect of this approximation is in general negligible when multiple
optimization cycles are performed. On the other hand, if the initial trial wave function is
poor, then some of the configuration generation/variance minimization cycles can be bypassed
by using the weights. There is also some evidence that the optimization of 'difficult'
parameters works better when VM_REWEIGHT is T.

### opt_fixnl

*Fix nonlocal energies in optimization* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Fix non-local contribution to local energy in optimization. In VARMIN, this gives a large
speedup for systems where pseudopotentials are used, and yields good results in most cases.
In EMIN the speedup is the same, but tends to destabilize the optimization. Defaults are T
for VARMIN, and F for EMIN. For both methods, when this is F, the non-local integration
grids are fixed for each electron in each configuration to ensure consistent, smooth
variance/energy surfaces.

### opt_maxeval

*Varmin max evaluations* — Integer, Intermediate, default `200` (monte_carlo.f90)

Maximum number of evaluations during optimization.

### vm_forgiving

*Varmin no whinge* — Boolean, Expert, default `.true.` (monte_carlo.f90)

If VM_FORGIVING is set to T [default], CASINO won't consider it an error if the config
energies from VMC do not agree with the initial energies in variance minimization.

### opt_complex

*Complex local energies in varmin* — Boolean, Expert, default `.true.` (monte_carlo.f90)

This flag determines whether the imaginary part of the local energy is taken into account in
VARMIN, MADMIN and VARMIN_LINJAS calculations when a complex trial wave function is used.
OPT_COMPLEX is T by default.

### opt_jastrow

*Optimize Jastrow factor* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

During optimization, allow the parameters in the Jastrow factor to be optimized.

### opt_det_coeff

*Optimize determinant coeffs* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

During optimization, allow the determinant coefficients to be optimized.

### opt_geminal

*Optimize geminal coeffs* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

During optimization, allow the optimization of the geminal coefficient matrix.

### postfit_vmc

*Perform post-fit VMC* — Boolean, Basic, default `.true.` (monte_carlo.f90)

If POSTFIT_VMC is set to T then an extra VMC calculation will be performed with the final
optimized wave function when RUNTYPE=vmc_opt or opt_vmc, to enable one to see the effect of
the final optimization on the energy etc. This is done by default. Unless POSTFIT_KEEP_CFG
is set to T, this final VMC run will not generate any configurations.

### postfit_keep_cfg

*Keep postfit VMC configurations* — Boolean, Basic, default `.false.` (monte_carlo.f90)

If POSTFIT_KEEP_CFG is set to T then the configurations generated in the post-fit VMC
calculation will be written to config.out (the default is F.

### lcutofftol

*Local PP cutoff tol* — Double Precision, Expert, default `0.00001` (monte_carlo.f90)

LCUTOFFTOL is used to define the cutoff radius for the local part of the pseudopotential. It
is the maximum deviation of the local potential from -z/r at the local cutoff radius. The
default of 1.e-5 is normally adequate.

### nlcutofftol

*Nonlocal PP cutoff tol* — Double Precision, Expert, default `0.00001` (monte_carlo.f90)

NLCUTOFFTOL is used to define the cutoff radius for the non-local parts of the
pseudopotential. It is defined as the maximum deviation of the non_local potentials from the
local_potential at the non-local cutoff radius. The default of 1.e-5 is normally adequate.

### orbbuf

*DMC orbital buffering* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

Setting ORBBUF=T turns on DMC orbital buffering. This is an efficiency device in which
buffered copies of orbitals/gradients/Laplacians are kept for later reuse. This has a high
memory cost. Orbital buffering should always be used unless you start running out of memory,
hence the ability to turn it off.

### blip_periodicity

*Periodicity with blip basis* — Integer, Intermediate, default `-1` (monte_carlo.f90)

Orbitals expanded in a blip basis can be periodic in zero, one, two or three dimensions.
BLIP_PERIODICITY specifies the number of dimensions in which the orbitals are periodic
(-1,0,1,2,3). Note that if BLIP_PERIODICITY is 1 then the system is assumed to be periodic
in the direction of lattice vector 1, while if BLIP_PERIODICITY is 2 then the system is
periodic in the directions of lattice vectors 1 and 2. If BLIP_PERIODICITY is -1 (the
default) then the periodicity is deduced from the value of the PERIODIC keyword (F-->0,
T-->3). In all cases, the simulation cell is the parallelepiped defined by the lattice
vectors placed at the origin so in non-3D cases make sure you put your atoms somewhere near
the middle of it. Note that k points may only be used in periodic directions.

### expot

*Use external potential* — Boolean, Basic, default `.false.` (monte_carlo.f90)

If EXPOT is set to T, then an external potential is read from the file 'expot.data' and
included as a summed contribution to the total energy.

### magnetic_field

*Use external magnetic field* — Boolean, Basic, default `.false.` (monte_carlo.f90)

If MAGNETIC_FIELD is set to T, then an external magnetic field is read from the file
'expot.data'. The magnetic field contributes to the kinetic energy.

### opt_orbitals

*Optimize orbital parameters* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

During optimization, allow the parameters in the orbitals to be optimized.

### structure_factor

*Accumulate structure factor* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If STRUCTURE_FACTOR is set to true then the structure factor will be accumulated in the
expval.data file (periodic systems only).

### cerefdmc

*DMC EREF update const* — Double Precision, Expert, default `1.` (monte_carlo.f90)

Constant used in updating the reference energy EREF during initial DMC diffusion to ground
state using the latest population control algorithm. A value of 1.0 is usually appropriate.

### makemovie

*Make a movie* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Plot the electron positions every movieplot moves.

### movieplot

*Frame length for movie* — Integer, Intermediate, default `1` (monte_carlo.f90)

Plot the electron positions every movieplot moves.

### movieproc

*Movie processor* — Integer, Intermediate, default `-1` (monte_carlo.f90)

Plot the electron positions on MPI process MOVIEPROC.

### moviecells

*Plot n.n. supercells in movie* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If false makemovie will plot the unit cell, if true, the n.n. cells in the xy plane will
also be written.

### edist_by_ion

*Initial electron distribution* — Block, Intermediate

The optional EDIST_BY_ION block allows fine control of the initial distribution of the
electrons before equilibration starts. The standard algorithm shares out the electrons
amongst the various ions weighted by the pseudo-charge/atomic number of the ion. Each
electron is placed randomly on the surface of a sphere surrounding its parent ion. There are
certain situations, for example a simple crystal with a very large lattice constant, where
the standard algorithm in the POINTS routine may give a bad initial distribution which
cannot be undone by equilibrating for a reasonable amount of time. This keyword allows a
user-defined set of electron/ion associations to be supplied. The syntax is to supply N_ion
lines within the block which look like e.g. 1 4 4, where the three numbers are: the ion
sequence number; the number of up-spin electrons associated with this ion ; the number of
down-spin electrons associated with this ion. Alternatively one may use the EDIST_BY_IONTYPE
keyword block where you replace the ion sequence number with the ion type sequence number
and the information is supplied only for each particular type of ion. This generally saves
typing.

### edist_by_iontype

*Initial electron distribution* — Block, Intermediate

The optional EDIST_BY_IONTYPE block allows fine control of the initial distribution of the
electrons before equilibration starts. The standard algorithm shares out the electrons
amongst the various ions weighted by the pseudo-charge/atomic number of the ion. Each
electron is placed randomly on the surface of a sphere surrounding its parent ion. There are
certain situations, for example a simple crystal with a very large lattice constant, where
the standard algorithm in the POINTS routine may give a bad initial distribution which
cannot be undone by equilibrating for a reasonable amount of time. This keyword allows a
user-defined set of electron/ion associations to be supplied. The syntax is to supply N_ion
lines within the block which look like e.g. 1 4 4, where the three numbers are: the ion type
sequence number; the number of up-spin electrons associated with each atom of this type ;
the number of down-spin electrons associated with each atom of this type. Alternatively one
may use the EDIST_BY_ION keyword block where you replace the sequence number of each type of
atom with the sequence number of each ion. This will obviously be useful in magnetic
systems, and in putting zigzag stripes of oxygen holes in those manganites and high-Tc
superconductors which CASINO is so good at.

### ewald_control

*Ewald accuracy control* — Double Precision, Expert, default `0.` (monte_carlo.f90)

EWALD_CONTROL is the percentage increase (from the default) of the cutoff radius for the
reciprocal space sum in the Ewald interaction - used for calculating electrostatic
interactions between particles in periodic systems. Its default value is zero. Increasing
this will cause more vectors to be included in the sum, the effect of which is to increase
the range of the Ewald gamma parameter over which the energy is constant (the default gamma
should lie somewhere in the middle of this range). This need only be done in exceptional
circumstances and the default should be fine for the general user.

### orb_norm

*Orbital normalization* — Double Precision, Intermediate, default `1.` (monte_carlo.f90)

Allows user to change normalization of orbitals by multiplying all of them by this constant.
Of course this should have no effect on the energy but it can be useful if the Slater
determinant starts going singular, as it might for example for some very dilute/low density
systems.

### printgscreening

*Print Gaussian screening info* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Before doing a periodic Gaussian calculation, CASINO prepares lists of potentially
significant (primitive) cells and sites in each such cell which could contain Gaussians
having a non-zero value in a reference primitive cell centred on the origin. Zero is defined
as 10^-GAUTOL. Turning on the PRINTGSCREENING flag prints out the important information
about this screening - it is turned off by default.

### qmc_plot

*Plot orbitals etc. in line/plane/volume* — Block, Intermediate

This block allows you to plot the value of certain quantities along a segment A-B /
parallelogram with sides A-B, A-C / parallelepiped with edges A-B, A-C, A-D. The data will
be plotted in a format suitable for xmgr/grace in the file 'lineplot.dat' or in a format
suitable for gnuplot in '2Dplot.dat' or '3Dplot.dat'. In order to produce the plot, RUNTYPE
must be set to 'plot'. The block has the following format: LINE 1: what_to_plot (orb,
orb_gradx, orb_grady, orb_gradz, orb_lap, wfn, nodes, energy, eipot, expot); LINE 2:
dimensionality ndim (1,2,3); LINE 3: no of points along the ndim directions; LINES 4-: xyz
coords of point A; of point B; of point C (if reqd.); of point D (if reqd.). Three
additional lines have to be added for orbital plots: LINE A1: number of orbitals (norb);
LINE A2: norb integers identifying the orbitals to be plotted; LINE A3: norb integers
identifying the spin/species for each of the orbitals. For wave-function and local-energy
plots, the coordinate defined by LINES 4- refer to electron 1 by default. The coordinates of
all remaining electrons are taken from a configuration obtained by VMC equilibration
(without Jastrow factor). Alternatively the positions of several electrons may be fixed by
the following lines: LINE B1: number of fixed electrons (>= 0); LINE B2 and onwards (if the
number of fixed electrons is nonzero): spin, number and (x, y, z) coordinates for each of
the fixed electrons. The default of moving the first electron can be overridden by adding
the lines: LINE B3: number of electrons to move (=1 or 2); LINE B4: spin, number of first
electron to move; LINE B5: spin, number and offset of 2nd electron relative to the 1st. For
electron-ion potential, external potential and node plots, no lines have to be added.

### plot

*Plot quantities in line/plane/volume* — Block, Expert

This block allows you to plot the value of certain quantities along a segment A-B /
parallelogram with sides A-B, A-C / parallelepiped with edges A-B, A-C, A-D. The data will
be plotted in a format suitable for post-processing by a CASINO utility (yet to be written),
to a file named '1Dplot.dat', '2Dplot.dat' or '3Dplot.dat'. This facility will replace
QMC_PLOT eventually. In order to produce the plot, RUNTYPE must be set to 'plot'. The block
has the following format: <what-to-plot> electron <ie> spin <ispin> <dimensionality> grid
<ngrid> A <A-coords> B <B-coords> C <C-coords> D <D-coords> fix electron <ie_fix1> spin
<ispin_fix1> @ <fix1-coords> where: - <what-to-plot> indicates what to plot, e.g. orb_grad,
wfn, energy, etc. The list of available plot subjects is dynamic, so you should set this to
'help' and run CASINO, which will display the list for the current system; - <ie> and
<ispin> are the particle and particle-type indices of the particle that is moved to generate
the plot. - <dimensionality> is the dimensionality of the plot. - <ngrid> is a set of
<dimensionality> integers defining the density of the plot grid - <A-coords>, <B-coords>,
<C-coords> and <D-coords> are the coordinates of A, B, C and D (only specify those required
according to <dimensionality>) - <ie_fix1> and <ispin_fix1> are the particle and particle-
type indices of a particle that ought to be fixed at <fix1-coords>; any number of particles
can be fixed with additional 'fix' lines.

### expval_cutoff

*G vector cutoff for expval* — Physical, Expert, default `30.,'hartree'` (monte_carlo.f90)

EXPVAL_CUTOFF is the energy cutoff for G vectors used in the evaluation of expectation
values accumulated in reciprocal space (e.g. the density, spin density, pair-correlation
function etc.). We use the convention that we include the set of G vectors such that
(1/2)|G|^2<mpc_cutoff (or |G|^2<mpc_cutoff before September 2013). The value of
EXPVAL_CUTOFF is ignored if an expval.data file is already present, in which case the G
vector set(s) given therein are used instead. If you set it to zero, then the program will
suggest a value. This is a keyword of type 'Physical' hence you need to supply units, such
as 'ev', 'ry', 'hartree', 'kcal/mol' etc. The default is 30.0 hartree.

### lwdmc

*Enable weighted DMC* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Enable weighted DMC, where each configuration carries a weight that is simply multiplied by
the branching factor after each move; only if the weight of a configuration goes outside
certain bounds (above WDMCMAX or below WDMCMIN) is it allowed to branch or be combined with
another configuration. This should reduce excessive population fluctuations, which is
generally held to be a good thing. Note that setting LWDMC=T means that your population will
generally fluctuate around a value other than DMC_TARGET_WEIGHT (after an initial
transient); the chances of being killed if your weight is below 1 or duplicated if your
weight is above 1 depend on the values of WDMCMIN and WDMCMAX, and in general this is not
symmetrical.

### dmc_equil_fixpop

*Fix pop and total weight during initial DMC equilibration* — Double Precision, Expert, default `0.` (monte_carlo.f90)

If the VMC and DMC energies are very different, the population increases during the initial
phase of equilibration before the reference energy can counteract. This parameter (between
0.0 and 1.0) specifies an initial fraction of the equilibration phase during which the
population and total weight are fixed to the target weight. Setting this parameter to e.g.
0.5 will prevent such explosions and should have a negligible effect on the equilibration
time.

### lwdmc_fixpop

*Fix population in lwdmc* — Boolean, Expert, default `.false.` (monte_carlo.f90)

This flag activates the LWDMC variant with fixed population. By interpreting WDMCMIN and
WDMCMAX relative towards the current population the population and the total weight are
decoupled. The population is nearly fixed while the total weight fluctuates as usual. While
this generally reduces the statistical efficiency of the DMC algorithm, it is a simple way
to eliminate population explosions or extinction in cases of small population and large
population fluctuation. WARNING: this is *not* a solution for walkers trapped in singular
points of the wave functions, nor is it a solution for populations that get trapped in high-
energy states. Be careful about this option when you do not know the reason for the
population problems in the first place.

### dmc_norm_conserve

*Enable norm-conserving DMC* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Use the norm-conserving DMC algorithm. This eliminates population fluctuations. Experimental
algorithm: use with caution.

### dmc_poprenorm

*Enable DMC pop-renormalization* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Control the DMC configuration population by randomly deleting or copying configurations
after branching with the reference energy set equal to the best estimate of the ground-state
energy. This can be used to maintain a constant population of configs per MPI process,
provided the value of DMC_TARGET_WEIGHT is an integer multiple of the number of processes.
Note that non-integer values of DMC_TARGET_WEIGHT are not allowed when using DMC_POPRENORM.
Note also that DMC_POPRENORM is not in general recommended because of the population control
errors it can theoretically introduce, though in general these are likely to be small.

### wdmcmin

*Minimum weight* — Double Precision, Expert, default `0.5` (monte_carlo.f90)

IF LWDMC=T then WDMCMIN is the minimum weight. Now type 'casinohelp lwdmc'.

### wdmcmax

*Maximum weight* — Double Precision, Expert, default `2.` (monte_carlo.f90)

IF LWDMC=T then WDMCMAX is the maximum weight. Now type 'casinohelp lwdmc.'

### checkwfn

*Numcheck analytic wfn derivs* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Enable a numerical check of the analytic orbital derivatives coded in the various routines
such as gauss_per/gauss_mol/bwfdet/pwfdet etc.

### vmc_ave_period

*Energy-averaging period in VMC* — Integer, Intermediate, default `1` (monte_carlo.f90)

Number of consecutive local energies that are averaged together in VMC before writing them
to the vmc.hist file. The only effect of this keyword is to reduce the size of vmc.hist: the
number of lines written in a VMC calculation is VMC_NSTEP/VMC_AVE_PERIOD.

### pair_corr_sph

*Accumulate real-space PCF* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If PAIR_CORR_SPH is set to true then the spherically-averaged real-space pair-correlation
function will be accumulated in the expval.data file (via a process of 'binning' the
electron-electron separations). This currently works for periodic homogeneous systems and
finite isotropic systems such as electron-hole biexcitons. For periodic systems with atoms
you can use the PAIR_CORR keyword instead which gives you the full (non-spherically
averaged) pair-correlation function accumulated in reciprocal space.

### pcfs_nbins

*Number of bins for real-space PCF* — Integer, Expert, default `-1` (monte_carlo.f90)

Number of bins to be used when accumulating the real-space pair-correlation function. Enter
a negative value or omit the keyword to use the default value. If an expval.data file is
present then the value given in expval.data will be used and the input keyword will be
ignored.

### pcfs_rcutoff

*Radius of binned region for real-space PCF* — Physical, Expert, default `-1.,'bohr'` (monte_carlo.f90)

Radius of region to be considered when accumulating the pair-correlation function. The
default value in periodic systems (the Wigner-Seitz cell radius) is generally appropriate;
however, for finite systems such as excitonic complexes, PCFS_RCUTOFF should be set to
something rather larger than the size of the complex. Note that units (e.g., bohr) should be
supplied after the value of PCFS_RCUTOFF. Enter a negative value or omit the keyword to use
the default value. If an expval.data file is present then the value given in expval.data
will be used and the input keyword will be ignored.

### kwarn

*Disable KE check in PW basis* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If the kwarn flag is set to T, then the routine PWFDET_SETUP will issue a warning whenever
the kinetic energy calculated from the supplied orbitals differs from the DFT kinetic energy
given in the pwfn.data file by more than an internal tolerance (usually set to 10^-6). If
the flag is F, then CASINO will stop with an error message on detecting this condition. Note
that in cases where the DFT calculation which generated the orbitals used fractional
occupation numbers, the kinetic energy mismatch is very likely to occur since QMC deals in
principle only with integer occupation numbers, hence the existence of this flag.

### writeout_vmc_hist

*Write vmc.hist file in VMC* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

If WRITEOUT_VMC_HIST is set to T then the energy components, etc., are written to the file
vmc.hist during a VMC simulation. This will not occur if WRITEOUT_VMC_HIST is set to F.
Furthermore, the config.out file required to continue a VMC calculation will only be
produced if WRITEOUT_VMC_HIST is T.

### writeout_dmc_hist

*Write dmc.hist file in DMC* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

If WRITEOUT_DMC_HIST is set to T then the energy components, etc., are written to the file
dmc.hist during a DMC simulation. This will not occur if WRITEOUT_DMC_HIST is set to F.

### vmc_local_dump

*Dump per-configuration local energies* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Setting VMC_LOCAL_DUMP causes the local energies and force components for each configuration
to be written to disk. Each MPI process writes its own dump, 'vmc_local_dump_<IPROC>.dat',
containing configuration weights (if VMC_SAMPLING is not 'standard') and the relevant local
values obtained every VMC_DECORR_PERIOD-th step.

### dmc_local_dump

*Dump per-configuration local energies* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Setting DMC_LOCAL_DUMP causes the local energies and force components for each configuration
to be written to disk. Each MPI process writes its own dump, 'dmc_local_dump_<IPROC>.dat',
containing configuration weights and relevant local values obtained every DMC_DECORR_PERIOD-
th step.

### vmc_method

*VMC method* — Integer, Intermediate, default `1` (monte_carlo.f90)

VMC_METHOD selects which version of VMC to use: 1 = Electron-by-electron algorithm,
evaluating configuration energies at the end of the configuration move; or 3 =
Configuration-by-configuration algorithm, evaluating configuration energies before and after
the move and adding a weighted sum of these to the accumulation arrays. Method 1 is the
default and is recommended. There used to be a Method 2 but after extensive testing we
concluded that it did not offer any advantage over the other methods and was hard to
support.

### vmc_ionjump

*Probability for trying jumps between ions in VMC* — Double Precision, Expert, default `0.` (monte_carlo.f90)

In the special case of nearly separated molecule fragments (e.g., for intermolecular forces)
electrons may get trapped in the energetically less favourable fragment for a long time
during a VMC run due to the large distance between the fragments. Setting vmc_ionjump to a
small, nonzero probability will cause the VMC routine to try a long-distance jump from one
ion to another once in a while, improving the sampling of disjoint areas of the
configuration space. Detailed balance is preserved.

### jasbuf

*Jastrow buffering* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

If JASBUF is set to T then the one-body terms in the Jastrow factor for each electron in
each configuration is buffered: saves time at the expense of memory. Clearly this will have
no effect in systems without one-body terms in the Jastrow (these are the Chi and Q terms at
present).

### neighprint

*Neighbour analysis* — Integer, Intermediate, default `0` (monte_carlo.f90)

NEIGHPRINT = n will generate a printout of the first n stars of neighbours of each atom in
the primitive cell, with the relevant interatomic distances given in both Angstrom and a.u.
If n=0 or if you are an atom-free electron or electron-hole fluid phase, then the keyword
has no effect. Note that activating cusp corrections when using a Gaussian basis (the
default) will trigger a neighbour analysis irrespective of the value of this keyword.

### ranluxlevel

*Quality/cost of random numbers* — Integer, highest possible luxury, all 24 bits chaotic., default `3` (monte_carlo.f90)

To generate parallel streams of pseudo-random numbers for its stochastic algorithms, CASINO
uses an implementation of RANLUX. This is an advanced pseudo-random number generator based
on the RCARRY algorithm proposed in 1991 by Marsaglia and Zaman. RCARRY used a subtract-and-
borrow algorithm with a period on the order of 10^171 but still had detectable correlations
between numbers. Martin Luescher proposed the RANLUX algorithm in 1993; RANLUX generates
pseudo-random numbers using RCARRY but throws away numbers to destroy correlations. RANLUX
trades execution speed for quality through the choice of a 'luxury level' given in CASINO by
the RANLUXLEVEL input keyword. By choosing a larger luxury setting one gets better random
numbers slower. By the tests available at the time it was proposed, RANLUX at its higher
settings appears to give a significant advance in quality over previous generators. The
luxury setting RANLUXLEVEL must be in the range 0-4. period, but fails many tests. gap test,
but still fails spectral test. chance of being observed.

### ranprint

*Print random numbers* — Integer, Expert, default `0` (monte_carlo.f90)

Setting this keyword to a value greater than zero will cause the first RANPRINT numbers
generated by the CASINO random number generator to be printed to a file 'random.log'. On
parallel machines the numbers generated on all MPI processes are printed. The run script
should pick out random.log files from different stages of a calculation (e.g. VMC config
gen/DMC equil/DMC stats accumulation) and rename them appropriately.

### sparse

*Activate sparse algorithms* — Boolean, Expert, default `.false.` (monte_carlo.f90)

CASINO is capable of using sparse matrix algebra in some algorithms for efficiency purposes.
For systems which are definitely not sparse (orbitals not well localized) then attempting to
use sparse algorithms might actually slow things down. Thus until we work out a better way
you can toggle this behaviour with the SPARSE flag. In these algorithms a matrix element is
considered to be zero if it less than the value of the input keyword SPARSE_THRESHOLD.

### sparse_threshold

*Sparsity threshold* — Double Precision, Expert, default `1.e-12` (monte_carlo.f90)

CASINO sometimes uses sparse matrix algebra for efficiency purposes. This keyword defines a
threshold such that a matrix element is taken to be zero if it is less than this threshold.
Changing this quantity might be used to trade speed for accuracy in some cases.

### cusp_correction

*Cusp-correct AE Gaussians and STOs* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

When expanded in a basis set of Gaussian functions, the electron-nucleus cusp present in
all-electron calculations is not represented correctly, since the gradient of an atom-
centred Gaussian is necessarily zero there. Clearly this only matters for s-type GTFs, since
all functions of higher angular momentum vanish at the nucleus. When the CUSP_CORRECTION
flag is activated, the s-type GTFs centred on each atom are replaced within a sphere of
radius r_c by a function which ensures that the electron-nucleus cusp condition is obeyed.
This procedure can be expected to greatly reduce fluctuations in the local energy in all-
electron Gaussian calculations. For STO wave functions, the cusp condition can be exactly
satisfied by a linear constraint. This flag determines whether this constraint is enforced
when reading in or optimizing a wave function.

### dmc_equil_nstep

*No of steps in DMC equil* — Integer, Basic, default `-1` (monte_carlo.f90)

Number of DMC steps performed on each MPI process in the DMC equilibration phase, and
consequently, the total number of local energy samples (averaged over configs and processes)
written to the dmc.hist file. The equilibration phase may be partitioned into
DMC_EQUIL_NBLOCK blocks, but this does not affect the total number of steps (just how
frequently stuff is written out). However, if DMC_EQUIL_NSTEP is not divisible by the number
of blocks, then it will be rounded up to the nearest multiple of DMC_EQUIL_NBLOCK.
Furthermore, DMC_AVE_PERIOD consecutive local energies may be averaged together in DMC
before writing them to the dmc.hist file (hence reducing its size), but again, if
DMC_EQUIL_NSTEP is not divisible by DMC_AVE_PERIOD, it will be rounded up to the nearest
multiple of it. Note the difference in parallel behaviour compared to VMC_NSTEP, which is
not a per-process quantity; this is because the DMC phase is parallelized over
configurations.

### dmc_equil_nblock

*Number of blocks in DMC equilibration* — Integer, Intermediate, default `1` (monte_carlo.f90)

In cases when the BLOCK_TIME keyword is not used this keyword defines the number of blocks
into which the DMC equilibration phase is divided (if DMC_EQUIL_NSTEP is not divisible by
DMC_EQUIL_NBLOCK, then the number of steps will be increased to the nearest multiple of the
number of blocks). Note that having multiple blocks does not increase the amount of data
collected, merely the frequency with which data is written to files; the final answer should
be essentially the same, irrespective of the number of blocks. Specifically, at the end of
each equilibration block, the following significant actions are performed: (1) Write
process- and config-averaged data to dmc.hist (one line for each step in the current block).
(2) Print monitoring data to the output file (block-averaged quantities). (3) Make a backup
copy of the config.out file (if catastrophe protection is turned on with the DMC_TRIP_WEIGHT
keyword). (4) Write the dmc.status file. (5) Write the current state of the system, and all
configs in the current population to the config.out file (note that by setting the
CHECKPOINT keyword to 0, this step can be skipped until the end of the final block, or
skipped completely if CHECKPOINT=-1, but this is not the default). Note that if accumulating
expectation values other than the energy, data is not written to the expval.data file after
each block, as it would be during the statistics accumulation phase. Note that having too
many blocks will make the code slower, and if the run is not massively long it is perfectly
in order to have only one DMC block (which is the default).

### dmc_stats_nstep

*No of steps in DMC stats accum* — Integer, Basic, default `-1` (monte_carlo.f90)

If the value of STOP_METHOD='nstep', then this is the number of DMC steps performed on each
MPI process in the statistics accumulation phase, and consequently, the total number of
local energy samples (averaged over configs and processes) written to the dmc.hist file. The
accumulation phase may be partitioned into DMC_STATS_NBLOCK blocks, but this does not affect
the total number of steps (just how frequently stuff is written out). However, if
DMC_STATS_NSTEP is not divisible by the number of blocks, then it will be rounded up to the
nearest multiple of DMC_STATS_NBLOCK. Furthermore, DMC_AVE_PERIOD consecutive local energies
may be averaged together in DMC before writing them to the dmc.hist file (hence reducing its
size), but again, if DMC_STATS_NSTEP is not divisible by DMC_AVE_PERIOD, it will be rounded
up to the nearest multiple of it. Note the difference in parallel behaviour compared to
VMC_NSTEP, which is not a per-process quantity; this is because the DMC phase is
parallelized over configs. Note that instead of executing a fixed number of DMC steps,
CASINO may be requested to do as many steps as are required to attain a target error bar or
to make the error bar as small as possible - see keyword STOP_METHOD.

### dmc_stats_nblock

*Number of blocks in DMC statistics accumulation* — Integer, Intermediate, default `1` (monte_carlo.f90)

In cases when the BLOCK_TIME keyword is not used this keyword defines the number of blocks
into which the DMC statistics accumulation phase is divided (if DMC_STATS_NSTEP is not
divisible by DMC_STATS_NBLOCK, then the number of steps will be increased to the nearest
multiple of the number of blocks). Note that having multiple blocks does not increase the
amount of data collected, merely the frequency with which data is written to files; the
final answer should be essentially the same, irrespective of the number of blocks.
Specifically, at the end of each accumulation block, the following significant actions are
performed: (1) Write process- and config-averaged data to dmc.hist (one line for each step
in the current block). (2) Write process- and config-averaged data to the expval.data file
(if accumulating expectation values other than the energy). (3) Print monitoring data to the
output file (block-averaged quantities). (4) Make a backup copy of the config.out file (if
catastrophe protection is turned on with the DMC_TRIP_WEIGHT keyword). (5) Make a backup
copy of the expval.data file (if it exists, and if catastrophe protection is turned on with
the DMC_TRIP_WEIGHT keyword). (6) Write the dmc.status file. (7) Write the current state of
the system, and all configs in the current population to the config.out file (note that by
setting the CHECKPOINT keyword to 0, this step can be skipped until the end of the final
block, or skipped completely if CHECKPOINT=-1, but this is not the default). Note that
having too many blocks will make the code slower, and if the run is not massively long it is
perfectly in order to have only one DMC block (which is the default).

### cusp_info

*Print Gaussian cusp info* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If CUSP_CORRECTION is set to TRUE for an all-electron Gaussian basis set calculation, then
CASINO will alter the orbitals inside a small radius around each nucleus in such a way that
they obey the electron-nuclear cusp condition. If CUSP_INFO is set to true, then information
about precisely how this is being done will be printed to the output file. Be aware that in
large systems, this may produce a lot of output. Furthermore, if you create a file called
'orbitals.in' containing an integer triplet specifying which orbital/ion/spin you want, the
code will additionally print graphs of the specified orbital, radial gradient, Laplacian and
'one-electron local energy' to the files orbitals.dat, gradients.dat, laplacians.dat and
local_energy.dat. These graphs may be viewed using xmgr/grace or similar plotting programs.

### opt_maxiter

*Max iterations in optimization* — Integer, Expert, default `10` (monte_carlo.f90)

OPT_MAXITER specifies the largest permitted number of iterations of the minimizer in both
VARMIN and EMIN. Default number: 10.

### ewald_check

*Perform Ewald accuracy check* — Boolean, Expert, default `.true.` (monte_carlo.f90)

CASINO and the wave function generating program should be able to calculate the same value
for the nuclear repulsion energy, given the same crystal structure. By default CASINO
therefore computes the Ewald interaction and compares it with the value given in the wave
function file. If they differ by more than 10^-5, then CASINO will stop and complain. If you
have a justifiable reason for doing so, you may turn off this check by setting EWALD_CHECK
to F.

### permit_den_symm

*Symmetrize QMC charge data* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If this flag is set to T then the symmetry of the SCF charge density (in mpc.data) will be
imposed upon the QMC charge-density data for use in the MPC interaction (with the
justification that imposing an exact condition on the charge density can''t hurt) and also
when writing the QMC density to expval.data. It is possible however that DMC will break the
symmetry of the SCF calculation; in this case the user should turn off PERMIT_DEN_SYMM.

### qmc_density_mpc

*Use QMC density in MPC int* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If this flag is set to T then the QMC charge density data at the end of the expval.data file
will be used to compute the MPC interaction, rather than the default SCF density in the
mpc.data file. This is likely to be useful in cases such as the Wigner crystal where the
Hartree-Fock charge density is very different to the true charge density (it is too
localized) as opposed to, say, the Fermi fluid where the Hartree-Fock charge density is
exact. Note that using this option is likely to increase the time taken to evaluate the MPC
interaction; in both DFT and QMC cases, the code counts backwards from the end of the list
of G vectors and discards all those before the first non-zero one (where zero is defined by
some threshold like 1.e-6). In the HF/DFT case this tends to give a large reduction in the
size of the vector to be evaluated. However, the random noise in the QMC density
coefficients is likely to exceed the zero threshold for all G, and the vector will likely be
untruncated. It is important therefore to use a value for EXPVAL_CUTOFF which is not too
large when using this facility, in order that the total number of G vectors in the expansion
is not too large.

### esupercell

*Energy/per supercell in output* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

By default total energies and their components are printed as energies per primitive cell.
Switching this flag to T forces printing of energies per simulation cell in the output file.

### dmc_trip_weight

*DMC catastrophe threshold* — Double Precision, Intermediate, default `0.` (monte_carlo.f90)

In the course of a DMC simulation, it is possible for a configuration "population explosion"
to occur. If DMC_TRIP_WEIGHT is set to 0 then nothing will be done about this. If
DMC_TRIP_WEIGHT>0 then it will attempt to restart with a different random number sequence
from the beginning of the previous block if the iteration weight exceeds DMC_TRIP_WEIGHT. A
general suggestion for its value would be 2-3 times DMC_TARGET_WEIGHT (but see the
discussion in the manual about this).

### max_rec_attempts

*Maximum number DMC recoveries* — Integer, Intermediate, default `5` (monte_carlo.f90)

This is the maximum number of times DMC will attempt to restart a block if it continues to
encounter catastrophes. Relevant only if the DMC_TRIP_WEIGHT keyword is set to a nonzero
value.

### molgscreening

*Screen molecular Gaussians* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Toggle on and off the use of screening in Gaussian basis set calculations of molecules i.e.
the division of space into boxes and the preparation of lists of which Gaussian basis
functions have a significant weight in each box. Should speed up the calculation of large
molecules. The screening information can take up a reasonable amount of memory, hence this
keyword.

### bsmooth

*Smooth truncate loc orbs* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If bsmooth is set to true then localized orbitals are interpolated smoothly to zero beyond
their cutoff radius. Otherwise, they are truncated abruptly. The latter is the default as it
generally produces better results.

### relativistic

*Relativistic correction* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If RELATIVISTIC is T, then calculate relativistic corrections to the energy using
perturbation theory. Note that for the moment this can only be done for closed-shell
systems.

### isotope_mass

*Nuclear mass override* — Double Precision, Expert, default `0.` (monte_carlo.f90)

This keyword can be used to define the nuclear mass (in amu) if one wishes to override the
default value (which is averaged over isotopes according to their abundances). The default
is used if ISOTOPE_MASS is set to zero. The atomic mass unit (amu) in this sense means 'the
ratio of the average mass per atom of the element to 1/12 of the mass of 12C.

### vm_w_min

*VM minimum weight* — Double Precision, Expert, default `0.` (monte_carlo.f90)

Minimum value that a configuration weight may take during weighted variance minimization.
This parameter should have a value between zero and one. Note that the limiting is not
applied if VM_W_MAX = 0.

### vm_w_max

*VM maximum weight* — Double Precision, Expert, default `0.` (monte_carlo.f90)

Maximum value that a configuration weight may take during weighted variance minimization.
Set this to zero if you do not wish to limit the weights; otherwise it should be greater
than 1.

### opt_dtvmc

*Optimize VMC time step* — Integer, Basic, default `1` (monte_carlo.f90)

IF OPT_DTVMC is set to 1 the VMC time step (initially the value of keyword DTVMC) is
optimized so that the VMC acceptance ratio is roughly 50%. This is the default. To prevent
this, set OPT_DTVMC to 0. The value of OPT_DTVMC is ignored if the input block DTVMCS is
supplied. CASINO can also maximize the diffusion constant with respect to DTVMC. This can be
enabled by setting OPT_DTVMC=2. In a first stage, DTVMC is varied to get an acceptance ratio
of 50%, so as to have decent statistics to perform the diffusion-constant maximization
stage. This is only useful for VMC_METHOD=3, where it is the default.

### pcf_rfix

*Fixed particle type and position in PCF calc.* — Block, Expert

This block contains two lines. The first line gives the type of particle to be fixed during
accumulation of the pair correlation function g(r,r') (e.g. 1-4 typically up/down spin
electron, up/down spin hole); the second line gives the coordinates of the position r at
which to fix it (in au). This applies to the reciprocal-space PCF activated with the
PAIR_CORR input keyword. It also applies *in principle* to the spherical real space PCF
activated with the PAIR_CORR_SPH input keyword, in the sense that the format of expval.data
allows it, but the accumulation of the spherical PCF with fixed particles has not yet been
implemented.

### on_top_pair

*Particles to place on top of each other in RPMD calc.* — Block, Expert

This block contains two lines consisting of the particle type and index (integers) of each
of two particles to be forced to stay on top of each other throughout a VMC calculation.
This is intended for the evaluation of recombining-pair momentum densities and 'one-body'
density matrices in hole-in-HEG systems -- wrong values will be reported for other
expectation values, including the energy. Note that the time step of the first-specified
particle applies to the pair. This block must not be used in DMC or optimization runs.

### spin_density

*Accumulate spin densities* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Setting SPIN_DENSITY to T will activate the accumulation of separate up- and down-spin
densities in the expval.data file.

### pos_mom_den

*Accumulate positron momentum densities* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Setting POS_MOM_DEN to T will activate the accumulation of positron momentum densities
(APMD) in the expval.data file.

### pair_corr

*Accumulate rec. space PCF* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Set PAIR_CORR to T to accumulate the reciprocal-space pair-correlation function in the
expval.data file. Currently restricted to periodic systems. Note you also need to give the
position and type of a fixed particle using the PCF_RFIX block. Note that if the density is
homogeneous, or if you want only the spherically-averaged pair-correlation function (for a
very restricted class of systems), you should use the PAIR_CORR_SPH keyword.

### opt_cycles

*Number of optimization cycles* — Integer, Basic, default `4` (monte_carlo.f90)

Number of cycles of configuration generation and optimization runs to be carried out if
RUNTYPE=vmc_opt or opt_vmc. For variance minimization, 3--6 cycles is typical; for energy
minimization, 5--10 cycles is usual unless only determinant coefficients are being
optimized, in which case 1--2 cycles will be enough.

### loc_tensor

*Accumulate localization tensor* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If LOC_TENSOR is set to true then the localization tensor will be accumulated in the
expval.data file (periodic systems only).

### jastrow_plot

*Plot components of Jastrow factor* — Block, Intermediate

This block allows you to plot the u(rij), w(rij), chi(ri), f(ri,rj,rij), p(rij) and q(ri)
terms in the Jastrow factor. Either THREE or SIX lines should be given in the input block:
LINE 1: flag for whether the Jastrow factor is to be plotted (0=NO, 1=YES); LINE 2: spin of
particle i (=1,2,..); LINE 3: spin of particle j (=1,2,..); The following three lines are
optional: LINE 4: (x,y,z)-position of particle j (used in plots of f and p); LINE 5: vector
with the direction in which i is moved (used in plots of f, p and q); LINE 6: position
vector of a point on the straight line along which electron i moves (used in plots of f, p
and q). Note that the nucleus is assumed to lie at the origin in plots of f. The
jastrow_value_f_?.dat files contain the value of f against the distance from the point given
in line 6. Likewise for plots of p and q. For u and chi we just plot u(r) and chi(r) and the
derivatives thereof.

### popstats

*Dump DMC population statistics* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Produce dmc.status file summarizing the statistical efficiency of DMC runs. This file is
updated every block.

### timing_info

*Activate subroutine timers* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Setting TIMING_INFO to T will turn on the collection of subroutine timings. Note, however,
that the timing routines can adversely affect performance on certain computers, especially
for small systems; hence timers are deactivated by default. If TIMING_INFO is T and CASINO
halts to report an error message, CASINO will be able to provide some traceback information
about where that error occurs.

### cusp_control

*All-electron cusp radius control* — Double Precision, Expert, default `50.` (monte_carlo.f90)

This undocumented parameter is required in the *OLD* procedure (which is in fact much better
than the NEW one, and is the default) used to make all-electron orbitals expanded in
Gaussian functions satisfy the electron-nuclear cusp conditions (activated by the
CUSP_CORRECTION input keyword). To activate the old algorithm for doing this, set
OLD_CUSP_RADIUS=.true. in the routine cusp_setup in gaussians.f90. The radius inside which
the form of the orbitals is modified is determined by looking at fluctuations in the local
energy. This radius ('rcusp') is set to the largest distance from the nucleus at which the
deviation from the 'ideal' curve has a magnitude of greater then (zion^2/CUSP_CONTROL),
where zion is the nuclear charge. The default value of CUSP_CONTROL is 50.0. Note that this
keyword will have no effect on hydrogen atoms, which are treated as a special case.

### vm_E_guess

*Estimate of GS energy* — Physical, Expert

If VM_ECENTRE is set to 'guess' then VM_E_GUESS will be used as the central energy in the
evaluation of the optimization target function. This is a keyword of type 'Physical' hence
you need to supply units, such as 'ev', 'ry', 'hartree', 'kcal/mol' etc.

### e_offset

*Energy offset* — Physical, Intermediate, default `0.,'hartree'` (monte_carlo.f90)

The E_OFFSET keyword gives a constant shift in the total energy per electron such that the
final results will be E = E_calc - E_offset. The default is zero. This is a keyword of type
'Physical' hence you need to supply units, such as 'ev', 'ry', 'hartree', 'kcal/mol' etc.

### vm_smooth_limits

*Smooth Jastrow cutoff limits* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

By setting this keyword to T (default), the optimizing routine used in variance minimization
is sent a smoothed version of the set of parameters. This only affects those which are to
remain bounded such as Jastrow cutoffs. The result is a set of parameters which can vary in
the range (-Inf,+Inf), which can be more convenient than ignoring out-of-range values
without the minimizer knowing. A suitable hyperbolic function is used for mapping 'limited'
values into 'extended' ones and vice versa.

### backflow

*Use backflow corrections* — Boolean, Basic, default `.false.` (monte_carlo.f90)

Turns on backflow corrections. Backflow parameters are read from correlation.data and, if
optimized (OPT_BACKFLOW = T), written to correlation.out(.x) .

### opt_backflow

*Optimize backflow parameters* — Boolean, Basic, default `.false.` (monte_carlo.f90)

During optimization, allow the backflow parameters to be optimized.

### use_jastrow

*Use a Jastrow function* — Boolean, Basic, default `.true.` (monte_carlo.f90)

Use a wave function of the Slater-Jastrow form, where the Jastrow factor exp(J) is an
optimizable object that multiplies the determinant part in order to introduce correlation in
the system. The Jastrow factor must be provided in the correlation.data file --see the
CASINO manual for information on the format.

### plot_backflow

*Generate plot of backflow transformation* — Block, Intermediate

This block allows plotting the backflow transformation after VMC equilibration. The block
should contain 2 lines, plus an optional line for plotting the Phi term: LINE 1: 0 or 1 to
(de-)activate this facility; LINE 2: kspin, knumber and zposition; LINE 3: value of fixed
electron-ion distance riI. This will produce various files: * bfconfig.dat (reference
config), * bfconfigx.dat (associated quasi-particle config), * bfions.dat (coordinates of
nuclei for which backflow terms exist), * bfeta_<s>.dat (eta vs rij for each spin-pair type
<s>), * bfmu_<s>_<set>.dat (mu vs ri for each spin type <s> in each set <set>), * bfphi.dat
(contribution of Phi to the 3D backflow displacement on electron j vs 2D projection of rjI
on the plane defined by electron j, ion I in set <set>, and electron i at distance riI from
the nucleus with spin such that <s> is the spin-pair type of i and j), * bffield.dat (3D
backflow displacement on electron (kspin,knumber) vs its 2D position on the plane
z=zposition).

### vm_linjas_method

*Optimization method for acc varmin* — String, Intermediate, default `'BFGS'` (monte_carlo.f90)

VM_LINJAS_METHOD specifies the method used to minimize the quartic LSF. VM_LINJAS_METHOD
should be one of: 'CG' (conjugate gradients), 'MC' (Monte Carlo), 'LM' (line minimization),
'SD' (steepest descents), 'BFGS' (Broyden-Fletcher-Goldfarb-Shanno), 'BFGS_MC' (BFGS and
Monte Carlo), 'CG_MC' (conjugate gradients and Monte Carlo), 'GN' (Gauss-Newton) or 'GN_MC'
(Gauss-Newton and Monte Carlo).

### vm_linjas_its

*Max iterations in acc varmin* — Integer, Expert, default `-1` (monte_carlo.f90)

VM_LINJAS_ITS specifies the maximum number of conjugate-gradients, steepest-descent or BFGS
iterations to be performed if VM_LINJAS_ITS is 'CG', 'SD', 'BFGS', 'CG_MC' or 'BFGS_MC'. If
VM_LINJAS_ITS is 'MC', 'LM', 'CG_MC' or 'BFGS_MC' then it specifies the number of line
minimizations to be performed.

### splot

*Line plotter s components* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Tell line plotter to do s component of orbitals rather than full plot. Useful for analysing
Gaussian cusp corrections.

### cusp_threshold

*Zero orbital threshold* — Double Precision, Expert, default `1.e-7` (monte_carlo.f90)

If the magnitude of the s component of a Gaussian orbital is less than this threshold, then
it will not be cusp corrected.

### vm_filter

*Filter outlying configs* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

This keyword activates filtering of configurations in VARMIN by making the weights
(artificially) energy-dependent, i.e., W_i = W(|E_i-E_ave|). This method uses two
parameters: VM_FILTER_THRES and VM_FILTER_WIDTH.

### vm_filter_thres

*Filter threshold* — Double Precision, Intermediate, default `4.` (monte_carlo.f90)

When limiting outlying configs in VARMIN (by setting the VM_FILTER flag to T), the maximum
deviation from the average energy at which the (artificial) weight of a configuration W_i =
W(|E_i-E_ave|) is kept equal to unity is VM_FILTER_THRES times the square root of the
unreweighted variance. After such limit, the weight is brought to zero using a gaussian of
width VM_FILTER_WIDTH times the square root of the unreweighted variance. By default,
VM_FILTER_THRES is 4.0 and VM_FILTER_WIDTH is 2.0.

### vm_filter_width

*Gaussian filter width* — Double Precision, Intermediate, default `2.` (monte_carlo.f90)

When limiting outlying configs in VARMIN (by setting the VM_FILTER flag to T), the maximum
deviation from the average energy at which the (artificial) weight of a configuration W_i =
W(|E_i-E_ave|) is kept equal to unity is VM_FILTER_THRES times the square root of the
unreweighted variance. After such limit, the weight is brought to zero using a gaussian of
width VM_FILTER_WIDTH times the square root of the unreweighted variance. By default,
VM_FILTER_THRES is 4.0 and VM_FILTER_WIDTH is 2.0.

### max_cpu_time

*Maximum CPU time* — Physical, Intermediate, default `0.,'s'` (monte_carlo.f90)

If the CPU time elapsed since the start of a QMC simulation exceeds MAX_CPU_TIME and a
suitable point in the algorithm is reached, then CASINO will halt gracefully. This should
make it easier to carry out e.g. multiple DMC runs on a computer with a queueing system,
particularly when used with the --continue or --auto-continue runqmc options. The way this
works is that at the end of each block of moves, CASINO will check whether doing one more
block will exceed the time limit. If so, it will perform an emergency stop, writing to the
output file any changes to the input file that must be made in order to restart the job (in
a form readable both by humans and by runqmc). The user must therefore define the block
length appropriately -- most usefully via the BLOCK_TIME keyword -- such that the time taken
per block is a sufficiently small fraction of MAX_CPU_TIME. Note that in DMC, if
CHECKPOINT=0 in input and there is a job time limit, it is strongly recommended that
MAX_CPU_TIME is used to ensure the config.out file is written out if the required CPU time
is longer than the time limit. MAX_CPU_TIME is a physical parameter with dimensions of time,
and the units must be specified as e.g. 1 day, 24 hr, 1440 min, or 86400 s. See also the
MAX_REAL_TIME keyword.

### max_real_time

*Maximum real time* — Physical, Intermediate, default `0.,'s'` (monte_carlo.f90)

If the wall-clock time elapsed since the start of a QMC simulation exceeds MAX_REAL_TIME and
a suitable point in the algorithm is reached, then CASINO will halt gracefully. This should
make it easier to carry out e.g. multiple DMC runs on a computer with a queueing system,
particularly when used with the --continue or --auto-continue runqmc options. The way this
works is that at the end of each block of moves, CASINO will check whether doing one more
block will exceed the time limit. If so, it will perform an emergency stop, writing to the
output file any changes to the input file that must be made in order to restart the job (in
a form readable both by humans and by runqmc). The user must therefore define the block
length appropriately -- most usefully via the BLOCK_TIME keyword -- such that the time taken
per block is a sufficiently small fraction of MAX_REAL_TIME. Note that in DMC, if
CHECKPOINT=0 in input and there is a job time limit, it is strongly recommended that
MAX_REAL_TIME is used to ensure the config.out file is written out if the required time is
longer than the time limit. MAX_REAL_TIME is a physical parameter with dimensions of time,
and the units must be specified as e.g. 1 day, 24 hr, 1440 min, or 86400 s. See also the
MAX_CPU_TIME keyword.

### use_orbmods

*Single particle orbital modifications* — Boolean, Basic, default `.false.` (monte_carlo.f90)

If USE_ORBMODS is set to T then the orbital-modification block in correlation.data will be
read, and the modifications will be applied to the single particle orbitals orbitals. This
only applies to ATOM_BASIS_TYPEs 'numerical', 'gaussian' or 'slater-type.

### particles

*Define custom quantum particles* — Block, Intermediate

Using the PARTICLES block the user can define quantum particles (other than electrons, which
can be introduced using NEU/NED) to be used in the QMC calculation. The format of each line
is: <i> <charge/|e|> <mass/m_e> <spin/hbar> <name> A negative value of the mass indicates
that the following three lines give an (anisotropic) 3x3 mass tensor. CASINO should decide
whether each particle type is a fermion or a boson (based on the spin), and select the
appropriate way to combine the one-particle orbitals (symmetric combination [not currently
implemented] or antisymmetric Slater determinants). The particles defined here can be
assigned orbitals using the FREE_PARTICLES block.

### free_particles

*Parameters for free particles* — Block, Intermediate

This block is used for orbitals that are not atom-related in the system. It is only required
if ATOM_BASIS_TYPE is 'none'. The format is as follows, with many items being optional: r_s
<rs> # Density parameter in a.u. dimensionality <d> # Particles move in d dimensions
periodicity <P> # System is periodic in P dimensions cell_geometry (d lines with d reals
holding the unscaled cell vectors) heg_nlayers <n> # Define number n of layers heg_ylayer
<l> <y> # Define y-coordinate of "layer" l heg_zlayer <l> <z> # Define z-coordinate of layer
l heg_layer <p> <l> # Assign particle type p to layer l quasi_1d <b> # Set "width" of 1D
harmonic wires to b transverse_model <m> # Softened (m=1); harmonic (m=2); hard-wall (m=3)
transverse_method <t> # Analytic int if t=0; sample transverse if t=1 k_offset <k_x k_y k_z>
# Offset of grid of k vectors in Cartesians k_offset_frac <k_1 k_2 k_3> # Offset of grid of
k vectors (fractional) The number and type of the orbitals can be given using lines of form:
particle <i> [det <det>] : <n> orbitals <orb> [orb-options] where <det> is the term in the
multideterminant expansion (this term is optional if all determinants contain the same
orbital type), <i> must be 1, 2 or a number given in the PARTICLES block (1/2 are up/down
electrons), <n> is the number of free particles/orbitals belonging in the <det>-th
determinant, and '<orb> [orb-options]' is one of the following: 'free', 'crystal siteset
<s>', 'harmonic', 'biex1', 'biex2', 'biex3' or 'pairing <j> [+ <n> orbitals free]', <j>
being the particle type which <i> is paired with, and <n> is the number of unpaired
particles of type <i>. Wigner crystal geometry is specified using: crystal_type <type> <n>
siteset[s] [repeat <r>] where type = 'cubic', 'fcc', 'bcc', 'rectangular', 'hexagonal' or
'triangular', or 'manual'), and siteset <s> [antiferro[magnetic]] offset <x y z> for
predefined lattices, where the offset is in fractional coordinates in the primitive cell,
and siteset <s> manual <n> site[s] (n lines <x y z> defining sites in primitive-cell
fractional coords) for manual lattices. In nonperiodic directions the site positions and
offsets are in absolute coordinates rather than fractional coordinates. Should the orbitals
have optimizable parameters, these must be provided in correlation.data.

### fixed_particles

*Create a set of fixed, charged particles* — Block, Expert

When setting up a model system, one can place fixed, charged particles within the primitive
cell by using the fixed_particles block. This can be used to study, e.g., electron-hole
complexes in the presence of fixed donor and acceptor ions. The block consists of one line
for each fixed particle, where the lines are of the form '<charge> <x> <y> <z>', where x, y
and z are the Cartesian components of the fixed charge's position. The charge must be an
integer. Periodic repeats of the fixed particle are generated automatically.

### plot_expval

*Plot exp values in line/plane/volume* — Block, Intermediate

The utility 'plot_expval' allows you to plot expectation values calculated by CASINO and
stored in the file 'expval.data'. It takes its instructions from this block in input which
tells it about the geometrical region over which the data will be plotted. Where the
geometry is clear, this input block is not required (e.g. spherical PCF/structure factor).
The geometrical region may be a line AB / plane AB-AC / volume AB-AC-AD. The data will be
plotted in a format suitable for xmgr/grace in the file 'lineplot.dat' or in a format
suitable for gnuplot in '2Dplot.dat' or '3Dplot.dat'. These latter two files can be quickly
visualized with the 'plot_2D' utility. The block, which is ignored by CASINO itself, has the
following format: LINE 1: dimensionality of plot ndim 1/2/3, OR EQUIVALENTLY,
line/plane/volume; LINE 2: No. of points along each of the ndim directions (ndim integers);
LINES 3-: xyz coords of point A ; of point B ; of point C (if reqd.) ; of point D (if
reqd.).

### ke_verbose

*Detailed info KE tests* — Boolean, Expert, default `.false.` (monte_carlo.f90)

CASINO performs numerical tests to determine whether the kinetic energies computed during
the run will be correct. Such tests are carried out after VMC equilibration, and will only
produce concise output about the outcome. However, if the flag KE_VERBOSE is set to T,
CASINO will print out information throughout the process. The default is F. See also
KE_FORGIVE.

### ke_forgive

*Allow failing KE tests* — Boolean, Expert, default `.true.` (monte_carlo.f90)

CASINO performs numerical tests to determine whether the kinetic energies computed during
the run will be correct. If KE_FORGIVE is set to F, CASINO will regard this as an error and
stop. The default is T. Note that although the procedure is rather stable, there may be
cases in which poor numerics causes narrow fails. See also KE_VERBOSE.

### dipole_moment

*Accumulate elec. dipole moment* — Boolean, Basic, default `.false.` (monte_carlo.f90)

If this flag is set to T then CASINO will accumulate the expectation value of the electric
dipole moment p. It will also evaluate the expectation of p^2. The data p_x, p_y, p_z and
p^2 are written to (v/d)mc.hist like energy components, rather than into expval.data, and
their value and error bars are determined by reblocking. The dipole moment is well-defined
in aperiodic systems, or in aperiodic directions in 1D- and 2D-periodic systems. Note that
the CASINO reblock utility reports only the components and not the magnitude of the dipole
moment in order to allow the user to decide how to deal with the symmetry. Suppose that
symmetry dictates the dipole moment will point in the x direction. The y and z components
should be zero, but there will be some noise when they are evaluated in QMC. If you work out
p=sqrt(p_x^2+p_y^2+p_z^2) then you will get something larger than p_x, tending to p_x in the
limit of perfect sampling (i.e. a biased estimate with finite sampling). You will also get
larger error bars on p than p_x.

### struc_factor_sph

*Accumulate sph. struc. factor* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If STRUC_FACTOR_SPH is set to T then the spherically-averaged structure factor will be
accumulated in the expval.data file. You should also define a one-dimensional k point grid
on which to calculate it using the EXPVAL_KGRID keyword block. Note this is implemented only
for homogeneous systems.

### expval_kgrid

*k point grid for exp values* — Block, Expert

This block contains a specification of one or more k point grids defined in 1, 2 or 3
dimensions (NOTE: this is more general than CASINO actually requires at the moment). One-
dimensional grid defined by line AB, two dimensional grid by plane AB-AC, three-dimensional
grid by parallelepiped AB-AC-AD, all with an appropriate number of k points along each
direction. These grids may be used in the calculation of various expectation values. Only
used if appropriate expval keywords are set to T in input. The block consists of the
following lines: LINE 1: No of k grids defined in this block; LINE 2: Which expectation
value uses this grid? (Currently 1=BLANK, 2=spherical structure factor, 3=BLANK); LINE 3:
dimensionality of current k grid NKDIM (1-3); LINE 4: k point A coordinates - origin (au);
LINE 5: k point B coordinates (au), number of k along AB; LINE 6: [IF NKDIM=2 or 3] k point
C coordinates (au), number of k along AC; LINE 7: [IF NKDIM=3] k point D coordinates (au),
number of k along AD ; Then repeat lines 2 to 7 for each additional k grid. Note that
spherically averaged quantities require only a one-dimensional k grid independently of the
dimensionality of the system, and that this is a radial coordinate so only one number is
required to specify each k point coordinate.

### opt_method

*Optimization method* — String, Intermediate, default `'varmin'` (monte_carlo.f90)

There are currently four optimization methods implemented in CASINO: 'varmin': variance
minimization 'varmin_linjas': an accelerated variance minimization technique for parameters
that appear linearly in the Jastrow; 'emin': linear least-squares energy minimization
'madmin': minimization of the mean absolute deviation of the set of local energies from the
median All these methods can be used to optimize all types of parameters (Jastrow, orbitals,
backflow and determinant coefficients), except 'varmin_linjas' which can only be applied to
optimize Jastrow factors. There are other keywords that affect the behaviour of each of
these methods. The default value of OPT_METHOD is 'varmin'.

### onep_density_mat

*Accumulate 1p density matrix* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If ONEP_DENSITY_MAT is set to T, then the spherically averaged one-particle density matrix
will be computed. This is only possible if the system is homogeneous for the moment.

### twop_density_mat

*Accumulate 2p density matrix* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If TWOP_DENSITY_MAT is set to T, then the spherically averaged two-particle density matrix
will be computed. This is only possible if the system is homogeneous for the moment, and
will increase the cost of the calculation significantly.

### int_sf

*Calc e-e int from struc fac* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If INT_SF is set to T, then the electron-electron interaction energy for a periodic system
will be calculated in terms of the structure factor. The structure factor should either have
been accumulated in a previous run and stored in an available expval.data file, or its
accumulation should be flagged for the current run. Using this method the total interaction
energy can be separated into Hartree and exchange-correlation terms. [NB: this is a
deprecated keyword, and should be replaced by HARTREE_XC.]

### hartree_xc

*Calc e-e int from struc fac/MPC* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Flag the computation of separate Hartree and exchange-correlation (XC) parts of the
electron-electron interaction energy for a periodic system. This may be done in two
different ways, namely the structure factor method and the MPC method. The computation thus
requires either (1) structure factor information from a previously accumulated expval.data
file or from setting STRUCTURE_FACTOR=T, or (2) the MPC interaction to be active (through
INTERACTION=mpc, mpc_ewald or ewald_mpc). If both these things are true then both methods
will be used to compute the hartree/XC energies (the resulting numbers should agree
reasonably closely). If neither are true, then this keyword has no effect. The default is T.
Note that the MPC version only works with 3D periodicity.

### future_walking

*Enable future walking* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If this flag is set to T, then future walking will be used to evaluate pure estimators in
DMC. This is currently implemented only for sampling the total energy.

### cond_fraction

*Accumulate cond fraction* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If COND_FRACTION is set to T, then an improved estimator of the spherically averaged two-
particle density matrix, from which one-body contributions are subtracted, will be computed.
This is only available if the system is homogeneous, for the moment.

### complex_wf

*Use complex Slater wfn* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If COMPLEX_WF is set to T then CASINO will use a complex wave function and the fixed-phase
(rather than fixed-node) approximation will be applied in DMC. Using complex arithmetic is
slower, and is only necessary if the Hamiltonian is complex (e.g. if a magnetic field is
present) or if the boundary conditions on the wave function force it to be complex (e.g. for
a periodic system with a general set of twist angles).

### virtual_node

*Node no in virtual parallel vm* — Integer, Expert, default `0` (monte_carlo.f90)

This parameter is not to be set manually.

### virtual_nconfig

*nconfigs in virtual parallel vm* — Integer, Expert, default `0` (monte_carlo.f90)

This parameter is not to be set manually.

### virtual_nnodes

*nnodes during virtual parallel vm* — Integer, Expert, default `1` (monte_carlo.f90)

This parameter is not to be set manually.

### use_tmove

*Casula nl PP scheme in DMC* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

If USE_TMOVE is T then the Casula nonlocal pseudopotential scheme will be used in DMC. So-
called 'T-moves' will be performed in order to give a DMC energy that is greater than or
equal to the ground-state energy. This violates the detailed-balance principle at finite
time steps, but greatly improves the stability of the DMC algorithm when nonlocal
pseudopotentials are used. The advantages of T-moves are that they restore the variational
principle and help to prevent population explosions; the disadvantages of T-moves are that
the magnitude of the error due to the locality approximation is generally larger, although
always positive, and the time-step bias is generally worse. [This latter problem is
alleviated, to some extent, by using a symmetric branching factor (Casula 2010) as opposed
to the asymmetric one suggested in his 2006 paper. This advice was implemented in CASINO in
June 2014.]. T-moves are now (as of 2018) used by default.

### use_detla

*Determinant localization approx. PP scheme* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If USE_DETLA is T the nonlocal contribution to the local energy will be calculated using the
Slater part of the wave function only. Note that, in contrast with USE_TMOVE which applies
to DMC calculations, USE_DETLA applies to both VMC and DMC. See A. Zen et al., J. Chem.
Phys. 151, 134105 (2019) for details.

### finite_size_corr

*Eval. finite size corrections* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Calculate finite size corrections to kinetic energy and electron-electron interaction energy
using the Chiesa-Ceperley-Martin-Holzmann or Drummond-Needs-Sorouri-Foulkes scheme.

### random_seed

*Random seed* — String, Expert, default `'timer'` (monte_carlo.f90)

This keyword determines which random seed to use for the RANLUX random-number generator. The
default value of RANDOM_SEED is 'timer', which causes the system timer to be used as the
seed. If RANDOM_SEED is set to 'standard', the seed 314159265 is used. If the value of
RANDOM_SEED is an integer, that integer will be used as the random seed. The seed is printed
to the output file so that calculations using RANDOM_SEED='timer' can be reproduced
afterwards. Note that, if RANDOM_SEED is an integer or 'standard' or 'timer' then, when
restarting from a previous calculation the value of RANDOM_SEED is ignored (except for any
initial setup, such as evaluation of the twist-averaged Hartree-Fock energy of a homogeneous
electron gas), and the random-number sequence will generally be continued from the saved
state of the random-number generator stored in the config.in file. However, if RANDOM_SEED
is 'timer_reset' then the generator will be re-initialized from the system clock after the
config.in file is read. This might be useful, for example, if a prior test has revealed that
the standard sequence will lead to a configuration giving rise to a population explosion.

### custom_striplet_dep

*Custom spin-triplet dependence* — Block, Expert

This input block can be used to create spin-triplet groupings (for the Jastrow H term only
so far). CASINO does not currently know how to generate non-trivial groupings automatically,
and this block is the way to define them. The format of the block is as in the following
example: %block custom_striplet_dep no_striplet_deps 1 striplet_dep -1 4
1=2-3,1=2-4,1=1-3,1=1-4,2=2-3,2=2-4 1-3=4,2-3=4,1-3=3,1-4=4,2-3=3,2-4=4
1=1=1,1=1=2,1=2=2,2=2=2 3=3=3,3=3=4,3=4=4,4=4=4 %endblock custom_striplet_dep The equal
signs denote that the two particle types are equivalent within the group, whereas the
hyphens denote they are different. All spin-triplets must be included in a group. See also
CUSTOM_SPAIR_DEP and CUSTOM_SSINGLE_DEP.

### custom_spair_dep

*Custom spin-pair dependence* — Block, Expert

This input block can be used to create new spin-pair groupings for the Jastrow factor, etc.
For example, if one were studying a paramagnetic fluid bilayer, with spin-up and spin-down
electrons in one plane (spins 1 and 2) and spin-up and spin-down electrons on the other
plane (spins 3 and 4) then one would want sets of u(r_ij) terms for same-plane, same-spin
pairs, same-plane, opposite-spin pairs and opposite-plane pairs. Here is an example: %block
custom_spair_dep no_spair_deps 1 # Number of custom spin dependences spair_dep -1 3 # Label
(-1,-2,-3,...) and number of spin groups 1-1,2-2,3-3,4-4 1-2,2-4 1-3,1-4,2-3,2-4 %endblock
custom_spair_dep All spin-pairs must be included in a group. See also CUSTOM_SSINGLE_DEP.

### custom_ssingle_dep

*Custom spin-single dependence* — Block, Expert

This input block can be used to create new spin-single groupings for the Jastrow factor,
etc. Here is an example: %block custom_ssingle_dep no_ssingle_deps 1 # Number of custom spin
dependences ssingle_dep -1 2 # Label (-1,-2,-3,...) and number of spin groups 1,2 3,4
%endblock custom_ssingle_dep All spins must be included in a group. See also
CUSTOM_SPAIR_DEP.

### positron_pp

*Use separate pseudopotential for positron* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

The pseudopotential must exist with name <atom>_positron_pp.data. Only one spin component is
considered.

### use_gpcc

*General-purpose cusp correction* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If use_gpcc is set to T then short-ranged functions will be added to the orbitals to ensure
that the Kato cusp conditions are satisfied.

### vmc_ntwist

*Number of twist angles in VMC* — Integer, Intermediate, default `0` (monte_carlo.f90)

Number of different 'twists' or offsets to the grid of k vectors to be applied during a VMC
twist-averaging run, per MPI process. Note that the usual keywords define the run length for
a single twist angle, thus the run length is increased by a factor of VMC_NTWIST. Note also
that if VMC_NTWIST > 0, the values of VMC_NBLOCK and BLOCK_TIME are ignored. Unlike other
keywords in the new keyword set (and unlike DMC_NTWIST), VMC_NTWIST is a per-MPI-process
quantity. Setting this keyword to a value greater than zero requires the use of a complex
wave function (COMPLEX_WF : T). Note twist-averaging wholly within CASINO can currently be
done only for electron(-hole) fluid phases; for real systems with atoms one needs to couple
with an external code to regenerate the wave function after each twist. The various
twistav_xxx scripts in CASINO/utils/twist can help with this - see the manual.

### vmc_reequil_nstep

*Number of post-twist equil steps* — Integer, Intermediate, default `500` (monte_carlo.f90)

Total number of steps to take for a re-equilibration when doing a twist-averaged VMC run.
Currently, a re-equilibration only takes place when the twist angle is changed. Note this is
a single-process quantity; all MPI processes run VMC_REEQUIL_NSTEP re-equilibration steps.
Note also that the value of VMC_DECORR_PERIOD is ignored in re-equilibrations.
Electron(-hole) fluid phases only. For more details, see the VMC_NTWIST keyword.

### dmc_ntwist

*Number of twist angles in DMC* — Integer, Intermediate, default `0` (monte_carlo.f90)

Number of different 'twists' or offsets to the grid of k vectors to be applied during DMC
statistics accumulation. If DMC_NTWIST is 0 then the twist angle is not changed during DMC.
After each change of twist angle, the set of configurations needs to be re-equilibrated:
hence a value needs to be specified for DMC_REEQUIL_NSTEP. Setting DMC_NTWIST>0 requires the
use of a complex wave function (COMPLEX_WF : T). Note that the usual keywords define the run
length for a single twist angle, thus the run length is increased by a factor of DMC_NTWIST.
(Note twist-averaging wholly within CASINO can currently be done only for electron(-hole)
fluid phases; for real systems with atoms one needs to couple with an external code to
regenerate the wave function after each twist. The various twistav_xxx scripts in
CASINO/utils/twist can help with this - see the manual.)

### dmc_reequil_nstep

*No of post-twist equil moves* — Integer, Intermediate, default `200` (monte_carlo.f90)

Number of steps to take for a re-equilibration when doing a twist-averaged DMC run.
Currently, a re-equilibration only takes place when the twist angle is changed.
Electron(-hole) fluid phases only. For more details, see the DMC_NTWIST keyword.

### dmc_reequil_nblock

*No. of post-twist equil blocks* — Integer, Intermediate, default `1` (monte_carlo.f90)

Number of blocks in which to divide a re-equilibration when doing a twist-averaged DMC run.
Currently, a re-equilibration only takes place when the twist angle is changed.
Electron(-hole) fluid phases only. For more details, see the DMC_NTWIST keyword.

### dmc_reweight_conf

*Update walker weights read in from config.in* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Weights of walkers are recomputed after reading config.in to correct for a modified wave
function. This allows continuous QMC-MD computations as described in PhysRevLett.94.056403.

### dmc_spacewarping

*Adjust electron coordinates to new wave function* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Electronic positions are adjusted to follow the ionic positions when adapting an existing
population to a new wave function. The method follows the description in PhysRevB.61.R16291
and is typically combined with DMC_REWEIGHT_CONF.

### emin_xi_value

*Emin semi-orthog param* — Double Precision, Expert, default `1.` (monte_carlo.f90)

During energy minimization, this parameter determines the wave function with respect to
which the linear basis of first derivatives is semi-orthogonalized. Please see the manual
for more details.

### ebest_av_window

*Av. window for DMC equil* — Integer, Expert, default `25` (monte_carlo.f90)

During DMC equilibration the best estimate of the ground-state energy is taken to be the
average local energy over the last EBEST_AV_WINDOW moves.

### xc_corr_method

*XC finite size corr algorithm* — Integer, Expert, default `1` (monte_carlo.f90)

If set to 1, the XC finite-size correction will be evaluated by determining the coefficient
of k^2 in the structure factor; if set to 2, the XC correction will be evaluated by fitting
the whole structure factor. Method 1 (the default) is recommended.

### emin_min_energy

*Emin min. E threshold* — Physical, Intermediate, default `0.,'hartree'` (monte_carlo.f90)

The stability of energy minimization can be improved by supplying a threshold below which
energies will be ignored (difficult optimizations can produce spurious too-low energies). If
this keyword is not present in the input file, no threshold will be applied. This is a
keyword of type 'Physical' hence you need to supply units, such as 'ev', 'ry', 'hartree',
'kcal/mol' etc.

### expval_error_bars

*Error bars for exp values* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If this flag is set, CASINO will, where practicable, accumulate the additional quantities
required to evaluate error bars on requested expectation values. This will increase the size
of the expval.data file and slow down the calculation slightly. At present this
functionality is limited to : structure_factor.

### bf_sparse

*Woodbury formula det update* — Boolean, Expert, default `.false.` (monte_carlo.f90)

BF_SPARSE can be used to speed up calculations on large systems by using the Woodbury update
formula instead of recalculating the determinants. Let N=number of electrons per
determinant, and M=number of electrons within the backflow range of a given electron (on
average). The Woodbury update scales as M * N**2, whereas recomputing the determinant scales
as N**3. It is advantageous to turn BF_SPARSE to T when M/N < 1/3 for acceptance ratios of
about 1/2. For calculating non-local energies using the Woodbury formula is advantageous
when M/N < 1/2. Note that when pairing wave functions are used scaling is rather worse than
this. If for a given system M is a constant (backflow range is independent of N), using this
feature makes backflow calculations scale as N**3, like Slater-Jastrow calculations. The
default value for BF_SPARSE is F.

### opt_strict

*Halt opt on apparent divergence* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Setting OPT_STRICT=T will cause CASINO to stop a vmc_opt or opt_vmc run if the VMC energies
are incremented within a 99.7% confidence interval during two consecutive cycles. Intended
for not wasting CPU time in times of scarcity. Default value is F.

### opt_noctf_cycles

*Fix cutoffs for some cycles* — Integer, Expert, default `0` (monte_carlo.f90)

Supplying a positive integer X for this keyword will cause all 'shallow' parameters (cut-off
lengths in the Jastrow factor, backflow transformation and orbitals) to remain fixed for the
final X cycles of a multi-cycle optimization run. This is potentially useful for energy
minimization, which can be adversely affected by the presence of optimizable cut-off
parameters. OPT_NOCTF_CYCLES defaults to 0 (i.e., cut-offs are never fixed).

### mom_den

*Accumulate momentum density* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If set to T the momentum density will be accumulated. Exclusively for HEGs at the moment.

### dtvmcs

*List of VMC time steps* — Block, Expert

Use this keyword to specify a VMC time step for each particle family explicitly, as well as
to determine whether to optimize each of them individually. The contents of this block
override the values of DTVMC and OPT_DTVMC. One line is to be written for each 'family' of
particles, the format of each line being: <dtvmc> <opt_dtvmc> where <dtvmc> is the value of
the time step, and <opt_dtvmc> can be 0 or 1, indicating whether to optimize the
corresponding <dtvmc> or not.

### initial_config

*Initial VMC positions* — Block, Expert

Use this keyword if you want to specify the initial VMC configuration to use instead of the
random one generated by the POINTS routine. It is possible to specify the positions of only
some of the particles. The format of each line in this block is: <spin> <number> <x> <y> <z>
where <spin> is the spin index of the particle, <number> is the index of the particle within
its spin channel, and <x> <y> <z> is the position of the particle.

### dmc_decorr_period

*DMC decorrelation period* — Integer, Intermediate, default `1` (monte_carlo.f90)

Length of the inner decorrelation loop in DMC. The algorithm will perform DMC_DECORR_PERIOD
configuration moves between successive evaluations of expectation values other than the
energy. Setting DMC_DECORR_PERIOD to a value greater than 1 should reduce the serial
correlation of the data. Notice that DMC_DECORR_PERIOD differs from its VMC counterpart in
that in DMC local energies are calculated at intermediate steps (they must), and these
additional values are averaged into the energy data. Therefore, for calculations which do
not require expectation values other than the energy, changing DMC_DECORR_PERIOD from 1 to
some value x is equivalent to multiplying both DMC_[EQUIL|STATS]_NSTEP and DMC_AVE_PERIOD by
x. In a preliminary DMC calculation DMC_DECORR_PERIOD specifies the frequency with which
configurations are written out.

### dmc_ave_period

*Energy-averaging period in DMC* — Integer, Intermediate, default `1` (monte_carlo.f90)

Number of consecutive local energies that are averaged together in DMC before writing them
to the dmc.hist file. The only effect of this keyword is reduce the number of lines in
dmc.hist by a factor of 1/DMC_AVE_PERIOD. Note that if DMC_EQUIL_NSTEP or DMC_STATS_NSTEP
are not divisible by DMC_AVE_PERIOD, they will be rounded up to the nearest integer multiple
of it.

### forces

*Calculate atomic forces in VMC or DMC* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Forces are only implemented for the Gaussian basis set and are only supposed to work with
pseudopotentials in order to eliminate the electron-nucleus singularity. The keyword
forces_info can be chosen to vary the level of output.

### forces_info

*Forces information level* — Integer, Expert, default `2` (monte_carlo.f90)

Controls the amount of information calculated/displayed during calculations: <= 1: displays
no extra information; the Hellmann-Feynman force is evaluated with the d-channel of the
pseudopotential chosen as local and the s-d and p-d channels as nonlocal == 2 (default):
displays ion information during set-up; the Hellmann-Feynman force is evaluated with the
d-channel of the pseudopotential chosen as local and the s-d and p-d channels as nonlocal >=
3: displays ion information during set-up; calculates/displays two additional Hellmann-
Feynman force estimators where the s and p channels of the pseudopotential components are
chosen as local.

### checkpoint

*Checkpointing level* — Integer, Expert, default `1` (monte_carlo.f90)

This integer-valued keyword determines how much CASINO should worry about saving checkpoint
data to config.* files (which can take a significant amount of time, especially with large
systems done on many MPI processes and can reduce the parallel efficiency - since the slower
blocking redistribution algorithm must be used at the end of every block when we write out a
config file). CHECKPOINT can take four values: '2' : save data after every block in both VMC
and DMC, and save the state of the random number generator in OPT runs. '1' [default] : as
'2', but save data in VMC only after the last block when RUNTYPE=vmc_opt, opt_vmc or vmc_dmc
(still after every block if RUNTYPE=vmc). '0' : only save data at the end of the run, for
continuation purposes. This is safe only if used in conjunction with the MAX_CPU_TIME
keyword (since then the config file will be automatically written out if CASINO sees the job
is about to run into an imposed time limit, even if we have not completed the full number of
requested blocks). '-1' : do not write config file at all, ever. Note this value should be
chosen only if you *know* that the job will fit in any imposed time limit, and that such a
run will be long enough to give an acceptably small error bar, since it will be impossible
to subsequently continue the run. CHECKPOINT=0 or -1 clashes with the DMC catastrophe-
recovery facility, for which each DMC block needs to be checkpointed. The value of
CHECKPOINT is thus set to 1 regardless of the input value if DMC_TRIP_WEIGHT > 0 .

### fix_holes

*Choose constraint for BIEX3* — Boolean, Expert, default `.false.` (monte_carlo.f90)

This keyword is used to define the reference points for the exciton-exciton separation when
using BIEX3. Setting fix_holes to T means that the two holes are fixed at a distance xx_sep
apart. The default is F, in which case the centres of mass of the two excitons are fixed
instead. If BIEX3 is not being used this keyword is ignored.

### allow_nochi_atoms

*Permit atoms no chi (etc) terms* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If this keyword is set to T then CASINO will issue a warning message when some atoms are not
included in any sets of chi or f terms in the Jastrow factor and mu and Phi terms in the
backflow function. Otherwise, CASINO halts with an error if some atoms are not included in
these terms.

### psi_s

*Wave function form for Psi_S* — String, Intermediate, default `'slater'` (monte_carlo.f90)

This keyword can take one of the following values: 'none': sets Psi_S to one 'slater': uses
Slater determinants or multi-determinant expansion (default) 'exmol': uses a specially-
crafted wave function for excitonic and positronic molecules 'geminal': uses a single
geminal 'mahan': uses custom wave function for impurity-in-HEG calculations.

### sp_blips

*Blip orbitals single prec coeffs* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Single particle orbitals that appear in the determinants can take a great deal of memory
when expanded in a blip basis. With SP_BLIPS=T CASINO represents the blip coefficients using
single precision real/complex numbers, which will halve the memory required. This parameter
is only relevant when ATOM_BASIS_TYPE=blip. Default value is F.

### write_binary_blips

*Write formatted bwfn.data as binary bwfn.data.bin* — Boolean, Expert, default `.true.` (monte_carlo.f90)

The formatted blip data file bwfn.data can be very large. Consequently reading this file can
be slow. Setting WRITE_BINARY_BLIPS to T will cause CASINO to write the binary file
bwfn.data.bin provided there are no pre-existing bwfn.data.bin files in the run directory.
When CASINO is run again it will first attempt to read bwfn.data.bin rather then read
bwfn.data, so start up will be faster. If one wishes to change the occupied orbitals delete
the existing binary file bwfn.data.bin in the run directory. This parameter is only relevant
when the ATOM_BASIS_TYPE=blip. Default value is T.

### conv_binary_blips

*Convert old bwfn.data.b1 to new bwfn.data.bin.* — Boolean, Expert, default `.false.` (monte_carlo.f90)

In November 2011, a new format binary blip file (bwfn.data.bin) was introduced, which is now
written out by default when CASINO reads in formatted bwfn.data files. The previous binary
format - bwfn.data.b1 - is still supported, not least because at the time of the
introduction of the new format, DFT codes such as PWSCF still produced the old-format b1
file natively (without the intermediate formatted file ever having existed). By default, b1
files are treated exactly as bin files. If the value of CONV_BINARY_BLIPS is set to T, then
after reading in a b1 file, the data will be converted and written out as bwfn.data.bin and
the old b1 file will be deleted, prior to continuing the calculation as normal. This can
save disk space since bin files are generally smaller than b1 files, and they can be read in
somewhat faster (which is advantageous if the same bin file is to be used in multiple
calculations). The bin files are also more portable. Default value is F.

### blip_mpc

*Blip long-range part of MPC* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If BLIP_MPC is set to T and one is using the MPC interaction in a system that is periodic in
all three dimensions and consists of only electrons, then the long-range portion of the MPC
potential will be evaluated using three-dimensional B-splines, 'Blips'. In some systems
setting this to T can greatly speed up the calculation. The default is F.

### hankel_step

*Control step size in hankel transform subroutine* — Double Precision, Expert

If this keyword is set, then it defines the step size (h) that the hankel subroutine uses.
Default is 1.e-9. This is a value which should be set on an integrand-by-integrand basis for
the hankel transform. Current default is reasonable for evaluation of the keldysh_bilayer
manual interaction.

### hankel_eps

*Control an error parameter in hankel transform subroutine* — Double Precision, Expert

If this keyword is set, then it defines the epsilon value (eps) that the hankel subroutine
uses. Default is 1.e-11. This is a value which should be set on an integrand-by-integrand
basis for the hankel transform. Current default is reasonable for evaluation of the
keldysh_bilayer manual interaction.

### small_transfer

*Prevent transfer of large arrays* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If SMALL_TRANSFER is set to T, the DBAR matrices and any potentially large optional data are
not transferred between processes in DMC configuration redistribution. The default is F. Set
to T if you run into problems with parallel transfers.

### opt_small_buffers

*Prevent buffering large arrays in opt* — Boolean, Expert, default `.false.` (monte_carlo.f90)

If OPT_SMALL_BUFFERS is set to T, large intermediate arrays will not be buffered during
optimization. This only affects optimizations involving determinant coefficients in the
absence of backflow.

### opt_plan

*Multi-cycle optimization plan* — Block, Intermediate

This block allows specifying different parameters for each optimization cycle for RUNTYPE =
'vmc_opt', 'opt_vmc' or 'opt'. The block has one line per optimization cycle (the block
length overrides the value of OPT_CYCLES), each containing the cycle index followed by any
number of blank-separated '<keyword>=<value>' assignments. Valid keywords are: * method:
sets OPT_METHOD to <value> for the cycle (string) * ecentre: sets VM_ECENTRE to <value> for
the cycle (string) * reweight: sets VM_REWEIGHT to <value> for the cycle (Boolean) * w_max:
sets VM_W_MAX to <value> for the cycle (real) * w_min: sets VM_W_MIN to <value> for the
cycle (real) * sample_hf: sets VMC_SAMPLE_HF to <value> for the cycle (Boolean) * jastrow:
sets OPT_JASTROW to <value> for the cycle (Boolean) * backflow: sets OPT_BACKFLOW to <value>
for the cycle (Boolean) * det_coeff: sets OPT_DET_COEFF to <value> for the cycle (Boolean) *
orbitals: sets OPT_ORBITALS to <value> for the cycle (Boolean) * geminal: sets OPT_GEMINAL
to <value> for the cycle (Boolean) * maxiter: sets OPT_MAXITER to <value> for the cycle
(Boolean) * fix_cutoffs: determines whether to fix cut-offs (T) or not (F) for the cycle
(Boolean; analogous to OPT_NOCTF_CYCLES) Input keywords will remain at their
provided/default values for all cycles for which they are not modified by the corresponding
OPT_PLAN line.

### rmc_rep_length

*No of configs in reptile* — Integer, Basic, default `-1` (monte_carlo.f90)

Determines the number of configurations in a reptile for reptation.

### dtrmc

*RMC time step* — Double Precision, Basic, default `-1.` (monte_carlo.f90)

Reptation time step.

### rmc_move_length

*No of configs in RMC move* — Integer, Intermediate, default `1` (monte_carlo.f90)

Determines the number of configurations in an RMC move (should be 1 if RMC_BOUNCE=T).
Default is 1.

### rmc_bounce

*Use bounce algorithm for RMC* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

Determines if bounce algorithm is used. Default is T.

### rmc_meas_pos

*Measure electron pos for RMC* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Determines if electron positions are measure in RMC. Default is F.

### rmc_decorr_period

*RMC decorrelation period* — Integer, Intermediate, default `1` (monte_carlo.f90)

Length of the inner decorrelation loop in RMC. The algorithm will perform RMC_DECORR_PERIOD
configuration moves between successive evaluations of the local energy and other expectation
values. Setting RMC_DECORR_PERIOD to a value greater than 1 should reduce the serial
correlation of the data, but notice that the length of the run will be increased.

### rmc_ave_period

*Energy-averaging period in RMC* — Integer, Intermediate, default `1` (monte_carlo.f90)

Number of consecutive local energies that are averaged together in RMC before writing them
to the rmc.hist file. The only effect of this keyword is reduce the number of lines in
rmc.hist by a factor of 1/RMC_AVE_PERIOD. NOTE: THIS FUNCTIONALITY IS NOT YET IMPLEMENTED -
RMC_AVE_PERIOD MUST BE SET TO ONE AT PRESENT.

### rmc_equil_nstep

*No of steps in RMC equil* — Integer, Basic, default `-1` (monte_carlo.f90)

Total number of RMC steps performed in the RMC equilibration stage. Notice that this number
will be rounded up to the nearest multiple of RMC_EQUIL_NBLOCK times RMC_AVE_PERIOD.

### rmc_equil_nblock

*No of blocks in RMC equil* — Integer, Intermediate, default `1` (monte_carlo.f90)

Number of blocks into which the total RMC equilibration run length is divided. The value of
RMC_EQUIL_NBLOCK determines how often the output file is written to. Default: 1.

### rmc_stats_nstep

*Number of steps in RMC stats accum* — Integer, Basic, default `-1` (monte_carlo.f90)

Total number of RMC steps performed in the RMC statistics-accumulation stage. Notice that
this number will be rounded up to the nearest multiple of RMC_STATS_NBLOCK times
RMC_AVE_PERIOD.

### rmc_stats_nblock

*No of blocks in RMC stats accum* — Integer, Intermediate, default `1` (monte_carlo.f90)

Number of blocks into which the total RMC statistics-accumulation run is divided. The value
of RMC_STATS_NBLOCK determines how often the output file is written to. Default: 1.

### dmc_init_eref

*Initial reference energy* — Physical, Expert, default `0.,'hartree'` (monte_carlo.f90)

If set, DMC_INIT_EREF defines the initial reference energy for a DMC calculation. If unset,
the VMC energy is used instead (default). This keyword is ignored if the initial
configurations come from DMC, in which case the previous DMC best estimate of the energy is
used instead. This is a keyword of type 'Physical' hence you need to supply units, such as
'ev', 'ry', 'hartree', 'kcal/mol' etc.

### use_gjastrow

*Use 'gjastrow' Jastrow function* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Use the 'gjastrow' Jastrow factor (T) or the Drummond-Towler-Needs Jastrow factor (F).
CASINO automatically detects the presence of JASTROW blocks in the parameters.casl and
correlation.data files to initialize the value of this keyword. Explicitly setting the value
of USE_GJASTROW is useful when both correlation.data and parameters.casl are present.

### use_gbackflow

*DEV: use gbackflow* — Boolean, Expert, default `.false.` (monte_carlo.f90)

RESERVED KEYWORD FOR DEVELOPMENT, NO EFFECT.

### gen_gjastrow

*Convert to gjastrow* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Setting this flag to T triggers the conversion of a Jastrow factor read from
correlation.data into a gjastrow, which is written to parameters.casl_converted . Renaming
this file to parameters.casl will cause CASINO to use the resulting gjastrow in subsequent
runs. The default value of this keyword is F.

### checkpoint_ncpu

*Num MPI processes checkpt read groups* — Integer, Expert, default `nprocs` (monte_carlo.f90)

This keyword can be used to specify how to group CPUs for reading 'config.in' checkpoint
files. Having many CPUs access the same file at the same time is not a good idea; therefore
we form groups of CHECKPOINT_NCPU MPI processes in which only one of them accesses data. The
default value is the total number of MPI processes (NPROCS internally), but depending on the
hardware you run on you may want to set CHECKPOINT_NCPU to a different value (between 1 and
NPROCS). Note that in the case that NPROCS is not exactly divisible by CHECKPOINT_NCPU, then
the remainder will be distributed over the existing groups, and some of the groups will
therefore contain CHECKPOINT_NCPU+1 MPI processes.

### contact_den

*Accumulate spatial overlap* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If this flag is set then CASINO will accumulate the spatial overlap between electrons and a
positron.

### dtvmc_shift

*VMC transition probability shift* — Double Precision, Expert, default `-1.` (monte_carlo.f90)

DTVMC_SHIFT is an optional shift in the VMC transition probability which can be used to
'encourage' electrons to be more mobile. DTVMC_SHIFT is expressed in units of the square
root of DTVMC.

### dmc_nconf_prelim

*# of configs to generate in prelim DMC calc* — Integer, Expert, default `-1` (monte_carlo.f90)

This is the approximate number of configurations to generate in a preliminary DMC
calculation.

### redist_grp_size

*Number of MPI processes in redist group* — Integer, Expert, default `500` (monte_carlo.f90)

In the branch_and_redist algorithm (which does redistribution of configs across MPI
processes in DMC) we must decide which pairs of processes are involved in config transfers,
and how many configs are to be transferred in each operation. There is an optimal algorithm
for doing this (involving looking at individual config multiplicities and the exact excess
or deficit of configs relative to a target on each MPI process). If we consider *all* the
processes, then this algorithm scales linearly with the number of processes, eventually
becoming so expensive that for a fixed number of configs the code actually becomes slower if
we increase the number of processes. We therefore parallelize the algorithm; to do with this
we form groups of processes ('redist groups') of size REDIST_GRP_SIZE (plus some remainder).
When calculating the vector of instructions, only transfers within these groups are
contemplated, and the cost for working out what to send where no longer increases with the
number of processes (above a certain size).

### allow_slave_write

*Toggle slave write to output* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

The ALLOW_SLAVE_WRITE flag can be used to allow/disallow slave process output to the main
output file. The default is to allow it. The ability to turn this off can be useful when
you''re running on a million cores.

### dmc_md

*DMC molecular dynamics* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If DMC_MD is T then in a DMC calculation we assume we are doing molecular dynamics and that
we are restarting from a converged wave function for a slightly different nuclear
configuration. In practice, all this means is that the number of steps performed are given
by DMCMD_EQUIL_NSTEP and DMCMD_STATS_NSTEP, rather than DMC_EQUIL_NSTEP/DMC_STATS_NSTEP (the
number of blocks is assumed to be 1 in the MD case, and the value of BLOCK_TIME is ignored).
The number of moves necessary will be greatly reduced from the normal case. See also
DMC_REWEIGHT_CONF and DMC_SPACEWARPING. The necessary manipulations are automated by the
runqmcmd script.

### dmcmd_equil_nstep

*No of steps in DMC-MD equil* — Integer, Basic, default `-1` (monte_carlo.f90)

Total number of DMC steps performed in the DMC equilibration stage when we are doing a non-
initial step in a DMC molecular dynamics calculation (we already have a quasi-converged wave
function for a slightly different nuclear configuration). The number of blocks is assumed to
be 1.

### dmcmd_stats_nstep

*No of steps in DMC-MD stats accum* — Integer, Basic, default `-1` (monte_carlo.f90)

Total number of DMC steps performed in the DMC statistics-accumulation stage when we are
doing a non-initial step in a DMC molecular dynamics calculation (we already have a quasi-
converged wave function for a slightly different nuclear configuration). The number of
blocks is assumed to be 1.

### rng_restart_safe

*Continuity of RNG through restart* — Boolean, Expert, default `.true.` (monte_carlo.f90)

We would like e.g. a 1000 move VMC run, and two 500 move VMC runs linked together by a
restart, to give the same answer (in the sense that we end up with the same vmc.hist file).
Unfortunately they do not in general since the pseudorandom number sequence is affected by
the restart. This is because random numbers are generated something like 63 at a time and
stored in a buffer until needed (this buffer being refilled when necessary). In the normal
way of saving a point in the random number sequence, any unused numbers in the buffer are
discarded, which means the final answer will be different to the unrestarted case. If the
keyword RNG_RESTART_SAFE is T (which is actually now the default) then the whole current
buffer is saved in the final config.out file as well as the current state of the random
sequence (necessarily fixed at the end of the current buffer). This allows multiple step
runs to give the same answer as single step runs, at the expense of slightly larger config
files.

### shm_size_nproc

*No of assumed MPI processes in Shm test* — Integer, Expert, default `0` (monte_carlo.f90)

In Shm calculations on Blue Gene machines one needs to know in advance the number of MB of
shared memory required, so that one may set the BG_SHAREDMEMSIZE environment variable (which
can be done by means of the --user.shemsize argument to runqmc). CASINO will calculate this
number and print it to output at the end of the setup process (within the scope of
TESTRUN=T). However, the amount of shared memory required depends on the the number of MPI
processes per node (or per shared memory partition). If one ultimately wishes to run on,
say, half a million cores, it may be desirable to execute a test run on just a few cores on
your personal laptop, rather than waiting a week for the full job to sit in a queue. For the
purposes of computing the size of the shared memory partition, one may therefore set the
number of desired processes/node by setting SHM_SIZE_NPROC, and this value will be used in
computation of the shared memory size rather than the actual number of processes/node being
used in the test run (unless SHM_SIZE_NPROC=0 - which is the default). Note that the CASINO
test run must be done in Shm mode.

### vmc_sampling

*Type of VMC sampling distribution* — String, Expert, default `'standard'` (monte_carlo.f90)

This keyword allows using alternative sampling distributions instead of the square of the
trial wave function in VMC and wave function optimization. Each sampling distribution has
its own set of advantages and disadvantages. Possible values of this keyword are: -
'standard': use the square of the wave function (default). This is the usual way of running
VMC calculations. This form has the drawback of poor sampling near the nodes, which
negatively impacts optimization. This form should be used when generating configurations for
DMC. - 'optimum': use the optimum sampling distribution, which achieves the smallest
possible variance of the local energies. This form enhances sampling near the nodes of the
trial wave function, potentially improving optimization. This form is expensive because it
requires the local energy to be evaluated at every move. With this form you should set the
additional input parameters VMC_OPTIMUM_E0 and VMC_OPTIMUM_EW. - 'HF optimum': use the
optimum sampling distribution for the HF wave function. This does not necessarily reduce the
variance with respect to 'standard', but otherwise offers the same advantages as 'optimum'
at a much reduced cost. Again, you should set the additional input parameters VMC_OPTIMUM_E0
and VMC_OPTIMUM_EW. - 'efficient': use a probability distribution designed to be inexpensive
to evaluate and has just the correct properties to enhance sampling near the nodes. This
form tends to yield the best performance, but it is only applicable to multideterminant wave
functions.

### vmc_optimum_e0

*Centre parameter for optimum VMC sampling* — Physical, Expert, default `0.,'hartree'` (monte_carlo.f90)

This keyword controls the centre parameter used for optimum VMC sampling, which is enabled
by setting VMC_SAMPLING to 'optimum' or 'HF optimum'. By default this is 0.0 hartree. It
should be set to an estimate of the ground-state energy of the system under consideration.
This is a keyword of type 'Physical' hence you need to supply units, such as 'ev', 'ry',
'hartree', 'kcal/mol' etc.

### vmc_optimum_ew

*Width parameter for optimum VMC sampling* — Physical, Expert, default `100.,'hartree'` (monte_carlo.f90)

This keyword controls the width parameter used for optimum VMC sampling, which is enabled by
setting VMC_SAMPLING to 'optimum' or 'HF optimum'. By default this is 100.0 hartree. It
should be set to an estimate of the expected width of the local energy distribution. This is
a keyword of type 'Physical' hence you need to supply units, such as 'ev', 'ry', 'hartree',
'kcal/mol' etc.

### population

*Accumulate ionic populations* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If set to T, ionic populations will be evaluated by Voronoi partitioning of the charge
density.

### block_time

*Approx CPU time per block* — Physical, Intermediate, default `0.,'s'` (monte_carlo.f90)

If BLOCK_TIME is greater than 0.0, then the number of blocks of moves implied by VMC_NBLOCK,
DMC_EQUIL_NBLOCK, or DMC_STATS_NBLOCK will be ignored. Instead, CASINO will do everything it
normally does at the end of a block approximately every BLOCK_TIME seconds of CPU time. For
VMC, the actions performed after a block are: (1) write data to out, vmc.hist, and possibly
expval.data; (2) write current VMC state plus any accumulated configs to config.out (this
latter only if CHECKPOINT is increased to 2 from its default of 1 - otherwise config.out is
only written at the end of the last block). For DMC, the actions performed after a block
are: (1) write data to out, dmc.hist, and possibly expval.data (the latter not during
equilibration); (2) make a backup copy of the config.out/expval.data file (if catastrophe
protection is turned on via DMC_TRIP_WEIGHT); (3) Write the dmc.status file (except after
the last block); (4) Write current state of the system, and all configs in the current
population to config.out (note that by setting CHECKPOINT to 0, this step can be skipped
until the end of the last block, or skipped completely if CHECKPOINT=-1, but this is not the
default). Note the above actions can take a long time, especially if they involve writing to
disk, so it is better to do them as infrequently as possible (i.e. large value of
BLOCK_TIME). Obviously if the stopping criterion (number of moves, target error bar...)
implies that the run will stop before BLOCK_TIME minutes have elapsed, then the total run
time can be shorter than BLOCK_TIME. Note using BLOCK_TIME implies that multiple repetitions
of the same run will not necessarily lead to the same answer in parallel calculations (as
the number of runs done in BLOCK_TIME seconds is defined by the master and what happens on
the other slaves can mess around with their random number sequences in an unpredictable
way). Note finally that BLOCK_TIME is a physical parameter with dimensions of time; the
units must be specified as e.g. 1 day, 24 hr, 1440 min, or 86400 s.

### stop_method

*How to terminate VMC/DMC run* — String, Basic, default `'nstep'` (monte_carlo.f90)

The STOP_METHOD keyword defines how VMC and DMC runs are to be terminated. It may take the
values 'nstep', 'target_error', or 'small_error'. The classic method is 'nstep' which means
simply: perform the number of VMC/DMC steps implied by the input keywords VMC_NSTEP or
DMC_STATS_NSTEP then stop. The error bar then is what it is (it may be too large or smaller
than required). Note that in the VMC case STOP_METHODs other than 'nstep' are used only for
pure VMC calculations i.e. for RUNTYPE='vmc'. The value of STOP_METHOD is implicitly assumed
to be 'nstep' for the VMC stage of optimization or DMC calculations. Any STOP_METHOD other
than 'nstep' requires BLOCK_TIME and STOP_TIME to be set to positive values in the input
file, since the stopping criterion is tested at the end of a block. If STOP_METHOD =
'target_error' then the run will continue until the error bar on the total energy (corrected
on the fly for serial correlation) is approximately equal to that defined by the
TARGET_ERROR input keyword, subject to the constraint that the *estimated* CPU time required
on the master process will not exceed STOP_TIME. CASINO is able to approximately estimate
the required time by analysing how the error bar decreases as a function of the number of
moves, and as soon as it is reasonably confident that the desired target_error is too small
and cannot be reached, then the code will stop (in a restartable condition). On halting in
this manner, an estimate of the CPU time required to get a range of error bars will be
written to the output file. Note that the method used to estimate the required time assumes
the validity of the central limit theorem, which is only approximately valid in most cases.
The criterion is applied at the end of each block and is deliberately conservative, because
stopping at the moment a fluctuating error estimate first dips below the target is not the
same thing as having reached the target: it fires on transients, and when it does not, it
preferentially catches downward fluctuations of the estimate, so that the error bar the run
reports is smaller than the scatter such runs actually have. Three conditions must therefore
hold. (i) The reblocked estimate must be worth believing before it is used at all. CASINO
requires the on-the-fly reblocking to have found a plateau, the standard error on the error
bar to be no more than 15 per cent of the error bar itself (equivalently, at least 23
reblocking blo

### target_error

*Target error bar on the energy* — Double Precision, Basic, default `0.` (monte_carlo.f90)

If STOP_METHOD = 'target_error' then the run will continue until the error bar on the total
energy (corrected on the fly for serial correlation) is approximately equal to that defined
by the TARGET_ERROR keyword, subject to the constraint that the *estimated* CPU time
required on the master process will not exceed STOP_TIME. CASINO is able to approximately
estimate the required time by analysing how the error bar decreases as a function of the
number of moves, and as soon as it is reasonably confident that the desired target_error is
too small and cannot be reached, then the code will stop (in a restartable condition). On
halting in this manner, an estimate of the CPU time required to get a range of error bars
will be written to the output file. Note that the method used to estimate the required time
assumes the validity of the central limit theorem, which is only approximately valid in most
cases. Note also that the code will block unfeasibly small target errors - where
'unfeasible' is currently and arbitrarily defined to be less than 1.d-6 au. The target error
bar is not compared directly with the reblocked error bar. See STOP_METHOD for the validity
gate that must open before the comparison is made at all, for the confidence adjustment that
is applied to the error bar, and for the number of consecutive block ends over which the
result must hold before the run stops.

### stop_time

*Stop method definition of 'reasonable time'* — Physical, Basic, default `0.,'s'` (monte_carlo.f90)

If STOP_METHOD = 'target_error' then the run will continue until the error bar on the total
energy (corrected on the fly for serial correlation) is approximately equal to that defined
by the TARGET_ERROR input keyword, subject to the constraint that the *estimated* CPU time
required on the master process will not exceed STOP_TIME. CASINO is able to approximately
estimate the required time by analysing how the error bar decreases as a function of the
number of moves, and as soon as it is reasonably confident that the desired target_error is
too small and cannot be reached, then the code will stop (in a restartable condition). On
halting in this manner, an estimate of the CPU time required to get a range of error bars
will be written to the output file. Note that the method used to estimate the required time
assumes the validity of the central limit theorem, which is only approximately valid in most
cases. The decision to give up is taken in the same sustained way as the decision to stop.
At the end of each block CASINO projects the total CPU time from the current statistics, and
abandons the run only when the median of the last 5 such projections has exceeded STOP_TIME
at 3 consecutive block ends. A single projection is far too noisy to abandon a run on, since
the reblocking transformation moving to a longer block length can treble it between one
check and the next. The run then stops at a block boundary with config.out written, and the
output file carries a table of the error bars that are reachable within STOP_TIME together
with the moves and CPU time each of them would need. Note that this accounting covers the
CPU time of the current run only. Time consumed by previous restarts of the same calculation
is not recorded in config.in and is not known to the code, so for a restarted calculation
STOP_TIME is effectively a limit on the time of the current segment. Note also that
STOP_TIME is not a hard time limit and cannot act as one. Nothing can be projected until the
validity gate described under STOP_METHOD has opened, so a run whose STOP_TIME is shorter
than the time needed to obtain a trustworthy error bar in the first place will overshoot it
before giving up. Use MAX_CPU_TIME or MAX_REAL_TIME if you need a hard limit. If STOP_METHOD
= 'small_error', CASINO will attempt to make the error bar as small as possible in a
'reasonable time' defined by the value of STOP_TIME. 'As small as possible' mean

### twop_dm_mom

*Accum 2p momentum density* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If TWOP_DM_MOM is set to T, then the Fourier transform of the two-particle density matrix
will be computed. This is only possible if the system is homogeneous for the moment, and
will increase the cost of the calculation significantly.

### cond_fraction_mom

*Accum strict 2p momentum density* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

If COND_FRACTION_MOM is set to T, then an improved estimator of the Fourier transform of the
two-particle density matrix, from which one-body contributions are subtracted, will be
computed. This is only available if the system is homogeneous, for the moment.

### blip_xmul

*Blip grid multiplicity* — Double Precision, Basic, default `2.` (monte_carlo.f90)

Specify the multiplicity for the blip grid in a GEN_BLIP calculation. The default value of 2
is often appropriate.

### blip_nrandpoints

*Number of random points for test of blip orbitals* — Integer, Intermediate, default `0` (monte_carlo.f90)

Number of random points for the Monte Carlo evaluation of the overlap between blip and
plane-wave orbitals in a GEN_BLIP calculation.

### blip_calc_ke

*Evaluate the norm squared and kinetic energy of each blip* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Perform a numerical evaluation of the norm squared and kinetic energy of blip orbitals, to
be compared with the kinetic energies evaluated in a plane-wave basis, in a GEN_BLIP
calculation.

### blip_nband_max

*Maximum number of bands to include in blip transformation* — Block, Expert

This block consists of a single line with a list of NSPIN integers, these being the maximum
numbers of bands to be transformed from a plane-wave basis to a blip basis in a blip-
generation calculation. If a negative integer is given then all bands for that spin will be
re-represented in a blip basis (this is the default behaviour).

### vmc_sample_hf

*Use HF wave function as sampling function in VMC* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Setting VMC_SAMPLE_HF to T causes CASINO to ignore the Jastrow factor during the
accept/reject step, which results in the HF wave function being sampled. The Jastrow factor
is still used to evaluate the local energy, so the reported VMC energy is an estimate of
<E_SJ>_|Psi_HF|^2, which is the reference energy in similarity-transformed FCIQMC
calculations. This keyword lets one run consistency checks between VMC and ST-FCIQMC, and
optimize Jastrow factors by minimizing the spread of E_SJ over |Psi_HF|^2 - set OPT_METHOD
to 'varmin', 'madmin', or 'varmin_linjas' for this.

### nnjas_w_amp

*Amplitude for random default weights in neural network Ja* — Double Precision, Expert, default `1.e-3` (monte_carlo.f90)

Neural network weights that are not specified in the correlation.data file are chosen from a
uniform distribution on the interval [-NNJAS_W_AMP,NNJAS_W_AMP).

### nnjas_b_amp

*Amplitude for random default biases in neural network Jas* — Double Precision, Expert, default `0.` (monte_carlo.f90)

Neural network biases that are not specified in the correlation.data file are chosen from a
uniform distribution on the interval [-NNJAS_B_AMP,NNJAS_B_AMP).

### check_aderivs

*Compare analytical and numerical derivatives w.r.t. w.f.* — Boolean, Expert, default `.false.` (monte_carlo.f90)

Check that analytical derivatives of the wave function and its gradient and Laplacian with
respect to wave function parameters are coded correctly, by comparing with numerical
derivatives. At present, this only applies to pjastrow.f90.

### plot_FT_jastrow

*Plot the Fourier transform of the u+p Jastrow terms* — Boolean, Expert

Plot the Fourier transform of the two-body u and p Jastrow terms against wavevector k to
file ft_of_jastrow.dat. Periodic systems only.

### nequil

*(OLD) Number of equilibration steps* — Integer, Basic, default `5000` (monte_carlo.f90)

NEQUIL is the number of Metropolis equilibration steps. Note that CORPER is not accounted
for, i.e., NEQUIL configuration move attempts are made. [NOTE: THIS KEYWORD IS NOW SEVERELY
DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE
vmc_equil_nstep INSTEAD.]

### nmove

*(OLD) Number of VMC moves* — Integer, Basic, default `-1` (monte_carlo.f90)

Number of moves per block in the main VMC loop, which corresponds to the number of
configurations in a block for which the energy and other expectation values are stored. Note
that there is an inner decorrelation loop of length CORPER, and an additional expectation-
value averaging loop of length NVMCAVE, so the total number of configuration moves attempted
in a block is NMOVE*CORPER*NVMCAVE. If PARALLEL_KEYWORDS is set to 'per_node', then NMOVE is
a per-process quantity, and the total number of moves per block in the main VMC loop is
NMOVE times the number of MPI processes. If PARALLEL_KEYWORDS is 'total', then NMOVE is a
total quantity. [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE
REMOVED (VIA TIME WARP) IN EARLY 2015. USE vmc_nstep INSTEAD.]

### nblock

*(OLD) Number of VMC blocks* — Integer, Basic, default `1` (monte_carlo.f90)

NBLOCK is the total number of blocks of NMOVE moves in a VMC run. NB, you should use the
reblock utility to investigate the effect of varying the block size on the variance after
the calculation is completed. In VMC, NBLOCK just determines how often block averaged
quantities are written to the output file. [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED -
SUPPORT FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE vmc_nblock INSTEAD.]

### corper

*(OLD) VMC energy evaluation period* — Integer, Basic, default `corper_default_vmc` (monte_carlo.f90)

VMC only. The local energy is calculated only once every CORPER configuration moves. If
CORPER is sufficiently large then the VMC energy data are uncorrelated, so that the naive
error bars displayed in the out file are accurate, and the VMC-generated configurations are
uncorrelated, which is very important when performing wave-function optimization or
generating the initial configuration population for a DMC calculation. If one is simply
interested in obtaining a VMC energy then CORPER should be 3 or 4; if one is using VMC to
generate configurations for DMC or variance minimization then CORPER should usually be in
excess of 10 (e.g. 15 is typical). [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT
FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE vmc_decorr_period INSTEAD.]

### nwrcon

*(OLD) Number of configs to write* — Integer, Basic, default `0` (monte_carlo.f90)

NWRCON is the number of configurations to be written out in VMC, for later use (wave-
function optimization or DMC). If PARALLEL_KEYWORDS is set to 'per_node' (the default) then
NWRCON is a per-process quantity, i.e. the total number of configurations written is NWRCON
multiplied by the number of MPI processes. If PARALLEL_KEYWORDS is set to 'total' then
NWRCON is the total number of configurations written. [NOTE: THIS KEYWORD IS NOW SEVERELY
DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE
vmc_nconfig_write INSTEAD.]

### nconfig

*(OLD) DMC target weight* — Double Precision, Basic, default `0.` (monte_carlo.f90)

Target weight in DMC. This is synonymous with "target population", except that NCONFIG is
allowed to be a non-integer. If PARALLEL_KEYWORDS is set to 'per_node' (the default) then
NCONFIG is a per-process quantity, i.e. the total weight is NCONFIG multiplied by the number
of MPI processes. If PARALLEL_KEYWORDS is set to 'total' then NCONFIG is the total target
weight. [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA
TIME WARP) IN EARLY 2015. USE dmc_target_weight INSTEAD.]

### nvmcave

*(OLD) Average successive points* — Integer, Intermediate, default `1` (monte_carlo.f90)

Instead of writing out VMC energies etc every time they are calculated, we average over
NVMCAVE evaluations before writing to the vmc.hist file. Note that the total number of moves
of all electrons in a block is given by NMOVE*CORPER*NVMCAVE so you will need to reduce
NMOVE proportionately if you increase NVMCAVE. [NOTE: THIS KEYWORD IS NOW SEVERELY
DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE
vmc_ave_period INSTEAD.]

### nmove_dmc_equil

*(OLD) Number of moves DMC equil* — Integer, Basic, default `-1` (monte_carlo.f90)

NMOVE_DMC_EQUIL is the number of moves of all electrons in a block during DMC equilibration.
[NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA TIME
WARP) IN EARLY 2015. USE dmc_equil_nstep INSTEAD.]

### nblock_dmc_equil

*(OLD) Number of blocks DMC equil* — Integer, Basic, default `-1` (monte_carlo.f90)

NBLOCK_DMC_EQUIL is the total number of blocks of NMOVE_DMC_EQUIL moves during the DMC
equilibration phase. [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE
REMOVED (VIA TIME WARP) IN EARLY 2015. USE dmc_equil_nblock INSTEAD.]

### nmove_dmc_stats

*(OLD) Number of moves DMC stats accum* — Integer, Basic, default `-1` (monte_carlo.f90)

NMOVE_DMC_STATS is the number of moves of all electrons in a block during the DMC statistics
accumulation phase. [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE
REMOVED (VIA TIME WARP) IN EARLY 2015. USE dmc_stats_nstep INSTEAD.]

### nblock_dmc_stats

*(OLD) Number of blocks DMC stats accum* — Integer, Basic, default `-1` (monte_carlo.f90)

NBLOCK_DMC_STATS is the total number of blocks of NMOVE_DMC_STATS moves during the DMC
statistics accumulation phase. NB, you should use the reblock utility in the utils directory
to see the effect of varying the block size on the variance after the calculation is
completed. [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE REMOVED
(VIA TIME WARP) IN EARLY 2015. USE dmc_stats_nblock INSTEAD.]

### trip_popn

*(OLD) DMC recovery population* — Double Precision, Intermediate, default `0.` (monte_carlo.f90)

In the course of a DMC simulation, it is possible for a configuration "population explosion"
to occur. If TRIP_POPN is set to 0 then nothing will be done about this. If TRIP_POPN>0 then
it will attempt to restart the block if the iteration weight exceeds TRIP_POPN. A general
suggestion for its value would be three times NCONFIG (but see the discussion in the manual
about this). If PARALLEL_KEYWORDS is set to 'per_node', then TRIP_POPN is a per-process
quantity, and the threshold for catastrophe is TRIP_POPN times the number of MPI processes.
If PARALLEL_KEYWORDS is 'total', then TRIP_POPN is a total quantity. [NOTE: THIS KEYWORD IS
NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE
dmc_trip_weight INSTEAD.]

### vmc_twist_av

*(OLD) Perform VMC twist averaging* — Boolean, Intermediate, default `.false.` (monte_carlo.f90)

Perform random changes of twist angle during a VMC simulation. This can only be done for
electron(-hole) fluid phases at present. NEQUIL_TA must be given a positive value in this
case. The k-vector offset is changed at the start of each block in VMC. [NOTE: THIS KEYWORD
IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015.
USE vmc_ntwist INSTEAD.]

### nequil_ta

*(OLD) Number of VMC post-twist-change equilibration moves* — Integer, Intermediate, default `500` (monte_carlo.f90)

Number of equilibration VMC moves to make after each change of k-vector offset. HEG only.
[NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA TIME
WARP) IN EARLY 2015. USE vmc_reequil_nstep INSTEAD.]

### num_dmc_twists

*(OLD) Number of DMC twist angles* — Integer, Intermediate, default `0` (monte_carlo.f90)

Number of different offsets to the grid of k vectors to be applied during DMC statistics
accumulation. HEG only. [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL
BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE dmc_ntwist INSTEAD.]

### nmove_dmct_equil

*(OLD) Number of DMC moves per block during post-twist-cha* — Integer, Intermediate, default `40` (monte_carlo.f90)

Number of equilibration DMC moves per block to make after each change of k-vector offset.
Electron(-hole) fluid phases only. [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT
FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE dmc_reequil_nstep INSTEAD.]

### nblock_dmct_equil

*(OLD) Number of blocks of DMC moves during post-twist-cha* — Integer, Intermediate, default `4` (monte_carlo.f90)

Number of blocks of equilibration DMC moves after each change of k-vector offset. HEG only.
[NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA TIME
WARP) IN EARLY 2015. USE dmc_reequil_nblock INSTEAD.]

### corper_dmc

*(OLD) DMC correlation period* — Integer, Intermediate, default `1` (monte_carlo.f90)

When gathering expectation values in DMC, it is inefficient to compute the expectation
values at every iteration. Instead one can calculate the expectation values every
CORPER_DMCth iteration. Note that, unlike its VMC counterpart, CORPER_DMC does not affect
the number of moves carried out. In a preliminary DMC calculation, CORPER_DMC specifies the
frequency with which configurations are written out. [NOTE: THIS KEYWORD IS NOW SEVERELY
DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE
dmc_decorr_period INSTEAD.]

### ndmcave

*(OLD) Average successive points in DMC* — Integer, Intermediate, default `1` (monte_carlo.f90)

Instead of writing out DMC energies etc every time they are calculated, we average over
NDMCAVE evaluations before writing to the dmc.hist file. Note that the total number of moves
of all electrons in a block is given by NMOVE_DMC_[EQUIL,STATS]*NDMCAVE so you will need to
reduce NMOVE_DMC_[EQUIL,STATS] proportionately if you increase NDMCAVE. [NOTE: THIS KEYWORD
IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015.
USE dmc_ave_period INSTEAD.]

### nmove_dmcmd_equil

*(OLD) Number of moves DMC-MD equil* — Integer, Basic, default `-1` (monte_carlo.f90)

NMOVE_DMC_EQUIL_MD is the number of moves of all electrons in a block during DMC
equilibration, when we are doing a non-initial step in a DMC molecular dynamics calculation
(i.e. we already have a quasi-converged wave function for a slightly different nuclear
configuration). [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT WILL BE
REMOVED (VIA TIME WARP) IN EARLY 2015. USE dmcmd_equil_nstep INSTEAD.]

### nmove_dmcmd_stats

*(OLD) Number of moves DMC-MD stats accum* — Integer, Basic, default `-1` (monte_carlo.f90)

NMOVE_DMC_STATS_MD is the number of moves of all electrons in a block during the DMC
statistics accumulation phase, when we are doing a non-initial step in a DMC molecular
dynamics calculation (i.e. we already have a quasi-converged wave function for a slightly
different nuclear configuration). [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT
FOR IT WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE dmcmd_stats_nstep INSTEAD.]

### nconfig_prelim

*# of configs to generate in prelim DMC calc* — Integer, Expert, default `-1` (monte_carlo.f90)

This is the approximate number of configurations per MPI process to generate in a
preliminary DMC calculation. [NOTE: THIS KEYWORD IS NOW SEVERELY DEPRECATED - SUPPORT FOR IT
WILL BE REMOVED (VIA TIME WARP) IN EARLY 2015. USE dmc_nconf_prelim INSTEAD.]

### emin_var_prefactor

*Emin target function variance prefactor* — Double Precision, Expert, default `-1.` (monte_carlo.f90)

The target function during the line minimization stage of energy minimization is by default
energy + 3*error. Setting EMIN_VAR_PREFACTOR>0.0 causes the target function energy +
EMIN_VAR_PREFACTOR*sqrt(variance) to be used instead.

### dmc_trim_pop_vmc

*Trim initial DMC population to target weight* — Boolean, Expert, default `.true.` (monte_carlo.f90)

When VMC configurations are loaded in DMC, this keyword determines whether to ignore
configurations beyond the target weight. This is a particularly useful thing to do in high-
process-count runs since VMC configurations get rounded up to a multiple of the number of
processes, and loading all of them may increase the equilibration time significantly. The
default is true.

### vm_ecentre

*Choice of central energy in varmin/madmin* — String, Expert, default `'default'` (monte_carlo.f90)

VM_ECENTRE can be set to 'mean', 'median', or 'guess', triggering the use of the
(reweighted/unreweighted) mean local energy, the median local energy, or the value of
VM_E_GUESS, respectively, as the centre with respect to which the target function in varmin
and madmin is computed. By default VM_ECENTRE is set to 'default', which translates to
'mean' for OPT_METHOD='varmin' and to 'median' for OPT_METHOD='madmin'.

### emin_opt_variance

*Use variance at target function in EMIN* — Boolean, Expert, default `.false.` (monte_carlo.f90)

The linear-least-squares (LLS) optimizer in EMIN uses the VMC variance as the target
function when EMIN_OPT_VARIANCE is set to T (the default is F). (Note that we use 'EMIN' to
refer to our LLS optimizer implementation but this is obviously a misnomer in this case.).

### single_precision_blips

*REDUNDANT Blip orbital single prec coeffs* — Boolean, Expert, default `.false.` (monte_carlo.f90)

SINGLE_PRECISION_BLIPS is redundant. Use SP_BLIPS instead.

### btype

*REDUNDANT: Basis set type* — Integer, Basic, default `-1` (monte_carlo.f90)

BTYPE is redundant. Use ATOM_BASIS_TYPE instead.

### special_wfn

*REDUNDANT: Special wave function* — String, Expert, default `'none'` (monte_carlo.f90)

SPECIAL_WFN is redundant. Use ATOM_BASIS_TYPE instead.

### iterac

*REDUNDANT: ee interaction type* — Integer, Intermediate, default `-1` (monte_carlo.f90)

ITERAC is redundant. Use INTERACTION instead.

### no_ee_int

*REDUNDANT: Turn off e-e interaction* — Boolean, Expert, default `.false.` (monte_carlo.f90)

NO_EE_INT is redundant. Use INTERACTION instead.

### nlrule1

*REDUNDANT: NL int rule (VMC/DMC)* — Integer, Intermediate, default `-1` (monte_carlo.f90)

NLRULE1 is redundant. Use NON_LOCAL_GRID instead.

### iaccumulate

*REDUNDANT: DMC stage* — Boolean, Intermediate, default `.true.` (monte_carlo.f90)

IACCUMULATE is redundant. Use RUNTYPE = dmc_equil or dmc_stats instead.

### use_molorbmods

*REDUNDANT: Molecular-orbital mods* — Boolean, Basic, default `.false.` (monte_carlo.f90)

USE_MOLORBMODS is redundant. Use USE_ORBMODS instead.

### nlrule2

*REDUNDANT: NL int rule (configs)* — Integer, Intermediate

NLRULE2 is redundant and its value is ignored. See NON_LOCAL_GRID.

### calc_variance

*REDUNDANT: Calculate VMC variance* — Boolean, Intermediate

CALC_VARIANCE is redundant and its value is ignored.

### vm_deriv_buffer

*REDUNDANT: Buffer WF sections* — Boolean, Intermediate

VM_DERIV_BUFFER is redundant and its value is ignored.

### vm_dist_buffer

*REDUNDANT: Buffer distances* — Boolean, Expert

VM_DIST_BUFFER is redundant and its value is ignored.

### bf_save_memory

*REDUNDANT: Disable backflow buffers* — Boolean, Expert

BF_SAVE_MEMORY is redundant and its value is ignored.

### emin_sampling

*REDUNDANT: Type of sampling in EMIN* — String, Expert

EMIN_SAMPLING is redundant and its value is ignored.

### dmc_npops

*REDUNDANT: Number of independent populations in DMC* — Integer, Expert, default `1` (monte_carlo.f90)

Removed because it doesn''t help.

### spin_density_mat

*REDUNDANT: Keyword* — Boolean, Intermediate

SPINDENSITYMAT is redundant. Implied by DENSITY/SPIN_DENSITY=T in non-collinear system.

### redist_period

*REDUNDANT: Redistribution period* — Integer, Expert

REDIST_PERIOD is redundant and its value is ignored.

### num_cpus_in_group

*REDUNDANT: Num CPUs to share blip orbs* — Integer, Intermediate

NUM_CPUS_IN_GROUP is redundant and its value is ignored.

### use_mpiio

*REDUNDANT: MPI IO to r/w binary blip file* — Boolean, Expert

USE_MPIIO is redundant and its value is ignored.

### have_ae

*REDUNDANT: System contains all-electron nuclei* — Boolean, Basic

HAVE_AE is redundant and its value is ignored.

### allow_ae_ppots

*REDUNDANT: Allow mixed ae/pp* — Boolean, Intermediate

ALLOW_AE_PPOTS is redundant and its value is ignored.

### parallel_keywords

*REDUNDANT: Parallel keyword interpretation* — String, Expert, default `'per_node'` (monte_carlo.f90)

When PARALLEL_KEYWORDS was set to 'per_node' [default], keywords NMOVE, NWRCON, NCONFIG and
TRIP_POPN were interpreted as numbers per process, as in previous versions of CASINO. When
PARALLEL_KEYWORDS was set to 'total', the value of these keywords was divided by the number
of MPI processes, so that they represent total quantities. This was an initial attempt at
providing simplified input for parallel runs. However, the preferred approach is now to
replace: * NMOVE, NWRCON, CORPER, NVMCAVE, NBLOCK, NEQUIL, NCONFIG, NMOVE_DMC_[EQUIL|STATS],
NBLOCK_DMC_[EQUIL|STATS], NDMCAVE, CORPER_DMC, TRIP_POPN, VMC_TWIST_AV, NEQUIL_TA,
NUM_DMC_TWISTS, NMOVE_DMCT_EQUIL, and NBLOCK_DMCT_EQUIL with * VMC_NSTEP, VMC_NCONFIG_WRITE,
VMC_DECORR_PERIOD, VMC_AVE_PERIOD, VMC_NBLOCK, VMC_EQUIL_NSTEP, DMC_TARGET_WEIGHT,
DMC_[EQUIL|STATS]_NSTEP, DMC_[EQUIL|STATS]_NBLOCK, DMC_AVE_PERIOD, DMC_DECORR_PERIOD,
DMC_TRIP_WEIGHT, VMC_NTWIST, VMC_REEQUIL_NSTEP, DMC_NTWIST, DMC_REEQUIL_NSTEP, and
DMC_REEQUIL_NBLOCK, respectively, to achieve the same effect (but notice the slightly
different meanings of some of the keywords in the two sets!).

### movienode

*REDUNDANT: MPI proc to make movie* — Integer, Intermediate, default `-1` (monte_carlo.f90)

Use MOVIEPROC instead.

### pN_cusp

*REDUNDANT Impose positron-nucleus cusp* — Integer, Expert

Simply use the "impose particle-nucleus cusp" flag in the PJASTROW block of
correlation.data.

### vm_use_E_guess

*REDUNDANT Use guess of GS energy* — Boolean, Expert

Redundant - set VM_ECENTRE to 'guess' instead.

### emin_auto_varmin

*REDUNDANT Min variance in 1st emin cycle* — Boolean, Expert

Redundant - use OPT_PLAN instead.
