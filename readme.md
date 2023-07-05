# Quantum monte carlo package

The Pycasino program implements some of the methods from the well-known [Casino](https://vallico.net/casinoqmc/) program.

## Basic functionality

1. supported orbital file formats:gwfn.data, stowfn.data
2. using multi-determinant expansions
3. capable of doing 3-term Jastrow factor and 3-term Backflow
4. configuration-by-configuration sampling (CBCS) and electron-by-electron sampling (EBES)
5. partial Ma CUSP correction (no rc optimization yet)
6. use only MPI parallelization
7. support VMC and DMC energy calculation, varmin and emin optimization

## List of supported keywords in input file

### General keywords:
* **NEU**, **NED** Number of electrons of up and down spin
* **RUNTYPE** Type of QMC calculation: ‘vmc’, ‘vmc_dmc’, ‘vmc_opt’
* **ATOM_BASIS_TYPE** The type of orbitals to be used: ‘gaussian’, ‘slater-type‘

### VMC keywords:
* **VMC_EQUIL_NSTEP** Number of equilibration steps
* **VMC_NSTEP** Number of VMC energy-evaluation steps
* **VMC_DECORR_PERIOD** Number of steps between VMC energy-evaluation moves.
* **VMC_NCONFIG_WRITE** Number of VMC configurations stored for later use in DMC or optimization
* **VMC_NBLOCK** number of blocks into which the total VMC run is divided post-equilibration
* **DTVMC** VMC time step (size of trial steps in random walk)
* **VMC_METHOD** (1) - EBES (work in progress), (3) - CBCS.

### Optimization keywords:
* **OPT_CYCLES** Number of optimization VMC cycles to perform.
* **OPT_METHOD** Optimization method to use: ‘varmin’, ‘emin’
* **OPT_JASTROW** Optimize the Jastrow factor in wave-function optimization.
* **OPT_BACKFLOW** Optimize backflow parameters in wave-function optimization.
* **OPT_DET_COEFF** Optimize the coefficients of the determinants in wave-function optimization.
* **OPT_MAXEVAL** Maximum number of evaluations of the variance during variance minimization (default 50).
* **VM_REWEIGHT** If set then the reweighted variance-minimization algorithm will be used, else the unreweighted algorithm will be used.
Unreweighted variance minimization is recommended.

### DMC keywords:
* **DMC_TARGET_WEIGHT** Target number of configurations in DMC
* **DMC_EQUIL_NSTEP** Number of DMC steps in equilibration
* **DMC_STATS_NSTEP** Number of DMC steps in statistics accumulation
* **DMC EQUIL NBLOCK** Number of blocks into which the DMC equilibration phase is divided
* **DMC STATS NBLOCK** Number of blocks into which the DMC statistics accumulation phase is divided.
* **DTDMC** DMC time step
* **DMC_METHOD** (1) - EBES (work in progress), (2) - CBCS.
* **LIMDMC** Set modifications to Green’s function in DMC. Only (4) Umrigar mods to drift velocity, Zen–Sorella–Alfè mods to energy
* **ALIMIT** Parameter required by DMC drift-velocity- and energy-limiting schemes.
* **NUCLEUS_GF_MODS** This keyword is the switch for enabling the use of the modifications to the DMC Green’s function for the presence of bare nuclei
* **EBEST_AV_WINDOW** Averaging window for calculating the ground-state energy during equilibration (work in progress).

### WFN definition keywords:
* **BACKFLOW** Turns on backflow corrections. Backflow parameters are read from correlation.data
* **USE_JASTROW**  Use a wave function of the Slater-Jastrow form. The Jastrow factor is read from the ‘JASTROW’ block in correlation.data
* **USE_GJASTROW** Use gjastrow Jastrow factor. This Jastrow factor is defined in a parameters.casl file (work in progress).

### Cusp correction keywords:
* **CUSP_CORRECTION**  When the cusp correction flag is activated, the s-type Gaussian basis functions centred on each atom are replaced
within a small sphere by a function which ensures that the electron–nucleus cusp condition is obeyed.
* **CUSP_INFO** If set then information about precisely how this is done will be printed to the log-file.
* **CUSP_THRESHOLD** If the magnitude of the s-component of a Gaussian orbital is less than this threshold then it will not be cusp corrected.
