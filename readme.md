# Pycasino python quantum package

The Pycasino program implements some of the methods from the well-known Casino program.
(available on site https://vallico.net/casinoqmc/)

1. supported orbital file formats:gwfn.data, stowfn.data
2. using multi-determinant expansions
3. capable of doing 3-term Jastrow factor and 3-term Backflow
4. support only configuration-by-configuration sampling (CBCS)
5. partial Ma CUSP correction (no rc optimization yet)
6. use only MPI parallelization
7. support VMC and DMC energy calculation, varmin and emin optimization

List of supported keywords in input file:
General (system-dependent) keywords:
* NEU, NED Number of electrons of up and down spin
* RUNTYPE Type of QMC calculation: ‘vmc’, ‘vmc_dmc’, ‘vmc_opt’
* ATOM_BASIS_TYPE The type of orbitals to be used: ‘gaussian’, ‘slater-type‘

Important VMC keywords:
* VMC_EQUIL_NSTEP Number of equilibration steps
* VMC_NSTEP Number of VMC energy-evaluation steps
* VMC_DECORR_PERIOD Number of steps between VMC energy-evaluation moves.
* VMC_NCONFIG_WRITE Number of VMC configurations stored for later use in DMC or optimization
* VMC_NBLOCK number of blocks into which the total VMC run is divided post-equilibration
* DTVMC VMC time step (size of trial steps in random walk)
* VMC_METHOD (1) - EBES (work in progress), (3) - CBCS.

Important optimization keywords:
* OPT_CYCLES Number of optimization+VMC cycles to perform.
* OPT_METHOD Optimization method to use: ‘varmin’, ‘emin’
* OPT_JASTROW Optimize the Jastrow factor in wave-function optimization.
* OPT_BACKFLOW Optimize backflow parameters in wave-function optimization.
* OPT_DET_COEFF Optimize the coefficients of the determinants in wave-function optimization.
* OPT_CYCLES Number of cycles of configuration generation and optimization to be carried out if runtype=‘vmc_opt’
* VM_REWEIGHT If set then the reweighted variance-minimization algorithm will be used, else the unreweighted algorithm will be used.
Unreweighted variance minimization is recommended.

Important DMC keywords:
* DMC_TARGET WEIGHT Target number of configurations in DMC
* DMC_EQUIL NSTEP Number of DMC steps in equilibration
* DMC_STATS NSTEP Number of DMC steps in statistics accumulation
* DTDMC DMC time step
* DMC_METHOD (1) - EBES (work in progress), (2) - CBCS.
* LIMDMC Set modifications to Green’s function in DMC. Only (4) Umrigar mods to drift velocity, Zen–Sorella–Alfè mods to energy
* ALIMIT Parameter required by DMC drift-velocity- and energy-limiting schemes.
* NUCLEUS_GF_MODS This keyword is the switch for enabling the use of the modifications to the DMC Green’s function for the presence of bare nuclei

WFN definition keywords:
* BACKFLOW Turns on backflow corrections (see Sec. 23). Backflow parameters are read from correlation.data
* USE_JASTROW  Use a wave function of the Slater-Jastrow form, where the Jastrow factor exp(J)
is an optimizable object that multiplies the determinant part in order to introduce correlations in the system.
The Jastrow factor is read from the ‘JASTROW’ block in correlation.data
* USE_GJASTROW If set to T, the gjastrow Jastrow factor will be used. This Jastrow factor is defined in a parameters.casl file (work in progress).

Cusp correction keywords:
* CUSP_CORRECTION When expanded in a basis set of Gaussian functions, the electron–nucleus cusp that should be present
in all-electron calculations is not represented correctly.
* CUSP_INFO If set then information about precisely how this is done will be printed to the log-file.
* CUSP_THRESHOLD If the magnitude of the s component of a Gaussian orbital is less than this threshold then it will not be cusp corrected.
