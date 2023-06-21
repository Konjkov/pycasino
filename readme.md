
The Pycasino program implements some of the methods from the well-known Casino program.
(available on site https://vallico.net/casinoqmc/)

1. supported orbital file formats:gwfn.data, stowfn.data
2. using multi-determinant expansions
3. capabal of doing 3-term Jastrow and 3-term Backflow factors
4. support only configuration-by-configuration sampling (CBCS)
5. partial Ma CUSP correction (no rc optimization yet)
6. use only MPI parallelization
7. support VMC and DMC energy calculation, varmin and emin optimization

list of supported keywords in input file:
1. NEU, NED Number of electrons of up and down spin
2. ATOM_BASIS_TYPE The type of orbitals to be used: ‘gaussian’, ‘slater-type‘
3. RUNTYPE Type of QMC calculation: ‘vmc’, ‘vmc_dmc’, ‘vmc_opt’
4. VMC_EQUIL_NSTEP Number of equilibration steps
5. VMC_NSTEP_Number of VMC energy-evaluation steps
6. VMC_DECORR_PERIOD Number of steps between VMC energy-evaluation moves.
7. VMC_NCONFIG_WRITE Number of VMC configurations stored for later use in DMC or
optimization
8. VMC_NBLOCK number of blocks into which the total VMC run is divided post-equilibration
9. OPT_CYCLES Number of optimization+VMC cycles to perform.
10. OPT_METHOD Optimization method to use: ‘varmin’, ‘emin’
11. DMC_TARGET WEIGHT Target number of configurations in DMC
12. DMC_EQUIL NSTEP Number of DMC steps in equilibration
13. DMC_STATS NSTEP Number of DMC steps in statistics accumulation
14. DTDMC DMC time step
