.. _config:

Supported configuration files and their contents
================================================

Pycasino can read input files in well-known `Casino <https://vallico.net/casinoqmc/>`_ format.

Input files are supported
-------------------------

- **input** is the main input parameter file.
- **correlation.data** file contains all optimizable parameters together with accompanying data (for example, the parameters used to define a Jastrow factor or backflow function).
- **gwfn.data** file contains the data that defines the geometry, the gaussian-type orbitals and, if appropriate, the determinant expansion coefficients.
- **stowfn.data** file contains the data that defines the geometry and the slater-type orbitals.
- **x_pp.data** (where x is the chemical symbol of an element in lower-case letters.) This file contains the pseudopotential data for the corresponding element.

Input parameter are supported
-----------------------------

General keywords
~~~~~~~~~~~~~~~~

- **NEU**, **NED** - number of electrons of *up* and *down* spin
- **RUNTYPE** - type of QMC calculation: *vmc*, *vmc_dmc*, *vmc_opt*
- **TESTRUN** - if this flag is T then read input files, print information and stop
- **ATOM_BASIS_TYPE** - the type of orbitals to be used: *gaussian*, *slater-type*

VMC keywords
~~~~~~~~~~~~

- **VMC_EQUIL_NSTEP** - number of equilibration steps
- **VMC_NSTEP** - number of VMC energy-evaluation steps
- **VMC_DECORR_PERIOD** - number of steps between VMC energy-evaluation moves
- **VMC_NCONFIG_WRITE** - number of VMC configurations stored for later use in DMC or optimization
- **VMC_NBLOCK** - number of blocks into which the total VMC run is divided post-equilibration
- **DTVMC** - VMC time step (size of trial steps in random walk)
- **VMC_METHOD** - (1) - EBES (work in progress), (3) - CBCS.

Optimization keywords
~~~~~~~~~~~~~~~~~~~~~

- **OPT_CYCLES** - number of optimization VMC cycles to perform
- **OPT_METHOD** - optimization method to use: *varmin*, *emin*
- **OPT_JASTROW** - optimize the Jastrow factor in wave-function optimization
- **OPT_BACKFLOW** - optimize backflow parameters in wave-function optimization
- **OPT_DET_COEFF** - optimize the coefficients of the determinants in wave-function optimization
- **OPT_MAXEVAL** - maximum number of evaluations of the variance during variance minimization (default 50)
- **OPT_PLAN** - allows specifying different parameters for each optimization cycle
- **VM_REWEIGHT** - if set then the reweighted variance-minimization algorithm will be used, else the unreweighted algorithm will be used Unreweighted variance minimization is recommended
- **EMIN_METHOD** - energy minimization method to use: *newton*, *linear* (default), *reconf*

DMC keywords
~~~~~~~~~~~~

- **DMC_TARGET_WEIGHT** - target number of configurations in DMC
- **DMC_EQUIL_NSTEP** - number of DMC steps in equilibration
- **DMC_STATS_NSTEP** - number of DMC steps in statistics accumulation
- **DMC EQUIL NBLOCK** - number of blocks into which the DMC equilibration phase is divided
- **DMC STATS NBLOCK** - number of blocks into which the DMC statistics accumulation phase is divided
- **DTDMC** - DMC time step
- **DMC_METHOD** - (1) - EBES, (2) - CBCS
- **LIMDMC** - set modifications to Green’s function in DMC. Only (4) Umrigar mods to drift velocity, Zen–Sorella–Alfè mods to energy
- **ALIMIT** - parameter required by DMC drift-velocity- and energy-limiting schemes
- **NUCLEUS_GF_MODS** - this keyword is the switch for enabling the use of the modifications to the DMC Green’s function for the presence of bare nuclei
- **EBEST_AV_WINDOW** - averaging window for calculating the ground-state energy during equilibration (work in progress)

WFN definition keywords
~~~~~~~~~~~~~~~~~~~~~~~

- **BACKFLOW** - turns on backflow corrections. Backflow parameters are read from correlation.data
- **USE_JASTROW** - use a wave function of the Slater-Jastrow form. The Jastrow factor is read from correlation.data
- **USE_GJASTROW** - use gjastrow Jastrow factor. This Jastrow factor is defined in a parameters.casl file (work in progress)

Cusp correction keywords
~~~~~~~~~~~~~~~~~~~~~~~~

- **CUSP_CORRECTION** - when the cusp correction flag is activated, the s-type Gaussian basis functions centred on each atom are replaced within a small sphere by a function which ensures that the electron–nucleus cusp condition is obeyed
- **CUSP_INFO** - if set then information about how cusp correction is done will be printed to the log-file
- **CUSP_THRESHOLD** - if the magnitude of the s-component of a Gaussian orbital is less than this threshold then it will not be cusp corrected

Pseudopotential keywords
~~~~~~~~~~~~~~~~~~~~~~~~

- **NON_LOCAL_GRID** - selects the grid for nonlocal integration, can take values between 1 and 7, the default being 4
- **LCUTOFFTOL** - this is used to define the cutoff radius for the local part of the pseudopotential, the default being 10\ :sup:`-5`
- **NLCUTOFFTOL** - this is used to define the cutoff radius for the nonlocal parts of the pseudopotential, the default being 10\ :sup:`-5`

Configs read/write
------------------

Config loader is represented by the :class:`casino.readers.CasinoConfig` class.
The :class:`~casino.readers.CasinoConfig` instance represents the relevant information in various attributes:

- config.input - data from input file
- config.wfn - data from gwfn.data or stowfn.data files
- config.mdet - multideterminant data from correlation.data files
- config.jastrow - jastrow related data from correlation.data files
- config.backflow - backflow related data from correlation.data files

Config files can be read::

    from casino.readers import CasinoConfig

    config_path = <path to a directory containing input file>
    config = CasinoConfig(config_path)
    config.read()

modified::

    config.title = 'New config title'
    config.mdet.title = 'New mdet section title'
    config.jastrow.title = 'New jastrow section title'
    config.backflow.title = 'New backflow section title'

and written to a new destination::

    config.write('.', version=0)
