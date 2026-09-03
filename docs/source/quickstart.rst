.. _quickstart:

:tocdepth: 2

Quick start
===========

This page shows how to run a complete VMC calculation for a helium atom, from input files to
energy output.

Prerequisites
-------------

Install Pycasino and verify it works::

    $ pipx install casino
    $ casino --help

You will also need a wavefunction file (``gwfn.data`` or ``stowfn.data``) produced by a quantum
chemistry program such as `ORCA <https://www.faccts.de/orca/>`_ or
`ADF <https://www.scm.com/product/adf/>`_.

Example: VMC energy for helium
-------------------------------

The ``examples/gwfn/He/HF/cc-pVQZ/CBCS/Jastrow/`` directory in the repository contains a
ready-to-run example. Below is the minimal ``input`` file for a VMC energy calculation:

.. code-block:: text

    #-------------------#
    # CASINO input file #
    #-------------------#

    # He atom (ground state)

    # SYSTEM
    neu               : 1              # number of up electrons
    ned               : 1              # number of down electrons
    atom_basis_type   : gaussian       # gaussian or slater-type

    # RUN
    runtype           : vmc            # vmc, vmc_dmc, or vmc_opt
    testrun           : F

    # VMC
    vmc_method        : 3              # 3 = CBCS (recommended)
    vmc_equil_nstep   : 5000           # equilibration steps
    vmc_nstep         : 1000000        # production steps
    vmc_nblock        : 10             # blocks for error estimation
    vmc_nconfig_write : 0
    vmc_decorr_period : 1

The directory must also contain ``gwfn.data`` (the wavefunction) and optionally
``correlation.data`` (Jastrow parameters).

Running the calculation
-----------------------

From the directory containing ``input`` and ``gwfn.data``::

    $ casino .

With MPI parallelisation across 4 processes::

    $ mpirun -n 4 casino .

The program prints block-by-block energy estimates and a final mean ± standard error::

    ====================================
     VMC energy (au)
    ------------------------------------
     Mean                = -2.903485
     Standard error      =  0.000031
    ====================================

Workflow: optimization then DMC
---------------------------------

A typical production workflow has three stages:

1. **Jastrow optimization** (``runtype : vmc_opt``) — minimise variance or energy to find
   optimal Jastrow parameters.
2. **VMC energy accumulation** (``runtype : vmc``) — collect a large sample with the optimised
   wavefunction.
3. **DMC** (``runtype : vmc_dmc``) — diffusion Monte Carlo for higher accuracy, starting from
   VMC configurations (requires ``vmc_nconfig_write > 0`` in the VMC run).

See :ref:`config` for a full description of all input keywords and :ref:`tutorial` for the
underlying mathematics of each wavefunction component.
