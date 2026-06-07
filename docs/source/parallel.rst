.. _parallel:

:tocdepth: 2

MPI parallelisation
===================

Pycasino uses `mpi4py <https://mpi4py.readthedocs.io/>`_ to distribute VMC and DMC workloads
across multiple processes via the Message Passing Interface (MPI) standard.

Installation
------------

Before installing mpi4py, install the system MPI library::

    $ sudo apt install libopenmpi-dev   # Debian/Ubuntu
    $ brew install open-mpi             # macOS

mpi4py is installed automatically with Pycasino, but requires the system library to be present
at install time.

Running with MPI
----------------

Pass the number of processes to ``mpirun`` (or ``mpiexec``)::

    $ mpirun -n 4 casino /path/to/input/directory

Each MPI rank runs an independent Markov chain starting from a different random initial
electron configuration. The ranks synchronise after every VMC block to accumulate statistics
and to broadcast optimised parameters.

.. note::

    The total number of VMC steps is divided evenly among ranks. Setting ``vmc_nstep : 1000000``
    with 4 ranks means each rank performs 250 000 steps. The statistical error therefore scales
    as :math:`1/\sqrt{N_\text{rank} \times N_\text{step}}`.

Memory model
------------

Pycasino uses ``MPI.Win.Allocate_shared`` to create a shared-memory energy buffer on each
compute node. This means:

- Ranks on the **same node** read each other's energy arrays directly from shared memory —
  no network transfer.
- Ranks on **different nodes** exchange data through the normal MPI network layer after each
  block (``MPI.Barrier`` at block boundaries).

This design works well when ``vmc_nblock`` is large enough so that the synchronisation overhead
is small compared to the per-block compute time.

Choosing the number of ranks
----------------------------

A practical starting point is one rank per physical CPU core. MPI parallelisation in Pycasino
is purely statistical (independent walkers), so efficiency does not degrade with large rank
counts as long as each rank accumulates enough steps for meaningful statistics.

For **wavefunction optimisation** (``runtype : vmc_opt``), the parameter broadcast uses
``mpi_comm.bcast`` so all ranks always start the next optimisation cycle from identical
parameters.

Combining MPI with Numba threading
-----------------------------------

Pycasino's inner loops are compiled with Numba (``@nb.njit``). The JIT kernels currently use
``parallel=False``, so each rank occupies one CPU core. If you want to use fewer MPI ranks and
let Numba use multiple threads per rank, set the ``NUMBA_NUM_THREADS`` environment variable::

    $ NUMBA_NUM_THREADS=4 mpirun -n 2 casino .

This runs 2 MPI walkers each using up to 4 Numba threads, for a total of 8 cores. Whether this
is faster than 8 single-threaded ranks depends on the system size.
