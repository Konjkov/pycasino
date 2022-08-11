#!/usr/bin/env python3

import os
from timeit import default_timer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import numba as nb
from numba.core.runtime import rtsys

from logger import logging
from casino import Casino


logger = logging.getLogger('vmc')


@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def profiling_simple_random_walk(markovchain, steps, r_initial, decorr_period):
    walker = markovchain.simple_random_walker(steps, r_initial, decorr_period)
    for _ in range(steps):
        next(walker)


class Profiler(Casino):

    def __init__(self, config_path):
        super().__init__(config_path)
        self.dx = 3.0
        self.steps, self.atom_positions = self.config.input.vmc_nstep, self.config.wfn.atom_positions

    def initial_position(self, ne, atom_positions, atom_charges):
        """Initial positions of electrons."""
        natoms = atom_positions.shape[0]
        r_e = np.zeros((ne, 3))
        for i in range(ne):
            r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
        return r_e

    def slater_profiling(self):
        """For multithreaded
        https://numba.pydata.org/numba-doc/latest/user/threading-layer.html
        """
        start = default_timer()
        self.wfn.slater.profile_value(self.dx, self.steps, self.atom_positions, self.r_initial)
        end = default_timer()
        logger.info(' slater value       %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.wfn.slater.profile_laplacian(self.dx, self.steps, self.atom_positions, self.r_initial)
        end = default_timer()
        logger.info(' slater laplacian   %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.wfn.slater.profile_gradient(self.dx, self.steps, self.atom_positions, self.r_initial)
        end = default_timer()
        logger.info(' slater gradient    %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.wfn.slater.profile_hessian(self.dx, self.steps, self.atom_positions, self.r_initial)
        end = default_timer()
        logger.info(' slater hessian     %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    def jastrow_profiling(self):

        start = default_timer()
        self.wfn.jastrow.profile_value(self.dx, self.steps, self.atom_positions, self.r_initial)
        end = default_timer()
        logger.info(' jastrow value      %8.1f', end - start)

        start = default_timer()
        self.wfn.jastrow.profile_laplacian(self.dx, self.steps, self.atom_positions, self.r_initial)
        end = default_timer()
        logger.info(' jastrow laplacian  %8.1f', end - start)

        start = default_timer()
        self.wfn.jastrow.profile_gradient(self.dx, self.steps, self.atom_positions, self.r_initial)
        end = default_timer()
        logger.info(' jastrow gradient   %8.1f', end - start)

    def backflow_profiling(self):

        start = default_timer()
        self.wfn.backflow.profile_value(self.dx, self.steps, self.atom_positions, self.r_initial)
        end = default_timer()
        logger.info(' backflow value     %8.1f', end - start)

        start = default_timer()
        self.wfn.backflow.profile_gradient(self.dx, self.steps, self.atom_positions, self.r_initial)
        end = default_timer()
        logger.info(' backflow gradient  %8.1f', end - start)

        start = default_timer()
        self.wfn.backflow.profile_laplacian(self.dx, self.steps, self.atom_positions, self.r_initial)
        end = default_timer()
        logger.info(' backflow laplacian %8.1f', end - start)

    def markovchain_profiling(self):

        start = default_timer()
        profiling_simple_random_walk(self.markovchain, self.config.input.vmc_nstep, self.r_initial, 1)
        end = default_timer()
        logger.info(' markovchain value     %8.1f', end - start)
        stats = rtsys.get_allocation_stats()
        logger.info(f'{stats} total: {stats[0] - stats[1]}')


if __name__ == '__main__':

    """
    He:
     slater value           33.1
     slater laplacian       61.1
     slater gradient        85.7
     slater hessian        274.0
     jastrow value          31.2
     jastrow laplacian      40.0
     jastrow gradient       47.4
     backflow value         43.6
     backflow gradient     121.4
     backflow laplacian    162.8
    Be:
     slater value           49.4
     slater laplacian      103.1
     slater gradient       147.3
     slater hessian        381.8
     jastrow value          70.0
     jastrow laplacian     114.0
     jastrow gradient      134.3
     backflow value        133.5
     backflow gradient     536.1
     backflow laplacian    696.7
    Ne:
     slater value          126.8
     slater laplacian      240.8
     slater gradient       323.7
     slater hessian        782.0
     jastrow value         338.2
     jastrow laplacian     648.3
     jastrow gradient      689.7
     backflow value        551.8
     backflow gradient    2152.2
     backflow laplacian   2688.3
    """

    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr', 'O3'):
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Backflow/'
        logger.info('%s:', mol)
        profileler = Profiler(path)
        profileler.slater_profiling()
        profileler.jastrow_profiling()
        profileler.backflow_profiling()
        # profileler.markovchain_profiling()

    """
    Slater:
        He:
         value         28.7
         laplacian     53.7
         gradient      74.1
         hessian      251.2
        Be:
         value         50.5
         laplacian    100.9
         gradient     136.9
         hessian      365.3
        Ne:
         value        116.1
         laplacian    228.9
         gradient     294.9
         hessian      777.2
        Ar:
         value        269.5
         laplacian    555.6
         gradient     657.0
         hessian     1670.7
        -- old --
        Kr:
         value        725.7
         laplacian   1529.8
         gradient    1918.6
         hessian     5085.5
        O3:
         value        657.9
         laplacian   1302.7
         gradient    1648.3
         hessian     3853.4
    Jatrow
    He:
     value         25.6
     laplacian     30.4
     gradient      37.6
    Be:
     value         57.5
     laplacian     93.9
     gradient     112.4
    Ne:
     value        277.5
     laplacian    481.7
     gradient     536.5
    Ar:
     value        875.4
     laplacian   1612.5
     gradient    1771.5
    Kr:
     value       3174.8
    Backflow
    He:
     value         40.0
     gradient     121.6
     laplacian    138.8
    Be:
     value         99.5
     gradient     481.4
     laplacian    573.4
    Ne:
     value        415.0
     gradient    1897.3
     laplacian   2247.9
    Ar:
     value       1501.9
    """
