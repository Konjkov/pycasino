#!/usr/bin/env python3

import os
from timeit import default_timer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from numba.core.runtime import rtsys

from logger import logging
from casino import Casino


logger = logging.getLogger('vmc')


class Profiler(Casino):

    def __init__(self, config_path):
        super().__init__(config_path)
        self.dr = 3.0  # AU
        self.steps, self.atom_positions = self.config.input.vmc_nstep, self.config.wfn.atom_positions

    def slater_profiling(self):

        start = default_timer()
        self.markovchain.wfn.slater.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater value       %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.markovchain.wfn.slater.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater laplacian   %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.markovchain.wfn.slater.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater gradient    %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.markovchain.wfn.slater.profile_hessian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater hessian     %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    def jastrow_profiling(self):

        start = default_timer()
        self.markovchain.wfn.jastrow.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow value      %8.1f', end - start)

        start = default_timer()
        self.markovchain.wfn.jastrow.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow laplacian  %8.1f', end - start)

        start = default_timer()
        self.markovchain.wfn.jastrow.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow gradient   %8.1f', end - start)

    def backflow_profiling(self):

        start = default_timer()
        self.markovchain.wfn.backflow.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow value     %8.1f', end - start)

        start = default_timer()
        self.markovchain.wfn.backflow.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow gradient  %8.1f', end - start)

        start = default_timer()
        self.markovchain.wfn.backflow.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow laplacian %8.1f', end - start)

    def markovchain_profiling(self):

        start = default_timer()
        self.markovchain.profiling_simple_random_walk(self.config.input.vmc_nstep, self.r_e, 1)
        end = default_timer()
        logger.info(' markovchain value  %8.1f', end - start)
        stats = rtsys.get_allocation_stats()
        logger.info(f'{stats} total: {stats[0] - stats[1]}')


if __name__ == '__main__':
    """
    He:
     slater value           29.3
     slater laplacian       54.6
     slater gradient        75.5
     slater hessian        252.4
     jastrow value          28.1
     jastrow laplacian      36.4
     jastrow gradient       44.7
     backflow value         42.3
     backflow gradient     116.5
     backflow laplacian    147.5
    Be:
     slater value           49.6
     slater laplacian       99.6
     slater gradient       121.8
     slater hessian        336.0
     jastrow value          63.6
     jastrow laplacian     103.6
     jastrow gradient      116.9
     backflow value        109.5
     backflow gradient     500.4
     backflow laplacian    595.4
    Ne:
     slater value          105.9
     slater laplacian      221.1
     slater gradient       275.2
     slater hessian        727.2
     jastrow value         302.6
     jastrow laplacian     599.1
     jastrow gradient      668.7
     backflow value        458.5
     backflow gradient    2017.4
     backflow laplacian   2448.1
    Ar:
     slater value          238.4
     slater laplacian      514.4
     slater gradient       644.3
     slater hessian       1677.0
     jastrow value         958.2
     jastrow laplacian    1910.5
     jastrow gradient     2109.0
     backflow value       1698.0
     backflow gradient    9133.1
     backflow laplacian  11405.6
    """

    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr', 'O3'):
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Backflow/'
        logger.info('%s:', mol)
        profileler = Profiler(path)
        profileler.slater_profiling()
        profileler.jastrow_profiling()
        profileler.backflow_profiling()
        # profileler.markovchain_profiling()
