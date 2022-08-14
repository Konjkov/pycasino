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
        self.dr = 3.0
        self.steps, self.atom_positions = self.config.input.vmc_nstep, self.config.wfn.atom_positions

    def profile_slater_value(self):
        self.markovchain.wfn.slater.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)

    def profile_slater_gradient(self):
        self.markovchain.wfn.slater.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)

    def profile_slater_laplacian(self):
        self.markovchain.wfn.slater.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)

    def profile_slater_hessian(self):
        self.markovchain.wfn.slater.profile_hessian(self.dr, self.steps, self.atom_positions, self.r_e)

    def profile_jastrow_value(self):
        self.markovchain.wfn.jastrow.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)

    def profile_jastrow_gradient(self):
        self.markovchain.wfn.jastrow.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)

    def profile_jastrow_laplacian(self):
        self.markovchain.wfn.jastrow.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)

    def profile_backflow_value(self):
        self.markovchain.wfn.backflow.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)

    def profile_backflow_gradient(self):
        self.markovchain.wfn.backflow.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)

    def profile_backflow_laplacian(self):
        self.markovchain.wfn.backflow.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)

    def profiling_simple_random_walk(self):
        self.markovchain.profiling_simple_random_walk(self.steps, self.r_e, 1)

    def slater_profiling(self):

        start = default_timer()
        self.parallel_execution(self.profile_slater_value)
        end = default_timer()
        logger.info(' slater value       %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.parallel_execution(self.profile_slater_laplacian)
        end = default_timer()
        logger.info(' slater laplacian   %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.parallel_execution(self.profile_slater_gradient)
        end = default_timer()
        logger.info(' slater gradient    %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.parallel_execution(self.profile_slater_hessian)
        end = default_timer()
        logger.info(' slater hessian     %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    def jastrow_profiling(self):

        start = default_timer()
        self.parallel_execution(self.profile_jastrow_value)
        end = default_timer()
        logger.info(' jastrow value      %8.1f', end - start)

        start = default_timer()
        self.parallel_execution(self.profile_jastrow_laplacian)
        end = default_timer()
        logger.info(' jastrow laplacian  %8.1f', end - start)

        start = default_timer()
        self.parallel_execution(self.profile_jastrow_gradient)
        end = default_timer()
        logger.info(' jastrow gradient   %8.1f', end - start)

    def backflow_profiling(self):

        start = default_timer()
        self.parallel_execution(self.profile_backflow_value)
        end = default_timer()
        logger.info(' backflow value     %8.1f', end - start)

        start = default_timer()
        self.parallel_execution(self.profile_backflow_gradient)
        end = default_timer()
        logger.info(' backflow gradient  %8.1f', end - start)

        start = default_timer()
        self.parallel_execution(self.profile_backflow_laplacian)
        end = default_timer()
        logger.info(' backflow laplacian %8.1f', end - start)

    def markovchain_profiling(self):

        start = default_timer()
        self.parallel_execution(self.markovchain.profiling_simple_random_walk, self.config.input.vmc_nstep, self.r_e, 1)
        end = default_timer()
        logger.info(' markovchain value     %8.1f', end - start)
        stats = rtsys.get_allocation_stats()
        logger.info(f'{stats} total: {stats[0] - stats[1]}')


if __name__ == '__main__':
    """
    He:
     slater value           31.5
     slater laplacian       69.6
     slater gradient       104.4
     slater hessian        363.5
     jastrow value          38.8
     jastrow laplacian      51.1
     jastrow gradient       62.1
     backflow value         59.9
     backflow gradient     162.8
     backflow laplacian    200.1
    """

    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr', 'O3'):
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Backflow/'
        logger.info('%s:', mol)
        profileler = Profiler(path)
        profileler.slater_profiling()
        profileler.jastrow_profiling()
        profileler.backflow_profiling()
        # profileler.markovchain_profiling()
