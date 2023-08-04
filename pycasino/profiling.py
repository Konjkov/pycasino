#!/usr/bin/env python3
import logging

from timeit import default_timer
from numba.core.runtime import rtsys
from pycasino.casino import Casino


logger = logging.getLogger(__name__)


class Profiler(Casino):

    def __init__(self, config_path):
        super().__init__(config_path)
        self.dr = 3.0  # AU
        self.r_e = self.initial_position(self.config.wfn.atom_positions, self.config.wfn.atom_charges)
        self.steps, self.atom_positions = self.config.input.vmc_nstep, self.config.wfn.atom_positions

    def cusp_profiling(self):

        start = default_timer()
        self.wfn.slater.cusp.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp value                        %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.wfn.slater.cusp.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp laplacian                    %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.wfn.slater.cusp.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp gradient                     %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.wfn.slater.cusp.profile_hessian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp hessian                      %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    def slater_profiling(self):

        start = default_timer()
        self.wfn.slater.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater value                      %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.wfn.slater.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater laplacian                  %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.wfn.slater.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater gradient                   %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.wfn.slater.profile_hessian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater hessian                    %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.wfn.slater.profile_tressian(self.dr, self.steps // 10, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater tressian                   %8.1f', (end - start) * 10)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    def jastrow_profiling(self):

        start = default_timer()
        self.wfn.jastrow.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow value                     %8.1f', end - start)

        start = default_timer()
        self.wfn.jastrow.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow laplacian                 %8.1f', end - start)

        start = default_timer()
        self.wfn.jastrow.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow gradient                  %8.1f', end - start)

        start = default_timer()
        self.wfn.jastrow.profile_value_parameters_d1(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow value parameters d1       %8.1f', (end - start))

        start = default_timer()
        self.wfn.jastrow.profile_laplacian_parameters_d1(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow laplacian parameters d1   %8.1f', (end - start))

        start = default_timer()
        self.wfn.jastrow.profile_gradient_parameters_d1(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow gradient parameters d1    %8.1f', (end - start))

    def backflow_profiling(self):

        start = default_timer()
        self.wfn.backflow.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow value                    %8.1f', end - start)

        start = default_timer()
        self.wfn.backflow.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow gradient                 %8.1f', end - start)

        start = default_timer()
        self.wfn.backflow.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow laplacian                %8.1f', end - start)

        start = default_timer()
        self.wfn.backflow.profile_value_parameters_d1(self.dr, self.steps // 10, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow value parameters d1      %8.1f', (end - start) * 10)

        start = default_timer()
        self.wfn.backflow.profile_laplacian_parameters_d1(self.dr, self.steps // 10, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow laplacian parameters d1  %8.1f', (end - start) * 10)

        start = default_timer()
        self.wfn.backflow.profile_gradient_parameters_d1(self.dr, self.steps // 10, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow gradient parameters d1   %8.1f', (end - start) * 10)

    def markovchain_profiling(self):

        start = default_timer()
        self.vmc_markovchain.profiling_simple_random_walk(self.config.input.vmc_nstep, self.r_e, 1)
        end = default_timer()
        logger.info(' markovchain value                 %8.1f', end - start)
        stats = rtsys.get_allocation_stats()
        logger.info(f'{stats} total: {stats[0] - stats[1]}')


if __name__ == '__main__':
    """Profiling"""
    for mol in ('He', 'He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3'):
        path = f'../tests/stowfn/{mol}/HF/QZ4P/CBCS/Backflow/'
        logger.info('%s:', mol)
        profiler = Profiler(path)
        profiler.slater_profiling()
        profiler.jastrow_profiling()
        profiler.backflow_profiling()
        # profiler.markovchain_profiling()

    for mol in ('He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3'):
        path = f'../tests/gwfn/{mol}/HF/cc-pVQZ/CBCS/Jastrow/'
        logger.info('%s:', mol)
        profiler = Profiler(path)
        profiler.cusp_profiling()

    for method in ('HF', 'MP2-CASSCF(2.4)'):
        path = f'../tests/gwfn/Be/{method}/cc-pVQZ/CBCS/Jastrow/'
        logger.info('%s:', method)
        profiler = Profiler(path)
        profiler.slater_profiling()
