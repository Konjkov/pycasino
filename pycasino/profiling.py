#!/usr/bin/env python3

from timeit import default_timer
from numba.core.runtime import rtsys
from pycasino.logger import logging
from pycasino.casino import Casino


logger = logging.getLogger('vmc')


class Profiler(Casino):

    def __init__(self, config_path):
        super().__init__(config_path)
        self.dr = 3.0  # AU
        self.r_e = self.initial_position(self.config.wfn.atom_positions, self.config.wfn.atom_charges)
        self.steps, self.atom_positions = self.config.input.vmc_nstep, self.config.wfn.atom_positions

    def cusp_profiling(self):

        start = default_timer()
        self.vmc_markovchain.wfn.slater.cusp.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp value       %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.vmc_markovchain.wfn.slater.cusp.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp laplacian   %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.vmc_markovchain.wfn.slater.cusp.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp gradient    %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.vmc_markovchain.wfn.slater.cusp.profile_hessian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp hessian     %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    def slater_profiling(self):

        start = default_timer()
        self.vmc_markovchain.wfn.slater.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater value       %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.vmc_markovchain.wfn.slater.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater laplacian   %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.vmc_markovchain.wfn.slater.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater gradient    %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.vmc_markovchain.wfn.slater.profile_hessian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater hessian     %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    def jastrow_profiling(self):

        start = default_timer()
        self.vmc_markovchain.wfn.jastrow.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow value      %8.1f', end - start)

        start = default_timer()
        self.vmc_markovchain.wfn.jastrow.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow laplacian  %8.1f', end - start)

        start = default_timer()
        self.vmc_markovchain.wfn.jastrow.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' jastrow gradient   %8.1f', end - start)

    def backflow_profiling(self):

        start = default_timer()
        self.vmc_markovchain.wfn.backflow.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow value     %8.1f', end - start)

        start = default_timer()
        self.vmc_markovchain.wfn.backflow.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow value gradient  %8.1f', end - start)

        start = default_timer()
        self.vmc_markovchain.wfn.backflow.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow value gradient laplacian %8.1f', end - start)

    def markovchain_profiling(self):

        start = default_timer()
        self.vmc_markovchain.profiling_simple_random_walk(self.config.input.vmc_nstep, self.r_e, 1)
        end = default_timer()
        logger.info(' markovchain value  %8.1f', end - start)
        stats = rtsys.get_allocation_stats()
        logger.info(f'{stats} total: {stats[0] - stats[1]}')


if __name__ == '__main__':
    """
    He:
     slater value           31.6
     slater laplacian       58.0
     slater gradient        78.3
     slater hessian        241.1
     jastrow value          26.7
     jastrow laplacian      33.5
     jastrow gradient       41.2
     backflow value         40.7
     backflow gradient     107.4
     backflow laplacian    129.2
    Be:
     slater value           46.4
     slater laplacian       92.2
     slater gradient       120.4
     slater hessian        305.4
     jastrow value          59.9
     jastrow laplacian      89.2
     jastrow gradient      106.1
     backflow value        101.3
     backflow gradient     439.3
     backflow laplacian    545.4
    Ne:
     slater value          111.7
     slater laplacian      217.7
     slater gradient       280.5
     slater hessian        752.1
     jastrow value         284.2
     jastrow laplacian     496.6
     jastrow gradient      563.9
     backflow value        389.7
     backflow gradient    1525.8
     backflow laplacian   1840.7
    Ar:
     slater value          229.1
     slater laplacian      465.2
     slater gradient       579.8
     slater hessian       1507.7
     jastrow value         857.8
     jastrow laplacian    1560.6
     jastrow gradient     1767.0
     backflow value       1409.5
     backflow gradient    7217.1
     backflow laplacian  11405.6
    Kr:
     slater value          695.6
     slater laplacian     1505.7
     slater gradient      1869.0
     slater hessian       4965.5
     jastrow value        3552.8
     jastrow laplacian    5946.7
     jastrow gradient     6377.9
    """
    for mol in ('He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3'):
        path = f'../tests/stowfn/{mol}/HF/QZ4P/CBCS/Backflow/'
        logger.info('%s:', mol)
        profiler = Profiler(path)
        profiler.slater_profiling()
        profiler.jastrow_profiling()
        profiler.backflow_profiling()
        # profiler.markovchain_profiling()/

    """
    He:
     cusp value            9.8
     cusp laplacian        9.4
     cusp gradient         9.4
     cusp hessian         12.4
    Be:
     cusp value           10.2
     cusp laplacian       10.4
     cusp gradient        10.8
     cusp hessian         11.3
    Ne:
     cusp value           21.1
     cusp laplacian       19.8
     cusp gradient        20.9
     cusp hessian         21.4
    Ar:
     cusp value           36.3
     cusp laplacian       39.3
     cusp gradient        38.7
     cusp hessian         39.9
    Kr:
     cusp value           85.6
     cusp laplacian       87.6
     cusp gradient        92.8
     cusp hessian        108.9
    O3:
     cusp value          110.1
     cusp laplacian      102.7
     cusp gradient       109.4
     cusp hessian        115.4
    """
    for mol in ('He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3'):
        path = f'../tests/gwfn/{mol}/HF/cc-pVQZ/CBCS/Jastrow/'
        logger.info('%s:', mol)
        profiler = Profiler(path)
        profiler.cusp_profiling()

    """
    HF:
     slater value           58.0
     slater laplacian      114.2
     slater gradient       156.0
     slater hessian        464.3
    MP2-CASSCF(2.4):
     slater value          119.4
     slater laplacian      272.5
     slater gradient       359.6
     slater hessian       1197.0
    """
    for method in ('HF', 'MP2-CASSCF(2.4)'):
        path = f'../tests/gwfn/Be/{method}/cc-pVQZ/CBCS/Jastrow/'
        logger.info('%s:', method)
        profiler = Profiler(path)
        profiler.slater_profiling()
