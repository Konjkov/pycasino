#!/usr/bin/env python3

from timeit import default_timer
from numba.core.runtime import rtsys
from logger import logging
from casino import Casino


logger = logging.getLogger('vmc')


class Profiler(Casino):

    def __init__(self, config_path):
        super().__init__(config_path)
        self.dr = 3.0  # AU
        self.steps, self.atom_positions = self.config.input.vmc_nstep, self.config.wfn.atom_positions

    def cusp_profiling(self):

        start = default_timer()
        self.markovchain.wfn.slater.cusp.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp value       %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.markovchain.wfn.slater.cusp.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp laplacian   %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.markovchain.wfn.slater.cusp.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp gradient    %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.markovchain.wfn.slater.cusp.profile_hessian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp hessian     %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

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
    Kr:
     slater value          695.6
     slater laplacian     1505.7
     slater gradient      1869.0
     slater hessian       4965.5
     jastrow value        3552.8
     jastrow laplacian    5946.7
     jastrow gradient     6377.9
    """
    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr', 'O3'):
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Backflow/'
        logger.info('%s:', mol)
        profiler = Profiler(path)
        profiler.slater_profiling()
        profiler.jastrow_profiling()
        profiler.backflow_profiling()
        # profiler.markovchain_profiling()

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
    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr', 'O3'):
        path = f'test/gwfn/{mol}/HF/cc-pVQZ/CBCS/Jastrow/'
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
        path = f'test/gwfn/Be/{method}/cc-pVQZ/CBCS/Jastrow/'
        logger.info('%s:', method)
        profiler = Profiler(path)
        profiler.slater_profiling()
