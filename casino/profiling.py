#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from timeit import default_timer

from casino.pycasino import Casino

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

        start = default_timer()
        self.wfn.slater.cusp.profile_tressian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' cusp tressian                     %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    def slater_value_profiling(self):
        start = default_timer()
        self.wfn.slater.profile_value(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater value                      %8.1f', end - start)

    def slater_laplacian_profiling(self):
        start = default_timer()
        self.wfn.slater.profile_laplacian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater laplacian                  %8.1f', end - start)

    def slater_gradient_profiling(self):
        start = default_timer()
        self.wfn.slater.profile_gradient(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater gradient                   %8.1f', end - start)

    def slater_hessian_profiling(self):
        start = default_timer()
        self.wfn.slater.profile_hessian(self.dr, self.steps, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater hessian                    %8.1f', end - start)

    def slater_tressian_profiling(self):
        start = default_timer()
        self.wfn.slater.profile_tressian(self.dr, self.steps // 10, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater tressian                   %8.1f * 10', (end - start))

    def slater_tressian_dot_profiling(self):
        start = default_timer()
        self.wfn.slater.profile_tressian_dot(self.dr, self.steps // 10, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' slater tressian_dot               %8.1f * 10', end - start)

    def slater_profiling(self):
        self.slater_value_profiling()
        self.slater_laplacian_profiling()
        self.slater_gradient_profiling()
        self.slater_hessian_profiling()
        self.slater_tressian_profiling()

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
        logger.info(' backflow value parameters d1      %8.1f * 10', (end - start))

        start = default_timer()
        self.wfn.backflow.profile_laplacian_parameters_d1(self.dr, self.steps // 10, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow laplacian parameters d1  %8.1f * 10', (end - start))

        start = default_timer()
        self.wfn.backflow.profile_gradient_parameters_d1(self.dr, self.steps // 100, self.atom_positions, self.r_e)
        end = default_timer()
        logger.info(' backflow gradient parameters d1   %8.1f * 100', (end - start))

    def markovchain_profiling(self):
        start = default_timer()
        self.vmc.random_walk(self.config.input.vmc_nstep, 1)
        end = default_timer()
        logger.info(' markovchain value                 %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')


MOLECULES = ('He', 'Be', 'N', 'Ne', 'Ar', 'O3', 'Kr')
SLATER_KERNELS = ('value', 'laplacian', 'gradient', 'hessian', 'tressian', 'tressian_dot')


if __name__ == '__main__':
    """Slater profiling: select molecules and kernels, e.g.
    python casino/profiling.py O3 Kr -k tressian
    python casino/profiling.py He            # all kernels for He
    """
    logging.getLogger().addHandler(logging.StreamHandler(sys.__stdout__))
    parser = argparse.ArgumentParser(description='Slater profiling')
    parser.add_argument('mols', nargs='*', choices=MOLECULES, help='molecules, default all')
    parser.add_argument('-k', '--kernel', action='append', choices=SLATER_KERNELS, help='slater kernel(s), default all')
    parser.add_argument('-s', '--steps', type=int, help='override vmc_nstep (tressian uses steps // 10)')
    args = parser.parse_args()
    mols = args.mols or MOLECULES
    kernels = args.kernel or SLATER_KERNELS
    root = Path(__file__).resolve().parents[1]
    for mol in mols:
        path = root / 'examples' / 'stowfn' / mol / 'HF' / 'QZ4P' / 'CBCS' / 'Backflow'
        logger.info('%s:', mol)
        profiler = Profiler(str(path))
        if args.steps:
            profiler.steps = args.steps
        for kernel in kernels:
            getattr(profiler, f'slater_{kernel}_profiling')()
