#!/usr/bin/env python3

import os
from concurrent.futures import ProcessPoolExecutor
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
from psutil import cpu_count
from readers.casino import CasinoConfig
from casino import Casino


logger = logging.getLogger('vmc')


# Sharing big NumPy arrays across python processes
# https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2


class Profiler:

    def __init__(self, config_path):
        self.dx = 3.0
        self.num_proc = cpu_count(logical=False)
        config = CasinoConfig(config_path)
        neu, ned = config.input.neu, config.input.ned
        r_initial = self.initial_position(neu + ned, config.wfn.atom_positions, config.wfn.atom_charges)
        self.steps, self.atom_positions = config.input.vmc_nstep, config.wfn.atom_positions
        self.args = (self.dx, self.steps, self.atom_positions, config_path, r_initial)

    def initial_position(self, ne, atom_positions, atom_charges):
        """Initial positions of electrons."""
        natoms = atom_positions.shape[0]
        r_e = np.zeros((ne, 3))
        for i in range(ne):
            r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
        return r_e

    def parallel_execution(self, function, *args):
        with ProcessPoolExecutor(max_workers=self.num_proc) as executor:
            futures = [executor.submit(function, *args) for _ in range(self.num_proc)]
            # get task results in order they were submitted
            return [res.result() for res in futures]

    def slater_profiling(self):

        start = default_timer()
        self.parallel_execution(slater_value, *self.args)
        end = default_timer()
        logger.info(' slater value       %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.parallel_execution(slater_laplacian, *self.args)
        end = default_timer()
        logger.info(' slater laplacian   %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.parallel_execution(slater_gradient, *self.args)
        end = default_timer()
        logger.info(' slater gradient    %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

        start = default_timer()
        self.parallel_execution(slater_hessian, *self.args)
        end = default_timer()
        logger.info(' slater hessian     %8.1f', end - start)
        # stats = rtsys.get_allocation_stats()
        # logger.info(f'{stats} total: {stats[0] - stats[1]}')

    def jastrow_profiling(self):

        start = default_timer()
        self.parallel_execution(jastrow_value, *self.args)
        end = default_timer()
        logger.info(' jastrow value      %8.1f', end - start)

        start = default_timer()
        self.parallel_execution(jastrow_laplacian, *self.args)
        end = default_timer()
        logger.info(' jastrow laplacian  %8.1f', end - start)

        start = default_timer()
        self.parallel_execution(jastrow_gradient, *self.args)
        end = default_timer()
        logger.info(' jastrow gradient   %8.1f', end - start)

    def backflow_profiling(self):

        start = default_timer()
        self.parallel_execution(backflow_value, *self.args)
        end = default_timer()
        logger.info(' backflow value     %8.1f', end - start)

        start = default_timer()
        self.parallel_execution(backflow_gradient, *self.args)
        end = default_timer()
        logger.info(' backflow gradient  %8.1f', end - start)

        start = default_timer()
        self.parallel_execution(backflow_laplacian, *self.args)
        end = default_timer()
        logger.info(' backflow laplacian %8.1f', end - start)

    # def markovchain_profiling(self):
    #
    #     start = default_timer()
    #     profiling_simple_random_walk(self.markovchain, self.config.input.vmc_nstep, self.r_initial, 1)
    #     end = default_timer()
    #     logger.info(' markovchain value     %8.1f', end - start)
    #     stats = rtsys.get_allocation_stats()
    #     logger.info(f'{stats} total: {stats[0] - stats[1]}')


def slater_value(dx, steps, atom_positions, config_path, r_initial):
    Casino(config_path).wfn.slater.profile_value(dx, steps, atom_positions, r_initial)


def slater_gradient(dx, steps, atom_positions, config_path, r_initial):
    Casino(config_path).wfn.slater.profile_gradient(dx, steps, atom_positions, r_initial)


def slater_laplacian(dx, steps, atom_positions, config_path, r_initial):
    Casino(config_path).wfn.slater.profile_laplacian(dx, steps, atom_positions, r_initial)


def slater_hessian(dx, steps, atom_positions, config_path, r_initial):
    Casino(config_path).wfn.slater.profile_hessian(dx, steps, atom_positions, r_initial)


def jastrow_value(dx, steps, atom_positions, config_path, r_initial):
    Casino(config_path).wfn.jastrow.profile_value(dx, steps, atom_positions, r_initial)


def jastrow_gradient(dx, steps, atom_positions, config_path, r_initial):
    Casino(config_path).wfn.jastrow.profile_gradient(dx, steps, atom_positions, r_initial)


def jastrow_laplacian(dx, steps, atom_positions, config_path, r_initial):
    Casino(config_path).wfn.jastrow.profile_laplacian(dx, steps, atom_positions, r_initial)


def backflow_value(dx, steps, atom_positions, config_path, r_initial):
    Casino(config_path).wfn.backflow.profile_value(dx, steps, atom_positions, r_initial)


def backflow_gradient(dx, steps, atom_positions, config_path, r_initial):
    Casino(config_path).wfn.backflow.profile_gradient(dx, steps, atom_positions, r_initial)


def backflow_laplacian(dx, steps, atom_positions, config_path, r_initial):
    Casino(config_path).wfn.backflow.profile_laplacian(dx, steps, atom_positions, r_initial)


# @nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
# def profiling_simple_random_walk(markovchain, steps, r_initial, decorr_period):
#     walker = markovchain.simple_random_walker(steps, r_initial, decorr_period)
#     for _ in range(steps):
#         next(walker)


if __name__ == '__main__':
    """
    He:
     slater value           46.8
     slater laplacian       91.3
     slater gradient       127.0
     slater hessian        388.5
     jastrow value          54.6
     jastrow laplacian      67.4
     jastrow gradient       78.5
     backflow value         75.6
     backflow gradient     177.3
     backflow laplacian    216.8
    """

    for mol in ('He', ):
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Backflow/'
        logger.info('%s:', mol)
        profileler = Profiler(path)
        profileler.slater_profiling()
        profileler.jastrow_profiling()
        profileler.backflow_profiling()
        # profileler.markovchain_profiling()
