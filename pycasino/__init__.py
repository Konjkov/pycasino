__version__ = '0.1.0'
__author__ = 'Vladimir Konkov'
__credits__ = 'Research Institute for Pythonic Quantum Chemistry'

disclamer = f"""
 ------------------------------------------------------------------------------
 ########::'##:::'##::'######:::::'###:::::'######::'####:'##::: ##::'#######::
 ##.... ##:. ##:'##::'##... ##:::'## ##:::'##... ##:. ##:: ###:: ##:'##.... ##:
 ##:::: ##::. ####::: ##:::..:::'##:. ##:: ##:::..::: ##:: ####: ##: ##:::: ##:
 ########::::. ##:::: ##:::::::'##:::. ##:. ######::: ##:: ## ## ##: ##:::: ##:
 ##.....:::::: ##:::: ##::::::: #########::..... ##:: ##:: ##. ####: ##:::: ##:
 ##::::::::::: ##:::: ##::: ##: ##.... ##:'##::: ##:: ##:: ##:. ###: ##:::: ##:
 ##::::::::::: ##::::. ######:: ##:::: ##:. ######::'####: ##::. ##:. #######::
 .::::::::::::..::::::......:::..:::::..:::......:::....::..::::..:::.......:::

                          Python Quantum Package
                   v{__version__} [{__author__}] (30 August 2023)

    Main Author : {__author__}
 ------------------------------------------------------------------------------
"""  # created with art python package

import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import numpy as np

np.random.seed(31415926)
np.set_printoptions(threshold=sys.maxsize)

# https://scicomp.stackexchange.com/questions/14355/choosing-epsilons
# delta = np.sqrt(sys.float_info.epsilon)
delta = np.finfo(np.float64).eps ** (1/2)
delta_2 = np.finfo(np.float64).eps ** (1/3)
delta_3 = np.finfo(np.float64).eps ** (1/4)

# np.show_config()

# os.environ['NUMBA_CAPTURED_ERRORS'] = 'new_style'
# os.environ['NUMBA_DEBUG_CACHE'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import logging

logging.basicConfig(
    level=logging.INFO,
    filename='pycasino.log',
    filemode='w',
    format='%(message)s'
)

logger = logging.getLogger(__name__)

from mpi4py import MPI

if MPI.COMM_WORLD.rank == 0:
    # to redirect scipy.optimize stdout to log-file
    from pycasino.loggers import StreamToLogger
    sys.stdout = StreamToLogger(logger, logging.INFO)
    # sys.stderr = StreamToLogger(self.logger, logging.ERROR)
else:
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

logger.info(disclamer)

import numba as nb
import scipy as sp

logger.info(
    f' Python {sys.version}\n'
    f' Numba {nb.__version__}\n'
    f' Numpy {np.__version__}\n'
    f' Scipy {sp.__version__}\n'
)
if MPI.COMM_WORLD.size > 1:
    logger.info(' Running in parallel using %i MPI processes.\n', MPI.COMM_WORLD.size)
else:
    logger.info(' Sequential run: not using MPI.\n')
