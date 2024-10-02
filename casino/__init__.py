import datetime
import logging
import os
import sys
import numba as nb
import numpy as np
import scipy as sp
from mpi4py import MPI


__version__ = '0.2.0'
__author__ = 'Vladimir Konkov'
__credits__ = 'Research Institute for Pythonic Quantum Chemistry'


# created with art python package
logo = f"""
 ------------------------------------------------------------------------------
 ########::'##:::'##::'######:::::'###:::::'######::'####:'##::: ##::'#######::
 ##.... ##:. ##:'##::'##... ##:::'## ##:::'##... ##:. ##:: ###:: ##:'##.... ##:
 ##:::: ##::. ####::: ##:::..:::'##:. ##:: ##:::..::: ##:: ####: ##: ##:::: ##:
 ########::::. ##:::: ##:::::::'##:::. ##:. ######::: ##:: ## ## ##: ##:::: ##:
 ##.....:::::: ##:::: ##::::::: #########::..... ##:: ##:: ##. ####: ##:::: ##:
 ##::::::::::: ##:::: ##::: ##: ##.... ##:'##::: ##:: ##:: ##:. ###: ##:::: ##:
 ##::::::::::: ##::::. ######:: ##:::: ##:. ######::'####: ##::. ##:. #######::
 .::::::::::::..::::::......:::..:::::..:::......:::....::..::::..:::.......:::

                     Python Quantum Monte Carlo Package
                        v {__version__} [{__author__}]

    Main Author : {__author__}
 ------------------------------------------------------------------------------
 Started {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}

 Python {sys.version}
 Numba {nb.__version__}
 Numpy {np.__version__}
 Scipy {sp.__version__}
"""

os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr
os.environ["NUMBA_NUM_THREADS"] = "1"  # numba
os.environ["NUMBA_FULL_TRACEBACKS"] = "1"
# os.environ['NUMBA_CAPTURED_ERRORS'] = 'new_style'
# os.environ['NUMBA_DEBUG_CACHE'] = '1'

np.seterr(all='warn')
np.random.seed(31415926)
np.set_printoptions(threshold=sys.maxsize)

# https://scicomp.stackexchange.com/questions/14355/choosing-epsilons
# delta = np.sqrt(sys.float_info.epsilon)
delta = np.finfo(np.float_).eps ** (1/2)
delta_2 = np.finfo(np.float_).eps ** (1/3)
delta_3 = np.finfo(np.float_).eps ** (1/4)

# np.show_config()
logging.basicConfig(
    level=logging.INFO,
    filename='pycasino.log',
    filemode='w',
    format='%(message)s'
)

logger = logging.getLogger(__name__)

if MPI.COMM_WORLD.rank == 0:
    # to redirect scipy.optimize stdout to log-file
    from casino.loggers import StreamToLogger
    sys.stdout = StreamToLogger(logger, logging.INFO)
    # sys.stderr = StreamToLogger(self.logger, logging.ERROR)
else:
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

logger.info(logo)

if MPI.COMM_WORLD.size > 1:
    logger.info(' Running in parallel using %i MPI processes.\n', MPI.COMM_WORLD.size)
else:
    logger.info(' Sequential run: not using MPI.\n')
