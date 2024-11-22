import os
import sys

import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'  # openmp
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # openblas
os.environ['MKL_NUM_THREADS'] = '1'  # mkl
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # accelerate
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # numexpr
os.environ['NUMBA_NUM_THREADS'] = '1'  # numba
os.environ['NUMBA_FULL_TRACEBACKS'] = '1'
# os.environ['NUMBA_CAPTURED_ERRORS'] = 'new_style'
# os.environ['NUMBA_DEBUG_CACHE'] = '1'

np.seterr(all='warn')
np.random.seed(31415926)
np.set_printoptions(threshold=sys.maxsize)

# https://scicomp.stackexchange.com/questions/14355/choosing-epsilons
# delta = np.sqrt(sys.float_info.epsilon)
delta = np.finfo(float).eps ** (1 / 2)
delta_2 = np.finfo(float).eps ** (1 / 3)
delta_3 = np.finfo(float).eps ** (1 / 4)

# np.show_config()
