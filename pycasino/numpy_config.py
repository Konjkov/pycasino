import os
os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr
import numpy as np

np.random.seed(31415926)

# https://scicomp.stackexchange.com/questions/14355/choosing-epsilons
# delta = np.sqrt(sys.float_info.epsilon)
delta = np.sqrt(np.finfo(np.float).eps)

# np.show_config()
