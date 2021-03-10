#!/usr/bin/env python3

import numpy as np
import numba as nb
# import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from readers.casino import Casino

eta_parameters_type = nb.float64[:, :]

spec = [
    ('enabled', nb.boolean),
    ('trunc', nb.int64),
    ('eta_parameters', eta_parameters_type),
]


@nb.experimental.jitclass(spec)
class Backflow:

    def __init__(self, trunc, eta_parameters):
        self.enabled = True
        self.trunc = trunc
        self.eta_parameters = eta_parameters


if __name__ == '__main__':
    """Plot Backflow terms
    """

    term = 'eta'

    path = 'test/stowfn/ne/HF/QZ4P/VMC_OPT_BF/emin_BF/8_8_44__9_9_33/'

    casino = Casino(path)
    backflow = Backflow(
        casino.backflow.trunc, casino.backflow.eta_parameters
    )

    print(backflow.eta_parameters)
