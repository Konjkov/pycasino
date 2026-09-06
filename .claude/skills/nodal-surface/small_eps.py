"""Nodal domain averages on a small-epsilon grid only, skipping the VMC energy accumulation that
--nodal would run first."""

import sys

import numpy as np

from casino.pycasino import Casino, configure_logging

configure_logging()
casino = Casino(sys.argv[1])
casino.nodal_domain_accumulation(np.geomspace(0.002, 0.024, 9))
