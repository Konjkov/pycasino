#!/usr/bin/env python3

import numpy as np
from yaml import safe_load


class Jastrow:
    """Jastrow reader from file.
    Framework for constructing generic Jastrow correlation factors
    P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
    Phys. Rev. E 86, 036703 – Published 27 September 2012
    """

    def __init__(self, file):
        with open(file, 'r') as f:
            self._jastrow_data = safe_load(f)['JASTROW']
        for key in self._jastrow_data.keys():
            if not key.startswith('TERM'):
                continue
            if self._jastrow_data[key]['Rank'] == [2, 0]:
                print('possible u-term')
            elif self._jastrow_data[key]['Rank'] == [1, 1]:
                print('possible chi-term')
            elif self._jastrow_data[key]['Rank'] == [2, 1]:
                print('possible f-term')


if __name__ == '__main__':
    """
    """
    path = '../test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/casl/8_8_44/1000000_9/parameters.casl'

    Jastrow(path)
