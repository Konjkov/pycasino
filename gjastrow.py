#!/usr/bin/env python3

import numpy as np
import numba as nb

from readers.casino import Casino


spec = [
    ('trunc', nb.int64),
]


@nb.experimental.jitclass(spec)
class Gjastrow:

    def __init__(self, trunc):
        self.trunc = trunc


if __name__ == '__main__':
    """
    """

    term = 'chi'

    path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/casl/8__2/'

    casino = Casino(path)
    gjastrow = Gjastrow(
        casino.jastrow.trunc
    )

    steps = 100



