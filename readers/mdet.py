#!/usr/bin/env python3

import numpy as np


class Mdet:
    """"""

    def __init__(self, file, neu, ned):
        mdet = False

        # with open(file, 'r') as f:
        #     line = f.readline()
        #     while line:
        #         line = f.readline()
        #         if line.strip().startswith('START MDET'):
        #             mdet = True
        #         elif line.strip().startswith('END MDET'):
        #             mdet = False
        #         elif line.strip().startswith('MD'):
        #             num_dets = int(f.readline())
        #             dets = np.zeros(num_dets)
        #             for i in range(num_dets):
        #                 dets[i] = int(f.readline().split()[0])
        #         elif line.strip().startswith('DET'):
        #             f.readline().split()

        coeff = [0.949672, - 0.180853, - 0.180853, - 0.180853]
        u = [[0, 1], [0, 2], [0, 3], [0, 4]]
        d = [[0, 1], [0, 2], [0, 3], [0, 4]]

        _mdet = [(
            1.0,
            np.arange(neu),
            np.arange(ned),
        ) for i in range(1)]
        self.mdet = np.array(_mdet, dtype=[
            ('coeff', np.float),
            ('up', np.int, neu),
            ('down', np.int, ned),
        ])
