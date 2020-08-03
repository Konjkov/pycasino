#!/usr/bin/env python3

import numpy as np


class Mdet:
    """"""

    def __init__(self, file, atoms):
        mdet = False

        with open(file, 'r') as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.strip().startswith('START MDET'):
                    mdet = True
                elif line.strip().startswith('END MDET'):
                    mdet = False
                elif line.strip().startswith('MD'):
                    num_dets = int(f.readline())
                    dets = np.zeros(num_dets)
                    for i in range(num_dets):
                        dets[i] = int(f.readline().split()[0])
                elif line.strip().startswith('DET'):
                    f.readline().split()
