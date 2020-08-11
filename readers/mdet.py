#!/usr/bin/env python3

import os

import numpy as np


class Mdet:
    """"""

    def __init__(self, file, neu, ned):
        mdet = False
        self._n_dets = 1
        self._coeff = np.ones(self._n_dets)
        self._up = np.stack([np.arange(neu)] * self._n_dets)
        self._down = np.stack([np.arange(ned)] * self._n_dets)

        if not os.path.isfile(file):
            self.mdet = self.set_mdet(neu, ned)
            return

        with open(file, 'r') as f:
            line = True
            while line:
                line = f.readline()
                if line.strip().startswith('START MDET'):
                    mdet = True
                elif line.strip().startswith('END MDET'):
                    mdet = False
                if mdet:
                    if line.strip().startswith('MD'):
                        self._n_dets = int(f.readline())
                        self._coeff = np.ones(self._n_dets)
                        self._up = np.stack([np.arange(neu)] * self._n_dets)
                        self._down = np.stack([np.arange(ned)] * self._n_dets)
                        for i in range(self._n_dets):
                            self._coeff[i] = float(f.readline().split()[0])
                    elif line.strip().startswith('DET'):
                        _, n_det, spin, operation, from_orb, _, to_orb, _ = line.split()
                        if operation == 'PR':
                            if int(spin) == 1:
                                self._up[int(n_det)-1, int(from_orb)-1] = int(to_orb)-1
                            elif int(spin) == 2:
                                self._down[int(n_det)-1, int(from_orb)-1] = int(to_orb)-1

        self.mdet = self.set_mdet(neu, ned)

    def set_mdet(self, neu, ned):
        _mdet = [(
            self._coeff[i],
            self._up[i],
            self._down[i],
        ) for i in range(self._n_dets)]
        return np.array(_mdet, dtype=[
            ('coeff', np.float),
            ('up', np.int, neu),
            ('down', np.int, ned),
        ])
