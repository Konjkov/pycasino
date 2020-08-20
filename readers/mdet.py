#!/usr/bin/env python3

import os

import numpy as np


class Mdet:
    """"""

    def __init__(self, file, neu, ned, mo_up, mo_down):
        mdet = False
        n_dets = 1
        self.coeff = np.ones(n_dets)
        up = np.stack([np.arange(neu)] * n_dets)
        down = np.stack([np.arange(ned)] * n_dets)

        if not os.path.isfile(file):
            self.mo_up = np.zeros((n_dets, neu, mo_up.shape[1]), np.float)
            self.mo_down = np.zeros((n_dets, ned, mo_down.shape[1]), np.float)
            for i in range(n_dets):
                self.mo_up[i] = mo_up[up[i]]
                self.mo_down[i] = mo_down[down[i]]
            return

        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('START MDET'):
                    mdet = True
                elif line.startswith('END MDET'):
                    break
                if mdet:
                    if line.startswith('MD'):
                        n_dets = int(f.readline())
                        self.coeff = np.ones(n_dets)
                        up = np.stack([np.arange(neu)] * n_dets)
                        down = np.stack([np.arange(ned)] * n_dets)
                        for i in range(n_dets):
                            self.coeff[i] = float(f.readline().split()[0])
                    elif line.startswith('DET'):
                        _, n_det, spin, operation, from_orb, _, to_orb, _ = line.split()
                        if operation == 'PR':
                            if int(spin) == 1:
                                up[int(n_det)-1, int(from_orb)-1] = int(to_orb)-1
                            elif int(spin) == 2:
                                down[int(n_det)-1, int(from_orb)-1] = int(to_orb)-1

        self.mo_up = np.zeros((n_dets, neu, mo_up.shape[1]), np.float)
        self.mo_down = np.zeros((n_dets, ned, mo_down.shape[1]), np.float)
        for i in range(n_dets):
            self.mo_up[i] = mo_up[up[i]]
            self.mo_down[i] = mo_down[down[i]]
