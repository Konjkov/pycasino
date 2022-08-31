import os

import numpy as np

mdet_template = """\
START MDET
Title
 {title}
MD
  4
  0.950037497  1 0
 -0.180211686  2 1
 -0.180211686  2 1
 -0.180211686  2 1
  DET 2 1 PR 2 1 3 1
  DET 2 2 PR 2 1 3 1
  DET 3 1 PR 2 1 4 1
  DET 3 2 PR 2 1 4 1
  DET 4 1 PR 2 1 5 1
  DET 4 2 PR 2 1 5 1
END MDET
"""


class Mdet:
    """"""

    def __init__(self, neu, ned):
        n_dets = 1
        self.neu = neu
        self.ned = ned
        self.coeff = np.ones(n_dets)
        self.up = np.stack([np.arange(neu)] * n_dets)
        self.down = np.stack([np.arange(ned)] * n_dets)

    def read(self, base_path):
        file_path = os.path.join(base_path, 'correlation.data')
        if os.path.isfile(file_path):
            mdet = False
            with open(file_path, 'r') as f:
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
                            self.up = np.stack([np.arange(self.neu)] * n_dets)
                            self.down = np.stack([np.arange(self.ned)] * n_dets)
                            for i in range(n_dets):
                                self.coeff[i] = float(f.readline().split()[0])
                        elif line.startswith('DET'):
                            _, n_det, spin, operation, from_orb, _, to_orb, _ = line.split()
                            if operation == 'PR':
                                if int(spin) == 1:
                                    self.up[int(n_det)-1, int(from_orb)-1] = int(to_orb)-1
                                elif int(spin) == 2:
                                    self.down[int(n_det)-1, int(from_orb)-1] = int(to_orb)-1

    def write(self):
        mdet = mdet_template.format(
            title='no title given',
        )
        return mdet
