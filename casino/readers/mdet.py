import os

import numpy as np

mdet_template = """\
START MDET
Title
 {title}
MD
  {n_dets}
  {det_weights}
  {promote}
END MDET
"""


class Mdet:
    """"""

    def __init__(self, neu, ned):
        n_dets = 1
        self.neu = neu
        self.ned = ned
        self.coeff = np.ones(n_dets)
        self.promote_lines = []
        self.permutation_up = np.stack([np.arange(neu)] * n_dets)
        self.permutation_down = np.stack([np.arange(ned)] * n_dets)

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
                            #  MD
                            #    4                                  ! Number of dets
                            #    0.95003749699999995        1   0      ! c_1 ; label ; opt-flag
                            #   -0.16308105069005557        2   1      ! c_2 ; label ; opt-flag
                            #   * 1.0000000000000000        2   1      ! c_3/c_2 ; label ; opt-flag
                            #   * 1.0000000000000000        2   1      ! c_4/c_2 ; label ; opt-flag
                            n_dets = int(f.readline().split()[0])
                            self.coeff = np.ones(n_dets)
                            self.permutation_up = np.stack([np.arange(self.neu)] * n_dets)
                            self.permutation_down = np.stack([np.arange(self.ned)] * n_dets)
                            for i in range(n_dets):
                                line = f.readline().split()
                                if line[0] != '*':
                                    self.coeff[i] = coeff = float(line[0])
                                else:
                                    self.coeff[i] = coeff * float(line[1])
                        elif line.startswith('DET'):
                            # DET 2 1 PR 2 1 3 1
                            self.promote_lines.append(line)
                            _, n_det, spin, operation, from_orb, _, to_orb, _ = line.split()
                            if operation == 'PR':
                                if spin == '1':
                                    self.permutation_up[int(n_det) - 1, int(from_orb) - 1] = int(to_orb) - 1
                                elif spin == '2':
                                    self.permutation_down[int(n_det) - 1, int(from_orb) - 1] = int(to_orb) - 1
                # normalisation
                self.coeff /= np.linalg.norm(self.coeff)

    def write(self, title='no title given'):
        mdet = ''
        if self.coeff.size > 1:
            det_weights_list = []
            for coeff in self.coeff:
                det_weights_list.append(f'{coeff: .16e}            1       0')
            mdet = mdet_template.format(
                title=title,
                n_dets=self.coeff.size,
                det_weights='\n  '.join(det_weights_list),
                promote='\n  '.join(self.promote_lines),
            )
        return mdet
