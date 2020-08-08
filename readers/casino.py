
import os

from readers.input import Input
from readers.wfn import Gwfn, Stowfn
from readers.jastrow import Jastrow
from readers.mdet import Mdet


class Casino:
    """Casino inputs reader."""

    def __init__(self, base_path):
        self.input = Input(os.path.join(base_path, 'input'))
        if self.input.atom_basis_type == 'gaussian':
            self.wfn = Gwfn(os.path.join(base_path, 'gwfn.data'))
        elif self.input.atom_basis_type == 'slater-type':
            self.wfn = Stowfn(os.path.join(base_path, 'stowfn.data'))
        if getattr(self.input, 'use_jastrow', False) or getattr(self.input, 'opt_jastrow', False):
            self.jastrow = Jastrow(os.path.join(base_path, 'correlation.out.5'), self.wfn.atoms)
        else:
            self.jastrow = False
        # self.mdet = Mdet(os.path.join(base_path, 'correlation.out.5')
