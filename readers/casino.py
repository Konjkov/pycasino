import os

from readers.input import Input
from readers.wfn import Gwfn, Stowfn
from readers.jastrow import Jastrow
from readers.gjastrow import Gjastrow
from readers.mdet import Mdet
from readers.backflow import Backflow


class CasinoConfig:
    """Casino inputs reader."""

    def __init__(self, base_path):
        self.input = Input(os.path.join(base_path, 'input'))
        if self.input.atom_basis_type == 'gaussian':
            self.wfn = Gwfn(os.path.join(base_path, 'gwfn.data'))
        elif self.input.atom_basis_type == 'slater-type':
            self.wfn = Stowfn(os.path.join(base_path, 'stowfn.data'))
        self.jastrow = None
        if getattr(self.input, 'use_gjastrow', False):
            self.jastrow = Gjastrow(os.path.join(base_path, 'parameters.casl'), self.wfn.atom_charges)
        if getattr(self.input, 'use_jastrow', False):
            self.jastrow = Jastrow(os.path.join(base_path, 'correlation.data'))
        self.mdet = Mdet(os.path.join(base_path, 'correlation.data'), self.input.neu, self.input.ned, self.wfn.mo_up, self.wfn.mo_down)
        self.backflow = None
        if getattr(self.input, 'backflow', False):
            self.backflow = Backflow(os.path.join(base_path, 'correlation.data'), self.wfn.atom_positions)
