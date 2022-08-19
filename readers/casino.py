import os

from readers.input import Input
from readers.wfn import Gwfn, Stowfn
from readers.jastrow import Jastrow
from readers.gjastrow import Gjastrow
from readers.mdet import Mdet
from readers.backflow import Backflow

template = """\
 START HEADER
  {title}
 END HEADER

 START VERSION
   1
 END VERSION

"""


class CasinoConfig:
    """Casino inputs reader."""

    def __init__(self, base_path):
        self.input = Input()
        self.input.read(base_path)
        if self.input.atom_basis_type == 'gaussian':
            self.wfn = Gwfn()
        elif self.input.atom_basis_type == 'slater-type':
            self.wfn = Stowfn()
        self.wfn.read(base_path)
        if getattr(self.input, 'use_gjastrow', False):
            self.jastrow = Gjastrow()
        elif getattr(self.input, 'use_jastrow', False):
            self.jastrow = Jastrow()
        else:
            self.jastrow = None
        self.mdet = Mdet(base_path, self.input.neu, self.input.ned, self.wfn.mo_up, self.wfn.mo_down)
        if getattr(self.input, 'backflow', False):
            self.backflow = Backflow()
        else:
            self.backflow = None

    def read(self, base_path):
        # if self.wfn:
        #     self.wfn.read(base_path)
        # if self.mdet:
        #     self.mdet.read(base_path)
        if self.jastrow:
            self.jastrow.read(base_path)
        if self.backflow:
            self.backflow.read(base_path)

    def write(self, base_path, version):
        title = 'no title given'
        correlation = template.format(title=title)

        # if self.wfn:
        #     self.wfn.write()
        # if self.mdet:
        #     self.mdet.write()
        if self.jastrow:
            correlation += self.jastrow.write()
        if self.backflow:
            correlation += self.backflow.write()

        file_path = os.path.join(base_path, f'correlation.out.{version}')
        with open(file_path, 'w') as f:
            f.write(correlation)

