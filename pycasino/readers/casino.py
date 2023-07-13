import os

from pycasino.readers.input import Input
from pycasino.readers.wfn import Gwfn, Stowfn
from pycasino.readers.jastrow import Jastrow
from pycasino.readers.gjastrow import Gjastrow
from pycasino.readers.mdet import Mdet
from pycasino.readers.backflow import Backflow

correlation_data_template = """\
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
        self.base_path = base_path
        self.input.read(self.base_path)
        if self.input.atom_basis_type == 'gaussian':
            self.wfn = Gwfn()
        elif self.input.atom_basis_type == 'slater-type':
            self.wfn = Stowfn()
        self.mdet = Mdet(self.input.neu, self.input.ned)
        if self.input.use_gjastrow:
            self.jastrow = Gjastrow()
        elif self.input.use_jastrow:
            self.jastrow = Jastrow()
        else:
            self.jastrow = None
        if self.input.backflow:
            self.backflow = Backflow()
        else:
            self.backflow = None

    def read(self):
        if self.wfn:
            self.wfn.read(self.base_path)
        if self.mdet:
            self.mdet.read(self.base_path)
        if self.jastrow:
            self.jastrow.read(self.base_path)
        if self.backflow:
            self.backflow.read(self.base_path)

    def write(self, base_path, version):
        correlation_data = correlation_data_template.format(title='no title given')

        # if self.wfn:
        #     self.wfn.write()
        if self.jastrow:
            if self.input.use_gjastrow:
                self.jastrow.write(base_path, version)
            else:
                correlation_data += self.jastrow.write()
        if self.backflow:
            correlation_data += self.backflow.write()
        if self.mdet:
            correlation_data += self.mdet.write()

        file_path = os.path.join(base_path, f'correlation.out.{version}')
        with open(file_path, 'w') as f:
            f.write(correlation_data)
