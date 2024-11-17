import logging
import os

from .backflow import Backflow
from .gjastrow import Gjastrow
from .input import Input
from .jastrow import Jastrow
from .mdet import Mdet
from .wfn import Gwfn, Stowfn

logger = logging.getLogger(__name__)

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
        self.title = 'no title given'
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
            self.backflow.set_ae_cutoff(self.wfn.is_pseudoatom)
            self.backflow.read(self.base_path)
        logger.info(self.input.log())

    def write(self, base_path, version):
        correlation_data = correlation_data_template.format(title=self.title)

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
