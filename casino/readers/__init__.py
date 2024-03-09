import os
import logging

from .inp import Input
from .wfn import Gwfn, Stowfn
from .jastrow import Jastrow
from .gjastrow import Gjastrow
from .mdet import Mdet
from .backflow import Backflow

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
        self.log()

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

    def log(self):
        logger.info(
            f' General input parameters\n'
            f' ========================\n'
            f' NEU (num up spin electrons)              :  {self.input.neu}\n'
            f' NED (num down spin electrons)            :  {self.input.ned}\n'
            f' RUNTYPE (type of run)                    :  {self.input.runtype}\n'
            f' PSI_S  (form for [anti]symmetrizing wfn) :  slater\n'
            f' ATOM_BASIS_TYPE (atom-centred orb basis) :  {self.input.atom_basis_type}\n'
            f' INTERACTION (interaction type)           :  coulomb\n'
            f' TESTRUN (read input data,print and stop) :  F\n'
            f' PERIODIC                                 :  F\n'
            f' COMPLEX_WF (complex Slater wave fn.)     :  F\n'
            f' NEIGHPRINT (neighbour analysis)          :  0\n'
            f' USE_JASTROW (use Jastrow factor)         :  {str(self.input.use_jastrow)[0]}\n'
            f' BACKFLOW (use backflow corrections)      :  {str(self.input.backflow)[0]}\n'
            # f' DBARRC (DBAR recalculation period)       :  100000\n'
            f' USE_ORBMODS (use orbitals modifications) :  F\n'
            # f' CUSP_CORRECTION                          :  F\n'
            # f' MOLGSCREENING                            :  F\n'
            # f' USE_DETLA (DLA approx. to NL PP)         :  F\n'
            # f' NON_LOCAL_GRID (NL integration grid)     :  4\n'
            f' E_OFFSET (energy offset)                 :  0.0000\n'
            # f' ESUPERCELL                               :  F\n'
            # f' GAUTOL  (Gaussian evaluation tolerance)  :  7.0\n'
            # f' SPARSE                                   :  F\n'
            # f' DIPOLE_MOMENT                            :  F\n'
            f' CHECKPOINT (checkpointing level)         :  1\n'
            # f' CHECKPOINT_NCPU (chkpnt group size)      :  4\n'
            f' CON_LOC (Dir to read/write config.*)     :  ./\n'
            f' RELATIVISTIC                             :  F\n'
            # f'\n VMC input parameters\n'
            # f' ===================='
            # f' NEWRUN (start new run)                   :  T\n'
            # f' VMC_METHOD (choice of VMC algorithm)     :  3\n'
            # f' DTVMC (VMC time step)                    :  1.0000E-02\n'
            # f' OPT_DTVMC (VMC time-step optimization)   :  1\n'
            # f' VMC_NSTEP (num VMC steps)                :  10000000\n'
            # f' VMC_NCONFIG_WRITE (num configs to write) :  0\n'
            # f' VMC_NBLOCK (num VMC blocks)              :  1\n'
            # f' VMC_EQUIL_NSTEP (num equil steps)        :  5000\n'
            # f' VMC_DECORR_PERIOD (length of inner loop) :  1\n'
            # f' VMC_AVE_PERIOD (hist reduction factor)   :  1\n'
            # f' VMC_SAMPLING                             :  standard\n'
            # f' MAKEMOVIE                                :  F\n'
            # f' FORCES                                   :  F\n'
        )
