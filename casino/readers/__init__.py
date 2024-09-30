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
        def to_fortran(val):
            return 'T' if val else 'F'

        if self.input.runtype == 'vmc':
            runtype = 'VMC input parameters'
        elif self.input.runtype == 'vmc_opt':
            runtype = 'VMC/optimization input parameters'
        elif self.input.runtype == 'vmc_dmc':
            runtype = 'VMC/DMC input parameters'
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
            f' USE_JASTROW (use Jastrow factor)         :  {to_fortran(self.input.use_jastrow)}\n'
            f' BACKFLOW (use backflow corrections)      :  {to_fortran(self.input.backflow)}\n'
            # f' DBARRC (DBAR recalculation period)       :  100000\n'
            f' USE_ORBMODS (use orbitals modifications) :  F\n'
            f' CUSP_CORRECTION                          :  {to_fortran(self.input.cusp_correction)}\n'
            # f' MOLGSCREENING                            :  F\n'
            # f' USE_DETLA (DLA approx. to NL PP)         :  F\n'
            f' NON_LOCAL_GRID (NL integration grid)     :  {self.input.non_local_grid}\n'
            f' E_OFFSET (energy offset)                 :  0.0000\n'
            # f' ESUPERCELL                               :  F\n'
            # f' GAUTOL  (Gaussian evaluation tolerance)  :  7.0\n'
            f' SPARSE                                   :  F\n'
            f' DIPOLE_MOMENT                            :  F\n'
            f' CHECKPOINT (checkpointing level)         :  1\n'
            # f' CHECKPOINT_NCPU (chkpnt group size)      :  4\n'
            f' CON_LOC (Dir to read/write config.*)     :  ./\n'
            f' RELATIVISTIC                             :  F\n'
            f'\n {runtype}\n'
            f' ====================\n'
            f' NEWRUN (start new run)                   :  T\n'
            f' VMC_METHOD (choice of VMC algorithm)     :  {self.input.vmc_method}\n'
            f' DTVMC (VMC time step)                    :  {self.input.dtvmc}\n'
            f' OPT_DTVMC (VMC time-step optimization)   :  {self.input.opt_dtvmc}\n'
            f' VMC_NSTEP (num VMC steps)                :  {self.input.vmc_nstep}\n'
            f' VMC_NCONFIG_WRITE (num configs to write) :  {self.input.vmc_nconfig_write}\n'
            f' VMC_NBLOCK (num VMC blocks)              :  {self.input.vmc_nblock}\n'
            f' VMC_EQUIL_NSTEP (num equil steps)        :  {self.input.vmc_equil_nstep}\n'
            f' VMC_DECORR_PERIOD (length of inner loop) :  {self.input.vmc_decorr_period}\n'
            f' VMC_AVE_PERIOD (hist reduction factor)   :  1\n'
            f' VMC_SAMPLING                             :  standard'
        )
        if self.input.runtype == 'vmc_opt':
            logger.info(
                f' OPT_CYCLES (num optimization cycles)     :  {self.input.opt_cycles}\n'
                f' POSTFIT_VMC (perform post-fit VMC calc)  :  T\n'
                f' POSTFIT_KEEP_CFG (keep post-fit VMC cfgs):  F\n'
                f' OPT_NOCTF_CYCLES (fixed cutoff cycles)   :  {self.input.opt_noctf_cycles}\n'
                f' OPT_INFO (information level)             :  2\n'
                f' OPT_JASTROW (opt Jastrow factor)         :  {to_fortran(self.input.opt_jastrow)}\n'
                f' OPT_DET_COEFF (opt det coeffs)           :  {to_fortran(self.input.opt_det_coeff)}\n'
                f' OPT_ORBITALS (opt orbitals)              :  {to_fortran(self.input.opt_orbitals)}\n'
                f' OPT_BACKFLOW (opt backflow params)       :  {to_fortran(self.input.opt_backflow)}\n'
                f' OPT_FIXNL (fix nonlocal energy)          :  F\n'
                f' OPT_MAXITER (max num iterations)         :  10\n'
                f' OPT_MAXEVAL (max num evaluations)        :  200\n'
                f' VM_SMOOTH_LIMITS (smooth limiting)       :  T\n'
                f' VM_REWEIGHT (reweighting)                :  {to_fortran(self.input.vm_reweight)}\n'
                f' VM_FILTER (filter outlying configs)      :  F\n'
                f' VM_USE_E_GUESS (use guess energy)        :  F\n'
                f' EMIN_XI_VALUE (xi parameter)             :  1.0'
            )
        elif self.input.runtype == 'vmc_dmc':
            logger.info(
                f' DMC_TARGET_WEIGHT                        :  {self.input.dmc_target_weight}\n'
                f' DMC_MD                                   :  F\n'
                f' DMC_EQUIL_NSTEP (num equil steps)        :  {self.input.dmc_equil_nstep}\n'
                f' DMC_EQUIL_NBLOCK (num blocks)            :  {self.input.dmc_equil_nblock}\n'
                f' DMC_STATS_NSTEP (num stats steps)        :  {self.input.dmc_stats_nstep}\n'
                f' DMC_STATS_NBLOCK (num blocks)            :  {self.input.dmc_stats_nblock}\n'
                f' DMC_DECORR_PERIOD (length of inner loop) :  1\n'
                f' DMC_AVE_PERIOD (hist reduction factor)   :  1\n'
                f' DMC_TRIP_WEIGHT (catastrophe thres)      :  0.00\n'
                f' EBEST_AV_WINDOW (running av for energy)  :  25\n'
                f' DMC_METHOD (choice of DMC algorithm)     :  {self.input.dmc_method}\n'
                f' DMC_REWEIGHT_CONF (Update weights)       :  F\n'
                f' DMC_SPACEWARPING (adjust e to new wfn)   :  F\n'
                f' REDIST_GRP_SIZE (size of redist groups)  :  500\n'
                f' DTDMC (DMC time step)                    :  {self.input.dtdmc}\n'
                f' TPDMC (DMC T_p parameter)                :  0\n'
                f' CEREFDMC (constant for EREF [DMC])       :  1.000\n'
                # f' LIMDMC (limit type for drift vel/energy) :  4 [ZSGMA, PRB 93 241118(R) (2016)]\n'
                f' NUCLEUS_GF_MODS (DMC GF mods for nuclei) :  {to_fortran(self.input.nucleus_gf_mods)}\n'
                f' ALPHALIMIT                               :  {self.input.alimit}\n'
                f' IACCUM (flag for statistics run [DMC])   :  T\n'
                f' IBRAN (flag to enable branching [DMC])   :  T\n'
                f' LWDMC (flag for enabling weighted DMC)   :  F\n'
                f' LWDMC_FIXPOP (fixed population LWDMC)    :  F\n'
                f' DMC_NORM_CONSERVE                        :  F\n'
                f' DMC_POPRENORM (renormalize config popn)  :  F\n'
                f' GROWTH_ESTIMATOR (calc growth estimator) :  F\n'
                f' USE_TMOVE                                :  {self.input.use_tmove}\n'
                f' FUTURE_WALKING                           :  F'
            )
        logger.info(
            f' MAKEMOVIE                                :  F\n'
            f' FORCES                                   :  F\n'
        )
