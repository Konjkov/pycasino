import os


class Input:
    """Input reader from file."""

    def __init__(self):
        """Default values"""
        self.lines = []

    def read_bool(self, keyword, value=None):
        for line in self.lines:
            if line.startswith(keyword):
                value = line.split(':')[1].strip() == 'T'
        setattr(self, keyword, value)

    def read_int(self, keyword, value=None):
        for line in self.lines:
            if line.startswith(keyword):
                value = int(line.split(':')[1].strip())
        setattr(self, keyword, value)

    def read_float(self, keyword, value=None):
        for line in self.lines:
            if line.startswith(keyword):
                value = float(line.split(':')[1].strip())
        setattr(self, keyword, value)

    def read_str(self, keyword, value=None):
        for line in self.lines:
            if line.startswith(keyword):
                value = str(line.split(':')[1].strip())
        setattr(self, keyword, value)

    def read_opt_plan(self):
        value = []
        block_start = False
        for line in self.lines:
            if line.startswith('%block opt_plan'):
                block_start = True
                continue
            elif line.startswith('%endblock opt_plan'):
                break
            if block_start:
                block_line = dict()
                for i, token in enumerate(line.split()):
                    if i == 0:
                        continue
                    k, v = token.split('=')
                    if v.startswith('T'):
                        block_line[k] = True
                    elif v.startswith('F'):
                        block_line[k] = False
                    elif k == 'method':
                        block_line[k] = v
                    else:
                        block_line[k] = float(v)
                value.append(block_line)
        setattr(self, 'opt_plan', value)

    def read(self, base_path):
        """Read input config."""
        self.file_path = os.path.join(base_path, 'input')
        with open(self.file_path, 'r') as f:
            # remove comments
            self.lines = [line.partition('#')[0].strip() for line in f if line.partition('#')[0].strip()]
        # General keywords
        self.read_int('neu')
        self.read_int('ned')
        self.read_str('runtype')
        self.read_bool('testrun', False)
        self.read_str('atom_basis_type')
        self.read_float('gautol', 7.0)
        # VMC keywords
        self.read_int('vmc_equil_nstep')
        self.read_int('vmc_nstep')
        self.read_int('vmc_decorr_period', 3)
        self.read_int('vmc_nblock')
        self.read_int('vmc_nconfig_write')
        self.read_float('dtvmc', 0.02)
        self.read_bool('opt_dtvmc', True)
        self.read_int('vmc_method', 1)
        # Optimization keywords
        self.read_str('opt_method')
        self.read_str('emin_method', 'linear')
        self.read_opt_plan()
        self.read_int('opt_cycles', len(self.opt_plan))
        self.read_bool('postfit_vmc', True)
        self.read_int('opt_noctf_cycles', 0)
        self.read_bool('opt_jastrow', bool(self.opt_method))
        self.read_bool('opt_det_coeff', False)
        self.read_bool('opt_orbitals', False)
        self.read_bool('opt_backflow', False)
        self.read_bool('opt_fixnl', self.opt_method == 'varmin')
        self.read_int('opt_maxiter', 10)
        self.read_int('opt_maxeval', 200)
        self.read_bool('vm_smooth_limit', True)
        self.read_bool('vm_reweight', False)
        self.read_bool('vm_w_max', 0.0)
        self.read_bool('vm_w_min', 0.0)
        self.read_bool('emin_xi_value', 1.0)
        # DMC keywords
        self.read_float('dmc_target_weight')
        self.read_int('dmc_equil_nstep')
        self.read_int('dmc_stats_nstep')
        self.read_int('dmc_equil_nblock')
        self.read_int('dmc_stats_nblock')
        self.read_float('dtdmc')
        self.read_int('dmc_method', 1)
        self.read_int('limdmct', 4)
        self.read_float('alimit', 0.5)
        self.read_bool('nucleus_gf_mods', True)
        self.read_bool('use_tmove', True)
        self.read_int('ebest_av_window', 25)
        # WFN definition keywords
        self.read_bool('backflow', self.opt_backflow)
        self.read_bool('use_jastrow', self.opt_jastrow)
        self.read_bool('use_gjastrow', False)
        # Cusp correction keywords
        self.read_bool('cusp_correction', self.atom_basis_type == 'gaussian')
        self.read_float('cusp_threshold', 1e-7)
        self.read_bool('cusp_info', False)
        # Pseudopotential keywords
        self.read_bool('non_local_grid', 4)
        self.read_float('lcutofftol', 1e-5)
        self.read_float('nlcutofftol', 1e-5)

        self.ppotential = False
        for file_name in os.listdir(base_path):
            if file_name.endswith('_pp.data'):
                self.ppotential = True

    def log(self):
        """Write log"""
        def to_fortran(val):
            return 'T' if val else 'F'
        if self.runtype == 'vmc':
            runtype = 'VMC input parameters'
        elif self.runtype == 'vmc_opt':
            runtype = 'VMC/optimization input parameters'
        elif self.runtype == 'vmc_dmc':
            runtype = 'VMC/DMC input parameters'
        msg = (
            f' General input parameters\n'
            f' ========================\n'
            f' NEU (num up spin electrons)              :  {self.neu}\n'
            f' NED (num down spin electrons)            :  {self.ned}\n'
            f' RUNTYPE (type of run)                    :  {self.runtype}\n'
            f' PSI_S  (form for [anti]symmetrizing wfn) :  slater\n'
            f' ATOM_BASIS_TYPE (atom-centred orb basis) :  {self.atom_basis_type}\n'
            f' INTERACTION (interaction type)           :  coulomb\n'
            f' TESTRUN (read input data,print and stop) :  F\n'
            f' PERIODIC                                 :  F\n'
            f' COMPLEX_WF (complex Slater wave fn.)     :  F\n'
            f' NEIGHPRINT (neighbour analysis)          :  0\n'
            f' USE_JASTROW (use Jastrow factor)         :  {to_fortran(self.use_jastrow)}\n'
            f' BACKFLOW (use backflow corrections)      :  {to_fortran(self.backflow)}\n'
            # f' DBARRC (DBAR recalculation period)       :  100000\n'
            f' USE_ORBMODS (use orbitals modifications) :  F\n'
            f' CUSP_CORRECTION                          :  {to_fortran(self.cusp_correction)}\n'
            # f' MOLGSCREENING                            :  F\n'
            # f' USE_DETLA (DLA approx. to NL PP)         :  F\n'
            f' NON_LOCAL_GRID (NL integration grid)     :  {self.non_local_grid}\n'
            f' E_OFFSET (energy offset)                 :  0.0000\n'
            # f' ESUPERCELL                               :  F\n'
            f' GAUTOL  (Gaussian evaluation tolerance)  :  {self.gautol}\n'
            f' SPARSE                                   :  F\n'
            f' DIPOLE_MOMENT                            :  F\n'
            # f' CHECKPOINT (checkpointing level)         :  1\n'
            # f' CHECKPOINT_NCPU (chkpnt group size)      :  4\n'
            # f' CON_LOC (Dir to read/write config.*)     :  ./\n'
            f' RELATIVISTIC                             :  F\n'
            f'\n {runtype}\n'
            f' ====================\n'
            f' NEWRUN (start new run)                   :  T\n'
            f' VMC_METHOD (choice of VMC algorithm)     :  {self.vmc_method}\n'
            f' DTVMC (VMC time step)                    :  {self.dtvmc}\n'
            f' OPT_DTVMC (VMC time-step optimization)   :  {to_fortran(self.opt_dtvmc)}\n'
            f' VMC_NSTEP (num VMC steps)                :  {self.vmc_nstep}\n'
            f' VMC_NCONFIG_WRITE (num configs to write) :  {self.vmc_nconfig_write}\n'
            f' VMC_NBLOCK (num VMC blocks)              :  {self.vmc_nblock}\n'
            f' VMC_EQUIL_NSTEP (num equil steps)        :  {self.vmc_equil_nstep}\n'
            f' VMC_DECORR_PERIOD (length of inner loop) :  {self.vmc_decorr_period}\n'
            f' VMC_AVE_PERIOD (hist reduction factor)   :  1\n'
            f' VMC_SAMPLING                             :  standard\n'
        )
        if self.runtype == 'vmc_opt':
            msg += (
                f' OPT_CYCLES (num optimization cycles)     :  {self.opt_cycles}\n'
                f' POSTFIT_VMC (perform post-fit VMC calc)  :  {to_fortran(self.postfit_vmc)}\n'
                f' POSTFIT_KEEP_CFG (keep post-fit VMC cfgs):  F\n'
                f' OPT_NOCTF_CYCLES (fixed cutoff cycles)   :  {self.opt_noctf_cycles}\n'
                f' OPT_INFO (information level)             :  2\n'
                f' OPT_JASTROW (opt Jastrow factor)         :  {to_fortran(self.opt_jastrow)}\n'
                f' OPT_DET_COEFF (opt det coeffs)           :  {to_fortran(self.opt_det_coeff)}\n'
                f' OPT_ORBITALS (opt orbitals)              :  {to_fortran(self.opt_orbitals)}\n'
                f' OPT_BACKFLOW (opt backflow params)       :  {to_fortran(self.opt_backflow)}\n'
                f' OPT_FIXNL (fix nonlocal energy)          :  {to_fortran(self.opt_fixnl)}\n'
                f' OPT_MAXITER (max num iterations)         :  {self.opt_maxiter}\n'
                f' OPT_MAXEVAL (max num evaluations)        :  {self.opt_maxeval}\n'
                f' VM_SMOOTH_LIMITS (smooth limiting)       :  F\n'
                f' VM_REWEIGHT (reweighting)                :  {to_fortran(self.vm_reweight)}\n'
                f' VM_FILTER (filter outlying configs)      :  F\n'
                f' VM_USE_E_GUESS (use guess energy)        :  F\n'
                f' EMIN_XI_VALUE (xi parameter)             :  {self.emin_xi_value}\n'
            )
        elif self.runtype == 'vmc_dmc':
            msg += (
                f' DMC_TARGET_WEIGHT                        :  {self.dmc_target_weight}\n'
                f' DMC_MD                                   :  F\n'
                f' DMC_EQUIL_NSTEP (num equil steps)        :  {self.dmc_equil_nstep}\n'
                f' DMC_EQUIL_NBLOCK (num blocks)            :  {self.dmc_equil_nblock}\n'
                f' DMC_STATS_NSTEP (num stats steps)        :  {self.dmc_stats_nstep}\n'
                f' DMC_STATS_NBLOCK (num blocks)            :  {self.dmc_stats_nblock}\n'
                f' DMC_DECORR_PERIOD (length of inner loop) :  1\n'
                f' DMC_AVE_PERIOD (hist reduction factor)   :  1\n'
                f' DMC_TRIP_WEIGHT (catastrophe thres)      :  0.00\n'
                f' EBEST_AV_WINDOW (running av for energy)  :  25\n'
                f' DMC_METHOD (choice of DMC algorithm)     :  {self.dmc_method}\n'
                f' DMC_REWEIGHT_CONF (Update weights)       :  F\n'
                f' DMC_SPACEWARPING (adjust e to new wfn)   :  F\n'
                f' REDIST_GRP_SIZE (size of redist groups)  :  500\n'
                f' DTDMC (DMC time step)                    :  {self.dtdmc}\n'
                f' TPDMC (DMC T_p parameter)                :  0\n'
                f' CEREFDMC (constant for EREF [DMC])       :  1.000\n'
                # f' LIMDMC (limit type for drift vel/energy) :  4 [ZSGMA, PRB 93 241118(R) (2016)]\n'
                f' NUCLEUS_GF_MODS (DMC GF mods for nuclei) :  {to_fortran(self.nucleus_gf_mods)}\n'
                f' ALPHALIMIT                               :  {self.alimit}\n'
                f' IACCUM (flag for statistics run [DMC])   :  T\n'
                f' IBRAN (flag to enable branching [DMC])   :  T\n'
                f' LWDMC (flag for enabling weighted DMC)   :  F\n'
                f' LWDMC_FIXPOP (fixed population LWDMC)    :  F\n'
                f' DMC_NORM_CONSERVE                        :  F\n'
                f' DMC_POPRENORM (renormalize config popn)  :  F\n'
                f' GROWTH_ESTIMATOR (calc growth estimator) :  F\n'
                f' USE_TMOVE                                :  {self.use_tmove}\n'
                f' FUTURE_WALKING                           :  F\n'
            )
        msg += (
            ' MAKEMOVIE                                :  F\n'
            ' FORCES                                   :  F\n'
        )
        return msg
