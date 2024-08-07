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
            if line.startswith(f'%block opt_plan'):
                block_start = True
                continue
            elif line.startswith(f'%endblock opt_plan'):
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
        self.file_path = os.path.join(base_path, 'input')
        with open(self.file_path, 'r') as f:
            # remove comments
            self.lines = [line.partition('#')[0].strip() for line in f if line.partition('#')[0].strip()]
        # General keywords
        self.read_int('neu')
        self.read_int('ned')
        self.read_str('atom_basis_type')
        self.read_str('runtype')
        self.read_bool('testrun', False)
        # VMC keywords
        self.read_int('vmc_equil_nstep')
        self.read_int('vmc_nstep')
        self.read_int('vmc_decorr_period', 3)
        self.read_int('vmc_nblock')
        self.read_int('vmc_nconfig_write')
        self.read_float('dtvmc')
        self.read_int('vmc_method', 1)
        # Optimization keywords
        self.read_int('opt_cycles')
        self.read_str('opt_method')
        self.read_str('emin_method', 'linear')
        self.read_bool('opt_jastrow', bool(self.opt_method))
        self.read_bool('opt_backflow', False)
        self.read_bool('opt_orbitals', False)
        self.read_bool('opt_det_coeff', False)
        self.read_int('opt_maxeval', 40)
        self.read_int('opt_maxiter', 10)
        self.read_opt_plan()
        self.read_int('opt_noctf_cycles', 0)
        self.read_bool('opt_fixnl', self.opt_method == 'varmin')
        self.read_bool('vm_reweight', False)
        self.read_bool('vm_w_max', 0.0)
        self.read_bool('vm_w_min', 0.0)
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
