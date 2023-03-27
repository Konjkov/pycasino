import os


class Input:
    """Input reader from file."""

    def __init__(self):
        """Default values"""
        self.vmc_method = 1
        self.dmc_method = 1
        self.limdmc = 4
        self.vmc_decorr_period = 3
        self.ebest_av_window = 25
        self.nucleus_gf_mods = True
        self.cusp_correction = None
        self.cusp_threshold = 1e-7
        self.use_gpcc = None
        self.opt_backflow = False
        self.nucleus_gf_mods = True
        self.alimit = 0.5

    def read(self, base_path):
        def read_bool(line):
            return line.split(':')[1].strip() == 'T'

        def read_int(line):
            return int(line.split(':')[1].strip())

        def read_float(line):
            return float(line.split(':')[1].strip())

        def read_str(line):
            return str(line.split(':')[1].strip())

        file_path = os.path.join(base_path, 'input')
        with open(file_path, 'r') as f:
            for line in f:
                # remove comments
                line = line.partition('#')[0].strip()
                if not line:
                    continue
                if line.startswith('neu'):
                    self.neu = read_int(line)
                elif line.startswith('ned'):
                    self.ned = read_int(line)
                elif line.startswith('vmc_equil_nstep'):
                    self.vmc_equil_nstep = read_int(line)
                elif line.startswith('vmc_nstep'):
                    self.vmc_nstep = read_int(line)
                elif line.startswith('vmc_nblock'):
                    self.vmc_nblock = read_int(line)
                elif line.startswith('atom_basis_type'):
                    self.atom_basis_type = read_str(line)
                elif line.startswith('use_jastrow'):
                    self.use_jastrow = read_bool(line)
                elif line.startswith('use_gjastrow'):
                    self.use_gjastrow = read_bool(line)
                elif line.startswith('backflow'):
                    self.backflow = read_bool(line)
                elif line.startswith('runtype'):
                    self.runtype = read_str(line)
                elif line.startswith('opt_method'):
                    self.opt_jastrow = self.use_jastrow = True
                    self.opt_method = read_str(line)
                elif line.startswith('opt_cycles'):
                    self.opt_cycles = read_int(line)
                elif line.startswith('opt_jastrow'):
                    self.opt_jastrow = self.use_jastrow = read_bool(line)
                elif line.startswith('opt_backflow'):
                    self.opt_backflow = read_bool(line)
                elif line.startswith('vmc_method'):
                    self.vmc_method = read_int(line)
                elif line.startswith('vmc_nconfig_write'):
                    self.vmc_nconfig_write = read_int(line)
                elif line.startswith('dmc_method'):
                    self.dmc_method = read_int(line)
                elif line.startswith('dmc_equil_nstep'):
                    self.dmc_equil_nstep = read_int(line)
                elif line.startswith('dmc_equil_nblock'):
                    self.dmc_equil_nblock = read_int(line)
                elif line.startswith('dmc_stats_nstep'):
                    self.dmc_stats_nstep = read_int(line)
                elif line.startswith('dmc_stats_nblock'):
                    self.dmc_stats_nblock = read_int(line)
                elif line.startswith('dtdmc'):
                    self.dtdmc = read_float(line)
                elif line.startswith('dmc_target_weight'):
                    self.dmc_target_weight = read_float(line)
                elif line.startswith('vmc_decorr_period'):
                    self.vmc_decorr_period = read_int(line)
                elif line.startswith('cusp_correction'):
                    self.cusp_correction = read_bool(line)
                elif line.startswith('nucleus_gf_mods'):
                    self.nucleus_gf_mods = read_bool(line)
                elif line.startswith('alimit'):
                    self.alimit = read_float(line)
        if self.cusp_correction is None:
            if self.atom_basis_type == 'gaussian':
                self.cusp_correction = True
            if self.atom_basis_type == 'slater-type':
                self.cusp_correction = False
