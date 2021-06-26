

class Input:
    """Input reader from file."""

    def __init__(self, file_name):
        def read_bool(line):
            return line.split(':')[1].strip() == 'T'

        def read_int(line):
            return int(line.split(':')[1].strip())

        def read_str(line):
            return str(line.split(':')[1].strip())

        with open(file_name, 'r') as f:
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
                self.vmc_opt_nstep = 1000000
