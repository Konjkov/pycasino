

class Input:
    """Input reader from file."""

    def __init__(self, file_name):
        def read_int(line):
            return int(line.split(':')[1])

        with open(file_name, 'r') as fp:
            for line in fp:
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
