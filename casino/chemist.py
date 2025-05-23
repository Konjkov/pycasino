import argparse
import os
import sys

import numpy as np

from casino.hartreefock import HartreeFock
from casino.readers import CasinoConfig


class Chemist:
    def __init__(self, config_path: str):
        """Chemist workflow.
        :param config_path: path to config file
        """
        self.config = CasinoConfig(config_path)
        self.config.read()
        self.hf = HartreeFock(self.config)

    def run(self):
        print(np.diag(self.hf.S()))
        print(np.diag(self.hf.T()))
        print(np.diag(self.hf.V()))


def main():
    parser = argparse.ArgumentParser(description='This script run Chemist workflow.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config_path', type=str, help='path to CASINO config dir')
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.config_path, 'input')):
        Chemist(args.config_path).run()
    else:
        print(f'File {args.config_path}input not found...')
        sys.exit(1)
