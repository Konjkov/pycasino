import numpy as np
import unittest

from casino.readers import CasinoConfig
from casino.slater import Slater
from casino.wfn import Wfn


class TestPPotential(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        config_path = 'inputs/Slater/He'
        self.config = CasinoConfig(config_path)
        self.config.read()
        slater = Slater(self.config, cusp=None)
        self.wfn = Wfn(self.config, slater, jastrow=None, backflow=None, ppotential=None)
        self.wfn.set_parameters_projector()
        position = self.initial_position()
        _, self.n_vectors = self.wfn._relative_coordinates(position)

    def initial_position(self):
        """Initial positions of electrons."""
        ne = self.config.input.neu + self.config.input.ned
        atom_charges = self.config.wfn.atom_charges
        atom_positions = self.config.wfn.atom_positions
        natoms = atom_positions.shape[0]
        r_e = np.zeros((ne, 3))
        for i in range(ne):
            # electrons randomly centered on atoms
            r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
        return r_e + np.random.uniform(-1, 1, ne * 3).reshape(ne, 3)


if __name__ == "__main__":
    unittest.main()
