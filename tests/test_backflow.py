import numpy as np
import unittest

from casino.readers import CasinoConfig
from casino.slater import Slater
from casino.jastrow import Jastrow
from casino.backflow import Backflow
from casino.wfn import Wfn


class TestBackflow(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        config_path = 'inputs/Backflow/He'
        self.config = CasinoConfig(config_path)
        self.config.read()
        slater = Slater(self.config, cusp=None)
        jastrow = Jastrow(self.config)
        backflow = Backflow(self.config)
        self.wfn = Wfn(self.config, slater, jastrow, backflow, ppotential=None)
        self.wfn.set_parameters_projector()
        position = self.initial_position()
        self.e_vectors, self.n_vectors = self.wfn._relative_coordinates(position)

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

    def test_gradient(self):
        assert np.allclose(self.wfn.backflow.gradient(self.e_vectors, self.n_vectors), self.wfn.backflow.numerical_gradient(self.e_vectors, self.n_vectors))


if __name__ == "__main__":
    unittest.main()
