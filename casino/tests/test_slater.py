import unittest
from pathlib import Path

import numpy as np

from casino.readers import CasinoConfig
from casino.slater import Slater
from casino.wfn import Wfn


class TestSlater(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        config_path = Path(__file__).resolve().parent / 'inputs/Slater/He'
        self.config = CasinoConfig(config_path)
        self.config.read()
        slater = Slater(self.config, cusp=None)
        self.wfn = Wfn(self.config, slater, jastrow=None, backflow=None, ppotential=None)
        self.wfn.set_parameters_projector()
        self.r_e = self.initial_position()
        _, self.n_vectors = self.wfn._relative_coordinates(self.r_e)

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
        assert np.allclose(self.wfn.slater.gradient(self.n_vectors), self.wfn.slater.numerical_gradient(self.n_vectors))

    def test_laplacian(self):
        assert np.allclose(self.wfn.slater.laplacian(self.n_vectors), self.wfn.slater.numerical_laplacian(self.n_vectors), rtol=0.001)

    def test_hessian(self):
        assert np.allclose(self.wfn.slater.hessian(self.n_vectors)[0], self.wfn.slater.numerical_hessian(self.n_vectors), rtol=0.001)

    def test_tressian(self):
        assert np.allclose(self.wfn.slater.tressian(self.n_vectors)[0], self.wfn.slater.numerical_tressian(self.n_vectors), rtol=0.001)

    def test_tressian_v2(self):
        assert np.allclose(self.wfn.slater.tressian(self.n_vectors)[0], self.wfn.slater.tressian_v2(self.n_vectors)[0])

    def test_wfn_laplacian(self):
        assert np.allclose(self.wfn.kinetic_energy(self.r_e), -self.wfn.numerical_laplacian(self.r_e) / 2, rtol=0.004)


if __name__ == '__main__':
    unittest.main()
