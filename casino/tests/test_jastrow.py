import unittest
from pathlib import Path

import numpy as np

from casino.jastrow import Jastrow
from casino.readers import CasinoConfig
from casino.slater import Slater
from casino.wfn import Wfn


class TestJastrow(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        config_path = Path(__file__).resolve().parent / 'inputs/Jastrow/He'
        self.config = CasinoConfig(config_path)
        self.config.read()
        slater = Slater(self.config, cusp=None)
        jastrow = Jastrow(self.config)
        self.wfn = Wfn(self.config, slater, jastrow, backflow=None, ppotential=None)
        self.wfn.opt_jastrow = True
        self.wfn.set_parameters_projector()
        self.r_e = self.initial_position()
        self.e_vectors, self.n_vectors = self.wfn._relative_coordinates(self.r_e)

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
        assert np.allclose(
            self.wfn.jastrow.gradient(self.e_vectors, self.n_vectors), self.wfn.jastrow.numerical_gradient(self.e_vectors, self.n_vectors)
        )

    def test_laplacian(self):
        assert np.allclose(
            self.wfn.jastrow.laplacian(self.e_vectors, self.n_vectors)[0],
            self.wfn.jastrow.numerical_laplacian(self.e_vectors, self.n_vectors),
            rtol=0.001,
        )

    def test_value_parameters_d1(self):
        assert np.allclose(
            self.wfn.jastrow.value_parameters_d1(self.e_vectors, self.n_vectors),
            self.wfn.jastrow.value_parameters_numerical_d1(self.e_vectors, self.n_vectors, False),
        )

    def test_gradient_parameters_d1(self):
        assert np.allclose(
            self.wfn.jastrow.gradient_parameters_d1(self.e_vectors, self.n_vectors),
            self.wfn.jastrow.gradient_parameters_numerical_d1(self.e_vectors, self.n_vectors, False),
        )

    def test_laplacian_parameters_d1(self):
        assert np.allclose(
            self.wfn.jastrow.laplacian_parameters_d1(self.e_vectors, self.n_vectors),
            self.wfn.jastrow.laplacian_parameters_numerical_d1(self.e_vectors, self.n_vectors, False),
        )

    def test_wfn_laplacian(self):
        assert np.allclose(self.wfn.kinetic_energy(self.r_e), -self.wfn.numerical_laplacian(self.r_e) / 2, rtol=0.0001)

    def test_wfn_value_parameters_d1(self):
        assert np.allclose(self.wfn.value_parameters_d1(self.r_e), self.wfn.value_parameters_numerical_d1(self.r_e))

    def test_wfn_energy_parameters_d1(self):
        assert np.allclose(self.wfn.energy_parameters_d1(self.r_e), self.wfn.energy_parameters_numerical_d1(self.r_e))


if __name__ == '__main__':
    unittest.main()
