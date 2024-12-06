import unittest
from pathlib import Path

import numpy as np

from casino.backflow import Backflow
from casino.jastrow import Jastrow
from casino.readers import CasinoConfig
from casino.slater import Slater
from casino.wfn import Wfn


class TestBackflow(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        config_path = Path(__file__).resolve().parent / 'inputs/Backflow/He'
        self.config = CasinoConfig(config_path)
        self.config.read()
        slater = Slater(self.config, cusp=None)
        jastrow = Jastrow(self.config)
        backflow = Backflow(self.config)
        self.wfn = Wfn(self.config, slater, jastrow, backflow, ppotential=None)
        self.wfn.opt_backflow = True
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
        ne = self.config.input.neu + self.config.input.ned
        assert np.allclose(
            self.wfn.backflow.gradient(self.e_vectors, self.n_vectors)[0],
            self.wfn.backflow.numerical_gradient(self.e_vectors, self.n_vectors) + np.eye(ne * 3),
        )

    def test_laplacian(self):
        assert np.allclose(
            self.wfn.backflow.laplacian(self.e_vectors, self.n_vectors)[0],
            self.wfn.backflow.numerical_laplacian(self.e_vectors, self.n_vectors),
            rtol=0.001,
        )

    def test_value_parameters_d1(self):
        assert np.allclose(
            self.wfn.backflow.value_parameters_d1(self.e_vectors, self.n_vectors),
            self.wfn.backflow.value_parameters_numerical_d1(self.e_vectors, self.n_vectors, False),
        )

    def test_gradient_parameters_d1(self):
        projector = self.wfn.backflow.parameters_projector.T
        gradient_parameters_d1 = self.wfn.backflow.gradient_parameters_d1(self.e_vectors, self.n_vectors)[0]
        gradient_parameters_d1 = gradient_parameters_d1.reshape(gradient_parameters_d1.shape[0], -1)
        gradient_parameters_numerical_d1 = self.wfn.backflow.gradient_parameters_numerical_d1(self.e_vectors, self.n_vectors, False).reshape(
            projector.shape[0], -1
        )
        assert np.allclose(projector @ gradient_parameters_d1, gradient_parameters_numerical_d1)

    def test_laplacian_parameters_d1(self):
        assert np.allclose(
            self.wfn.backflow.parameters_projector.T @ self.wfn.backflow.laplacian_parameters_d1(self.e_vectors, self.n_vectors)[0],
            self.wfn.backflow.laplacian_parameters_numerical_d1(self.e_vectors, self.n_vectors, False),
        )

    def test_wfn_value_parameters_d1(self):
        assert np.allclose(self.wfn.value_parameters_d1(self.r_e), self.wfn.value_parameters_numerical_d1(self.r_e))

    def test_wfn_energy_parameters_d1(self):
        assert np.allclose(self.wfn.energy_parameters_d1(self.r_e), self.wfn.energy_parameters_numerical_d1(self.r_e))


if __name__ == '__main__':
    unittest.main()
