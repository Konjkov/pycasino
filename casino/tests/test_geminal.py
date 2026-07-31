import unittest
from pathlib import Path

import numpy as np
import pytest

from casino.geminal import Geminal
from casino.readers import CasinoConfig
from casino.readers.geminal import Geminal as GeminalReader
from casino.slater import Slater
from casino.wfn import Wfn


class TestGeminal(unittest.TestCase):
    """The Hartree-Fock default geminal (lambda = 1 on the occupied orbitals) must
    reproduce the Slater single determinant value, gradient and Laplacian."""

    def setUp(self):
        np.random.seed(1)
        config_path = Path(__file__).resolve().parent / 'inputs/Slater/He'
        self.config = CasinoConfig(config_path)
        self.config.read()
        self.config.geminal = GeminalReader(self.config.input.neu, self.config.input.ned)
        slater = Slater(self.config, cusp=None)
        self.geminal = Geminal(self.config)
        self.wfn = Wfn(self.config, slater, geminal=self.geminal)
        self.wfn.set_parameters_projector()
        self.r_e = self.initial_position()
        _, self.n_vectors = self.wfn._relative_coordinates(self.r_e)

    def initial_position(self):
        ne = self.config.input.neu + self.config.input.ned
        atom_charges = self.config.wfn.atom_charges
        atom_positions = self.config.wfn.atom_positions
        natoms = atom_positions.shape[0]
        r_e = np.zeros((ne, 3))
        for i in range(ne):
            r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
        return r_e + np.random.uniform(-1, 1, ne * 3).reshape(ne, 3)

    def test_value(self):
        assert self.geminal.value(self.n_vectors) == pytest.approx(self.wfn.slater.value(self.n_vectors))

    def test_gradient_vs_slater(self):
        assert self.geminal.gradient(self.n_vectors) == pytest.approx(self.wfn.slater.gradient(self.n_vectors))

    def test_laplacian_vs_slater(self):
        assert self.geminal.laplacian(self.n_vectors) == pytest.approx(self.wfn.slater.laplacian(self.n_vectors))

    def test_gradient(self):
        assert self.geminal.gradient(self.n_vectors) == pytest.approx(self.geminal.numerical_gradient(self.n_vectors))

    def test_laplacian(self):
        assert self.geminal.laplacian(self.n_vectors) == pytest.approx(self.geminal.numerical_laplacian(self.n_vectors), rel=1e-5)


if __name__ == '__main__':
    unittest.main()
