import unittest
from pathlib import Path

import numpy as np
import pytest

from casino.cusp import CuspFactory
from casino.readers import CasinoConfig
from casino.slater import Slater
from casino.wfn import Wfn


class TestCusp(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/He'
        self.config = CasinoConfig(config_path)
        self.config.read()
        cusp_factory = CuspFactory(self.config)
        cusp = cusp_factory.create()
        slater = Slater(self.config, cusp)
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

    def test_gradient(self):
        analytical = self.wfn.slater.cusp.gradient(self.n_vectors)
        numerical = self.wfn.slater.cusp.numerical_gradient(self.n_vectors)
        assert analytical[0] == pytest.approx(numerical[0])
        assert analytical[1] == pytest.approx(numerical[1])

    def test_laplacian(self):
        analytical = self.wfn.slater.cusp.laplacian(self.n_vectors)
        numerical = self.wfn.slater.cusp.numerical_laplacian(self.n_vectors)
        assert analytical[0] == pytest.approx(numerical[0])
        assert analytical[1] == pytest.approx(numerical[1])

    def test_hessian(self):
        analytical = self.wfn.slater.cusp.hessian(self.n_vectors)
        numerical = self.wfn.slater.cusp.numerical_hessian(self.n_vectors)
        assert analytical[0] == pytest.approx(numerical[0])
        assert analytical[1] == pytest.approx(numerical[1])

    def test_tressian(self):
        analytical = self.wfn.slater.cusp.tressian(self.n_vectors)
        numerical = self.wfn.slater.cusp.numerical_tressian(self.n_vectors)
        assert analytical[0] == pytest.approx(numerical[0])
        assert analytical[1] == pytest.approx(numerical[1])


if __name__ == '__main__':
    unittest.main()
