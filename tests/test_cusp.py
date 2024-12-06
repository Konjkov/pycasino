import numpy as np
import unittest

from casino.readers import CasinoConfig
from casino.cusp import CuspFactory
from casino.slater import Slater
from casino.wfn import Wfn


class TestCusp(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        config_path = 'inputs/Cusp/He'
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
        assert np.allclose(self.wfn.slater.cusp.gradient(self.n_vectors), self.wfn.slater.cusp.numerical_gradient(self.n_vectors))

    def test_laplacian(self):
        assert np.allclose(self.wfn.slater.cusp.laplacian(self.n_vectors), self.wfn.slater.cusp.numerical_laplacian(self.n_vectors))

    def test_hessian(self):
        assert np.allclose(self.wfn.slater.cusp.hessian(self.n_vectors)[0], self.wfn.slater.cusp.numerical_hessian(self.n_vectors))

    def test_tressian(self):
        assert np.allclose(self.wfn.slater.cusp.tressian(self.n_vectors)[0], self.wfn.slater.cusp.numerical_tressian(self.n_vectors))


if __name__ == "__main__":
    unittest.main()
