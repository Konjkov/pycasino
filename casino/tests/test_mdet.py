import unittest
from pathlib import Path

import numpy as np
import pytest

from casino.jastrow import Jastrow
from casino.readers import CasinoConfig
from casino.slater import Slater
from casino.wfn import Wfn


class TestMdet(unittest.TestCase):
    """Derivatives w.r.t. the coefficients of a multideterminant expansion, on the CASSCF(2,4)
    four-determinant Be of examples/gwfn. The Jastrow stays in the wave function and out of the
    parameter vector, so what is compared is the determinant part alone, cross term included.
    """

    def setUp(self):
        np.random.seed(1)
        config_path = Path(__file__).resolve().parent / 'inputs/MDET/Be'
        self.config = CasinoConfig(config_path)
        self.config.read()
        slater = Slater(self.config, cusp=None)
        jastrow = Jastrow(self.config)
        self.wfn = Wfn(self.config, slater, jastrow=jastrow)
        self.wfn.opt_det_coeff = True
        self.wfn.set_parameters_projector()
        self.r_e = self.initial_position()

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

    def test_value_parameters_d1(self):
        analytical = self.wfn.value_parameters_d1(self.r_e)
        numerical = self.wfn.value_parameters_numerical_d1(self.r_e)
        assert analytical == pytest.approx(numerical, rel=1e-5, abs=1e-8)

    def test_energy_parameters_d1(self):
        analytical = self.wfn.energy_parameters_d1(self.r_e)
        numerical = self.wfn.energy_parameters_numerical_d1(self.r_e)
        assert analytical == pytest.approx(numerical, rel=1e-4, abs=1e-6)


if __name__ == '__main__':
    unittest.main()
