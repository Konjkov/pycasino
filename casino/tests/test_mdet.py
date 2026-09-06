import unittest
from pathlib import Path

import numpy as np
import pytest

from casino.backflow import Backflow
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


class TestMdetBackflow(TestMdet):
    """The same with backflow, where the determinant is read at the quasi-particle coordinates
    and the laplacian of the transformation goes against the gradient taken there, before the
    jacobian is applied to it. Without backflow that branch is never entered.
    """

    def setUp(self):
        super().setUp()
        self.wfn = Wfn(self.config, self.wfn.slater, jastrow=Jastrow(self.config), backflow=Backflow(self.config))
        self.wfn.opt_det_coeff = True
        self.wfn.set_parameters_projector()


class TestMdetSingularDeterminant(unittest.TestCase):
    """A determinant of the expansion may be exactly singular where their sum is not, and then
    its inverse does not exist while the wave function is perfectly finite. Both down electrons
    on the z axis is the cheapest such configuration: the p_x and p_y orbitals are zero for both
    of them, so two of the four determinants have a zero row, and the other two carry the value.
    """

    def setUp(self):
        config = CasinoConfig(Path(__file__).resolve().parent / 'inputs/MDET/Be')
        config.read()
        self.wfn = Wfn(config, Slater(config, cusp=None))
        self.r_e = np.array([[0.3, 0.1, 0.2], [-0.2, 0.4, 0.1], [0.0, 0.0, 0.5], [0.0, 0.0, -0.9]])

    def test_singular_determinant(self):
        assert np.isfinite(self.wfn.log_value(self.r_e)[0])
        assert np.all(np.isfinite(self.wfn.drift_velocity(self.r_e)))
        assert np.isfinite(self.wfn.energy(self.r_e))


if __name__ == '__main__':
    unittest.main()
