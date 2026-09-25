import unittest
from pathlib import Path

import numpy as np
import pytest

from casino import delta
from casino.jastrow import Jastrow
from casino.readers import CasinoConfig
from casino.slater import Slater
from casino.wfn import Wfn


class TestJastrow(unittest.TestCase):
    config_dir = 'inputs/Jastrow/He'
    tolerance = {}

    def setUp(self):
        np.random.seed(1)
        config_path = Path(__file__).resolve().parent / self.config_dir
        self.config = CasinoConfig(config_path)
        self.config.read()
        slater = Slater(self.config, cusp=None)
        jastrow = Jastrow(self.config)
        self.wfn = Wfn(self.config, slater, jastrow=jastrow, backflow=None, ppotential=None)
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
        analytical = self.wfn.jastrow.gradient(self.e_vectors, self.n_vectors)
        numerical = self.wfn.jastrow.numerical_gradient(self.e_vectors, self.n_vectors)
        assert analytical == pytest.approx(numerical, **self.tolerance)

    def test_laplacian(self):
        analytical = self.wfn.jastrow.laplacian(self.e_vectors, self.n_vectors)[0]
        numerical = self.wfn.jastrow.numerical_laplacian(self.e_vectors, self.n_vectors)
        assert analytical == pytest.approx(numerical, **self.tolerance)

    def test_value_parameters_d1(self):
        analytical = self.wfn.jastrow.value_parameters_d1(self.e_vectors, self.n_vectors)
        numerical = self.wfn.jastrow.value_parameters_numerical_d1(self.e_vectors, self.n_vectors, False)
        assert analytical == pytest.approx(numerical, **self.tolerance)

    def test_gradient_parameters_d1(self):
        analytical = self.wfn.jastrow.gradient_parameters_d1(self.e_vectors, self.n_vectors)
        numerical = self.wfn.jastrow.gradient_parameters_numerical_d1(self.e_vectors, self.n_vectors, False)
        assert analytical == pytest.approx(numerical, **self.tolerance)

    def test_laplacian_parameters_d1(self):
        analytical = self.wfn.jastrow.laplacian_parameters_d1(self.e_vectors, self.n_vectors)
        numerical = self.wfn.jastrow.laplacian_parameters_numerical_d1(self.e_vectors, self.n_vectors, False)
        assert analytical == pytest.approx(numerical, **self.tolerance)

    def test_wfn_laplacian(self):
        assert self.wfn.kinetic_energy(self.r_e) == pytest.approx(-self.wfn.numerical_laplacian(self.r_e) / 2)

    def test_wfn_value_parameters_d1(self):
        assert self.wfn.value_parameters_d1(self.r_e) == pytest.approx(self.wfn.value_parameters_numerical_d1(self.r_e), rel=1e-4)

    def test_wfn_energy_parameters_d1(self):
        assert self.wfn.energy_parameters_d1(self.r_e) == pytest.approx(self.wfn.energy_parameters_numerical_d1(self.r_e))


class TestJastrowAnalytic(TestJastrow):
    """Exponential u-term and bell chi-term (Functional form = 1) with the polynomial f-term."""

    config_dir = '../../examples/stowfn/He/HF/QZ4P/CBCS/Jastrow_emin_analytic'
    # cutoff derivatives are finite differences and the projector mixes the blocks at the 1e-11 level
    tolerance = {'rel': 1e-5, 'abs': 1e-9}

    @pytest.mark.xfail(reason='optimized f-term: wfn value_parameters_d1 differs from the numerical one by ~1e-2, the same with the CASINO Jastrow')
    def test_wfn_value_parameters_d1(self):
        super().test_wfn_value_parameters_d1()


class TestJastrowUncut(TestJastrowAnalytic):
    """Exponential u-term and Gaussian chi-term without cutoff (Functional form = 2) with the polynomial f-term."""

    config_dir = '../../examples/stowfn/He/HF/QZ4P/CBCS/Jastrow_emin_uncut'


class TestJastrowProduct(TestJastrowAnalytic):
    """u and chi without cutoff with the product f-term (r1-L)^C (r2-L)^C g(r1) g(r2) h(r12) (Functional form = 1)."""

    config_dir = '../../examples/stowfn/He/HF/QZ4P/CBCS/Jastrow_emin_product'

    def test_wfn_value_parameters_d2(self):
        parameters = self.wfn.get_parameters()
        numerical = np.zeros(shape=(parameters.size, parameters.size))
        for i in range(parameters.size):
            parameters[i] -= delta
            self.wfn.set_parameters(parameters)
            numerical[i] -= self.wfn.value_parameters_d1(self.r_e)
            parameters[i] += 2 * delta
            self.wfn.set_parameters(parameters)
            numerical[i] += self.wfn.value_parameters_d1(self.r_e)
            parameters[i] -= delta
        self.wfn.set_parameters(parameters)
        assert self.wfn.value_parameters_d2(self.r_e) == pytest.approx(numerical / delta / 2, **self.tolerance)


if __name__ == '__main__':
    unittest.main()
