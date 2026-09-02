import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pytest

from casino.cusp import CuspFactory
from casino.geminal import Geminal
from casino.readers import CasinoConfig
from casino.readers.geminal import Geminal as GeminalReader
from casino.slater import Slater
from casino.wfn import Wfn


def initial_position(config):
    """Initial positions of electrons."""
    ne = config.input.neu + config.input.ned
    atom_charges = config.wfn.atom_charges
    atom_positions = config.wfn.atom_positions
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        # electrons randomly centered on atoms
        r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
    return r_e + np.random.uniform(-1, 1, ne * 3).reshape(ne, 3)


class HartreeFockGeminal:
    """The Hartree-Fock default geminal (lambda = 1 on the occupied orbitals, one column per
    unpaired electron) must reproduce the Slater single determinant value, gradient and Laplacian,
    with or without the cusp correction of the orbital pool.
    """

    config_path = None
    cusp = False

    def setUp(self):
        np.random.seed(1)
        self.config = CasinoConfig(Path(__file__).resolve().parent / self.config_path)
        self.config.read()
        self.config.geminal = GeminalReader(self.config.input.neu, self.config.input.ned)
        if self.cusp:
            slater_cusp = CuspFactory(self.config).create()
            geminal_cusp = CuspFactory(self.config, self.config.geminal.norb).create()
        else:
            slater_cusp = geminal_cusp = None
        slater = Slater(self.config, slater_cusp)
        self.geminal = Geminal(self.config, geminal_cusp)
        self.wfn = Wfn(self.config, slater, geminal=self.geminal)
        self.wfn.set_parameters_projector()
        self.r_e = initial_position(self.config)
        _, self.n_vectors = self.wfn._relative_coordinates(self.r_e)

    def test_value(self):
        assert self.geminal.value(self.n_vectors) == pytest.approx(self.wfn.slater.value(self.n_vectors))

    def test_gradient_vs_slater(self):
        assert self.geminal.gradient(self.n_vectors) == pytest.approx(self.wfn.slater.gradient(self.n_vectors))

    def test_laplacian_vs_slater(self):
        assert self.geminal.laplacian(self.n_vectors) == pytest.approx(self.wfn.slater.laplacian(self.n_vectors)[0])

    def test_gradient(self):
        assert self.geminal.gradient(self.n_vectors) == pytest.approx(self.geminal.numerical_gradient(self.n_vectors))

    def test_laplacian(self):
        assert self.geminal.laplacian(self.n_vectors) == pytest.approx(self.geminal.numerical_laplacian(self.n_vectors), rel=1e-5)


class TestGeminalHe(HartreeFockGeminal, unittest.TestCase):
    config_path = 'inputs/Slater/He'


class TestGeminalNe(HartreeFockGeminal, unittest.TestCase):
    """Closed shell on a gaussian basis: l > 0 orbitals and a pool wider than one column."""

    config_path = 'inputs/Cusp/Ne'


class TestGeminalNeCusp(HartreeFockGeminal, unittest.TestCase):
    """The pool of the geminal carries the same cusp correction as the determinant."""

    config_path = 'inputs/Cusp/Ne'
    cusp = True


class TestGeminalN(HartreeFockGeminal, unittest.TestCase):
    """Open shell, neu = 5 and ned = 2: three unpaired columns."""

    config_path = 'inputs/Cusp/N'


class GeminalParameters:
    """Derivatives w.r.t. the geminal parameters against finite differences of the wave function
    taken through the parameter interface of Wfn.
    """

    config_path = None

    def setUp(self):
        np.random.seed(1)
        self.config = CasinoConfig(Path(__file__).resolve().parent / self.config_path)
        self.config.read()
        self.config.geminal = GeminalReader(self.config.input.neu, self.config.input.ned)
        # the whole matrix is optimizable, the off-diagonal elements starting from zero
        self.config.geminal.c_mask[:] = True
        self.config.geminal.g_mask[:] = self.config.geminal.g_available[:] = True
        self.config.geminal.u_mask[:] = self.config.geminal.u_available[:] = True
        slater = Slater(self.config, cusp=None)
        self.geminal = Geminal(self.config)
        self.wfn = Wfn(self.config, slater, geminal=self.geminal)
        self.wfn.opt_geminal = True
        self.wfn.set_parameters_projector()
        self.r_e = initial_position(self.config)

    def test_value_parameters_d1(self):
        # the off-diagonal elements start from zero, where some of the derivatives vanish, so the
        # two sets of finite differences are compared with an absolute tolerance as well
        analytical = self.wfn.value_parameters_d1(self.r_e)
        numerical = self.wfn.value_parameters_numerical_d1(self.r_e)
        assert analytical == pytest.approx(numerical, rel=1e-5, abs=1e-8)

    def test_energy_parameters_d1(self):
        analytical = self.wfn.energy_parameters_d1(self.r_e)
        numerical = self.wfn.energy_parameters_numerical_d1(self.r_e)
        assert analytical == pytest.approx(numerical, rel=1e-4, abs=1e-6)


class TestGeminalParametersBe(GeminalParameters, unittest.TestCase):
    config_path = 'inputs/Cusp/Be'


class TestGeminalParametersN(GeminalParameters, unittest.TestCase):
    """The unpaired columns are parameters of their own."""

    config_path = 'inputs/Cusp/N'


class TestGeminalConstraints(unittest.TestCase):
    """A Constraints group is determined by the one member that carries an explicit flag: it gives
    the whole group its value and its optimizability, and only it is an independent parameter.
    """

    casl = textwrap.dedent("""\
        GEMINAL:
          Default g optimizability: fixed
          Default c optimizability: fixed
          Geminal 1:
            Parameters:
              c: [ 1.0, fixed ]
              g_1,1: [ 1.0, fixed ]
              g_2,2: [ 1.0, fixed ]
          Geminal 2:
            Parameters:
              c: [ -1.0, fixed ]
              g_3,3: [ -0.05, optimizable ]
              g_3,5: [ 0.02, optimizable ]
          Constraints:
            2^g_3,3=2^g_4,4
            2^g_3,5=2^g_5,3=2^g_4,6=2^g_6,4
        """)

    def read(self, casl):
        geminal = GeminalReader(2, 2)
        with tempfile.TemporaryDirectory() as base_path:
            with open(Path(base_path) / 'parameters.casl', 'w') as f:
                f.write(casl)
            geminal.read(base_path)
        return geminal

    def test_determined_values(self):
        geminal = self.read(self.casl)
        assert geminal.norb == 6
        assert geminal.g[1, 3, 3] == -0.05
        assert geminal.g[1, 3, 5] == geminal.g[1, 5, 3] == 0.02

    def test_determined_are_dependent(self):
        geminal = self.read(self.casl)
        assert geminal.g_mask[1, 2, 2] and not geminal.g_mask[1, 3, 3]
        assert geminal.g_mask[1, 2, 4] and not geminal.g_mask[1, 3, 5]
        assert geminal.g_available[1, 3, 3] and geminal.g_available[1, 3, 5]
        assert geminal.g_ties.shape == (2, 6)

    def test_write_skips_determined(self):
        text = self.read(self.casl).write()
        assert 'g_3,3: ' in text and 'g_4,4: ' not in text
        assert 'g_3,5: ' in text and 'g_4,6: ' not in text
        assert '2^g_3,3=2^g_4,4' in text

    def test_two_declared_members(self):
        casl = self.casl.replace('    2^g_3,3=2^g_4,4\n', '    2^g_3,3=2^g_3,5\n')
        with pytest.raises(ValueError):
            self.read(casl)


if __name__ == '__main__':
    unittest.main()
