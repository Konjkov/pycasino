import unittest
from pathlib import Path

import numpy as np

from casino.cusp import CasinoCuspFactory, CuspFactory
from casino.readers import CasinoConfig


class TestCuspFactory(unittest.TestCase):
    def test_cusp_He(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/He'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create(casino_rc=True, casino_phi_tilde_0=False)
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, atol=0.001)

    def test_cusp_Be(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/Be'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create(casino_rc=True, casino_phi_tilde_0=False)
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=0.02)

    def test_cusp_N(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/N'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create(casino_rc=True, casino_phi_tilde_0=False)
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=0.02)

    def test_cusp_Ne(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/Ne'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create(casino_rc=True, casino_phi_tilde_0=False)
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=0.01)

    def test_cusp_Ar(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/Ar'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create(casino_rc=True, casino_phi_tilde_0=False)
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=0.05)

    def test_cusp_Kr(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/Kr'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create(casino_rc=True, casino_phi_tilde_0=False)
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=0.2)

    # def test_cusp_O3(self):
    #     config_path = Path(__file__).resolve().parent / 'inputs/Cusp/O3'
    #     config = CasinoConfig(config_path)
    #     config.read()
    #     cusp = CuspFactory(config).create(casino_rc=True, casino_phi_tilde_0=False)
    #     cusp_test = CasinoCuspFactory(config).create()
    #     assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
    #     assert np.allclose(cusp.shift, cusp_test.shift)
    #     assert np.allclose(cusp.rc, cusp_test.rc)
    #     assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=1)


if __name__ == '__main__':
    unittest.main()
