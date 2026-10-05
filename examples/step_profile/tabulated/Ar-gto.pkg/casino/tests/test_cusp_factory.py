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
        cusp = CuspFactory(config).create()
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=1e-5)

    def test_cusp_Be(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/Be'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create()
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=1e-5)

    def test_cusp_N(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/N'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create()
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=1e-5)

    def test_cusp_Ne(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/Ne'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create()
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=1e-5)

    def test_cusp_Ar(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/Ar'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create()
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=1e-5)

    def test_cusp_Kr(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/Kr'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create()
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=1e-5)

    def test_cusp_O3(self):
        config_path = Path(__file__).resolve().parent / 'inputs/Cusp/O3'
        config = CasinoConfig(config_path)
        config.read()
        cusp = CuspFactory(config).create()
        cusp_test = CasinoCuspFactory(config).create()
        assert np.allclose(cusp.orbital_sign, cusp_test.orbital_sign)
        assert np.allclose(cusp.shift, cusp_test.shift)
        assert np.allclose(cusp.rc, cusp_test.rc)
        assert np.allclose(cusp.alpha, cusp_test.alpha, rtol=0.001)


if __name__ == '__main__':
    unittest.main()
