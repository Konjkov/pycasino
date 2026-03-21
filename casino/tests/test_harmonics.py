import unittest

import numpy as np
import pytest

from casino.harmonics import Harmonics


class TestHarmonics(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.harmonics = Harmonics(4)
        self.r_e = np.random.uniform(-1, 1, 3)

    def test_value(self):
        assert self.harmonics.get_value(*self.r_e) == pytest.approx(self.harmonics.simple_value(*self.r_e))

    def test_gradient(self):
        assert self.harmonics.get_gradient(*self.r_e) == pytest.approx(self.harmonics.simple_gradient(*self.r_e))

    def test_hessian(self):
        assert self.harmonics.get_hessian(*self.r_e) == pytest.approx(self.harmonics.simple_hessian(*self.r_e))

    def test_tressian(self):
        assert self.harmonics.get_tressian(*self.r_e) == pytest.approx(self.harmonics.simple_tressian(*self.r_e))


if __name__ == '__main__':
    unittest.main()
