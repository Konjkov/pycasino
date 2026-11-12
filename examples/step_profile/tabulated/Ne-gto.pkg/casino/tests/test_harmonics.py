import unittest

import numpy as np
import pytest

from casino.harmonics import Harmonics


class TestHarmonics(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.l_max_range = range(5)
        self.harmonics = [Harmonics(l_max) for l_max in self.l_max_range]
        self.r_e = np.random.uniform(-1, 1, 3)

    def test_value(self):
        for l_max, harmonics in zip(self.l_max_range, self.harmonics):
            assert harmonics.get_value(*self.r_e) == pytest.approx(harmonics.simple_value(*self.r_e)[: (l_max + 1) ** 2])

    def test_gradient(self):
        for l_max, harmonics in zip(self.l_max_range, self.harmonics):
            assert harmonics.get_gradient(*self.r_e) == pytest.approx(harmonics.simple_gradient(*self.r_e)[: (l_max + 1) ** 2])

    def test_hessian(self):
        for l_max, harmonics in zip(self.l_max_range, self.harmonics):
            assert harmonics.get_hessian(*self.r_e) == pytest.approx(harmonics.simple_hessian(*self.r_e)[: (l_max + 1) ** 2])

    def test_tressian(self):
        for l_max, harmonics in zip(self.l_max_range, self.harmonics):
            assert harmonics.get_tressian(*self.r_e) == pytest.approx(harmonics.simple_tressian(*self.r_e)[: (l_max + 1) ** 2])


if __name__ == '__main__':
    unittest.main()
