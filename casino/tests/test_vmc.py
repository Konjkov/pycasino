import unittest
from pathlib import Path

import numpy as np
import pytest

from casino.jastrow import Jastrow
from casino.readers import CasinoConfig
from casino.slater import Slater
from casino.vmc import VMC
from casino.wfn import Wfn


class TestVmc(unittest.TestCase):
    """The caches an electron-by-electron walk carries - the slater state and the powers of the
    e-n distances - must describe the same wave function as a configuration evaluated in full, on
    a multideterminant expansion with a Jastrow factor.
    """

    def setUp(self):
        np.random.seed(1)
        config_path = Path(__file__).resolve().parent / 'inputs/MDET/Be'
        self.config = CasinoConfig(config_path)
        self.config.read()
        slater = Slater(self.config, cusp=None)
        self.wfn = Wfn(self.config, slater, jastrow=Jastrow(self.config))
        self.ne = self.config.input.neu + self.config.input.ned
        self.r_e = self.initial_position()

    def initial_position(self):
        """Initial positions of electrons."""
        atom_charges = self.config.wfn.atom_charges
        atom_positions = self.config.wfn.atom_positions
        natoms = atom_positions.shape[0]
        r_e = np.zeros((self.ne, 3))
        for i in range(self.ne):
            # electrons randomly centered on atoms
            r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
        return r_e + np.random.uniform(-1, 1, self.ne * 3).reshape(self.ne, 3)

    def test_value_ratio_1e(self):
        """The ratio taken out of the caches against both ends of the move evaluated in full. The
        caches are handed to every electron in turn, so a ratio that left a mark on them would
        show up here.
        """
        state, n_powers = self.wfn.caches(self.r_e)
        log_value, sign = self.wfn.log_value(self.r_e)
        for e in range(self.ne):
            next_r_e = self.r_e.copy()
            next_r_e[e] += np.random.uniform(-1, 1, 3)
            log_ratio, ratio_sign, _, _ = self.wfn.value_ratio_1e(state, n_powers, self.r_e, e, next_r_e[e])
            next_log_value, next_sign = self.wfn.log_value(next_r_e)
            assert log_ratio == pytest.approx(next_log_value - log_value)
            assert ratio_sign == next_sign * sign

    def test_accept_1e(self):
        """The caches after a whole sweep of accepted moves against the caches of the
        configuration the sweep ends on.
        """
        r_e = self.r_e.copy()
        state, n_powers = self.wfn.caches(r_e)
        for e in range(self.ne):
            next_r_e = r_e.copy()
            next_r_e[e] += np.random.uniform(-1, 1, 3)
            _, _, orbitals, q = self.wfn.value_ratio_1e(state, n_powers, r_e, e, next_r_e[e])
            assert self.wfn.accept_1e(state, n_powers, e, next_r_e[e], orbitals, q)
            r_e = next_r_e
        next_state, next_n_powers = self.wfn.caches(r_e)
        assert state.log_value == pytest.approx(next_state.log_value)
        assert state.sign == next_state.sign
        assert n_powers == pytest.approx(next_n_powers)

    def test_random_walk(self):
        """The logarithm the walk accumulates from its ratios against the configuration it ends
        on, for both electron-by-electron methods. Under method 4 the proposal is asymmetric and
        its density must stay out of the wave function.
        """
        for method in (1, 4):
            vmc = VMC(self.r_e.copy(), 0.1, self.wfn, method)
            vmc.random_walk(100, 1)
            assert vmc.log_value == pytest.approx(self.wfn.log_value(vmc.r_e)[0])


if __name__ == '__main__':
    unittest.main()
