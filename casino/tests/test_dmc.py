import unittest
from pathlib import Path

import numpy as np
import pytest

from casino.dmc import DMC
from casino.jastrow import Jastrow
from casino.readers import CasinoConfig
from casino.slater import Slater
from casino.wfn import Wfn


class TestDmc(unittest.TestCase):
    """An electron-by-electron DMC step moves the electrons one at a time and carries the value of
    the wave function from ratio to ratio, so what the walkers hold at the end of a few steps must
    be what a fresh evaluation of their configurations gives.
    """

    walkers = 8

    def setUp(self):
        np.random.seed(1)
        config_path = Path(__file__).resolve().parent / 'inputs/MDET/Be'
        self.config = CasinoConfig(config_path)
        self.config.read()
        slater = Slater(self.config, cusp=None)
        self.wfn = Wfn(self.config, slater, jastrow=Jastrow(self.config))
        self.ne = self.config.input.neu + self.config.input.ned
        self.r_e_list = np.array([self.initial_position() for _ in range(self.walkers)])

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

    def dmc(self, nucleus_gf_mods):
        return DMC(self.r_e_list, 0.5, nucleus_gf_mods, False, 0.01, float(self.walkers), self.wfn, 1)

    def test_random_walk(self):
        """The simple gaussian proposal."""
        dmc = self.dmc(False)
        dmc.random_walk(5)
        for r_e, value in zip(dmc.r_e_list, dmc.wfn_value_list):
            assert value == pytest.approx(self.wfn.value(r_e))

    def test_random_walk_nucleus_gf_mods(self):
        """The proposal that treats the nucleus separately."""
        dmc = self.dmc(True)
        dmc.random_walk(5)
        for r_e, value in zip(dmc.r_e_list, dmc.wfn_value_list):
            assert value == pytest.approx(self.wfn.value(r_e))

    def test_drift_velocity_1e(self):
        """The drift of the electron a single-electron move is about to touch, taken out of the
        caches of the walker, against the drift of the whole configuration.
        """
        r_e = self.r_e_list[0]
        state, n_powers = self.wfn.caches(r_e)
        q_stay = np.ones(self.config.mdet.coeff.size)
        drift_velocity = self.wfn.drift_velocity(r_e).reshape(self.ne, 3)
        for e in range(self.ne):
            assert self.wfn.drift_velocity_1e(state, n_powers, r_e, e, r_e[e], q_stay) == pytest.approx(drift_velocity[e])

    def test_drift_velocity_1e_moved(self):
        """The same at the far end of a proposal, which the caches do not hold."""
        r_e = self.r_e_list[0]
        state, n_powers = self.wfn.caches(r_e)
        for e in range(self.ne):
            next_r_e = r_e.copy()
            next_r_e[e] += np.random.uniform(-0.5, 0.5, 3)
            _, _, _, q = self.wfn.value_ratio_1e(state, n_powers, r_e, e, next_r_e[e])
            drift_velocity = self.wfn.drift_velocity(next_r_e).reshape(self.ne, 3)
            assert self.wfn.drift_velocity_1e(state, n_powers, r_e, e, next_r_e[e], q) == pytest.approx(drift_velocity[e])


if __name__ == '__main__':
    unittest.main()
