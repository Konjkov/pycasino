import unittest

import numpy as np
import pytest

from casino.nodal import nodal_domain_sums


class TestNodalDomainSums(unittest.TestCase):
    """The 2p state of a hydrogen-like atom, whose weighted nodal domain average is known in
    closed form (Mitas & Annaberdiyev, arXiv:2109.01734, Eq. 23): with Ψ = exp(-Zr/2)z, whose
    node is the z = 0 plane, and the bosonic ground state Φ = exp(-Zr) as the weight,

        ∫_∂Ω Φ|∇Ψ|dS / ∫ Φ|Ψ|dR = 3Z²/8

    so that the energy E = 3Z²/8 + E_1s = -Z²/8. The sample is drawn from Φ|Ψ| directly, which
    is what the estimator expects and what a nodeless random walk is there to produce.
    """

    def setUp(self):
        np.random.seed(1)
        self.z = 2.0
        nconfig = 10**6
        # Φ|Ψ| = exp(-3Zr/2)·r|cosθ| in spherical coordinates
        r = np.random.gamma(4, 2 / (3 * self.z), nconfig)
        mu = np.sqrt(np.random.random(nconfig)) * np.sign(np.random.random(nconfig) - 0.5)
        z_e = r * mu
        self.integrand = np.stack(
            (
                -self.z * r / 2 + np.log(np.abs(z_e)),
                self.z**2 / 4 - self.z / r + 1 / z_e**2,
                -self.z / r,
            ),
            axis=-1,
        )

    def test_surface_integral(self):
        # the tube holds a number of configurations that falls off as ε², so a sample of this size
        # has nothing left to say below ε ≈ 0.01 bohr, and the bias shows above ε ≈ 0.05
        epsilon = np.geomspace(0.01, 0.05, 4)
        surface, overlap = nodal_domain_sums(self.integrand, epsilon)
        assert overlap[0] == self.integrand.shape[0]
        assert overlap[1] == pytest.approx(self.integrand.shape[0])
        # the potential of the sample, <-Z/r> over Φ|Ψ| = -Z²/2
        assert overlap[2] / overlap[1] == pytest.approx(-(self.z**2) / 2, rel=1e-2)
        assert surface[:, 0] / overlap[1] == pytest.approx(3 * self.z**2 / 8, rel=5e-2)


if __name__ == '__main__':
    unittest.main()
