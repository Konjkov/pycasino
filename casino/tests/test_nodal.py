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
                r,
                1 / r,
                np.ones(shape=nconfig),
                np.zeros(shape=nconfig),
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


class TestWeightedNodalDomainAverage(unittest.TestCase):
    """The same 2p state with a weight that need not be the bosonic ground state of the
    Hamiltonian. Eq. (18) is exact for any nodeless Φ once the potential Φ is the ground state of
    is taken out of the volume term, and Φ = exp(-ζr) fixes that potential by inversion,
    V_Φ = e_Φ + (∆Φ/2)/Φ = ζ²/2 - ζ/r with e_Φ = 0, leaving

        E = ∫_∂Ω Φ|∇Ψ|dS / ∫ Φ|Ψ|dR + <V - V_Φ>_{Φ|Ψ|} = -Z²/8

    for every ζ, the two terms moving against each other. ζ = Z is the case the paper writes out
    separately: there exp(-Zr) is the bosonic ground state, V - V_Φ is the constant -Z²/2, and
    Eq. (18) degenerates into Eq. (20), so this is Eq. (23) generalized off that point.

    Both terms are known in closed form - the surface integral is (ζ + Z/2)²/6 and the volume term
    (ζ - Z)(ζ + Z/2)/3 - ζ²/2 - and they are checked separately, because their sum is a
    cancellation of two numbers several times larger than the answer and a sample settles it far
    less accurately than it settles either of them. That cancellation is what makes ζ a variance
    knob and not a bias one.

    The sample is the walk's own |Ψ| and the weight is left to the estimator, which is the way a
    run has it: Φ and V_Φ are built there out of Σr and Σ1/r, and this is what checks that they are
    built right. Reweighting |Ψ| to Φ|Ψ| costs the effective sample size - ξ⁸(1 + 4ζ/Z)⁴ of it with
    ξ = Z/(Z + 2ζ), which is 10% at ζ = Z - and that is why the sum is given the loosest tolerance
    of the three. It is still tight enough: every sign and factor the weight can be given wrong
    moves it by 70% or more.
    """

    def setUp(self):
        np.random.seed(1)
        self.z = 2.0
        self.nconfig = 3 * 10**6
        self.epsilon = np.geomspace(0.04, 0.08, 3)

    def integrand(self):
        """a sample of |Ψ| = exp(-Zr/2)·r|cosθ| and what the estimator asks of it, for one electron
        and one nucleus at the origin, where Σ r_iI is r and Σ_i |Σ_I r̂_iI|² is one"""
        r = np.random.gamma(4, 2 / self.z, self.nconfig)
        mu = np.sqrt(np.random.random(self.nconfig)) * np.sign(np.random.random(self.nconfig) - 0.5)
        z_e = r * mu
        return np.stack(
            (
                -self.z * r / 2 + np.log(np.abs(z_e)),
                self.z**2 / 4 - self.z / r + 1 / z_e**2,
                -self.z / r,
                r,
                1 / r,
                np.ones(shape=self.nconfig),
                np.zeros(shape=self.nconfig),
                -self.z / r,
            ),
            axis=-1,
        )

    def test_energy(self):
        for zeta in (self.z / 4, self.z / 2, self.z):
            surface, overlap = nodal_domain_sums(self.integrand(), self.epsilon, zeta)
            a = zeta + self.z / 2
            volume = overlap[2] / overlap[1]
            assert volume == pytest.approx((zeta - self.z) * a / 3 - zeta**2 / 2, rel=5e-3)
            assert surface[:, 0] / overlap[1] == pytest.approx(a**2 / 6, rel=0.12)
            assert surface[:, 0] / overlap[1] + volume == pytest.approx(-(self.z**2) / 8, rel=0.3)


if __name__ == '__main__':
    unittest.main()
