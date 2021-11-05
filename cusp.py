#!/usr/bin/env python3

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import numba as nb
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval

from readers.casino import CasinoConfig
from logger import logging

logger = logging.getLogger('cusp')

cusp_spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('norm', nb.float64),
    ('rc', nb.float64[:, :]),
    ('shift', nb.float64[:, :]),
    ('orbital_sign', nb.int64[:, :]),
    ('alpha', nb.float64[:, :, :]),
]


@nb.experimental.jitclass(cusp_spec)
class Cusp:
    """
    Scheme for adding electron–nucleus cusps to Gaussian orbitals
    A. Ma, M. D. Towler, N. D. Drummond, and R. J. Needs

    An orbital, psi, expanded in a Gaussian basis set can be written as:

    psi = phi + eta

    where phi is the part of the orbital arising from the s-type Gaussian functions
    centered on the nucleus in question (which, for convenience is at r = 0)

    In our scheme we seek a corrected orbital, psi_tilde, which differs from psi
    only in the part arising from the s-type Gaussian functions centered on the nucleus,
    i.e., so that psi_tilde obeys the cusp condition at r=0

    psi_tilde = phi_tilde + eta

    We apply a cusp correction to each orbital at each nucleus at which it is nonzero.
    Inside some cusp correction radius rc we replace phi, the part of the orbital arising
    from s-type Gaussian functions centered on the nucleus in question, by:

    phi_tilde = C + sign[phi_tilde(0)] * exp(p(r))

    In this expression sign[phi_tilde(0)], reflecting the sign of tilde_phĩ at the nucleus,
    and C is a shift chosen so that phi_tilde − C is of one sign within rc. This shift is
    necessary since the uncorrected s-part of the orbital phi may have a node where it changes
    sign inside the cusp correction radius, and we wish to replace phi by an exponential
    function, which is necessarily of one sign everywhere. The polynomial p is given by:

        p = alpha_0 + alpha_1 * r + alpha_2 * r^2 + alpha_3 * r^3 + alpha_0 * r^4

    To get gaussian cusp information from CASINO output set the following settings in
    input:
        cusp_info         : T       #*! Print Gaussian cusp info
    and in gaussians.f90:
        POLYPRINT=.true. ! Include cusp polynomial coefficients in CUSP_INFO output.
    """
    def __init__(self, neu, ned, rc, shift, orbital_sign, alpha):
        """
        Cusp
        """
        self.neu = neu
        self.ned = ned
        self.rc = rc
        self.shift = shift
        self.orbital_sign = orbital_sign
        self.alpha = alpha
        self.norm = np.exp(np.math.lgamma(self.neu + 1) / self.neu / 2)

    def wfn(self, n_vectors: np.ndarray):
        """Cusp correction for s-part of orbitals."""
        orbital = np.zeros((self.neu + self.ned, self.neu + self.ned))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.rc[atom, i]:
                        orbital[i, j] = self.orbital_sign[atom, i] * np.exp(
                            self.alpha[0, atom, i] +
                            self.alpha[1, atom, i] * r +
                            self.alpha[2, atom, i] * r ** 2 +
                            self.alpha[3, atom, i] * r ** 3 +
                            self.alpha[4, atom, i] * r ** 4
                        ) + self.shift[atom, i]

        for i in range(self.neu, self.neu + self.ned):
            for j in range(self.neu, self.neu + self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.rc[atom, i]:
                        orbital[i, j] = self.orbital_sign[atom, i] * np.exp(
                            self.alpha[0, atom, i] +
                            self.alpha[1, atom, i] * r +
                            self.alpha[2, atom, i] * r ** 2 +
                            self.alpha[3, atom, i] * r ** 3 +
                            self.alpha[4, atom, i] * r ** 4
                        ) + self.shift[atom, i]

        return self.norm * orbital

    def gradient(self, n_vectors: np.ndarray):
        """Cusp part of gradient"""
        gradient = np.zeros((self.neu + self.ned, self.neu + self.ned, 3))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.rc[atom, i]:
                        gradient[i, j] = self.orbital_sign[atom, i] * (
                                self.alpha[1, atom, i] +
                                2 * self.alpha[2, atom, i] * r +
                                3 * self.alpha[3, atom, i] * r ** 2 +
                                4 * self.alpha[4, atom, i] * r ** 3
                        ) * np.exp(
                            self.alpha[0, atom, i] +
                            self.alpha[1, atom, i] * r +
                            self.alpha[2, atom, i] * r ** 2 +
                            self.alpha[3, atom, i] * r ** 3 +
                            self.alpha[4, atom, i] * r ** 4
                        ) * n_vectors[atom, j] / r + self.shift[atom, i]

        for i in range(self.neu, self.neu + self.ned):
            for j in range(self.neu, self.neu + self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.rc[atom, i]:
                        gradient[i, j] = self.orbital_sign[atom, i] * (
                                self.alpha[1, atom, i] +
                                2 * self.alpha[2, atom, i] * r +
                                3 * self.alpha[3, atom, i] * r ** 2 +
                                4 * self.alpha[4, atom, i] * r ** 3
                        ) * np.exp(
                            self.alpha[0, atom, i] +
                            self.alpha[1, atom, i] * r +
                            self.alpha[2, atom, i] * r ** 2 +
                            self.alpha[3, atom, i] * r ** 3 +
                            self.alpha[4, atom, i] * r ** 4
                        ) * n_vectors[atom, j] / r + self.shift[atom, i]

        return self.norm * gradient

    def laplacian(self, n_vectors: np.ndarray):
        """Cusp part of laplacian"""
        laplacian = np.zeros((self.neu + self.ned, self.neu + self.ned))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.rc[atom, i]:
                        laplacian[i, j] = self.orbital_sign[atom, i] * (
                            2 * self.alpha[1, atom, i] +
                            4 * self.alpha[2, atom, i] * r +
                            6 * self.alpha[3, atom, i] * r**2 +
                            8 * self.alpha[4, atom, i] * r**3 +
                            2 * r * (self.alpha[2, atom, i] + 3 * self.alpha[3, atom, i] * r + 6 * self.alpha[4, atom, i] * r**2) +
                            r * (self.alpha[1, atom, i] + 2 * self.alpha[2, atom, i] * r + 3*self.alpha[3, atom, i] * r**2 + 4*self.alpha[4, atom, i] * r**3)**2
                        ) * np.exp(
                            self.alpha[0, atom, i] +
                            self.alpha[1, atom, i] * r +
                            self.alpha[2, atom, i] * r ** 2 +
                            self.alpha[3, atom, i] * r ** 3 +
                            self.alpha[4, atom, i] * r ** 4
                        ) / r + self.shift[atom, i]

        for i in range(self.neu, self.neu + self.ned):
            for j in range(self.neu, self.neu + self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.rc[atom, i]:
                        laplacian[i, j] = self.orbital_sign[atom, i] * (
                            2 * self.alpha[1, atom, i] +
                            4 * self.alpha[2, atom, i] * r +
                            6 * self.alpha[3, atom, i] * r**2 +
                            8 * self.alpha[4, atom, i] * r**3 +
                            2 * r * (self.alpha[2, atom, i] + 3 * self.alpha[3, atom, i] * r + 6 * self.alpha[4, atom, i] * r**2) +
                            r * (self.alpha[1, atom, i] + 2 * self.alpha[2, atom, i] * r + 3*self.alpha[3, atom, i] * r**2 + 4*self.alpha[4, atom, i] * r**3)**2
                        ) * np.exp(
                            self.alpha[0, atom, i] +
                            self.alpha[1, atom, i] * r +
                            self.alpha[2, atom, i] * r ** 2 +
                            self.alpha[3, atom, i] * r ** 3 +
                            self.alpha[4, atom, i] * r ** 4
                        ) / r + self.shift[atom, i]

        return self.norm * laplacian

    def hessian(self, n_vectors: np.ndarray):
        """Cusp part of hessian"""
        hessian = np.zeros((self.neu + self.ned, self.neu + self.ned))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.rc[atom, i]:
                        pass

        for i in range(self.ned):
            for j in range(self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, self.neu + j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.rc[atom, i]:
                        pass

        return self.norm * hessian


class CuspFactory:

    def __init__(
            self, neu, ned, mo_up, mo_down, nbasis_functions, first_shells, shell_moments,
            primitives, coefficients, exponents, atom_positions, atom_charges
    ):
        self.neu = neu
        self.ned = ned
        self.mo = np.concatenate((mo_up, mo_down), axis=1)
        self.nbasis_functions = nbasis_functions
        self.first_shells = first_shells
        self.shell_moments = shell_moments
        self.primitives = primitives
        self.coefficients = coefficients
        self.exponents = exponents
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.norm = np.exp(np.math.lgamma(self.neu + 1) / self.neu / 2)
        self.cusp_threshold = 1e-7  # FIXME: take from config
        self.phi_0, _, _ = self.phi(np.zeros(shape=(self.atom_positions.shape[0], self.neu + self.ned)))
        self.orb_mask = np.abs(self.phi_0) > self.cusp_threshold
        self.beta = np.array([3.25819, -15.0126, 33.7308, -42.8705, 31.2276, -12.1316, 1.94692])
        # atoms, MO - sign of s-type Gaussian functions centered on the nucleus
        self.orbital_sign = self.phi_sign()

    def phi(self, rc):
        """Wfn of single electron of s-orbitals an each atom"""
        orbital = np.zeros((self.atom_positions.shape[0], self.neu + self.ned, self.mo[0].shape[1]))
        orbital_derivative = np.zeros((self.atom_positions.shape[0], self.neu + self.ned, self.mo[0].shape[1]))
        orbital_second_derivative = np.zeros((self.atom_positions.shape[0], self.neu + self.ned, self.mo[0].shape[1]))
        for orb in range(self.neu + self.ned):
            p = ao = 0
            for atom in range(self.atom_positions.shape[0]):
                for nshell in range(self.first_shells[atom] - 1, self.first_shells[atom + 1] - 1):
                    l = self.shell_moments[nshell]
                    s_part = s_derivative_part = s_second_derivative_part = 0.0
                    if self.shell_moments[nshell] == 0:
                        for primitive in range(self.primitives[nshell]):
                            r = rc[atom, orb]
                            alpha = self.exponents[p + primitive]
                            exponent = self.coefficients[p + primitive] * np.exp(-alpha * r * r)
                            s_part += exponent
                            s_derivative_part -= 2 * alpha * r * exponent
                            s_second_derivative_part += 2 * alpha * (2 * alpha * r * r - 1) * exponent
                        orbital[atom, orb, ao] = s_part
                        orbital_derivative[atom, orb, ao] = s_derivative_part
                        orbital_second_derivative[atom, orb, ao] = s_second_derivative_part
                    p += self.primitives[nshell]
                    ao += 2 * l + 1
        return (
            np.sum(orbital * self.mo[0], axis=2) / self.norm,
            np.sum(orbital_derivative * self.mo[0], axis=2) / self.norm,
            np.sum(orbital_second_derivative * self.mo[0], axis=2) / self.norm
        )

    def eta_data(self, wfn_0):
        """Contribution from Gaussians on other nuclei"""
        return wfn_0 - self.phi_0

    def rc_initial(self):
        """Initial rc"""
        return np.where(self.orb_mask, 1 / self.atom_charges[:, np.newaxis], 0)

    def phi_sign(self):
        """Calculate phi sign."""
        return np.where(self.orb_mask, np.sign(self.phi_0), 0).astype(np.int64)

    def alpha_data(self, rc, eta, phi_tilde_0, shift):
        """Calculate phi coefficients.
        shift variable chosen so that (phi−shift) is of one sign within rc.
        eta is a contribution from Gaussians on other nuclei.
        """
        alpha = np.zeros(shape=(5, self.atom_positions.shape[0], self.neu + self.ned))
        phi_rc, phi_diff_rc, phi_diff_2_rc = self.phi(rc)
        R = phi_tilde_0 - shift
        X1 = np.log(np.abs(phi_rc - shift))                             # (9)
        X2 = phi_diff_rc / (phi_rc - shift)                             # (10)
        X3 = phi_diff_2_rc / (phi_rc - shift)                           # (11)
        X4 = -self.atom_charges[:, np.newaxis] * (shift + R + eta) / R  # (12)
        X5 = np.log(np.abs(R))                                          # (13)
        # (14)
        alpha[0] = X5
        alpha[1] = X4
        alpha[2] = 6*X1/rc**2 - 3*X2/rc + X3/2 - 3*X4/rc - 6*X5/rc**2 - X2**2/2
        alpha[3] = -8*X1/rc**3 + 5*X2/rc**2 - X3/rc + 3*X4/rc**2 + 8 * X5/rc**3 + X2**2/rc
        alpha[4] = 3*X1/rc**4 - 2*X2/rc**3 + X3/2/rc**2 - X4/rc**3 - 3*X5/rc**4 - X2**2/2/rc**2
        return alpha

    def phi_energy(self, r, eta, shift):
        """Effective one-electron local energy for gaussian s-part orbital.
        :param r:
        :return: energy
        """
        R = self.phi_0 - shift
        phi_rc, phi_diff_rc, phi_diff_2_rc = self.phi(r)
        z_eff = self.atom_charges[:, np.newaxis] * (1 + eta / (R + shift))         # (16)
        return - 0.5 * (2 * phi_diff_rc / r + phi_diff_2_rc) / phi_rc - z_eff / r  # (15)

    def phi_tilde_energy(self, r, eta, shift, alpha):
        """effective one-electron local energy for corrected orbital.
        Equation (15)
        :param r:
        :return: energy
        """
        p = alpha[0] + alpha[1] * r + alpha[2] * r ** 2 + alpha[3] * r ** 3 + alpha[4] * r ** 4
        p_diff_1 = alpha[1] + 2 * alpha[2] * r + 3 * alpha[3] * r ** 2 + 4 * alpha[4] * r ** 3
        p_diff_2 = 2 * alpha[2] + 2 * 3 * alpha[3] * r + 3 * 4 * alpha[4] * r ** 2
        R = self.orbital_sign * np.exp(p)
        z_eff = self.atom_charges[:, np.newaxis] * (1 + eta / (R + shift))  # (16)
        # np.where is not lazy
        # https://pretagteam.com/question/numpy-where-function-can-not-avoid-evaluate-sqrtnegative
        return np.where(
            r == 0,
            # apply L'Hôpital's rule to find energy limit at r=0 in (15)
            -0.5 * R / (R + shift) * (3 * p_diff_2 + p_diff_1 ** 2),
            -0.5 * R / (R + shift) * (2 * p_diff_1 / r + p_diff_2 + p_diff_1 ** 2) - z_eff / r
        )  # (15)

    def ideal_energy(self, r, beta0):
        """Ideal energy.
        :param r:
        :param beta0:
        :return:
        """
        return (beta0 + np.where(self.atom_charges[:, np.newaxis] == 1, 0, polyval(r, self.beta) * r**2)) * self.atom_charges[:, np.newaxis] ** 2  # (17)

    def energy_diff_max(self, rc, eta, shift, alpha):
        """Maximum square deviation of phi_tilde energy from the ideal energy
        :return:
        """
        steps = 1000
        beta0 = (self.phi_tilde_energy(rc, eta, shift, alpha) - self.ideal_energy(rc, 0)) / self.atom_charges[:, np.newaxis] ** 2
        r = np.linspace(0, rc, steps)
        energy = np.abs(self.phi_tilde_energy(r, eta, shift, alpha) - self.ideal_energy(r, beta0))
        return np.max(energy, axis=0)

    def optimize_rc(self, rc, eta, shift):
        """Optimize rc
        from SUBROUTINE choose_init_rc in gpcc.f90

        REAL(dp) FUNCTION E_L_uncorr(iorb,jion,r)
            This function returns the one-electron local energy calculated using the
            spherical part of the uncorrected orbital.
        """
        beta0 = (self.phi_energy(rc, eta, shift) - self.ideal_energy(self.rc_initial(), 0)) / self.atom_charges[:, np.newaxis] ** 2
        energy_delta = np.abs(self.phi_energy(rc, eta, shift) - self.ideal_energy(rc, beta0))
        print(energy_delta / (self.atom_charges[:, np.newaxis] ** 2 / 50))

    def optimize_phi_tilde_0(self, rc, eta, phi_tilde_0, shift):
        """Optimize phi_tilde at r=0
        :param rc:
        :param eta:
        :param phi_tilde_0: initial value
        :param shift:
        :return:
        """
        for atom in range(self.atom_positions.shape[0]):
            for orb in range(self.neu + self.ned):
                r = rc[atom, orb]
                if r == 0.0:
                    phi_tilde_0[atom, orb] = 0.0
                    continue

                def callback(x, *args):
                    """"""
                    logger.info('phi_0 = %.5f', x)

                def f(x):
                    phi_tilde_0[atom, orb] = x
                    alpha = self.alpha_data(rc, eta, phi_tilde_0, shift)
                    return self.energy_diff_max(rc, eta, shift, alpha)[atom, orb]

                options = dict(disp=True)
                res = minimize(f, [phi_tilde_0[atom, orb]], method='TNC', options=options, callback=callback)
                phi_tilde_0[atom, orb] = res.x[0]
        return phi_tilde_0

    def create(self, debug=False):
        """Set phi_0
        :param cusp:
        :return:
        """
        if self.neu == 1 and self.ned == 1:
            # atoms, MO - Value of uncorrected orbital at nucleus
            wfn_0_up = wfn_0_down = np.array([[1.307524154011]])
            # atoms, MO - cusp correction radius
            rc_up = rc_down = np.array([[0.4375]])
            # atoms, MO - Optimum corrected s orbital at nucleus
            phi_tilde_0_up = phi_tilde_0_down = np.array([[1.338322724162]])
        elif self.neu == 2 and self.ned == 2:
            wfn_0_up = wfn_0_down = np.array([[-3.447246814709, -0.628316785317]])
            rc_up = rc_down = np.array([[0.1205, 0.1180]])
            phi_tilde_0_up = phi_tilde_0_down = np.array([[-3.481156233321, -0.634379297525]])
        elif self.neu == 5 and self.ned == 2:
            wfn_0_up = np.array([[6.069114031640, -1.397116693472, 0.0, 0.0, 0.0]])
            wfn_0_down = np.array([[6.095832387803, 1.268342737910]])
            rc_up = np.array([[0.0670, 0.0695, 0.0, 0.0, 0.0]])
            rc_down = np.array([[0.0675, 0.0680]])
            phi_tilde_0_up = np.array([[6.130043694767, -1.412040439372, 0.0, 0.0, 0.0]])
            phi_tilde_0_down = np.array([[6.155438260537, 1.280709246720]])
        elif self.neu == 5 and self.ned == 5:
            wfn_0_up = wfn_0_down = np.array([[10.523069754656, 2.470734575103, 0.0, 0.0, 0.0]])
            rc_up = rc_down = np.array([[0.0455, 0.0460, 0.0, 0.0, 0.0]])
            phi_tilde_0_up = phi_tilde_0_down = np.array([[10.624267229647, 2.494850990545, 0.0, 0.0, 0.0]])
        elif self.neu == 9 and self.ned == 9:
            wfn_0_up = wfn_0_down = np.array([[20.515046538335, 5.824658914949, 0.0, 0.0, 0.0, -1.820248905891, 0.0, 0.0, 0.0]])
            rc_up = rc_down = np.array([[0.0205, 0.0200, 0, 0, 0, 0.0205, 0, 0, 0]])
            phi_tilde_0_up = phi_tilde_0_down = np.array([[20.619199783780, 5.854393350981, 0.0, 0.0, 0.0, -1.829517070413, 0.0, 0.0, 0.0]])
        elif self.neu == 18 and self.ned == 18:
            wfn_0_up = wfn_0_down = np.array(([
                [43.608490133788, -13.720841107516, 0.0, 0.0, 0.0, -5.505781654931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.751185788791, 0.0, 0.0, 0.0],
            ]))
            rc_up = rc_down = np.array([[0.0045, 0.0045, 0, 0, 0, 0.0045, 0, 0, 0, 0, 0, 0, 0, 0, 0.0045, 0, 0, 0]])
            phi_tilde_0_up = phi_tilde_0_down = np.array(([
                [43.713171699758, -13.754783719428, 0.0, 0.0, 0.0, -5.518712340056, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.757882280257, 0.0, 0.0, 0.0],
            ]))
        elif self.neu == 12 and self.ned == 12:
            wfn_0_up = np.array(([
                [-5.245016636407, -0.025034008898,  0.019182670511, -0.839192164211,  0.229570396176, -0.697628545957, 0.0, -0.140965538444, -0.015299796091, 0.0, -0.084998032927,  0.220208807573],
                [-0.024547538656,  5.241296804923, -0.002693454373, -0.611438043012, -0.806215116184,  0.550648084416, 0.0, -0.250758940038, -0.185619271170, 0.0,  0.007450966720, -0.023495021763],
                [-0.018654332490, -0.002776929419, -5.248498638985, -0.386055559344,  0.686627203383,  0.707083323432, 0.0, -0.029625096851,  0.443458560481, 0.0, -0.034753046153, -0.008117407260],
            ]))
            wfn_0_down = np.array(([
                [-5.245016636416, -0.025034009046,  0.019182670402,  0.839192164264, -0.229570396203, -0.697628545936, 0.0, -0.140965538413, -0.015299796160, 0.0,  0.084998032932, -0.220208807501],
                [-0.018654332375, -0.002776929447, -5.248498638992,  0.386055559309, -0.686627203339,  0.707083323455, 0.0, -0.029625097018,  0.443458560519, 0.0,  0.034753046180,  0.008117407241],
                [-0.024547538802,  5.241296804930, -0.002693454404,  0.611438042982,  0.806215116191,  0.550648084418, 0.0, -0.250758940010, -0.185619271253, 0.0, -0.007450966721,  0.023495021734],
            ]))
            rc_up = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
            ])
            rc_down = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
            ])
            phi_tilde_0_up = np.array(([
                [-5.296049272683, -0.025239447913,  0.019411088359, -0.861113290693,  0.232146160162, -0.696713729469, 0.0, -0.138314540320, -0.014622271569, 0.0, -0.081945526891,  0.210756161800],
                [-0.024767570243,  5.292572118624, -0.002698702824, -0.625758024602, -0.816314024031,  0.547346193861, 0.0, -0.243109138669, -0.182823180724, 0.0,  0.009584833192, -0.026173722205],
                [-0.018824846998, -0.002828414099, -5.299444354520, -0.398174414650,  0.701172068607,  0.709930839484, 0.0, -0.026979314507,  0.436599565939, 0.0, -0.032554800126, -0.012216984830],
            ]))
            phi_tilde_0_down = np.array(([
                [-5.296049272690, -0.025239448062,  0.019411088253,  0.861113290740, -0.232146160186, -0.696713729439, 0.0, -0.138314540293, -0.014622271629, 0.0,  0.081945526895, -0.210756161728],
                [-0.018824846884, -0.002828414129, -5.299444354528,  0.398174414615, -0.701172068569,  0.709930839504, 0.0, -0.026979314668,  0.436599565976, 0.0,  0.032554800149,  0.012216984810],
                [-0.024767570392,  5.292572118630, -0.002698702856,  0.625758024578,  0.816314024042,  0.547346193858, 0.0, -0.243109138643, -0.182823180811, 0.0, -0.009584833190,  0.026173722176],
            ]))
        # atoms, MO - Value of uncorrected orbital at nucleus
        wfn_0 = np.concatenate((wfn_0_up, wfn_0_down), axis=1)
        # contribution from Gaussians on other nuclei
        eta = self.eta_data(wfn_0)
        # atoms, MO - shift chosen so that phi − shift is of one sign within rc
        shift = np.zeros((self.atom_positions.shape[0], self.neu + self.ned))
        # atoms, MO - cusp correction radius
        if debug:
            rc = np.concatenate((rc_up, rc_down), axis=1)
        else:
            rc = np.concatenate((rc_up, rc_down), axis=1)
            self.optimize_rc(rc, eta, shift)
        # atoms, MO - Value of corrected orbital at nucleus
        if debug:
            phi_tilde_0 = np.concatenate((phi_tilde_0_up, phi_tilde_0_down), axis=1)
        else:
            phi_tilde_0 = self.optimize_phi_tilde_0(rc, eta, self.phi_0, shift)

        alpha = self.alpha_data(rc, eta, phi_tilde_0, shift)
        return Cusp(self.neu, self.ned, rc, shift, self.orbital_sign, alpha)


class TestCuspFactory:

    def __init__(self, neu, ned):
        self.neu = neu
        self.ned = ned

    def create(self):
        if self.neu == 1 and self.ned == 1:
            # atoms, MO - Value of uncorrected orbital at nucleus
            wfn_0_up = wfn_0_down = np.array([[1.307524154011]])
            # atoms, MO
            shift_up = shift_down = np.array([[0.0]])
            # atoms, MO - sign of s-type Gaussian functions centered on the nucleus
            orbital_sign_up = orbital_sign_down = np.array([[1]])
            # atoms, MO
            rc_up = rc_down = np.array([[0.4375]])
            # atoms, MO, alpha index
            alpha_up = alpha_down = np.array([[
                [0.29141713, -2.0, 0.25262478, -0.098352818, 0.11124336],
            ]])
        elif self.neu == 2 and self.ned == 2:
            wfn_0_up = wfn_0_down = np.array([[-3.447246814709, -0.628316785317]])
            shift_up = shift_down = np.array([[0.0, 0.0]])
            orbital_sign_up = orbital_sign_down = np.array([[-1, -1]])
            rc_up = rc_down = np.array([[0.1205, 0.1180]])
            alpha_up = alpha_down = np.array([[
                [ 1.24736449, -4.0,  0.49675975, -0.30582868,  1.0897532],
                [-0.45510824, -4.0, -0.73882727, -0.89716308, -5.8491770]
            ]])
        elif self.neu == 5 and self.ned == 2:
            wfn_0_up = np.array([[6.069114031640, -1.397116693472, 0.0, 0.0, 0.0]])
            wfn_0_down = np.array([[6.095832387803, 1.268342737910]])
            shift_up = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            shift_down = np.array([[0.0, 0.0]])
            orbital_sign_up = np.array([[1, -1, 0, 0, 0]])
            orbital_sign_down = np.array([[1, 1]])
            rc_up = np.array([[0.0670, 0.0695, 0.0, 0.0, 0.0]])
            rc_down = np.array([[0.0675, 0.0680]])
            alpha_up = np.array([[
                [1.81320188, -7.0,  0.66956651, 0.60574099E+01, -0.42786390E+02],
                [0.34503578, -7.0, -0.34059064E+01, -0.10410228E+02, -0.22372391E+02],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
            alpha_down = np.array([[
                [1.81733596, -7.0, 0.72913009, 0.19258618E+01, -0.12077748E+02],
                [0.24741402, -7.0, -0.36101513E+01, -0.11720244E+02, -0.17700238E+02],
            ]])
        elif self.neu == 5 and self.ned == 5:
            wfn_0_up = wfn_0_down = np.array([[10.523069754656, 2.470734575103, 0.0, 0.0, 0.0]])
            shift_up = shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            orbital_sign_up = orbital_sign_down = np.array([[1, 1, 0, 0, 0]])
            rc_up = rc_down = np.array([[0.0455, 0.0460, 0.0, 0.0, 0.0]])
            alpha_up = alpha_down = np.array([[
                [2.36314075, -10.0,  0.81732253,  0.15573932E+02, -0.15756663E+03],
                [0.91422900, -10.0, -0.84570201E+01, -0.26889022E+02, -0.17583628E+03],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
        elif self.neu == 9 and self.ned == 9:
            wfn_0_up = wfn_0_down = np.array([[20.515046538335, 5.824658914949, 0.0, 0.0, 0.0, -1.820248905891, 0.0, 0.0, 0.0]])
            shift_up = shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            orbital_sign_up = orbital_sign_down = np.array([[1, 1, 0, 0, 0, -1, 0, 0, 0]])
            rc_up = rc_down = np.array([[0.0205, 0.0200, 0, 0, 0, 0.0205, 0, 0, 0]])
            alpha_up = alpha_down = np.array([[
                [3.02622267, -18.0,  0.22734669E+01,  0.79076581E+02, -0.15595740E+04],
                [1.76719238, -18.0, -0.30835348E+02, -0.23112278E+03, -0.45351148E+03],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.60405204, -18.0, -0.35203155E+02, -0.13904842E+03, -0.35690426E+04],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
        elif self.neu == 18 and self.ned == 18:
            wfn_0_up = wfn_0_down = np.array(([
                [43.608490133788, -13.720841107516, 0.0, 0.0, 0.0, -5.505781654931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.751185788791, 0.0, 0.0, 0.0],
            ]))
            shift_up = shift_down = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            orbital_sign_up = orbital_sign_down = np.array([[1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]])
            rc_up = rc_down = np.array([[0.0045, 0.0045, 0, 0, 0, 0.0045, 0, 0, 0, 0, 0, 0, 0, 0, 0.0045, 0, 0, 0]])
            alpha_up = alpha_down = np.array([[
                [3.77764947, -36.0,  0.22235586E+02, -0.56621947E+04, 0.62983424E+06],
                [2.62138667, -36.0, -0.12558804E+03, -0.72801257E+04, 0.58905979E+06],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.70814456, -36.0, -0.14280857E+03, -0.80481344E+04, 0.63438487E+06],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.56410983, -36.0, -0.14519895E+03, -0.85628812E+04, 0.69239963E+06],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]])
        elif self.neu == 12 and self.ned == 12:
            wfn_0_up = np.array(([
                [-5.245016636407, -0.025034008898,  0.019182670511, -0.839192164211,  0.229570396176, -0.697628545957, 0.0, -0.140965538444, -0.015299796091, 0.0, -0.084998032927,  0.220208807573],
                [-0.024547538656,  5.241296804923, -0.002693454373, -0.611438043012, -0.806215116184,  0.550648084416, 0.0, -0.250758940038, -0.185619271170, 0.0,  0.007450966720, -0.023495021763],
                [-0.018654332490, -0.002776929419, -5.248498638985, -0.386055559344,  0.686627203383,  0.707083323432, 0.0, -0.029625096851,  0.443458560481, 0.0, -0.034753046153, -0.008117407260],
            ]))
            wfn_0_down = np.array(([
                [-5.245016636416, -0.025034009046,  0.019182670402,  0.839192164264, -0.229570396203, -0.697628545936, 0.0, -0.140965538413, -0.015299796160, 0.0,  0.084998032932, -0.220208807501],
                [-0.018654332375, -0.002776929447, -5.248498638992,  0.386055559309, -0.686627203339,  0.707083323455, 0.0, -0.029625097018,  0.443458560519, 0.0,  0.034753046180,  0.008117407241],
                [-0.024547538802,  5.241296804930, -0.002693454404,  0.611438042982,  0.806215116191,  0.550648084418, 0.0, -0.250758940010, -0.185619271253, 0.0, -0.007450966721,  0.023495021734],
            ]))
            shift_up = shift_down = np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])
            orbital_sign_up = np.array([
                [-1, -1,  1, -1,  1, -1, 0, -1, -1, 0, -1,  1],
                [-1,  1, -1, -1, -1,  1, 0, -1, -1, 0,  1, -1],
                [-1, -1, -1, -1,  1,  1, 0, -1,  1, 0, -1, -1],
            ])
            orbital_sign_down = np.array([
                [-1, -1,  1,  1, -1, -1, 0, -1, -1, 0,  1, -1],
                [-1, -1, -1,  1, -1,  1, 0, -1,  1, 0,  1,  1],
                [-1,  1, -1,  1,  1,  1, 0, -1, -1, 0, -1,  1],
            ])
            rc_up = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
            ])
            rc_down = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
            ])
            alpha_up = np.array([
                [
                    [ 1.66696112, -0.80000242E+01,  0.72538040E+00,  0.74822749E+01, -0.59832829E+02],
                    [-3.67934712, -0.80068146E+01,  0.52306712E+00,  0.74024477E+01, -0.66331792E+02],
                    [-3.94191081, -0.79787241E+01,  0.77594866E+00,  0.17932088E+01, -0.10979109E+02],
                    [-0.14952920, -0.78746605E+01, -0.43071992E+01, -0.96038217E+01, -0.73806352E+02],
                    [-1.46038810, -0.79908365E+01, -0.50007568E+01, -0.10260692E+02, -0.95143069E+02],
                    [-0.36138067, -0.80924033E+01, -0.55877946E+01, -0.12746613E+02, -0.98421943E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.97822491, -0.82393021E+01, -0.65029776E+01, -0.21458170E+02, -0.62457982E+02],
                    [-4.22520946, -0.84474893E+01, -0.73574279E+01, -0.66355424E+01, -0.22665330E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-2.50170056, -0.83886665E+01, -0.73121006E+01, -0.21788297E+02, -0.94723331E+02],
                    [-1.55705345, -0.84492012E+01, -0.78325729E+01, -0.13722665E+02, -0.18661446E+03],
                ],
                [
                    [-3.69822013, -0.80095431E+01,  0.11302149E+01, -0.43409054E+01,  0.46442078E+02],
                    [ 1.66630435, -0.80000141E+01,  0.72320378E+00,  0.81261796E+01, -0.65047801E+02],
                    [-5.91498406, -0.80763347E+01,  0.90077681E+00,  0.31742107E+00,  0.35324292E+01],
                    [-0.46879152, -0.78900614E+01, -0.41907688E+01, -0.88090450E+01, -0.77431698E+02],
                    [-0.20295616, -0.79750959E+01, -0.44477740E+01, -0.11660536E+02, -0.61243691E+02],
                    [-0.60267378, -0.81319664E+01, -0.57277864E+01, -0.14396561E+02, -0.91691179E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.41424481, -0.83393407E+01, -0.69933925E+01, -0.15868367E+02, -0.13371167E+03],
                    [-1.69923582, -0.82051284E+01, -0.64601471E+01, -0.99697087E+01, -0.15524536E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-4.64757331, -0.62408167E+01,  0.11841009E+01, -0.30996021E+01,  0.22886716E+02],
                    [-3.64299934, -0.72053551E+01, -0.11465606E+01, -0.73818077E+01,  0.16054618E+00],
                ],
                [
                    [-3.97257763, -0.80089446E+01,  0.11382835E+01, -0.33320223E+01,  0.38514125E+02],
                    [-5.86803911, -0.79164537E+01,  0.92088515E+00, -0.96065381E+00,  0.16259254E+02],
                    [1.66760198,  -0.79999530E+01,  0.72544044E+00,  0.70833021E+01, -0.56621874E+02],
                    [-0.92086514, -0.78245771E+01, -0.39180783E+01, -0.73789685E+01, -0.76429773E+02],
                    [-0.35500196, -0.79059822E+01, -0.43591899E+01, -0.11018127E+02, -0.64511456E+02],
                    [-0.34258772, -0.80478675E+01, -0.55020049E+01, -0.69743091E+01, -0.14002539E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.61268484, -0.88953684E+01, -0.10239355E+02, -0.14393680E+02, -0.29851351E+03],
                    [-0.82873883, -0.82083731E+01, -0.64401279E+01, -0.99864145E+01, -0.15474953E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.42483045, -0.86471163E+01, -0.90649452E+01, -0.17907642E+02, -0.21389596E+03],
                    [-4.40492810, -0.52578816E+01,  0.37442246E+01,  0.60259540E+01,  0.24340638E+01],
                ],
            ])
            alpha_down = np.array([
                [
                    [ 1.66696112, -0.80000242E+01,  0.72538040E+00,  0.74822749E+01, -0.59832829E+02],
                    [-3.67934711, -0.80068146E+01,  0.52306718E+00,  0.74024468E+01, -0.66331787E+02],
                    [-3.94191082, -0.79787241E+01,  0.77594861E+00,  0.17932097E+01, -0.10979114E+02],
                    [-0.14952920, -0.78746605E+01, -0.43071992E+01, -0.96038217E+01, -0.73806352E+02],
                    [-1.46038810, -0.79908365E+01, -0.50007568E+01, -0.10260692E+02, -0.95143069E+02],
                    [-0.36138067, -0.80924033E+01, -0.55877946E+01, -0.12746613E+02, -0.98421943E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.97822491, -0.82393021E+01, -0.65029776E+01, -0.21458170E+02, -0.62457982E+02],
                    [-4.22520946, -0.84474893E+01, -0.73574279E+01, -0.66355420E+01, -0.22665330E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-2.50170056, -0.83886665E+01, -0.73121006E+01, -0.21788297E+02, -0.94723331E+02],
                    [-1.55705345, -0.84492012E+01, -0.78325729E+01, -0.13722665E+02, -0.18661446E+03],
                ],
                [
                    [-3.97257764, -0.80089446E+01,  0.11382835E+01, -0.33320231E+01,  0.38514130E+02],
                    [-5.86803910, -0.79164537E+01,  0.92088495E+00, -0.96065034E+00,  0.16259234E+02],
                    [ 1.66760198, -0.79999530E+01,  0.72544044E+00,  0.70833021E+01, -0.56621874E+02],
                    [-0.92086514, -0.78245771E+01, -0.39180783E+01, -0.73789685E+01, -0.76429773E+02],
                    [-0.35500196, -0.79059822E+01, -0.43591899E+01, -0.11018127E+02, -0.64511456E+02],
                    [-0.34258772, -0.80478675E+01, -0.55020049E+01, -0.69743091E+01, -0.14002539E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.61268483, -0.88953684E+01, -0.10239355E+02, -0.14393679E+02, -0.29851351E+03],
                    [-0.82873883, -0.82083731E+01, -0.64401279E+01, -0.99864145E+01, -0.15474953E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-3.42483045, -0.86471163E+01, -0.90649452E+01, -0.17907642E+02, -0.21389596E+03],
                    [-4.40492810, -0.52578816E+01,  0.37442246E+01,  0.60259539E+01,  0.24340639E+01],
                ],
                [
                    [-3.69822013, -0.80095431E+01,  0.11302149E+01, -0.43409047E+01,  0.46442075E+02],
                    [ 1.66630435, -0.80000141E+01,  0.72320378E+00,  0.81261796E+01, -0.65047801E+02],
                    [-5.91498405, -0.80763347E+01,  0.90077692E+00,  0.31741986E+00,  0.35324359E+01],
                    [-0.46879152, -0.78900614E+01, -0.41907688E+01, -0.88090450E+01, -0.77431698E+02],
                    [-0.20295616, -0.79750959E+01, -0.44477740E+01, -0.11660536E+02, -0.61243691E+02],
                    [-0.60267378, -0.81319664E+01, -0.57277864E+01, -0.14396561E+02, -0.91691179E+02],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.41424481, -0.83393407E+01, -0.69933925E+01, -0.15868367E+02, -0.13371167E+03],
                    [-1.69923582, -0.82051284E+01, -0.64601471E+01, -0.99697088E+01, -0.15524535E+03],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-4.64757331, -0.62408167E+01,  0.11841009E+01, -0.30996020E+01,  0.22886715E+02],
                    [-3.64299934, -0.72053551E+01, -0.11465606E+01, -0.73818076E+01,  0.16054597E+00],
                ]
            ])
        # atoms, MO - cusp correction radius
        rc = np.concatenate((rc_up, rc_down), axis=1)
        # atoms, MO - shift chosen so that phi − shift is of one sign within rc
        shift = np.concatenate((shift_up, shift_down), axis=1)
        # atoms, MO - sign of s-type Gaussian functions centered on the nucleus
        orbital_sign = np.concatenate((orbital_sign_up, orbital_sign_down), axis=1)
        # atoms, MO, alpha index
        alpha = np.concatenate((alpha_up, alpha_down), axis=1)
        alpha = np.moveaxis(alpha, -1, 0)
        return Cusp(self.neu, self.ned, rc, shift, orbital_sign, alpha)
        # atoms, MO - Optimum corrected s orbital at nucleus
        # phi_0 = np.concatenate((phi_0_up, phi_0_down), axis=1)
        # wfn_0 = np.concatenate((wfn_0_up, wfn_0_down), axis=1)


# def cusp_graph(config, atom, mo, shells, atoms):
#     """In nuclear position dln(phi)/dr|r=r_nucl = -Z_nucl
#     """
#     cusp = Cusp(config.input.neu, config.input.ned, config.wfn.nbasis_functions)
#
#     x = np.linspace(0, 1/atom[atom].charge**2, 1000)
#     args = (atom, mo, shells, atoms)
#     wfn = [cusp.wfn_s(r, *args)[1]/cusp.wfn_s(r, *args)[0] for r in x]
#     plt.plot(x, wfn, x, -atoms[atom].charge*np.ones(1000))
#     plt.show()


if __name__ == '__main__':
    """
    """

    # path = 'test/gwfn/He/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Be/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/N/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Ne/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Ar/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/Kr/HF/cc-pVQZ/CBCS/Slater/'
    # path = 'test/gwfn/O3/HF/cc-pVQZ/CBCS/Slater/'

    config = CasinoConfig(path)

    cusp = CuspFactory(
        config.input.neu, config.input.ned, config.mdet.mo_up, config.mdet.mo_down, config.wfn.nbasis_functions,
        config.wfn.first_shells, config.wfn.shell_moments, config.wfn.primitives,
        config.wfn.coefficients, config.wfn.exponents, config.wfn.atom_positions, config.wfn.atom_charges
    ).create(debug=False)

    cusp_test = TestCuspFactory(config.input.neu, config.input.ned).create()

    print(np.moveaxis(cusp.alpha, 0, -1) / np.moveaxis(cusp_test.alpha, 0, -1))

    # print(cusp.orbital_sign - cusp_test.orbital_sign)