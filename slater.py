#!/usr/bin/env python3

import os
from timeit import default_timer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import numba as nb

# np.show_config()

from decorators import pool, thread
from overload import subtract_outer
from logger import logging
from readers.wfn import GAUSSIAN_TYPE, SLATER_TYPE
from readers.casino import CasinoConfig

logger = logging.getLogger('vmc')


@nb.jit(nopython=True, nogil=True, parallel=False)
def angular_part(x, y, z):
    """Angular part of gaussian WFN.
    :return:
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    r2 = x2 + y2 + z2
    return np.array([
        1.0,
        x,
        y,
        z,
        (3.0 * z2 - r2) / 2.0,
        3.0 * x*z,
        3.0 * y*z,
        3.0 * (x2 - y2),
        6.0 * x*y,
        z * (5.0 * z2 - 3.0 * r2) / 2.0,
        1.5 * x * (5 * z2 - r2),
        1.5 * y * (5 * z2 - r2),
        15.0 * z * (x2 - y2),
        30.0 * x * y*z,
        15.0 * x * (x2 - 3 * y2),
        15.0 * y * (3 * x2 - y2),
        (35.0 * z**4 - 30.0 * z2 * r2 + 3.0 * r2**2) / 8.0,
        2.5 * x*z * (7 * z2 - 3 * r2),
        2.5 * y*z * (7 * z2 - 3 * r2),
        7.5 * (x2 - y2) * (7 * z2 - r2),
        15.0 * x*y * (7 * z2 - r2),
        105.0 * x*z * (x2 - 3 * y2),
        105.0 * y*z * (3 * x2 - y2),
        105.0 * (x2**2 - 6 * x2 * y2 + y2**2),
        420.0 * x*y * (x2 - y2)
    ])


@nb.jit(nopython=True, nogil=True, parallel=False)
def gradient_angular_part(x, y, z):
    """Angular part of gaussian WFN gradient.
    order: dx, dy, dz
    :return:
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-x, -y, 2.0*z],
        [3.0*z, 0.0, 3.0*x],
        [0.0, 3.0*z, 3.0*y],
        [6.0*x, -6.0*y, 0.0],
        [6.0*y, 6.0*x, 0.0],
        [-3.0*x*z, -3.0*y*z, -1.5*x2 - 1.5*y2 + 3.0*z2],
        [-4.5*x2 - 1.5*y2 + 6.0*z2, -3.0*x*y, 12.0*x*z],
        [-3.0*x*y, -1.5*x2 - 4.5*y2 + 6.0*z2, 12.0*y*z],
        [30.0*x*z, -30.0*y*z, 15.0*x2 - 15.0*y2],
        [30.0*y*z, 30.0*x*z, 30.0*x*y],
        [45.0*x2 - 45.0*y2, -90.0*x*y, 0],
        [90.0*x*y, 45.0*x2 - 45.0*y2, 0],
        [x*(1.5*x2 + 1.5*y2 - 6.0*z2), y*(1.5*x2 + 1.5*y2 - 6.0*z2), z*(-6.0*x2 - 6.0*y2 + 4.0*z2)],
        [z*(-22.5*x2 - 7.5*y2 + 10.0*z2), -15.0*x*y*z, x*(-7.5*x2 - 7.5*y2 + 30.0*z2)],
        [-15.0*x*y*z, z*(-7.5*x2 - 22.5*y2 + 10.0*z2), y*(-7.5*x2 - 7.5*y2 + 30.0*z2)],
        [x*(-30.0*x2 + 90.0*z2), y*(30.0*y2 - 90.0*z2), 90.0*z*(x2 - y2)],
        [y*(-45.0*x2 - 15.0*y2 + 90.0*z2), x*(-15.0*x2 - 45.0*y2 + 90.0*z2), 180.0*x*y*z],
        [315.0*z*(x2 - y2), -630.0*x*y*z, x*(105.0*x2 - 315.0*y2)],
        [630.0*x*y*z, 315.0*z*(x2 - y2), y*(315.0*x2 - 105.0*y2)],
        [x*(420.0*x2 - 1260.0*y2), y*(-1260.0*x2 + 420.0*y2), 0],
        [y*(1260.0*x2 - 420.0*y2), x*(420.0*x2 - 1260.0*y2), 0]
    ])


@nb.jit(nopython=True, nogil=True, parallel=False)
def hessian_angular_part(x, y, z):
    """Angular part of gaussian WFN hessian.
    order: dxdx, dxdy, dydy, dxdz, dydz, dzdz
    :return:
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    return np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, -1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        [6.0, 0.0, -6.0, 0.0, 0.0, 0.0],
        [0.0, 6.0, 0.0, 0.0, 0.0, 0.0],
        [-3.0*z, 0, -3.0*z, -3.0*x, -3.0*y, 6.0*z],
        [-9.0*x, -3.0*y, -3.0*x, 12.0*z, 0, 12.0*x],
        [-3.0*y, -3.0*x, -9.0*y, 0, 12.0*z, 12.0*y],
        [30.0*z, 0, -30.0*z, 30.0*x, -30.0*y, 0],
        [0, 30.0*z, 0, 30.0 * y, 30.0*x, 0],
        [90.0*x, -90.0*y, -90.0*x, 0, 0, 0],
        [90.0*y, 90.0*x, -90.0*y, 0, 0, 0],
        [4.5*x2 + 1.5*y2 - 6.0*z2, 3.0*x*y, 1.5*x2 + 4.5*y2 - 6.0*z2, -12.0*x*z, -12.0*y*z, -6.0*x2 - 6.0*y2 + 12.0*z2],
        [-45.0*x*z, -15.0*y*z, -15.0*x*z, -22.5*x2 - 7.5*y2 + 30.0*z2, -15.0*x*y, 60.0*x*z],
        [-15.0*y*z, -15.0*x*z, -45.0*y*z, -15.0*x*y, -7.5*x2 - 22.5*y2 + 30.0*z2, 60.0*y*z],
        [-90.0*x2 + 90.0*z2, 0, 90.0*y2 - 90.0*z2, 180.0*x*z, -180.0*y*z, 90.0*x2 - 90.0*y2],
        [-90.0*x*y, -45.0*x2 - 45.0*y2 + 90.0*z2, -90.0*x*y, 180.0*y*z, 180.0*x*z, 180.0*x*y],
        [630.0*x*z, -630.0*y*z, -630.0*x*z, 315.0*x2 - 315.0*y2, -630.0*x*y, 0],
        [630.0*y*z, 630.0*x*z, -630.0*y*z, 630.0*x*y, 315.0*x2 - 315.0*y2, 0],
        [1260.0*x2 - 1260.0*y2, -2520.0*x*y, -1260.0*x2 + 1260.0*y2, 0, 0, 0],
        [2520.0*x*y, 1260.0*x2 - 1260.0*y2, -2520.0*x*y, 0, 0, 0],
    ])


spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('nbasis_functions', nb.int64),
    ('first_shells', nb.int64[:]),
    ('orbital_types', nb.int64[:]),
    ('shell_moments', nb.int64[:]),
    ('slater_orders', nb.int64[:]),
    ('primitives', nb.int64[:]),
    ('coefficients', nb.float64[:]),
    ('exponents', nb.float64[:]),
    ('mo_up', nb.float64[:, :, :]),
    ('mo_down', nb.float64[:, :, :]),
    ('coeff', nb.float64[:]),
    ('orbital_sign', nb.optional(nb.int64[:, :])),
    ('cusp_r', nb.optional(nb.float64[:, :])),
    ('alpha', nb.optional(nb.float64[:, :, :])),
]


@nb.experimental.jitclass(spec)
class Slater:

    def __init__(
        self, neu, ned,
        nbasis_functions, first_shells, orbital_types, shell_moments, slater_orders, primitives, coefficients, exponents, mo_up, mo_down, coeff
    ):
        """
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param nbasis_functions:
        :param first_shells:
        :param orbital_types:
        :param shell_moments:
        :param slater_orders:
        :param primitives:
        :param coefficients:
        :param exponents:
        :param mo_up:
        :param mo_down:
        :param coeff:
        """
        self.neu = neu
        self.ned = ned
        self.nbasis_functions = nbasis_functions
        self.first_shells = first_shells
        self.orbital_types = orbital_types
        self.shell_moments = shell_moments
        self.slater_orders = slater_orders
        self.primitives = primitives
        self.coefficients = coefficients
        self.exponents = exponents
        self.mo_up = mo_up
        self.mo_down = mo_down
        self.coeff = coeff
        atom = 'Be'
        if atom == 'He':
            # atoms, MO
            self.orbital_sign = np.array([[1]])
            # atoms, MO
            self.cusp_r = np.array([[0.4375]])
            # atoms, MO, alpha index
            self.alpha = np.array([[
                [0.29141713, -2.0, 0.25262478E+00, -0.98352818E-01, 0.11124336E+00],
            ]])
        elif atom == 'Be':
            self.orbital_sign = np.array([[-1, -1]])
            self.cusp_r = np.array([[0.1205, 0.1180]])
            self.alpha = np.array([[
                [ 1.24736449, -4,  0.49675975E+00, -0.30582868E+00,  0.10897532E+01],
                [-0.45510824, -4, -0.73882727E+00, -0.89716308E+00, -0.58491770E+01],
            ]])
        elif atom == 'N':
            pass
        elif atom == 'Ne':
            self.orbital_sign = np.array([[1, 1, 0, 0, 0]])
            self.cusp_r = np.array([[0.0455, 0.0460, 0.0, 0.0, 0.0]])
            self.alpha = np.array([[
                [2.36314075, -10,  0.81732253E+00,  0.15573932E+02, -0.15756663E+03],
                [0.91422900, -10, -0.84570201E+01, -0.26889022E+02, -0.17583628E+03],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]])
        elif atom == 'Ar':
            self.orbital_sign = np.array([[1, 1, 0, 0, 0, -1, 0, 0, 0]])
            self.cusp_r = np.array([[0.0205, 0.0200, 0, 0, 0, 0.0205, 0, 0, 0]])
            self.alpha = np.array([[
                [3.02622267, -18,  0.22734669E+01,  0.79076581E+02, -0.15595740E+04],
                [1.76719238, -18, -0.30835348E+02, -0.23112278E+03, -0.45351148E+03],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0.60405204, -18, -0.35203155E+02, -0.13904842E+03, -0.35690426E+04],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]])
        elif atom == 'Kr':
            self.orbital_sign = np.array([[1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]])
            self.cusp_r = np.array([[0.0045, 0.0045, 0, 0, 0, 0.0045, 0, 0, 0, 0, 0, 0, 0, 0, 0.0045, 0, 0, 0]])
            self.alpha = np.array([[
                [3.77764947, -36,  0.22235586E+02, -0.56621947E+04, 0.62983424E+06],
                [2.62138667, -36, -0.12558804E+03, -0.72801257E+04, 0.58905979E+06],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1.70814456, -36, -0.14280857E+03, -0.80481344E+04, 0.63438487E+06],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0.56410983, -36, -0.14519895E+03, -0.85628812E+04, 0.69239963E+06],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]])
        elif atom == 'O3':
            self.orbital_sign = np.array([
                [-1, -1,  1, -1,  1, -1, 0, -1, -1, 0, -1,  1],
                [-1,  1, -1, -1, -1,  1, 0, -1, -1, 0,  1, -1],
                [-1, -1, -1, -1,  1,  1, 0, -1,  1, 0, -1, -1],
            ])
            self.cusp_r = np.array([
                [0.0580, 0.0570, 0.0580, 0.0580, 0.0580, 0.0585, 0, 0.0605, 0.0565, 0, 0.0615, 0.0595],
                [0.0605, 0.0580, 0.0620, 0.0790, 0.0415, 0.0590, 0, 0.0595, 0.0580, 0, 0.0935, 0.0910],
                [0.0605, 0.0565, 0.0580, 0.0805, 0.0780, 0.0575, 0, 0.0660, 0.0580, 0, 0.0680, 0.1345],
            ])
            self.alpha = np.array([[
                [1.66696112, -0.80000242E+01, 0.72538040E+00, 0.74822749E+01, -0.59832829E+02],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ]])
        else:
            self.orbital_sign = None
            self.cusp_r = None
            self.alpha = None

    def cusp_wfn(self, n_vectors: np.ndarray):
        """Calculate cusped correction for s-part of orbitals.
        We apply a cusp correction to each orbital at each nucleus at which it is nonzero. Inside some cusp
        correction radius rc we replace φ, the part of the orbital arising from s-type Gaussian functions centred
        on the nucleus in question, by
        φ̃ = C + sign[φ̃(0)] * exp[p(r)]
        in gaussians.f90 set
        POLYPRINT=.true. ! Include cusp polynomial coefficients in CUSP_INFO output.
        """
        orbital_up = np.zeros((self.neu, self.neu))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.cusp_r[atom, i]:
                        orbital_up[i, j] = self.orbital_sign[atom, i] * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r**2 +
                            self.alpha[atom, i, 3] * r**3 +
                            self.alpha[atom, i, 4] * r**4
                        )

        orbital_down = np.zeros((self.ned, self.ned))
        for i in range(self.ned):
            for j in range(self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, self.neu + j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.cusp_r[atom, i]:
                        orbital_down[i, j] = self.orbital_sign[atom, i] * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r**2 +
                            self.alpha[atom, i, 3] * r**3 +
                            self.alpha[atom, i, 4] * r**4
                        )

        return orbital_up, orbital_down

    def cusp_gradient(self, n_vectors: np.ndarray):
        """Cusp part of gradient"""
        gradient_up = np.zeros((self.neu, self.neu, 3))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.cusp_r[atom, i]:
                        gradient_up[i, j] = self.orbital_sign[atom, i] * (
                                self.alpha[atom, i, 1] + 2 * self.alpha[atom, i, 2]*r + 3 * self.alpha[atom, i, 3] * r**2 + 4 * self.alpha[atom, i, 4] * r**3
                        ) * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r ** 2 +
                            self.alpha[atom, i, 3] * r ** 3 +
                            self.alpha[atom, i, 4] * r ** 4
                        ) * n_vectors[atom, j] / r

        gradient_down = np.zeros((self.ned, self.ned, 3))
        for i in range(self.ned):
            for j in range(self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, self.neu + j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.cusp_r[atom, i]:
                        gradient_down[i, j] = self.orbital_sign[atom, i] * (
                                self.alpha[atom, i, 1] + 2 * self.alpha[atom, i, 2] * r + 3 * self.alpha[atom, i, 3] * r**2 + 4 * self.alpha[atom, i, 4] * r**3
                        ) * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r ** 2 +
                            self.alpha[atom, i, 3] * r ** 3 +
                            self.alpha[atom, i, 4] * r ** 4
                        ) * n_vectors[atom, self.neu + j] / r

        return gradient_up, gradient_down

    def cusp_laplacian(self, n_vectors: np.ndarray):
        """Cusp part of laplacian"""
        laplacian_up = np.zeros((self.neu, self.neu))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.cusp_r[atom, i]:
                        laplacian_up[i, j] = self.orbital_sign[atom, i] * (
                            2 * self.alpha[atom, i, 1] + 4 * self.alpha[atom, i, 2] * r + 6 * self.alpha[atom, i, 3] * r**2 + 8 * self.alpha[atom, i, 4] * r**3 +
                            2 * r * (self.alpha[atom, i, 2] + 3 * self.alpha[atom, i, 3] * r + 6 * self.alpha[atom, i, 4] * r**2) +
                            r * (self.alpha[atom, i, 1] + 2 * self.alpha[atom, i, 2] * r + 3*self.alpha[atom, i, 3] * r**2 + 4*self.alpha[atom, i, 4] * r**3)**2
                        ) * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r ** 2 +
                            self.alpha[atom, i, 3] * r ** 3 +
                            self.alpha[atom, i, 4] * r ** 4
                        ) / r

        laplacian_down = np.zeros((self.ned, self.ned))
        for i in range(self.ned):
            for j in range(self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, self.neu + j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.cusp_r[atom, i]:
                        laplacian_down[i, j] = self.orbital_sign[atom, i] * (
                            2 * self.alpha[atom, i, 1] + 4 * self.alpha[atom, i, 2] * r + 6 * self.alpha[atom, i, 3] * r**2 + 8 * self.alpha[atom, i, 4] * r**3 +
                            2 * r * (self.alpha[atom, i, 2] + 3 * self.alpha[atom, i, 3] * r + 6 * self.alpha[atom, i, 4] * r**2) +
                            r * (self.alpha[atom, i, 1] + 2 * self.alpha[atom, i, 2] * r + 3*self.alpha[atom, i, 3] * r**2 + 4*self.alpha[atom, i, 4] * r**3)**2
                        ) * np.exp(
                            self.alpha[atom, i, 0] +
                            self.alpha[atom, i, 1] * r +
                            self.alpha[atom, i, 2] * r ** 2 +
                            self.alpha[atom, i, 3] * r ** 3 +
                            self.alpha[atom, i, 4] * r ** 4
                        ) / r

        return laplacian_up, laplacian_down

    def cusp_hessian(self, n_vectors: np.ndarray):
        """Cusp part of laplacian"""
        hessian_up = np.zeros((self.neu, self.neu))
        for i in range(self.neu):
            for j in range(self.neu):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.cusp_r[atom, i]:
                        pass

        hessian_down = np.zeros((self.ned, self.ned))
        for i in range(self.ned):
            for j in range(self.ned):
                for atom in range(n_vectors.shape[0]):
                    x, y, z = n_vectors[atom, self.neu + j]
                    r = np.sqrt(x * x + y * y + z * z)
                    if r < self.cusp_r[atom, i]:
                        pass

        return hessian_up, hessian_down

    def AO_wfn(self, n_vectors: np.ndarray) -> np.ndarray:
        """
        Atomic orbitals for every electron
        :param n_vectors: electron-nuclei array(nelec, natom, 3)
        :return: AO array(nelec, nbasis_functions)
        """
        orbital = np.zeros((self.neu + self.ned, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = 0
            ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = x * x + y * y + z * z
                angular_1 = angular_part(x, y, z)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            radial_1 += self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r2)
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        for primitive in range(self.primitives[nshell]):
                            radial_1 += r**self.slater_orders[nshell] * self.coefficients[p + primitive] * np.exp(-self.exponents[p + primitive] * r)
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital[i, ao+m] = angular_1[l*l+m] * radial_1
                    ao += 2*l+1
        return orbital

    def AO_gradient(self, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient matrix.
        :param n_vectors: electron-nuclei - array(natom, nelec, 3)
        :return: AO gradient - array(3, nelec, nbasis_functions)
        """
        orbital = np.zeros((self.neu + self.ned, 3, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = 0
            ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = x * x + y * y + z * z
                angular_1 = angular_part(x, y, z)
                angular_2 = gradient_angular_part(x, y, z)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    radial_2 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            exponent = self.coefficients[p + primitive] * np.exp(-alpha * r2)
                            radial_1 -= 2 * alpha * exponent
                            radial_2 += exponent
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        n = self.slater_orders[nshell]
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            exponent = r**self.slater_orders[nshell] * self.coefficients[p + primitive] * np.exp(-alpha * r)
                            radial_1 -= (alpha*r - n)/r2 * exponent
                            radial_2 += exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital[i, 0, ao+m] = x * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 0] * radial_2
                        orbital[i, 1, ao+m] = y * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 1] * radial_2
                        orbital[i, 2, ao+m] = z * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 2] * radial_2
                    ao += 2*l+1
        return orbital.reshape(((self.neu + self.ned) * 3, self.nbasis_functions))

    def AO_laplacian(self, n_vectors: np.ndarray) -> np.ndarray:
        """Laplacian matrix.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        :return: AO laplacian - array(nelec, nbasis_functions)
        """
        orbital = np.zeros((self.neu + self.ned, self.nbasis_functions))
        for i in range(self.neu + self.ned):
            p = 0
            ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = x * x + y * y + z * z
                angular_1 = angular_part(x, y, z)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            radial_1 += 2 * alpha * (2 * alpha * r2 - 2 * l - 3) * self.coefficients[p + primitive] * np.exp(-alpha * r2)
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        n = self.slater_orders[nshell]
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            exponent = r**n * self.coefficients[p + primitive] * np.exp(-alpha * r)
                            radial_1 += (alpha**2 - 2*(l+n+1)*alpha/r + (2*l+n+1)*n/r2) * exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital[i, ao+m] = angular_1[l*l+m] * radial_1
                    ao += 2*l+1
        return orbital

    def AO_hessian(self, n_vectors: np.ndarray) -> np.ndarray:
        """hessian matrix.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        :return: AO hessian - array(6, nelec, nbasis_functions)
        """
        orbital = np.zeros((6, self.neu + self.ned, self.nbasis_functions))

        for i in range(self.neu + self.ned):
            p = 0
            ao = 0
            for atom in range(n_vectors.shape[0]):
                x, y, z = n_vectors[atom, i]
                r2 = x * x + y * y + z * z
                angular_1 = angular_part(x, y, z)
                angular_2 = gradient_angular_part(x, y, z)
                angular_3 = hessian_angular_part(x, y, z)
                for nshell in range(self.first_shells[atom]-1, self.first_shells[atom+1]-1):
                    l = self.shell_moments[nshell]
                    radial_1 = 0.0
                    radial_2 = 0.0
                    radial_3 = 0.0
                    if self.orbital_types[nshell] == GAUSSIAN_TYPE:
                        for primitive in range(self.primitives[nshell]):
                            alpha = self.exponents[p + primitive]
                            exponent = self.coefficients[p + primitive] * np.exp(-alpha * r2)
                            c = -2 * alpha
                            radial_1 += c**2 * exponent
                            radial_2 += c * exponent
                            radial_3 += exponent
                    elif self.orbital_types[nshell] == SLATER_TYPE:
                        r = np.sqrt(r2)
                        for primitive in range(self.primitives[nshell]):
                            n = self.slater_orders[nshell]
                            alpha = self.exponents[p + primitive]
                            exponent = r**self.slater_orders[nshell] * self.coefficients[p + primitive] * np.exp(-alpha * r)
                            c = -(alpha*r - n)/r2
                            d = c**2 + (alpha*r - 2*n)/r2**2
                            radial_1 += d * exponent
                            radial_2 += c * exponent
                            radial_3 += exponent
                    p += self.primitives[nshell]
                    for m in range(2 * l + 1):
                        orbital[0, i, ao+m] = x*x * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * x * angular_2[l*l+m, 0]) * radial_2 + angular_3[l*l+m, 0] * radial_3
                        orbital[1, i, ao+m] = x*y * angular_1[l*l+m] * radial_1 + (y * angular_2[l*l+m, 0] + x * angular_2[l*l+m, 1]) * radial_2 + angular_3[l*l+m, 1] * radial_3
                        orbital[2, i, ao+m] = y*y * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * y * angular_2[l*l+m, 1]) * radial_2 + angular_3[l*l+m, 2] * radial_3
                        orbital[3, i, ao+m] = x*z * angular_1[l*l+m] * radial_1 + (z * angular_2[l*l+m, 0] + x * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 3] * radial_3
                        orbital[4, i, ao+m] = y*z * angular_1[l*l+m] * radial_1 + (z * angular_2[l*l+m, 1] + y * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 4] * radial_3
                        orbital[5, i, ao+m] = z*z * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * z * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 5] * radial_3
                    ao += 2*l+1

        return orbital

    def value(self, n_vectors: np.ndarray) -> float:
        """Multideterminant wave function value.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao = self.AO_wfn(n_vectors)
        if self.orbital_sign is not None:
            cusp_wfn_up, cusp_wfn_down = self.cusp_wfn(n_vectors)

        val = 0.0
        for i in range(self.coeff.shape[0]):
            if self.orbital_sign is not None:
                wfn_u = np.where(cusp_wfn_up, cusp_wfn_up, self.mo_up[i] @ ao[:self.neu].T)
                wfn_d = np.where(cusp_wfn_down, cusp_wfn_down, self.mo_down[i] @ ao[self.neu:].T)
            else:
                wfn_u = self.mo_up[i] @ ao[:self.neu].T
                wfn_d = self.mo_down[i] @ ao[self.neu:].T
            val += self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
        return val

    def gradient(self, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient ∇(phi).
        d(det(slater))/dri = det(slater) * (tr(slater^-1 * B(n)) over n
        where the matrix B(n) is zero with the exception of the n-th column.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao = self.AO_wfn(n_vectors)
        gradient = self.AO_gradient(n_vectors)
        if self.orbital_sign is not None:
            cusp_wfn_up, cusp_wfn_down = self.cusp_wfn(n_vectors)
            cusp_gradient_up, cusp_gradient_down = self.cusp_gradient(n_vectors)

        val = 0.0
        grad = np.zeros((self.neu + self.ned, 3))
        for i in range(self.coeff.shape[0]):

            if self.orbital_sign is not None:
                wfn_u = np.where(cusp_wfn_up, cusp_wfn_up, self.mo_up[i] @ ao[:self.neu].T)
                grad_u = np.where(cusp_gradient_up, cusp_gradient_up, (self.mo_up[i] @ gradient[:self.neu * 3].T).reshape((self.neu, self.neu, 3)))
            else:
                wfn_u = self.mo_up[i] @ ao[:self.neu].T
                grad_u = (self.mo_up[i] @ gradient[:self.neu * 3].T).reshape((self.neu, self.neu, 3))
            inv_wfn_u = np.linalg.inv(wfn_u)
            res_u = (inv_wfn_u * grad_u.T).T.sum(axis=0)

            if self.orbital_sign is not None:
                wfn_d = np.where(cusp_wfn_down, cusp_wfn_down, self.mo_down[i] @ ao[self.neu:].T)
                grad_d = np.where(cusp_gradient_down, cusp_gradient_down, (self.mo_down[i] @ gradient[self.neu * 3:].T).reshape((self.ned, self.ned, 3)))
            else:
                wfn_d = self.mo_down[i] @ ao[self.neu:].T
                grad_d = (self.mo_down[i] @ gradient[self.neu * 3:].T).reshape((self.ned, self.ned, 3))
            inv_wfn_d = np.linalg.inv(wfn_d)
            res_d = (inv_wfn_d * grad_d.T).T.sum(axis=0)

            c = self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
            val += c
            grad += c * np.concatenate((res_u, res_d))

        return grad.ravel() / val

    def laplacian(self, n_vectors: np.ndarray) -> float:
        """Scalar laplacian Δ(phi).
        Δ(det(slater)) = det(slater) * sum(tr(slater^-1 * B(n)) over n
        where the matrix B(n) is zero with the exception of the n-th column
        as tr(A) + tr(B) = tr(A + B)
        Δ(det(slater)) = det(slater) * tr(slater^-1 * B)
        where the matrix Bij = ∆phi i (rj)
        then using np.trace(A @ B) = np.sum(A * B.T)
        Read for details:
        "Simple formalism for efficient derivatives and multi-determinant expansions in quantum Monte Carlo"
        C. Filippi, R. Assaraf, S. Moroni
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao = self.AO_wfn(n_vectors)
        ao_laplacian = self.AO_laplacian(n_vectors)
        if self.orbital_sign is not None:
            cusp_wfn_up, cusp_wfn_down = self.cusp_wfn(n_vectors)
            cusp_laplacian_up, cusp_laplacian_down = self.cusp_laplacian(n_vectors)

        val = lap = 0
        for i in range(self.coeff.shape[0]):

            if self.orbital_sign is not None:
                wfn_u = np.where(cusp_wfn_up, cusp_wfn_up, self.mo_up[i] @ ao[:self.neu].T)
                lap_u = np.where(cusp_laplacian_up, cusp_laplacian_up, self.mo_up[i] @ ao_laplacian[:self.neu].T)
            else:
                wfn_u = self.mo_up[i] @ ao[:self.neu].T
                lap_u = self.mo_up[i] @ ao_laplacian[:self.neu].T
            inv_wfn_u = np.linalg.inv(wfn_u)
            res_u = np.sum(inv_wfn_u * lap_u.T)

            if self.orbital_sign is not None:
                wfn_d = np.where(cusp_wfn_down, cusp_wfn_down, self.mo_down[i] @ ao[self.neu:].T)
                lap_d = np.where(cusp_laplacian_down, cusp_laplacian_down, self.mo_down[i] @ ao_laplacian[self.neu:].T)
            else:
                wfn_d = self.mo_down[i] @ ao[self.neu:].T
                lap_d = self.mo_down[i] @ ao_laplacian[self.neu:].T
            inv_wfn_d = np.linalg.inv(wfn_d)
            res_d = np.sum(inv_wfn_d * lap_d.T)

            c = self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
            val += c
            lap += c * (res_u + res_d)

        return lap / val

    def hessian(self, n_vectors: np.ndarray):
        """Hessian.
        d²det(A)/dxdy = det(A) * (
            tr(A^-1 * d²A/dxdy) +
            tr(A^-1 * dA/dx) * tr(A^-1 * dA/dy) -
            tr(A^-1 * dA/dx * A^-1 * dA/dy)
        )
        in case of x and y is a coordinates of different electrons first term is zero
        in other case a sum of last two terms is zero.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao = self.AO_wfn(n_vectors)
        gradient = self.AO_gradient(n_vectors)
        hessian = self.AO_hessian(n_vectors)
        # if self.orbital_sign is not None:
        #     cusp_wfn_up, cusp_wfn_down = self.cusp_wfn(n_vectors)
        #     cusp_gradient_up, cusp_gradient_down = self.cusp_gradient(n_vectors)
        #     cusp_hessian_up, cusp_hessian_down = self.cusp_hessian(n_vectors)

        val = 0
        hass = np.zeros((self.neu + self.ned, 3, self.neu + self.ned, 3))
        for i in range(self.coeff.shape[0]):

            wfn_u = self.mo_up[i] @ ao[:self.neu].T
            inv_wfn_u = np.linalg.inv(wfn_u)
            grad_u = self.mo_up[i] @ gradient[:self.neu * 3].T
            hess_xx = self.mo_up[i] @ hessian[0, :self.neu].T
            hess_xy = self.mo_up[i] @ hessian[1, :self.neu].T
            hess_yy = self.mo_up[i] @ hessian[2, :self.neu].T
            hess_xz = self.mo_up[i] @ hessian[3, :self.neu].T
            hess_yz = self.mo_up[i] @ hessian[4, :self.neu].T
            hess_zz = self.mo_up[i] @ hessian[5, :self.neu].T

            res_grad_u = np.zeros((self.neu, 3))
            res_u = np.zeros((self.neu, 3, self.neu, 3))

            temp = (inv_wfn_u @ grad_u).reshape((self.neu, self.neu, 3))
            dx = temp[:, :, 0]
            dy = temp[:, :, 1]
            dz = temp[:, :, 2]

            res_grad_u[:, 0] = np.diag(dx)
            res_grad_u[:, 1] = np.diag(dy)
            res_grad_u[:, 2] = np.diag(dz)

            res_u[:, 0, :, 0] = np.eye(self.neu) * (inv_wfn_u @ hess_xx) - dx.T * dx
            res_u[:, 0, :, 1] = np.eye(self.neu) * (inv_wfn_u @ hess_xy) - dx.T * dy
            res_u[:, 1, :, 0] = np.eye(self.neu) * (inv_wfn_u @ hess_xy) - dy.T * dx
            res_u[:, 1, :, 1] = np.eye(self.neu) * (inv_wfn_u @ hess_yy) - dy.T * dy
            res_u[:, 0, :, 2] = np.eye(self.neu) * (inv_wfn_u @ hess_xz) - dx.T * dz
            res_u[:, 2, :, 0] = np.eye(self.neu) * (inv_wfn_u @ hess_xz) - dz.T * dx
            res_u[:, 1, :, 2] = np.eye(self.neu) * (inv_wfn_u @ hess_yz) - dy.T * dz
            res_u[:, 2, :, 1] = np.eye(self.neu) * (inv_wfn_u @ hess_yz) - dz.T * dy
            res_u[:, 2, :, 2] = np.eye(self.neu) * (inv_wfn_u @ hess_zz) - dz.T * dz

            wfn_d = self.mo_down[i] @ ao[self.neu:].T
            inv_wfn_d = np.linalg.inv(wfn_d)
            grad_d = self.mo_down[i] @ gradient[self.neu * 3:].T
            hess_xx = self.mo_down[i] @ hessian[0, self.neu:].T
            hess_xy = self.mo_down[i] @ hessian[1, self.neu:].T
            hess_yy = self.mo_down[i] @ hessian[2, self.neu:].T
            hess_xz = self.mo_down[i] @ hessian[3, self.neu:].T
            hess_yz = self.mo_down[i] @ hessian[4, self.neu:].T
            hess_zz = self.mo_down[i] @ hessian[5, self.neu:].T

            res_grad_d = np.zeros((self.ned, 3))
            res_d = np.zeros((self.ned, 3, self.ned, 3))

            temp = (inv_wfn_d @ grad_d).reshape((self.ned, self.ned, 3))
            dx = temp[:, :, 0]
            dy = temp[:, :, 1]
            dz = temp[:, :, 2]

            res_grad_d[:, 0] = np.diag(dx)
            res_grad_d[:, 1] = np.diag(dy)
            res_grad_d[:, 2] = np.diag(dz)

            res_d[:, 0, :, 0] = np.eye(self.ned) * (inv_wfn_d @ hess_xx) - dx.T * dx
            res_d[:, 0, :, 1] = np.eye(self.ned) * (inv_wfn_d @ hess_xy) - dx.T * dy
            res_d[:, 1, :, 0] = np.eye(self.ned) * (inv_wfn_d @ hess_xy) - dy.T * dx
            res_d[:, 1, :, 1] = np.eye(self.ned) * (inv_wfn_d @ hess_yy) - dy.T * dy
            res_d[:, 0, :, 2] = np.eye(self.ned) * (inv_wfn_d @ hess_xz) - dx.T * dz
            res_d[:, 2, :, 0] = np.eye(self.ned) * (inv_wfn_d @ hess_xz) - dz.T * dx
            res_d[:, 1, :, 2] = np.eye(self.ned) * (inv_wfn_d @ hess_yz) - dy.T * dz
            res_d[:, 2, :, 1] = np.eye(self.ned) * (inv_wfn_d @ hess_yz) - dz.T * dy
            res_d[:, 2, :, 2] = np.eye(self.ned) * (inv_wfn_d @ hess_zz) - dz.T * dz

            c = self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
            val += c
            res_grad = np.concatenate((res_grad_u.ravel(), res_grad_d.ravel()))
            hass += c * np.outer(res_grad, res_grad).reshape((self.neu + self.ned), 3, (self.neu + self.ned), 3)
            hass[:self.neu, :, :self.neu, :] += c * res_u
            hass[self.neu:, :, self.neu:, :] += c * res_d

        return hass.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3) / val

    def numerical_gradient(self, n_vectors: np.ndarray) -> float:
        """Numerical gradient with respect to a e-coordinates
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        delta = 0.00001

        val = self.value(n_vectors)
        res = np.zeros((self.neu + self.ned, 3))
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= delta
                res[i, j] -= self.value(n_vectors)
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.value(n_vectors)
                n_vectors[:, i, j] -= delta

        return res.ravel() / delta / 2 / val

    def numerical_laplacian(self, n_vectors: np.ndarray) -> float:
        """Numerical laplacian with respect to a e-coordinates
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        delta = 0.00001

        val = self.value(n_vectors)
        res = - 6 * (self.neu + self.ned) * val
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= delta
                res += self.value(n_vectors)
                n_vectors[:, i, j] += 2 * delta
                res += self.value(n_vectors)
                n_vectors[:, i, j] -= delta

        return res / delta / delta / val

    def numerical_hessian(self, n_vectors: np.ndarray):
        """Numerical hessian with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        delta = 0.00001

        val = self.value(n_vectors)
        res = -2 * val * np.eye((self.neu + self.ned) * 3).reshape(self.neu + self.ned, 3, self.neu + self.ned, 3)
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= 2 * delta
                res[i, j, i, j] += self.value(n_vectors)
                n_vectors[:, i, j] += 4 * delta
                res[i, j, i, j] += self.value(n_vectors)
                n_vectors[:, i, j] -= 2 * delta

        for i1 in range(self.neu + self.ned):
            for j1 in range(3):
                for i2 in range(i1 + 1):
                    for j2 in range(3):
                        if i1 == i2 and j1 >= j2:
                            continue
                        n_vectors[:, i1, j1] -= delta
                        n_vectors[:, i2, j2] -= delta
                        res[i1, j1, i2, j2] += self.value(n_vectors)
                        n_vectors[:, i1, j1] += 2 * delta
                        res[i1, j1, i2, j2] -= self.value(n_vectors)
                        n_vectors[:, i2, j2] += 2 * delta
                        res[i1, j1, i2, j2] += self.value(n_vectors)
                        n_vectors[:, i1, j1] -= 2 * delta
                        res[i1, j1, i2, j2] -= self.value(n_vectors)
                        n_vectors[:, i1, j1] += delta
                        n_vectors[:, i2, j2] -= delta
                        res[i2, j2, i1, j1] = res[i1, j1, i2, j2]

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3) / delta / delta / 4 / val


@nb.jit(forceobj=True)
def initial_position(ne, atom_positions, atom_charges):
    """Initial positions of electrons."""
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
    return r_e + np.random.laplace(0, 1, ne * 3).reshape((ne, 3))


@nb.jit(nopython=True)
def random_step(step, ne):
    """Random N-dim square distributed step"""
    return step * np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))


# @pool
@nb.jit(nopython=True, nogil=True)
def profiling_value(dx, neu, ned, steps, atom_positions, slater, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        n_vectors = subtract_outer(atom_positions, r_e)
        slater.value(n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def profiling_gradient(dx, neu, ned, steps, atom_positions, slater, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        n_vectors = subtract_outer(atom_positions, r_e)
        slater.gradient(n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def profiling_laplacian(dx, neu, ned, steps, atom_positions, slater, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        n_vectors = subtract_outer(atom_positions, r_e)
        slater.laplacian(n_vectors)


# @pool
@nb.jit(nopython=True, nogil=True)
def profiling_hessian(dx, neu, ned, steps, atom_positions, slater, r_initial):

    for _ in range(steps):
        r_e = r_initial + random_step(dx, neu + ned)
        n_vectors = subtract_outer(atom_positions, r_e)
        slater.hessian(n_vectors)


def main(casino):
    dx = 3.0

    slater = Slater(
        casino.input.neu, casino.input.ned,
        casino.wfn.nbasis_functions, casino.wfn.first_shells, casino.wfn.orbital_types, casino.wfn.shell_moments,
        casino.wfn.slater_orders, casino.wfn.primitives, casino.wfn.coefficients, casino.wfn.exponents,
        casino.mdet.mo_up, casino.mdet.mo_down, casino.mdet.coeff
    )

    r_initial = initial_position(casino.input.neu + casino.input.ned, casino.wfn.atom_positions, casino.wfn.atom_charges)

    start = default_timer()
    profiling_value(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, slater, r_initial)
    end = default_timer()
    logger.info(' value     %8.1f', end - start)

    start = default_timer()
    profiling_laplacian(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, slater, r_initial)
    end = default_timer()
    logger.info(' laplacian %8.1f', end - start)

    start = default_timer()
    profiling_gradient(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, slater, r_initial)
    end = default_timer()
    logger.info(' gradient  %8.1f', end - start)

    start = default_timer()
    profiling_hessian(dx, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, slater, r_initial)
    end = default_timer()
    logger.info(' hessian   %8.1f', end - start)


if __name__ == '__main__':
    """
    He:
     value         25.7
     laplacian     55.8
     gradient     134.5
     hessian      413.7
    Be:
     value         45.5
     laplacian     98.2
     gradient     242.5
     hessian      729.8
    Ne:
     value        101.4
     laplacian    224.5
     gradient     529.8
     hessian     1616.1
    Ar:
     value        274.7
     laplacian    538.0
     gradient    1078.3
     hessian     3029.9
    Kr:
     value        751.1
     laplacian   1602.4
     gradient    2684.6
     hessian     7316.8
    O3:
     value        626.4
     laplacian   1272.6
    """

    for mol in ('He', ):
        path = f'test/gwfn/{mol}/HF/cc-pVQZ/CBCS/Slater/'
        # path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Slater/'
        logger.info('%s:', mol)
        main(CasinoConfig(path))
