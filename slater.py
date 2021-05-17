#!/usr/bin/env python3

import os
from typing import Tuple
from math import gamma
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
from readers.wfn import GAUSSIAN_TYPE, SLATER_TYPE
from readers.casino import Casino


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

    def AO_gradient(self, n_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gradient matrix.
        :param n_vectors: electron-nuclei - array(natom, nelec, 3)
        :return: AO gradient - list of array(nelec, nbasis_functions)
        """
        orbital_x = np.zeros((self.neu + self.ned, self.nbasis_functions))
        orbital_y = np.zeros((self.neu + self.ned, self.nbasis_functions))
        orbital_z = np.zeros((self.neu + self.ned, self.nbasis_functions))
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
                        orbital_x[i, ao+m] = x * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 0] * radial_2
                        orbital_y[i, ao+m] = y * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 1] * radial_2
                        orbital_z[i, ao+m] = z * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 2] * radial_2
                    ao += 2*l+1
        return orbital_x, orbital_y, orbital_z

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

    def AO_hessian(self, n_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """hessian matrix.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        :return: AO hessian - array(nelec, nbasis_functions)
        """
        orbital_xx = np.zeros((self.neu + self.ned, self.nbasis_functions))
        orbital_xy = np.zeros((self.neu + self.ned, self.nbasis_functions))
        orbital_yy = np.zeros((self.neu + self.ned, self.nbasis_functions))
        orbital_xz = np.zeros((self.neu + self.ned, self.nbasis_functions))
        orbital_yz = np.zeros((self.neu + self.ned, self.nbasis_functions))
        orbital_zz = np.zeros((self.neu + self.ned, self.nbasis_functions))

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
                        orbital_xx[i, ao+m] = x*x * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * x * angular_2[l*l+m, 0]) * radial_2 + angular_3[l*l+m, 0] * radial_3
                        orbital_xy[i, ao+m] = x*y * angular_1[l*l+m] * radial_1 + (y * angular_2[l*l+m, 0] + x * angular_2[l*l+m, 1]) * radial_2 + angular_3[l*l+m, 1] * radial_3
                        orbital_yy[i, ao+m] = y*y * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * y * angular_2[l*l+m, 1]) * radial_2 + angular_3[l*l+m, 2] * radial_3
                        orbital_xz[i, ao+m] = x*z * angular_1[l*l+m] * radial_1 + (z * angular_2[l*l+m, 0] + x * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 3] * radial_3
                        orbital_yz[i, ao+m] = y*z * angular_1[l*l+m] * radial_1 + (z * angular_2[l*l+m, 1] + y * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 4] * radial_3
                        orbital_zz[i, ao+m] = z*z * angular_1[l*l+m] * radial_1 + (angular_1[l*l+m] + 2 * z * angular_2[l*l+m, 2]) * radial_2 + angular_3[l*l+m, 5] * radial_3
                    ao += 2*l+1

        return orbital_xx, orbital_xy, orbital_yy, orbital_xz, orbital_yz, orbital_zz

    def value(self, n_vectors: np.ndarray) -> float:
        """Multideterminant wave function value.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao = self.AO_wfn(n_vectors)

        val = 0.0
        for i in range(self.coeff.shape[0]):
            wfn_u = self.mo_up[i] @ ao[:self.neu].T
            wfn_d = self.mo_down[i] @ ao[self.neu:].T
            val += self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
        return val

    def gradient(self, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient ∇(phi).
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)

        As numba overloaded function 'dot' and @ only supported on 1-D and 2-D arrays,
        so I use list of 2-D arrays (gradient_x, gradient_y, gradient_z) to represent gradient.
        """
        ao = self.AO_wfn(n_vectors)
        gradient_x, gradient_y, gradient_z = self.AO_gradient(n_vectors)

        val = 0.0
        grad = np.zeros((self.neu + self.ned, 3))
        for i in range(self.coeff.shape[0]):

            wfn_u = self.mo_up[i] @ ao[:self.neu].T
            inv_wfn_u = np.linalg.inv(wfn_u)
            grad_x = self.mo_up[i] @ gradient_x[:self.neu].T
            grad_y = self.mo_up[i] @ gradient_y[:self.neu].T
            grad_z = self.mo_up[i] @ gradient_z[:self.neu].T

            res_u = np.zeros((self.neu, 3))
            res_u[:, 0] = np.diag(inv_wfn_u @ grad_x)
            res_u[:, 1] = np.diag(inv_wfn_u @ grad_y)
            res_u[:, 2] = np.diag(inv_wfn_u @ grad_z)

            wfn_d = self.mo_down[i] @ ao[self.neu:].T
            inv_wfn_d = np.linalg.inv(wfn_d)
            grad_x = self.mo_down[i] @ gradient_x[self.neu:].T
            grad_y = self.mo_down[i] @ gradient_y[self.neu:].T
            grad_z = self.mo_down[i] @ gradient_z[self.neu:].T

            res_d = np.zeros((self.ned, 3))
            res_d[:, 0] = np.diag(inv_wfn_d @ grad_x)
            res_d[:, 1] = np.diag(inv_wfn_d @ grad_y)
            res_d[:, 2] = np.diag(inv_wfn_d @ grad_z)

            c = self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
            val += c
            grad += c * np.concatenate((res_u, res_d))

        return grad.ravel() / val

    def laplacian(self, n_vectors: np.ndarray) -> float:
        """Scalar laplacian Δ(phi).
        Δ(det(slater)) = det(slater) * sum(tr(slater**-1 * B(n)) over n
        where the matrix B(n) is zero with the exception of the n-th column
        as tr(A) + tr(B) = tr(A + B)
        Δ(det(slater)) = det(slater) * tr(slater**-1 * B)
        where the matrix Bij = ∆phi i (rj)
        then using np.trace(A @ B) = np.sum(A * B.T)
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao = self.AO_wfn(n_vectors)
        ao_laplacian = self.AO_laplacian(n_vectors)

        val = lap = 0
        for i in range(self.coeff.shape[0]):

            wfn_u = self.mo_up[i] @ ao[:self.neu].T
            lap_u = self.mo_up[i] @ ao_laplacian[:self.neu].T
            res_u = np.sum(np.linalg.inv(wfn_u) * lap_u.T)

            wfn_d = self.mo_down[i] @ ao[self.neu:].T
            lap_d = self.mo_down[i] @ ao_laplacian[self.neu:].T
            res_d = np.sum(np.linalg.inv(wfn_d) * lap_d.T)

            c = self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
            val += c
            lap += c * (res_u + res_d)

        return lap / val

    def hessian(self, n_vectors: np.ndarray):
        """Hessian.
        d²det(A)/dxdy = det(A) * (
            tr(A**-1 * d²A/dxdy) +
            tr(A**-1 * dA/dx) * tr(A**-1 * dA/dy) -
            tr(A**-1 * dA/dx * A**-1 * dA/dy)
        )
        in case of x and y is a coordinates of different electrons first term is zero
        in other case a sum of last two terms is zero.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao = self.AO_wfn(n_vectors)
        gradient_x, gradient_y, gradient_z = self.AO_gradient(n_vectors)
        hessian_xx, hessian_xy, hessian_yy, hessian_xz, hessian_yz, hessian_zz = self.AO_hessian(n_vectors)
        cond_u = np.arange(self.neu) * np.ones((self.neu, self.neu))
        cond_d = np.arange(self.ned) * np.ones((self.ned, self.ned))

        val = 0
        hass = np.zeros((self.neu + self.ned, 3, self.neu + self.ned, 3))
        for i in range(self.coeff.shape[0]):

            wfn_u = self.mo_up[i] @ ao[:self.neu].T
            inv_wfn_u = np.linalg.inv(wfn_u)
            grad_x = self.mo_up[i] @ gradient_x[:self.neu].T
            grad_y = self.mo_up[i] @ gradient_y[:self.neu].T
            grad_z = self.mo_up[i] @ gradient_z[:self.neu].T
            hess_xx = self.mo_up[i] @ hessian_xx[:self.neu].T
            hess_xy = self.mo_up[i] @ hessian_xy[:self.neu].T
            hess_yy = self.mo_up[i] @ hessian_yy[:self.neu].T
            hess_xz = self.mo_up[i] @ hessian_xz[:self.neu].T
            hess_yz = self.mo_up[i] @ hessian_yz[:self.neu].T
            hess_zz = self.mo_up[i] @ hessian_zz[:self.neu].T

            res_grad_u = np.zeros((self.neu, 3))
            res_u = np.zeros((self.neu, 3, self.neu, 3))
            res_grad_u[:, 0] = np.diag(inv_wfn_u @ grad_x)
            res_grad_u[:, 1] = np.diag(inv_wfn_u @ grad_y)
            res_grad_u[:, 2] = np.diag(inv_wfn_u @ grad_z)
            t = np.zeros((self.neu, 3, 3))
            t[:, 0, 0] = np.diag(inv_wfn_u @ hess_xx)
            t[:, 0, 1] = t[:, 1, 0] = np.diag(inv_wfn_u @ hess_xy)
            t[:, 1, 1] = np.diag(inv_wfn_u @ hess_yy)
            t[:, 2, 0] = t[:, 0, 2] = np.diag(inv_wfn_u @ hess_xz)
            t[:, 2, 1] = t[:, 1, 2] = np.diag(inv_wfn_u @ hess_yz)
            t[:, 2, 2] = np.diag(inv_wfn_u @ hess_zz)
            for j in range(self.neu):
                res_u[j, :, j, :] = t[j]

            dx = np.zeros((self.neu, self.neu, self.neu))
            dy = np.zeros((self.ned, self.ned, self.ned))
            dz = np.zeros((self.ned, self.ned, self.ned))
            for j in range(self.neu):
                dx[j] = inv_wfn_u @ np.where(cond_u == j, grad_x, 0)
                dy[j] = inv_wfn_u @ np.where(cond_u == j, grad_y, 0)
                dz[j] = inv_wfn_u @ np.where(cond_u == j, grad_z, 0)
                for k in range(j + 1):
                    res_u[j, 0, k, 0] = -np.trace(dx[j] @ dx[k])
                    res_u[j, 0, k, 1] = -np.trace(dx[j] @ dy[k])
                    res_u[j, 1, k, 0] = -np.trace(dy[j] @ dx[k])
                    res_u[j, 1, k, 1] = -np.trace(dy[j] @ dy[k])
                    res_u[j, 0, k, 2] = -np.trace(dx[j] @ dz[k])
                    res_u[j, 2, k, 0] = -np.trace(dz[j] @ dx[k])
                    res_u[j, 1, k, 2] = -np.trace(dy[j] @ dz[k])
                    res_u[j, 2, k, 1] = -np.trace(dz[j] @ dy[k])
                    res_u[j, 2, k, 2] = -np.trace(dz[j] @ dz[k])
                    if j != k:
                        res_u[k, :, j, :] = res_u[j, :, k, :].T

            t = np.zeros((self.neu, 3, 3))
            t[:, 0, 0] = np.diag(inv_wfn_u @ hess_xx)
            t[:, 0, 1] = t[:, 1, 0] = np.diag(inv_wfn_u @ hess_xy)
            t[:, 1, 1] = np.diag(inv_wfn_u @ hess_yy)
            t[:, 2, 0] = t[:, 0, 2] = np.diag(inv_wfn_u @ hess_xz)
            t[:, 2, 1] = t[:, 1, 2] = np.diag(inv_wfn_u @ hess_yz)
            t[:, 2, 2] = np.diag(inv_wfn_u @ hess_zz)
            for j in range(self.neu):
                res_u[j, :, j, :] += t[j]

            wfn_d = self.mo_down[i] @ ao[self.neu:].T
            inv_wfn_d = np.linalg.inv(wfn_d)
            grad_x = self.mo_down[i] @ gradient_x[self.neu:].T
            grad_y = self.mo_down[i] @ gradient_y[self.neu:].T
            grad_z = self.mo_down[i] @ gradient_z[self.neu:].T
            hess_xx = self.mo_down[i] @ hessian_xx[self.neu:].T
            hess_xy = self.mo_down[i] @ hessian_xy[self.neu:].T
            hess_yy = self.mo_down[i] @ hessian_yy[self.neu:].T
            hess_xz = self.mo_down[i] @ hessian_xz[self.neu:].T
            hess_yz = self.mo_down[i] @ hessian_yz[self.neu:].T
            hess_zz = self.mo_down[i] @ hessian_zz[self.neu:].T

            res_grad_d = np.zeros((self.ned, 3))
            res_d = np.zeros((self.ned, 3, self.ned, 3))
            res_grad_d[:, 0] = np.diag(inv_wfn_d @ grad_x)
            res_grad_d[:, 1] = np.diag(inv_wfn_d @ grad_y)
            res_grad_d[:, 2] = np.diag(inv_wfn_d @ grad_z)

            dx = np.zeros((self.ned, self.ned, self.ned))
            dy = np.zeros((self.ned, self.ned, self.ned))
            dz = np.zeros((self.ned, self.ned, self.ned))
            for j in range(self.ned):
                dx[j] = inv_wfn_d @ np.where(cond_d == j, grad_x, 0)
                dy[j] = inv_wfn_d @ np.where(cond_d == j, grad_y, 0)
                dz[j] = inv_wfn_d @ np.where(cond_d == j, grad_z, 0)
                for k in range(j + 1):
                    res_d[j, 0, k, 0] = -np.trace(dx[j] @ dx[k])
                    res_d[j, 0, k, 1] = -np.trace(dx[j] @ dy[k])
                    res_d[j, 1, k, 0] = -np.trace(dy[j] @ dx[k])
                    res_d[j, 1, k, 1] = -np.trace(dy[j] @ dy[k])
                    res_d[j, 0, k, 2] = -np.trace(dx[j] @ dz[k])
                    res_d[j, 2, k, 0] = -np.trace(dz[j] @ dx[k])
                    res_d[j, 1, k, 2] = -np.trace(dy[j] @ dz[k])
                    res_d[j, 2, k, 1] = -np.trace(dz[j] @ dy[k])
                    res_d[j, 2, k, 2] = -np.trace(dz[j] @ dz[k])
                    if j != k:
                        res_d[k, :, j, :] = res_d[j, :, k, :].T

            t = np.zeros((self.ned, 3, 3))
            t[:, 0, 0] = np.diag(inv_wfn_d @ hess_xx)
            t[:, 0, 1] = t[:, 1, 0] = np.diag(inv_wfn_d @ hess_xy)
            t[:, 1, 1] = np.diag(inv_wfn_d @ hess_yy)
            t[:, 2, 0] = t[:, 0, 2] = np.diag(inv_wfn_d @ hess_xz)
            t[:, 2, 1] = t[:, 1, 2] = np.diag(inv_wfn_d @ hess_yz)
            t[:, 2, 2] = np.diag(inv_wfn_d @ hess_zz)
            for j in range(self.ned):
                res_d[j, :, j, :] += t[j]

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

        res = np.zeros((self.neu + self.ned, 3))
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= delta
                res[i, j] -= self.value(n_vectors)
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.value(n_vectors)
                n_vectors[:, i, j] -= delta

        return res.ravel() / delta / 2

    def numerical_laplacian(self, n_vectors: np.ndarray) -> float:
        """Numerical laplacian with respect to a e-coordinates
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        delta = 0.00001

        res = - 6 * (self.neu + self.ned) * self.value(n_vectors)
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= delta
                res += self.value(n_vectors)
                n_vectors[:, i, j] += 2 * delta
                res += self.value(n_vectors)
                n_vectors[:, i, j] -= delta

        return res / delta / delta

    def numerical_hessian(self, n_vectors: np.ndarray):
        """Numerical hessian with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        delta = 0.00001

        res = -2 * self.value(n_vectors) * np.eye((self.neu + self.ned) * 3).reshape(self.neu + self.ned, 3, self.neu + self.ned, 3)
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

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3) / delta / delta / 4


@nb.jit(nopython=True)
def random_step(dx, ne):
    """Random N-dim square distributed step"""
    return np.random.uniform(-dx, dx, ne * 3).reshape((ne, 3))


# @pool
@nb.jit(nopython=True, nogil=True)
def integral(dX, neu, ned, steps, atom_positions, slater, r_initial):
    """https://en.wikipedia.org/wiki/Monte_Carlo_integration"""
    v = (2 * dX) ** (3 * (neu + ned))  # integration volume
    slater_determinant_normalization_factor = np.sqrt(1 / gamma(neu+1) / gamma(ned+1))

    result = 0.0
    for i in range(steps):
        r_e = r_initial + random_step(dX, neu + ned)
        n_vectors = subtract_outer(atom_positions, r_e)
        result += (slater_determinant_normalization_factor * slater.value(n_vectors)) ** 2

    return result * v / steps


def plot_graph(neu, ned, steps, atom_positions, slater):
    """Plot a graph along random line going through (0, 0, 0)"""
    import matplotlib.pyplot as plt
    for n in range(100000):
        res = np.random.uniform(0, 1, (neu + ned) * 3)
        res /= np.linalg.norm(res)
        res = res.reshape((neu + ned, 3))
        x_grid = np.linspace(0, 5, steps)
        y_grid = np.zeros((steps, ))
        n_unit_vectors = subtract_outer(atom_positions, res)
        for i in range(steps):
            n_vectors = n_unit_vectors * x_grid[i]
            y_grid[i] = slater.value(n_vectors, neu)
        a, b = np.min(y_grid), np.max(y_grid)
        if np.sign(a) * np.sign(b) < 0:
            print(f'{n}-th try, random normal vector: {res}, min {a}, max {b}')
            plt.plot(x_grid, y_grid)
            plt.xlabel('r_eN (au)')
            plt.ylabel('polynomial part')
            plt.title('JASTROW chi-term')
            plt.grid(True)
            plt.legend()
            plt.show()

    return 0


@nb.jit(forceobj=True)
def initial_position(ne, atom_positions, atom_charges):
    """Initial positions of electrons."""
    natoms = atom_positions.shape[0]
    r_e = np.zeros((ne, 3))
    for i in range(ne):
        r_e[i] = atom_positions[np.random.choice(natoms, p=atom_charges / atom_charges.sum())]
    return r_e


def main(casino):
    dX = 3.0

    slater = Slater(
        casino.input.neu, casino.input.ned,
        casino.wfn.nbasis_functions, casino.wfn.first_shells, casino.wfn.orbital_types, casino.wfn.shell_moments,
        casino.wfn.slater_orders, casino.wfn.primitives, casino.wfn.coefficients, casino.wfn.exponents,
        casino.mdet.mo_up, casino.mdet.mo_down, casino.mdet.coeff
    )

    r_initial = initial_position(casino.input.neu + casino.input.ned, casino.wfn.atom_positions, casino.wfn.atom_charges)
    return integral(dX, casino.input.neu, casino.input.ned, casino.input.vmc_nstep, casino.wfn.atom_positions, slater, r_initial)
    # return plot_graph(casino.input.neu, casino.input.ned, 100, casino.wfn.atom_positions, slater)


if __name__ == '__main__':
    """
    be HF/cc-pVQZ

    steps = 5 * 1000 * 1000 * 1000
    offset = 4.5

    0.925763438273841

    real    464m11,715s
    user    464m11,194s
    sys     0m0,488s
    """

    # path = 'test/gwfn/He/HF/cc-pVQZ/Slater/'
    # path = 'test/gwfn/Be/HF/cc-pVQZ/Slater/'
    # path = 'test/gwfn/Ne/HF/cc-pVQZ/Slater/'
    # path = 'test/gwfn/Ae/HF/cc-pVQZ/Slater/'
    # path = 'test/gwfn/Kr/HF/cc-pVQZ/Slater/'
    # path = 'test/gwfn/O3/HF/cc-pVQZ/Slater/'

    # path = 'test/stowfn/He/HF/QZ4P/Slater/'
    # path = 'test/stowfn/Be/HF/QZ4P/Slater/'
    # path = 'test/stowfn/Ne/HF/QZ4P/Slater/'
    # path = 'test/stowfn/Ar/HF/QZ4P/Slater/'
    # path = 'test/stowfn/Kr/HF/QZ4P/Slater/'
    # path = 'test/stowfn/O3/HF/QZ4P/Slater/'

    start = default_timer()
    res = main(Casino(path))
    print(np.mean(res), '+/-', np.var(res))
    end = default_timer()
    print(f'total time {end-start}')
