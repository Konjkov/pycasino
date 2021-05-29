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
from readers.casino import Casino

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

    def AO_gradient(self, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient matrix.
        :param n_vectors: electron-nuclei - array(natom, nelec, 3)
        :return: AO gradient - array(3, nelec, nbasis_functions)
        """
        orbital = np.zeros((3, self.neu + self.ned, self.nbasis_functions))
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
                        orbital[0, i, ao+m] = x * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 0] * radial_2
                        orbital[1, i, ao+m] = y * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 1] * radial_2
                        orbital[2, i, ao+m] = z * angular_1[l*l+m] * radial_1 + angular_2[l*l+m, 2] * radial_2
                    ao += 2*l+1
        return orbital

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

        val = 0.0
        for i in range(self.coeff.shape[0]):
            wfn_u = self.mo_up[i] @ ao[:self.neu].T
            wfn_d = self.mo_down[i] @ ao[self.neu:].T
            val += self.coeff[i] * np.linalg.det(wfn_u) * np.linalg.det(wfn_d)
        return val

    def gradient(self, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient ∇(phi).
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        ao = self.AO_wfn(n_vectors)
        gradient = self.AO_gradient(n_vectors)

        val = 0.0
        grad = np.zeros((self.neu + self.ned, 3))
        for i in range(self.coeff.shape[0]):

            wfn_u = self.mo_up[i] @ ao[:self.neu].T
            inv_wfn_u = np.linalg.inv(wfn_u)
            grad_x = self.mo_up[i] @ gradient[0, :self.neu].T
            grad_y = self.mo_up[i] @ gradient[1, :self.neu].T
            grad_z = self.mo_up[i] @ gradient[2, :self.neu].T

            res_u = np.zeros((self.neu, 3))
            res_u[:, 0] = np.diag(inv_wfn_u @ grad_x)
            res_u[:, 1] = np.diag(inv_wfn_u @ grad_y)
            res_u[:, 2] = np.diag(inv_wfn_u @ grad_z)

            wfn_d = self.mo_down[i] @ ao[self.neu:].T
            inv_wfn_d = np.linalg.inv(wfn_d)
            grad_x = self.mo_down[i] @ gradient[0, self.neu:].T
            grad_y = self.mo_down[i] @ gradient[1, self.neu:].T
            grad_z = self.mo_down[i] @ gradient[2, self.neu:].T

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
        gradient = self.AO_gradient(n_vectors)
        hessian = self.AO_hessian(n_vectors)

        val = 0
        hass = np.zeros((self.neu + self.ned, 3, self.neu + self.ned, 3))
        for i in range(self.coeff.shape[0]):

            wfn_u = self.mo_up[i] @ ao[:self.neu].T
            inv_wfn_u = np.linalg.inv(wfn_u)
            grad_x = self.mo_up[i] @ gradient[0, :self.neu].T
            grad_y = self.mo_up[i] @ gradient[1, :self.neu].T
            grad_z = self.mo_up[i] @ gradient[2, :self.neu].T
            hess_xx = self.mo_up[i] @ hessian[0, :self.neu].T
            hess_xy = self.mo_up[i] @ hessian[1, :self.neu].T
            hess_yy = self.mo_up[i] @ hessian[2, :self.neu].T
            hess_xz = self.mo_up[i] @ hessian[3, :self.neu].T
            hess_yz = self.mo_up[i] @ hessian[4, :self.neu].T
            hess_zz = self.mo_up[i] @ hessian[5, :self.neu].T

            res_grad_u = np.zeros((self.neu, 3))
            res_u = np.zeros((self.neu, 3, self.neu, 3))

            dx = inv_wfn_u @ grad_x
            dy = inv_wfn_u @ grad_y
            dz = inv_wfn_u @ grad_z

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
            grad_x = self.mo_down[i] @ gradient[0, self.neu:].T
            grad_y = self.mo_down[i] @ gradient[1, self.neu:].T
            grad_z = self.mo_down[i] @ gradient[2, self.neu:].T
            hess_xx = self.mo_down[i] @ hessian[0, self.neu:].T
            hess_xy = self.mo_down[i] @ hessian[1, self.neu:].T
            hess_yy = self.mo_down[i] @ hessian[2, self.neu:].T
            hess_xz = self.mo_down[i] @ hessian[3, self.neu:].T
            hess_yz = self.mo_down[i] @ hessian[4, self.neu:].T
            hess_zz = self.mo_down[i] @ hessian[5, self.neu:].T

            res_grad_d = np.zeros((self.ned, 3))
            res_d = np.zeros((self.ned, 3, self.ned, 3))

            dx = inv_wfn_d @ grad_x
            dy = inv_wfn_d @ grad_y
            dz = inv_wfn_d @ grad_z

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
    return r_e


@nb.jit(nopython=True)
def random_step(dx, ne):
    """Random N-dim square distributed step"""
    return np.random.uniform(-dx, dx, ne * 3).reshape((ne, 3))


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
     value        125.4
     laplacian    244.3
     gradient     555.3
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

    for mol in ('He', 'Be', 'Ne', 'Ar', 'Kr'):
        # path = f'test/gwfn/{mol}/HF/cc-pVQZ/CBCS/Slater/'
        path = f'test/stowfn/{mol}/HF/QZ4P/CBCS/Slater/'
        logger.info('%s:', mol)
        main(Casino(path))
