import numpy as np
from abc import ABC, abstractmethod
from pycasino import delta, delta_2, delta_3


class AbstractCusp:

    def __init__(self, neu, ned, orbitals_up, orbitals_down):
        """
        :param neu: number of up electrons
        :param ned: number of down electrons
        """
        self.neu = neu
        self.ned = ned
        self.orbitals_up = orbitals_up
        self.orbitals_down = orbitals_down

    # @abstractmethod
    def value(self, n_vectors: np.ndarray):
        """Value φ(r)
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """

    # @abstractmethod
    def numerical_gradient(self, n_vectors: np.ndarray):
        """Cusp part of gradient"""
        gradient = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned, 3))
        for j in range(3):
            n_vectors[:, :, j] -= delta
            value = self.value(n_vectors)
            gradient[:self.orbitals_up, :self.neu, j] -= value[0]
            gradient[self.orbitals_up:, self.neu:, j] -= value[1]
            n_vectors[:, :, j] += 2 * delta
            value = self.value(n_vectors)
            gradient[:self.orbitals_up, :self.neu, j] += value[0]
            gradient[self.orbitals_up:, self.neu:, j] += value[1]
            n_vectors[:, :, j] -= delta

        return gradient[:self.orbitals_up, :self.neu] / delta / 2, gradient[self.orbitals_up:, self.neu:] / delta / 2

    # @abstractmethod
    def numerical_laplacian(self, n_vectors: np.ndarray):
        """Cusp part of laplacian"""
        laplacian = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned))
        value = self.value(n_vectors)
        laplacian[:self.orbitals_up, :self.neu] -= 6 * value[0]
        laplacian[self.orbitals_up:, self.neu:] -= 6 * value[1]
        for j in range(3):
            n_vectors[:, :, j] -= delta
            value = self.value(n_vectors)
            laplacian[:self.orbitals_up, :self.neu] += value[0]
            laplacian[self.orbitals_up:, self.neu:] += value[1]
            n_vectors[:, :, j] += 2 * delta
            value = self.value(n_vectors)
            laplacian[:self.orbitals_up, :self.neu] += value[0]
            laplacian[self.orbitals_up:, self.neu:] += value[1]
            n_vectors[:, :, j] -= delta

        return laplacian[:self.orbitals_up, :self.neu] / delta / delta, laplacian[self.orbitals_up:, self.neu:] / delta / delta

    # @abstractmethod
    def numerical_hessian(self, n_vectors: np.ndarray):
        """Cusp part of hessian"""
        hessian = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned, 3, 3))
        for j in range(3):
            n_vectors[:, :, j] -= 2 * delta_2
            value = self.value(n_vectors)
            hessian[:self.orbitals_up, :self.neu, j, j] -= value[0]
            hessian[self.orbitals_up:, self.neu:, j, j] -= value[1]
            n_vectors[:, :, j] += 4 * delta_2
            value = self.value(n_vectors)
            hessian[:self.orbitals_up, :self.neu, j, j] += value[0]
            hessian[self.orbitals_up:, self.neu:, j, j] += value[1]
            n_vectors[:, :, j] -= 2 * delta_2

        return hessian[:self.orbitals_up, :self.neu] / delta_2 / delta_2 / 4, hessian[self.orbitals_up:, self.neu:] / delta_2 / delta_2 / 4

    # @abstractmethod
    def numerical_tressian(self, n_vectors: np.ndarray):
        """Cusp part of tressian"""
        tressian = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned, 3, 3, 3))
        return tressian[:self.orbitals_up, :self.neu], tressian[self.orbitals_up:, self.neu:]


class AbstractSlater:

    def __init__(self, neu, ned):
        """
        :param neu: number of up electrons
        :param ned: number of down electrons
        """
        self.neu = neu
        self.ned = ned

    # @abstractmethod
    def value(self, n_vectors: np.ndarray) -> float:
        """Value φ(r)
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """

    def numerical_gradient(self, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient ∇φ(r)/φ(r) w.r.t. e-coordinates.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        val = self.value(n_vectors)
        res = np.zeros(shape=(self.neu + self.ned, 3))
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= delta
                res[i, j] -= self.value(n_vectors)
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.value(n_vectors)
                n_vectors[:, i, j] -= delta

        return res.ravel() / delta / 2 / val

    def numerical_laplacian(self, n_vectors: np.ndarray) -> float:
        """Laplacian Δφ(r)/φ(r) w.r.t. e-coordinates.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
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

    def numerical_hessian(self, n_vectors: np.ndarray) -> np.ndarray:
        """Hessian H(φ(r))/φ(r) w.r.t. e-coordinates.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        :return:
        """
        val = self.value(n_vectors)
        res = -2 * val * np.eye((self.neu + self.ned) * 3).reshape(self.neu + self.ned, 3, self.neu + self.ned, 3)
        for i in range(self.neu + self.ned):
            for j in range(3):
                n_vectors[:, i, j] -= 2 * delta_2
                res[i, j, i, j] += self.value(n_vectors)
                n_vectors[:, i, j] += 4 * delta_2
                res[i, j, i, j] += self.value(n_vectors)
                n_vectors[:, i, j] -= 2 * delta_2

        for i1 in range(self.neu + self.ned):
            for j1 in range(3):
                for i2 in range(i1 + 1):
                    for j2 in range(3):
                        if i1 == i2 and j1 >= j2:
                            continue
                        n_vectors[:, i1, j1] -= delta_2
                        n_vectors[:, i2, j2] -= delta_2
                        res[i1, j1, i2, j2] += self.value(n_vectors)
                        n_vectors[:, i1, j1] += 2 * delta_2
                        res[i1, j1, i2, j2] -= self.value(n_vectors)
                        n_vectors[:, i2, j2] += 2 * delta_2
                        res[i1, j1, i2, j2] += self.value(n_vectors)
                        n_vectors[:, i1, j1] -= 2 * delta_2
                        res[i1, j1, i2, j2] -= self.value(n_vectors)
                        n_vectors[:, i1, j1] += delta_2
                        n_vectors[:, i2, j2] -= delta_2
                        res[i2, j2, i1, j1] = res[i1, j1, i2, j2]

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3) / delta_2 / delta_2 / 4 / val

    # @abstractmethod
    def numerical_tressian(self, n_vectors: np.ndarray) -> np.ndarray:
        """Third-order partial derivatives T(φ(r))/φ(r) w.r.t. e-coordinates.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        val = self.value(n_vectors)
        res = np.zeros(shape=(self.neu + self.ned, 3, self.neu + self.ned, 3, self.neu + self.ned, 3))

        for i1 in range(self.neu + self.ned):
            for j1 in range(3):
                for i2 in range(self.neu + self.ned):
                    for j2 in range(3):
                        for i3 in range(self.neu + self.ned):
                            for j3 in range(3):
                                n_vectors[:, i1, j1] -= delta_3
                                n_vectors[:, i2, j2] -= delta_3
                                n_vectors[:, i3, j3] -= delta_3
                                # (-1, -1, -1)
                                res[i1, j1, i2, j2, i3, j3] -= self.value(n_vectors)
                                n_vectors[:, i1, j1] += 2 * delta_3
                                # ( 1, -1, -1)
                                res[i1, j1, i2, j2, i3, j3] += self.value(n_vectors)
                                n_vectors[:, i2, j2] += 2 * delta_3
                                # ( 1,  1, -1)
                                res[i1, j1, i2, j2, i3, j3] -= self.value(n_vectors)
                                n_vectors[:, i1, j1] -= 2 * delta_3
                                # (-1,  1, -1)
                                res[i1, j1, i2, j2, i3, j3] += self.value(n_vectors)
                                n_vectors[:, i2, j2] -= 2 * delta_3
                                n_vectors[:, i3, j3] += 2 * delta_3
                                # (-1, -1,  1)
                                res[i1, j1, i2, j2, i3, j3] += self.value(n_vectors)
                                n_vectors[:, i1, j1] += 2 * delta_3
                                # ( 1, -1,  1)
                                res[i1, j1, i2, j2, i3, j3] -= self.value(n_vectors)
                                n_vectors[:, i2, j2] += 2 * delta_3
                                # ( 1,  1,  1)
                                res[i1, j1, i2, j2, i3, j3] += self.value(n_vectors)
                                n_vectors[:, i1, j1] -= 2 * delta_3
                                # (-1,  1,  1)
                                res[i1, j1, i2, j2, i3, j3] -= self.value(n_vectors)
                                n_vectors[:, i1, j1] += delta_3
                                n_vectors[:, i2, j2] -= delta_3
                                n_vectors[:, i3, j3] -= delta_3

                                # res[i1, j1, i3, j3, i2, j2] = res[i1, j1, i2, j2, i3, j3]
                                # res[i2, j2, i1, j1, i3, j3] = res[i1, j1, i2, j2, i3, j3]
                                # res[i2, j2, i3, j3, i1, j1] = res[i1, j1, i2, j2, i3, j3]
                                # res[i3, j3, i1, j1, i2, j2] = res[i1, j1, i2, j2, i3, j3]
                                # res[i3, j3, i2, j2, i1, j1] = res[i1, j1, i2, j2, i3, j3]

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3) / delta_3 / delta_3 / delta_3 / 8 / val

    # @abstractmethod
    def get_parameters(self) -> np.ndarray:
        """Get parameters"""

    # @abstractmethod
    def set_parameters(self, parameters: np.ndarray):
        """Set parameters"""


class AbstractJastrow:

    def __init__(self, neu, ned):
        """
        :param neu: number of up electrons
        :param ned: number of down electrons
        """
        self.neu = neu
        self.ned = ned

    # @abstractmethod
    def value(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> float:
        """Value
        :param e_vectors: electron-electron vectors shape = (nelec, nelec, 3)
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """

    def numerical_gradient(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient w.r.t. e-coordinates.
        :param e_vectors: electron-electron vectors shape = (nelec, nelec, 3)
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        :return:
        """
        res = np.zeros(shape=(self.neu + self.ned, 3))
        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res[i, j] -= self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res.ravel() / delta / 2

    def numerical_laplacian(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> float:
        """Laplacian w.r.t. e-coordinates.
        :param e_vectors: electron-electron vectors shape = (nelec, nelec, 3)
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        res = -6 * (self.neu + self.ned) * self.value(e_vectors, n_vectors)
        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res += self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res += self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res / delta / delta

    # @abstractmethod
    def get_parameters(self) -> np.ndarray:
        """Get parameters"""

    # @abstractmethod
    def set_parameters(self, parameters: np.ndarray):
        """Set parameters"""


class AbstractBackflow:

    def __init__(self, neu, ned):
        """
        :param neu: number of up electrons
        :param ned: number of down electrons
        """
        self.neu = neu
        self.ned = ned

    # @abstractmethod
    def value(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        """Value"""

    def numerical_gradient(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        """Gradient w.r.t. e-coordinates.
        :param e_vectors: electron-electron vectors shape = (nelec, nelec, 3)
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        :return: partial derivatives of displacements of electrons shape = (nelec * 3, nelec * 3)
        """
        res = np.zeros(shape=(self.neu + self.ned, 3, self.neu + self.ned, 3))

        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res[:, :, i, j] -= self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res[:, :, i, j] += self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3) / delta / 2

    def numerical_laplacian(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        """Laplacian w.r.t. e-coordinates.
        :param e_vectors: electron-electron vectors shape = (nelec, nelec, 3)
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        :return: vector laplacian shape = (nelec * 3)
        """
        res = -6 * (self.neu + self.ned) * self.value(e_vectors, n_vectors)
        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res += self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res += self.value(e_vectors, n_vectors)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res.ravel() / delta / delta

    # @abstractmethod
    def get_parameters(self) -> np.ndarray:
        """Get parameters"""

    # @abstractmethod
    def set_parameters(self, parameters: np.ndarray):
        """Set parameters"""
