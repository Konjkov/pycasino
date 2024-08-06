import numpy as np
import numba as nb
from abc import ABC, abstractmethod
from casino import delta, delta_2, delta_3
from casino.overload import random_step


class AbstractCusp:

    @nb.njit(nogil=True, parallel=False, cache=True)
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

    @nb.njit(nogil=True, parallel=False, cache=True)
    def numerical_laplacian(self, n_vectors: np.ndarray):
        """Cusp part of laplacian"""
        laplacian = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned))
        value = self.value(n_vectors)
        laplacian[:self.orbitals_up, :self.neu] -= 6 * value[0]
        laplacian[self.orbitals_up:, self.neu:] -= 6 * value[1]
        for j in range(3):
            n_vectors[:, :, j] -= delta_2
            value = self.value(n_vectors)
            laplacian[:self.orbitals_up, :self.neu] += value[0]
            laplacian[self.orbitals_up:, self.neu:] += value[1]
            n_vectors[:, :, j] += 2 * delta_2
            value = self.value(n_vectors)
            laplacian[:self.orbitals_up, :self.neu] += value[0]
            laplacian[self.orbitals_up:, self.neu:] += value[1]
            n_vectors[:, :, j] -= delta_2

        return laplacian[:self.orbitals_up, :self.neu] / delta_2 / delta_2, laplacian[self.orbitals_up:, self.neu:] / delta_2 / delta_2

    @nb.njit(nogil=True, parallel=False, cache=True)
    def numerical_hessian(self, n_vectors: np.ndarray):
        """Cusp part of hessian"""
        hessian = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned, 3, 3))
        for j in range(3):
            value = self.value(n_vectors)
            hessian[:self.orbitals_up, :self.neu, j, j] -= 2 * value[0]
            hessian[self.orbitals_up:, self.neu:, j, j] -= 2 * value[1]
            n_vectors[:, :, j] -= 2 * delta_2
            value = self.value(n_vectors)
            hessian[:self.orbitals_up, :self.neu, j, j] += value[0]
            hessian[self.orbitals_up:, self.neu:, j, j] += value[1]
            n_vectors[:, :, j] += 4 * delta_2
            value = self.value(n_vectors)
            hessian[:self.orbitals_up, :self.neu, j, j] += value[0]
            hessian[self.orbitals_up:, self.neu:, j, j] += value[1]
            n_vectors[:, :, j] -= 2 * delta_2

        for j1 in range(3):
                for j2 in range(3):
                    if j1 >= j2:
                        continue
                    n_vectors[:, :, j1] -= delta_2
                    n_vectors[:, :, j2] -= delta_2
                    value = self.value(n_vectors)
                    hessian[:self.orbitals_up, :self.neu, j1, j2] += value[0]
                    hessian[self.orbitals_up:, self.neu:, j1, j2] += value[1]
                    n_vectors[:, :, j1] += 2 * delta_2
                    value = self.value(n_vectors)
                    hessian[:self.orbitals_up, :self.neu, j1, j2] -= value[0]
                    hessian[self.orbitals_up:, self.neu:, j1, j2] -= value[1]
                    n_vectors[:, :, j2] += 2 * delta_2
                    value = self.value(n_vectors)
                    hessian[:self.orbitals_up, :self.neu, j1, j2] += value[0]
                    hessian[self.orbitals_up:, self.neu:, j1, j2] += value[1]
                    n_vectors[:, :, j1] -= 2 * delta_2
                    value = self.value(n_vectors)
                    hessian[:self.orbitals_up, :self.neu, j1, j2] -= value[0]
                    hessian[self.orbitals_up:, self.neu:, j1, j2] -= value[1]
                    n_vectors[:, :, j1] += delta_2
                    n_vectors[:, :, j2] -= delta_2
                    hessian[:self.orbitals_up, :self.neu, j2, j1] = hessian[:self.orbitals_up, :self.neu, j1, j2]
                    hessian[self.orbitals_up:, self.neu:, j2, j1] = hessian[self.orbitals_up:, self.neu:, j1, j2]

        return hessian[:self.orbitals_up, :self.neu] / delta_2 / delta_2 / 4, hessian[self.orbitals_up:, self.neu:] / delta_2 / delta_2 / 4

    @nb.njit(nogil=True, parallel=False, cache=True)
    def numerical_tressian(self, n_vectors: np.ndarray):
        """Cusp part of tressian"""
        tressian = np.zeros(shape=(self.orbitals_up + self.orbitals_down, self.neu + self.ned, 3, 3, 3))
        for j1 in range(3):
            for j2 in range(3):
                for j3 in range(3):
                    n_vectors[:, :, j1] -= delta_3
                    n_vectors[:, :, j2] -= delta_3
                    n_vectors[:, :, j3] -= delta_3
                    # (-1, -1, -1)
                    value = self.value(n_vectors)
                    tressian[:self.orbitals_up, :self.neu, j1, j2, j3] -= value[0]
                    tressian[self.orbitals_up:, self.neu:, j1, j2, j3] -= value[1]
                    n_vectors[:, :, j1] += 2 * delta_3
                    # ( 1, -1, -1)
                    value = self.value(n_vectors)
                    tressian[:self.orbitals_up, :self.neu, j1, j2, j3] += value[0]
                    tressian[self.orbitals_up:, self.neu:, j1, j2, j3] += value[1]
                    n_vectors[:, :, j2] += 2 * delta_3
                    # ( 1,  1, -1)
                    value = self.value(n_vectors)
                    tressian[:self.orbitals_up, :self.neu, j1, j2, j3] -= value[0]
                    tressian[self.orbitals_up:, self.neu:, j1, j2, j3] -= value[1]
                    n_vectors[:, :, j1] -= 2 * delta_3
                    # (-1,  1, -1)
                    value = self.value(n_vectors)
                    tressian[:self.orbitals_up, :self.neu, j1, j2, j3] += value[0]
                    tressian[self.orbitals_up:, self.neu:, j1, j2, j3] += value[1]
                    n_vectors[:, :, j2] -= 2 * delta_3
                    n_vectors[:, :, j3] += 2 * delta_3
                    # (-1, -1,  1)
                    value = self.value(n_vectors)
                    tressian[:self.orbitals_up, :self.neu, j1, j2, j3] += value[0]
                    tressian[self.orbitals_up:, self.neu:, j1, j2, j3] += value[1]
                    n_vectors[:, :, j1] += 2 * delta_3
                    # ( 1, -1,  1)
                    value = self.value(n_vectors)
                    tressian[:self.orbitals_up, :self.neu, j1, j2, j3] -= value[0]
                    tressian[self.orbitals_up:, self.neu:, j1, j2, j3] -= value[1]
                    n_vectors[:, :, j2] += 2 * delta_3
                    # ( 1,  1,  1)
                    value = self.value(n_vectors)
                    tressian[:self.orbitals_up, :self.neu, j1, j2, j3] += value[0]
                    tressian[self.orbitals_up:, self.neu:, j1, j2, j3] += value[1]
                    n_vectors[:, :, j1] -= 2 * delta_3
                    # (-1,  1,  1)
                    value = self.value(n_vectors)
                    tressian[:self.orbitals_up, :self.neu, j1, j2, j3] -= value[0]
                    tressian[self.orbitals_up:, self.neu:, j1, j2, j3] -= value[1]
                    n_vectors[:, :, j1] += delta_3
                    n_vectors[:, :, j2] -= delta_3
                    n_vectors[:, :, j3] -= delta_3

        return tressian[:self.orbitals_up, :self.neu] / delta_3 / delta_3 / delta_3 / 8, tressian[self.orbitals_up:, self.neu:] / delta_3 / delta_3 / delta_3 / 8

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_value(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.value(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_gradient(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.gradient(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_laplacian(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.laplacian(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_hessian(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.hessian(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_tressian(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.tressian(n_vectors)


class AbstractWfn:

    @nb.njit(nogil=True, parallel=False, cache=True)
    def numerical_gradient(self, r_e):
        """Numerical gradient of log wfn value w.r.t e-coordinates
        :param r_e: electron coordinates - array(nelec, 3)
        """
        val = self.value(r_e)
        res = np.zeros((self.neu + self.ned, 3))
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res[i, j] -= self.value(r_e)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res[i, j] += self.value(r_e)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res.ravel() / delta / 2 / val

    @nb.njit(nogil=True, parallel=False, cache=True)
    def numerical_laplacian(self, r_e):
        """Numerical laplacian  of log wfn value w.r.t e-coordinates
        :param r_e: electron coordinates - array(nelec, 3)
        """
        val = self.value(r_e)
        res = - 6 * (self.neu + self.ned) * self.value(r_e)
        e_vectors, n_vectors = self._relative_coordinates(r_e)
        for i in range(self.neu + self.ned):
            for j in range(3):
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta
                res += self.value(r_e)
                e_vectors[i, :, j] += 2 * delta
                e_vectors[:, i, j] -= 2 * delta
                n_vectors[:, i, j] += 2 * delta
                res += self.value(r_e)
                e_vectors[i, :, j] -= delta
                e_vectors[:, i, j] += delta
                n_vectors[:, i, j] -= delta

        return res / delta / delta / val

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value_parameters_numerical_d1(self, r_e, opt_jastrow, opt_backflow, opt_det_coeff, all_parameters=False):
        """First-order derivatives of log wfn value w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param opt_det_coeff: optimize coefficients of the determinants
        :param all_parameters: optimize all parameters or only independent
        :return:
        """
        scale = self.get_parameters_scale(opt_jastrow, opt_backflow)
        parameters = self.get_parameters(opt_jastrow, opt_backflow, opt_det_coeff, all_parameters)
        res = np.zeros(shape=parameters.shape)
        for i in range(parameters.size):
            parameters[i] -= delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, opt_det_coeff, all_parameters)
            res[i] -= self.value(r_e) / scale[i]
            parameters[i] += 2 * delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, opt_det_coeff, all_parameters)
            res[i] += self.value(r_e) / scale[i]
            parameters[i] -= delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, opt_det_coeff, all_parameters)

        return res / delta / 2 / self.value(r_e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def energy_parameters_numerical_d1(self, r_e, opt_jastrow, opt_backflow, opt_det_coeff, all_parameters=False):
        """First-order derivatives of local energy w.r.t parameters.
        :param r_e: electron coordinates - array(nelec, 3)
        :param opt_jastrow: optimize jastrow parameters
        :param opt_backflow: optimize backflow parameters
        :param opt_det_coeff: optimize coefficients of the determinants
        :param all_parameters: optimize all parameters or only independent
        :return:
        """
        scale = self.get_parameters_scale(opt_jastrow, opt_backflow)
        parameters = self.get_parameters(opt_jastrow, opt_backflow, all_parameters, opt_det_coeff)
        res = np.zeros(shape=parameters.shape)
        for i in range(parameters.size):
            parameters[i] -= delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, all_parameters, opt_det_coeff)
            res[i] -= self.energy(r_e) / scale[i]
            parameters[i] += 2 * delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, all_parameters, opt_det_coeff)
            res[i] += self.energy(r_e) / scale[i]
            parameters[i] -= delta * scale[i]
            self.set_parameters(parameters, opt_jastrow, opt_backflow, all_parameters, opt_det_coeff)

        return res / delta / 2


class AbstractSlater:

    @nb.njit(nogil=True, parallel=False, cache=True)
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

    @nb.njit(nogil=True, parallel=False, cache=True)
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

    @nb.njit(nogil=True, parallel=False, cache=True)
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

    @nb.njit(nogil=True, parallel=False, cache=True)
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

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_value(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.value(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_gradient(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.gradient(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_laplacian(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.laplacian(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_hessian(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.hessian(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_tressian(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.tressian(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_tressian_v2(self, dr, steps: int, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.tressian_v2(n_vectors)


class AbstractJastrow:

    @nb.njit(nogil=True, parallel=False, cache=True)
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

    @nb.njit(nogil=True, parallel=False, cache=True)
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

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value_parameters_numerical_d1(self, e_vectors, n_vectors, all_parameters) -> np.ndarray:
        """Numerical first derivatives of Jastrow value w.r.t. the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param all_parameters: all parameters or only independent
        :return:
        """
        parameters = self.get_parameters(all_parameters)
        res = np.zeros(shape=parameters.shape)
        for i in range(parameters.size):
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)
            res[i] -= self.value(e_vectors, n_vectors)
            parameters[i] += 2 * delta
            self.set_parameters(parameters, all_parameters)
            res[i] += self.value(e_vectors, n_vectors)
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)

        return res / delta / 2

    @nb.njit(nogil=True, parallel=False, cache=True)
    def gradient_parameters_numerical_d1(self, e_vectors, n_vectors, all_parameters) -> np.ndarray:
        """Numerical first derivatives of Jastrow gradient w.r.t. the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param all_parameters: all parameters or only independent
        :return:
        """
        parameters = self.get_parameters(all_parameters)
        res = np.zeros(shape=(parameters.size, (self.neu + self.ned) * 3))
        for i in range(parameters.size):
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)
            res[i] -= self.gradient(e_vectors, n_vectors)
            parameters[i] += 2 * delta
            self.set_parameters(parameters, all_parameters)
            res[i] += self.gradient(e_vectors, n_vectors)
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)

        return res / delta / 2

    @nb.njit(nogil=True, parallel=False, cache=True)
    def laplacian_parameters_numerical_d1(self, e_vectors, n_vectors, all_parameters) -> np.ndarray:
        """Numerical first derivatives of Jastrow laplacian w.r.t. the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param all_parameters: all parameters or only independent
        :return:
        """
        parameters = self.get_parameters(all_parameters)
        res = np.zeros(shape=parameters.shape)
        for i in range(parameters.size):
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)
            res[i] -= self.laplacian(e_vectors, n_vectors)
            parameters[i] += 2 * delta
            self.set_parameters(parameters, all_parameters)
            res[i] += self.laplacian(e_vectors, n_vectors)
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)

        return res / delta / 2

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_value(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.value(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_gradient(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.gradient(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_laplacian(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.laplacian(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_value_parameters_d1(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.value_parameters_d1(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_gradient_parameters_d1(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.gradient_parameters_d1(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_laplacian_parameters_d1(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.laplacian_parameters_d1(e_vectors, n_vectors)


class AbstractBackflow:

    @nb.njit(nogil=True, parallel=False, cache=True)
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

    @nb.njit(nogil=True, parallel=False, cache=True)
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

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value_parameters_numerical_d1(self, e_vectors, n_vectors, all_parameters) -> np.ndarray:
        """Numerical first derivatives of backflow value w.r.t. the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param all_parameters: all parameters or only independent
        :return:
        """
        parameters = self.get_parameters(all_parameters)
        res = np.zeros(shape=(parameters.size, (self.neu + self.ned), 3))
        for i in range(parameters.size):
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)
            res[i] -= self.value(e_vectors, n_vectors)
            parameters[i] += 2 * delta
            self.set_parameters(parameters, all_parameters)
            res[i] += self.value(e_vectors, n_vectors)
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)

        return res.reshape(parameters.size, (self.neu + self.ned) * 3) / delta / 2

    @nb.njit(nogil=True, parallel=False, cache=True)
    def gradient_parameters_numerical_d1(self, e_vectors, n_vectors, all_parameters) -> np.ndarray:
        """Numerical first derivatives of backflow gradient w.r.t. the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param all_parameters: all parameters or only independent
        :return:
        """
        parameters = self.get_parameters(all_parameters)
        res = np.zeros(shape=(parameters.size, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3))
        for i in range(parameters.size):
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)
            res[i] -= self.gradient(e_vectors, n_vectors)[0]
            parameters[i] += 2 * delta
            self.set_parameters(parameters, all_parameters)
            res[i] += self.gradient(e_vectors, n_vectors)[0]
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)

        return res / delta / 2

    @nb.njit(nogil=True, parallel=False, cache=True)
    def laplacian_parameters_numerical_d1(self, e_vectors, n_vectors, all_parameters) -> np.ndarray:
        """Numerical first derivatives of backflow laplacian w.r.t. the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param all_parameters: all parameters or only independent
        :return:
        """
        parameters = self.get_parameters(all_parameters)
        res = np.zeros(shape=(parameters.size, (self.neu + self.ned) * 3))
        for i in range(parameters.size):
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)
            res[i] -= self.laplacian(e_vectors, n_vectors)[0]
            parameters[i] += 2 * delta
            self.set_parameters(parameters, all_parameters)
            res[i] += self.laplacian(e_vectors, n_vectors)[0]
            parameters[i] -= delta
            self.set_parameters(parameters, all_parameters)

        return res / delta / 2

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_value(self, dr, steps, atom_positions, r_initial):
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.value(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_gradient(self, dr, steps, atom_positions, r_initial):
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.gradient(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_laplacian(self, dr, steps, atom_positions, r_initial):
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.laplacian(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_value_parameters_d1(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.value_parameters_d1(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_gradient_parameters_d1(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.gradient_parameters_d1(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def profile_laplacian_parameters_d1(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
            n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
            self.laplacian_parameters_d1(e_vectors, n_vectors)
