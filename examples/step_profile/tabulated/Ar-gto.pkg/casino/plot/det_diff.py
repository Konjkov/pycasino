#!/usr/bin/env python3

import numpy as np
from scipy.special import factorial


def radial_part(n, zeta, r):
    """Вычисляет радиальную часть Slater-type orbital (STO) для заданных n, Z и r.

    Параметры:
        n (int): Главное квантовое число.
        Z (float): Заряд ядра (для одноэлектронной системы Z_eff = Z).
        r (float или np.ndarray): Радиальное расстояние (может быть массивом).

    Возвращает:
        float или np.ndarray: Значение радиальной части R_{nl}(r).
    """
    # Нормировочный множитель
    norm_factor = np.sqrt((2*zeta)**(2*n + 1)/np.sqrt(factorial(2 * n)))
    # Радиальная часть STO
    r = np.linalg.norm(r)
    return norm_factor * r**(n - 1) * np.exp(-zeta * r)


def orbitals(zeta):
    """Slater orbitals."""
    return [
        lambda r: radial_part(1, zeta, r),                      # 1s    (0)
        lambda r: radial_part(2, zeta, r),                      # 2s    (2)
        lambda r: r[0] * radial_part(2, zeta, r),               # 2p_x  (3)
        lambda r: r[1] * radial_part(2, zeta, r),               # 2p_y  (4)
        lambda r: r[2] * radial_part(2, zeta, r),               # 2p_z  (5)
        lambda r: radial_part(3, zeta, r),                      # 3s    (7)
        lambda r: r[0] * radial_part(3, zeta, r),               # 3p_x
        lambda r: r[1] * radial_part(3, zeta, r),               # 3p_y
        lambda r: r[2] * radial_part(3, zeta, r),               # 3p_z
        lambda r: r[0]*r[1] * radial_part(3, zeta, r),          # 3d_xy
        lambda r: r[0]*r[2] * radial_part(3, zeta, r),          # 3d_xz
        lambda r: r[1]*r[2] * radial_part(3, zeta, r),          # 3d_yz
        lambda r: (r[0]**2-r[1]**2) * radial_part(3, zeta, r),  # 3d_x²-y²
        lambda r: (3*r[2]**2-r @ r) * radial_part(3, zeta, r),  # 3d_3z²-r²
        lambda r: radial_part(4, zeta, r),                      # 4s
        lambda r: r[0] * radial_part(4, zeta, r),               # 4p_x
        lambda r: r[1] * radial_part(4, zeta, r),               # 4p_y
        lambda r: r[2] * radial_part(4, zeta, r),               # 4p_z
        lambda r: r[0] * r[1] * radial_part(4, zeta, r),        # 4d_xy
        lambda r: r[0] * r[2] * radial_part(4, zeta, r),        # 4d_xz
        lambda r: r[1] * r[2] * radial_part(4, zeta, r),        # 4d_yz
        lambda r: (r[0]**2-r[1]**2) * radial_part(4, zeta, r),  # 4d_x²-y²
        lambda r: (3*r[2]**2-r @ r) * radial_part(4, zeta, r),  # 4d_3z²-r²
    ][:zeta]


class SlaterTaylor:
    def __init__(self, orbitals, h=1e-7, tol=1e-10, max_order=9):
        """
        :param orbitals: list of single-particle functions phi(r) accepting (3,) array.
        :param h: step for finite differences
        :param tol: tolerance for first non-zero derivative
        :param max_order: max Taylor expansion order to check
        """
        self.orbitals = orbitals
        self.n_elec = len(orbitals)
        self.h = h
        self.tol = tol
        self.max_order = max_order
        self.det_calls = 0
        self.cache = {}  # (electron_index, (dx,dy,dz)) -> np.array([...])
        self.points = []  # список всех сгенерированных точек

    def _generate_shifted_points(self, x0, alpha):
        """Генерируем все точки вида x0 ± h для данного многоиндекса alpha."""
        order = alpha.sum()
        axes = []
        for i, a in enumerate(alpha):
            axes.extend([i] * a)
        points = []
        for mask in range(1 << order):
            x = x0.copy()
            for n, axis in enumerate(axes):
                i, j = divmod(axis, 3)
                if (mask >> n) & 1:
                    x[i, j] += self.h
                else:
                    x[i, j] -= self.h
            points.append(x)
        return points

    def _precompute_orbitals(self, points):
        """Вычисляем и кешируем все одноэлектронные орбитали для списка точек."""
        for x in points:
            for i, r_i in enumerate(x):
                key = (i, tuple(r_i.round(12)))  # индекс электрона + координаты
                if key not in self.cache:
                    self.cache[key] = np.array([f(r_i) for f in self.orbitals])

    def det_from_cache(self, x):
        """Строим детерминант из закешированных одноэлектронных орбиталей."""
        self.det_calls += 1
        mat = np.empty((self.n_elec, len(self.orbitals)))
        for i, r_i in enumerate(x):
            key = (i, tuple(r_i.round(12)))
            mat[i, :] = self.cache[key]
        return np.linalg.det(mat)

    @staticmethod
    def multiindex(total, length):
        """Генератор многоиндексов длины `length` с суммой `total`."""
        if length == 1:
            yield np.array([total], dtype=np.int64)
            return
        for i in range(total + 1):
            for rest in SlaterTaylor.multiindex(total - i, length - 1):
                yield np.concatenate(([i], rest))

    def finite_diff(self, alpha, x0):
        """Смешанные частные производные с использованием центральных конечных разностей."""
        result = 0.0
        order = alpha.sum()
        axes = []
        for i, a in enumerate(alpha):
            axes.extend([i] * a)
        points = self._generate_shifted_points(x0, alpha)
        self._precompute_orbitals(points)

        for mask, x in zip(range(1 << order), points):
            coeff = 1.0
            for n in range(order):
                if not ((mask >> n) & 1):
                    coeff *= -1
            result += coeff * self.det_from_cache(x)
        return result / (2 * self.h) ** order

    def first_nonzero_term(self, x0):
        """Находит первый ненулевой член разложения Тейлора."""
        for k in range(1, self.max_order + 1):
            for alpha in self.multiindex(k, 3*self.n_elec):
                # FIXME: учитывает, что все электроны находятся в одной точке
                per_elec = alpha.reshape(self.n_elec, 3).sum(axis=1)
                if np.count_nonzero(per_elec == 0) > 1:
                    continue
                if abs(d := self.finite_diff(alpha, x0)) > self.tol:
                    print(f'Количество вычислений детерминанта для k={k}:', self.det_calls)
                    return alpha, d
            print(f'Количество вычислений детерминанта для k={k}:', self.det_calls)
        return None, None


if __name__ == '__main__':
    n_elec = 5
    x0 = np.zeros((n_elec, 3))
    orbitals = orbitals(n_elec)
    slater = SlaterTaylor(orbitals)
    alpha, coeff = slater.first_nonzero_term(x0)
    print('Многоиндекс:', alpha)
    print('Коэффициент ~', coeff)
