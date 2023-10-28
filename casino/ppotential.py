import numpy as np
import numba as nb


ppotential_spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('ppotential', nb.float64[:, :]),
]


@nb.experimental.jitclass(ppotential_spec)
class PPotential:

    def __init__(self, neu, ned, ppotential):
        """
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param ppotential: pseudopotential
        """
        self.neu = neu
        self.ned = ned
        self.ppotential = ppotential

    def pp_value(self, n_vectors: np.ndarray) -> np.ndarray:
        """Value φ(r)
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        value = np.zeros(shape=(n_vectors.shape[0], self.neu + self.ned, self.ppotential.shape[0]-1))
        for atom in range(n_vectors.shape[0]):
            for i in range(self.neu + self.ned):
                r = np.linalg.norm(n_vectors[atom, i])
                # self.ppotential[i-1] < r <= self.ppotential[i]
                idx = np.searchsorted(self.ppotential[0], r)
                value[atom, i] = self.ppotential[1:, idx] / r
        return value

    def grid(self, n_vectors: np.ndarray) -> np.ndarray:
        """nonlocal PP grid.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        # Tetrahedron symmetry quadrature.
        grid = np.zeros(shape=(n_vectors.shape[0], self.neu + self.ned, 4) + n_vectors.shape)
        quadrature = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        for atom in range(n_vectors.shape[0]):
            for i in range(self.neu + self.ned):
                r = np.linalg.norm(n_vectors[atom, i])
                for q in range(4):
                    grid[atom, i, q, atom, i] = quadrature[q] * r / np.sqrt(3)
        return grid

    def legendre_polynomial(self, l, x):
        """legendre polynomial"""
        if l == 0:
            return 1
        if l == 1:
            return x
        elif l == 2:
            return (3 * x ** 2 - 1) / 2
