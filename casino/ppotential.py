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

    def pseudo_charge(self, n_vectors: np.ndarray) -> np.ndarray:
        """Value φ(r) * r
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        charge = np.zeros(shape=(n_vectors.shape[0], self.neu + self.ned, self.ppotential.shape[0]-1))
        for atom in range(n_vectors.shape[0]):
            for i in range(self.neu + self.ned):
                r = np.linalg.norm(n_vectors[atom, i])
                # self.ppotential[i-1] < r <= self.ppotential[i]
                idx = np.searchsorted(self.ppotential[0], r)
                didx = (r - self.ppotential[0, idx-1]) / (self.ppotential[0, idx] - self.ppotential[0, idx-1])
                charge[atom, i] = self.ppotential[1:, idx-1] + (self.ppotential[1:, idx] - self.ppotential[1:, idx-1]) * didx
        charge[:, :, :2] -= charge[:, :, 2]
        return charge

    def integration_grid(self, n_vectors: np.ndarray) -> np.ndarray:
        """nonlocal PP grid.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        # Tetrahedron symmetry quadrature.
        grid = np.zeros(shape=(n_vectors.shape[0], self.neu + self.ned, 4) + n_vectors.shape)
        quadrature = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        for atom in range(n_vectors.shape[0]):
            for i in range(self.neu + self.ned):
                grid[atom, i] = n_vectors
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
        elif l == 3:
            return (5 * x ** 2 - 3) * x / 2
        elif l == 4:
            return (35 * x ** 4 - 30 * x ** 2 + 3) / 8
