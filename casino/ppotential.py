import numpy as np
import numba as nb


ppotential_spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('vmc_nonlocal_grid', nb.int64),
    ('dmc_nonlocal_grid', nb.int64),
    ('ppotential', nb.float64[:, :]),
    ('quadrature', nb.float64[:, :]),
]


@nb.experimental.jitclass(ppotential_spec)
class PPotential:

    def __init__(self, neu, ned, vmc_nonlocal_grid, dmc_nonlocal_grid, ppotential):
        """Pseudopotential.
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param ppotential: pseudopotential
        """
        self.neu = neu
        self.ned = ned
        self.vmc_nonlocal_grid = vmc_nonlocal_grid or 4
        self.dmc_nonlocal_grid = dmc_nonlocal_grid or 4
        self.ppotential = ppotential
        # From "Nonlocal pseudopotentials and diffusion Monte Carlo" by LuboS Mitas; Eric L. Shirley; David M. Ceperley
        if self.vmc_nonlocal_grid == 1:
            quadrature = [[1.0, 0.0, 0.0]]
        elif self.vmc_nonlocal_grid == 2:
            # Tetrahedron symmetry quadrature.
            q = 1 / np.sqrt(3)
            quadrature = [[q, q, q], [q, -q, -q], [-q, q, -q], [-q, -q, q]]
        elif self.vmc_nonlocal_grid in (3, 5, 6, 7):
            #  Octahedron symmetry quadratures.
            quadrature = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            if self.vmc_nonlocal_grid >= 5:
                p = 1 / np.sqrt(3)
                quadrature += [
                    [p, p, 0.0], [p, 0.0, p], [0.0, p, p], [-p, p, 0.0], [-p, 0.0, p], [0.0, -p, p],
                    [p, -p, 0.0], [p, 0.0, -p], [0.0, p, -p], [-p, -p, 0.0], [-p, 0.0, -p], [0.0, -p, -p]
                ]
            if self.vmc_nonlocal_grid >= 6:
                q = 1 / np.sqrt(3)
                quadrature += [
                    [q, q, q], [q, q, -q], [q, -q, q], [q, -q, -q], [-q, q, q], [-q, q, -q], [-q, -q, q], [-q, -q, -q]
                ]
            if self.vmc_nonlocal_grid == 7:
                r = 1 / np.sqrt(11)
                s = 3 / np.sqrt(11)
                quadrature += [
                    [r, r, s], [r, r, -s], [r, -r, s], [r, -r, -s], [-r, r, s], [-r, r, -s], [-r, -r, s], [-r, -r, -s],
                    [r, s, r], [r, s, -r], [r, -s, r], [r, -s, -r], [-r, s, r], [-r, s, -r], [-r, -s, r], [-r, -s, -r],
                    [s, r, r], [s, r, -r], [s, -r, r], [s, -r, -r], [-s, r, r], [-s, r, -r], [-s, -r, r], [-s, -r, -r]
                ]
        elif self.vmc_nonlocal_grid == 4:
            # Icosahedron symmetry quadratures.
            c1 = np.arctan(2)
            c2 = np.pi - np.arctan(2)
            quadrature = [
                self.to_cartesian(0, 0), self.to_cartesian(0, np.pi),
                self.to_cartesian(c1, 0), self.to_cartesian(c1, 2 * np.pi), self.to_cartesian(c1, 4 * np.pi), self.to_cartesian(c1, 6 * np.pi), self.to_cartesian(c1, 8 * np.pi),
                self.to_cartesian(c2, np.pi), self.to_cartesian(c2, 3 * np.pi), self.to_cartesian(c2, 5 * np.pi), self.to_cartesian(c2, 7 * np.pi), self.to_cartesian(c2, 9 * np.pi),
            ]
        # logger.info(
        #      f'Non-local integration grids\n'
        #      f'===========================\n'
        #      f'Ion type            :  1\n'
        #      f'Non-local grid no.  :  {self.vmc_nonlocal_grid}\n'
        #      f'Lexact              :  {[0, 2, 3, 5, 5, 7, 11][self.vmc_nonlocal_grid-1]}\n'
        #      f'Number of points    :  {len(quadrature)}\n'
        # )
        self.quadrature = np.array(quadrature, dtype=np.float64)

    @staticmethod
    def to_cartesian(theta: float, phi: float) -> list:
        """Converts a spherical coordinate (theta, phi) into a cartesian one (x, y, z)."""
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(theta)
        return [x, y, z]

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
        """Nonlocal PP grid.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        grid = np.zeros(shape=(n_vectors.shape[0], self.neu + self.ned, self.quadrature.shape[0]) + n_vectors.shape)
        for atom in range(n_vectors.shape[0]):
            for i in range(self.neu + self.ned):
                grid[atom, i] = n_vectors
                r = np.linalg.norm(n_vectors[atom, i])
                for q in range(self.quadrature.shape[0]):
                    grid[atom, i, q, atom, i] = self.quadrature[q] * r
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
