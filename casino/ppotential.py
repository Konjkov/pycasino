import numpy as np
import numba as nb


ppotential_type = nb.float64[:, :]
weight_type = nb.float64[:]
quadrature_type = nb.float64[:, :]

ppotential_spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('vmc_nonlocal_grid', nb.int64[:]),
    ('dmc_nonlocal_grid', nb.int64[:]),
    ('ppotential_list', nb.types.ListType(ppotential_type)),
    ('weight', nb.types.ListType(weight_type)),
    ('quadrature', nb.types.ListType(quadrature_type)),
]


@nb.experimental.jitclass(ppotential_spec)
class PPotential:

    def __init__(self, neu, ned, vmc_nonlocal_grid, dmc_nonlocal_grid, ppotential):
        """Pseudopotential.
        For more details
        https://vallico.net/casinoqmc/pplib/
        https://pseudopotentiallibrary.org/
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param vmc_nonlocal_grid:
        :param dmc_nonlocal_grid:
        :param ppotential: tabulated pseudopotential Casino style
        """
        self.neu = neu
        self.ned = ned
        self.vmc_nonlocal_grid = vmc_nonlocal_grid
        self.dmc_nonlocal_grid = dmc_nonlocal_grid
        self.ppotential_list = ppotential
        self.weight = nb.typed.List.empty_list(weight_type)
        self.quadrature = nb.typed.List.empty_list(quadrature_type)
        self.generate_quadratures()

    def generate_quadratures(self):
        """Formulae from "Nonlocal pseudopotentials and diffusion monte carlo"
        Lubos Mitas, Eric L. Shirley, David M. Ceperley J. Chem. Phys. 95, 3467 (1991).
        """
        for vmc_nonlocal_grid in self.vmc_nonlocal_grid:
            vmc_nonlocal_grid = vmc_nonlocal_grid or 4
            if vmc_nonlocal_grid == 1:
                weight = [1.0]
                quadrature = [[1.0, 0.0, 0.0]]
            elif vmc_nonlocal_grid == 2:
                # Tetrahedron symmetry quadrature.
                q = 1 / np.sqrt(3)
                weight = [1 / 4] * 4
                quadrature = [[q, q, q], [q, -q, -q], [-q, q, -q], [-q, -q, q]]
            elif vmc_nonlocal_grid in (3, 5, 6, 7):
                #  Octahedron symmetry quadratures.
                weight = [1 / 6] * 6
                quadrature = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
                if vmc_nonlocal_grid >= 5:
                    p = 1 / np.sqrt(3)
                    weight = [1 / 30] * 6 + [1 / 15] * 12
                    quadrature += [
                        [p, p, 0.0], [p, 0.0, p], [0.0, p, p], [-p, p, 0.0], [-p, 0.0, p], [0.0, -p, p],
                        [p, -p, 0.0], [p, 0.0, -p], [0.0, p, -p], [-p, -p, 0.0], [-p, 0.0, -p], [0.0, -p, -p]
                    ]
                if vmc_nonlocal_grid >= 6:
                    q = 1 / np.sqrt(3)
                    weight = [1 / 21] * 6 + [4 / 105] * 12 + [27 / 840] * 8
                    quadrature += [
                        [q, q, q], [q, q, -q], [q, -q, q], [q, -q, -q], [-q, q, q], [-q, q, -q], [-q, -q, q], [-q, -q, -q]
                    ]
                if vmc_nonlocal_grid == 7:
                    r = 1 / np.sqrt(11)
                    s = 3 / np.sqrt(11)
                    weight = [4 / 315] * 6 + [64 / 2835] * 12 + [27 / 1280] * 8 + [146411 / 725760] * 24
                    quadrature += [
                        [r, r, s], [r, r, -s], [r, -r, s], [r, -r, -s], [-r, r, s], [-r, r, -s], [-r, -r, s], [-r, -r, -s],
                        [r, s, r], [r, s, -r], [r, -s, r], [r, -s, -r], [-r, s, r], [-r, s, -r], [-r, -s, r], [-r, -s, -r],
                        [s, r, r], [s, r, -r], [s, -r, r], [s, -r, -r], [-s, r, r], [-s, r, -r], [-s, -r, r], [-s, -r, -r]
                    ]
            elif vmc_nonlocal_grid == 4:
                # Icosahedron symmetry quadratures.
                c1 = np.arctan(2)
                c2 = np.pi - np.arctan(2)
                weight = [1 / 12] * 12
                quadrature = [
                    self.to_cartesian(0, 0), self.to_cartesian(np.pi, 0),
                    self.to_cartesian(c1, 0), self.to_cartesian(c2, np.pi/5),
                    self.to_cartesian(c1, 2 * np.pi/5), self.to_cartesian(c2, 3 * np.pi/5),
                    self.to_cartesian(c1, 4 * np.pi/5), self.to_cartesian(c2, 5 * np.pi/5),
                    self.to_cartesian(c1, 6 * np.pi/5), self.to_cartesian(c2, 7 * np.pi/5),
                    self.to_cartesian(c1, 8 * np.pi/5), self.to_cartesian(c2, 9 * np.pi/5)
                ]
            else:
                weight = [0.0]
                quadrature = [[0.0, 0.0, 0.0]]
            # logger.info(
            #      f'Non-local integration grids\n'
            #      f'===========================\n'
            #      f'Ion type            :  1\n'
            #      f'Non-local grid no.  :  {self.vmc_nonlocal_grid}\n'
            #      f'Lexact              :  {[0, 2, 3, 5, 5, 7, 11][self.vmc_nonlocal_grid-1]}\n'
            #      f'Number of points    :  {len(quadrature)}\n'
            # )
            self.weight.append(np.array(weight, dtype=np.float64))
            self.quadrature.append(np.array(quadrature, dtype=np.float64))

    @staticmethod
    def to_cartesian(theta: float, phi: float) -> list:
        """Converts a spherical coordinate (theta, phi) into a cartesian one (x, y, z)."""
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(theta)
        return [x, y, z]

    @staticmethod
    def rotation_marix(vec1, vec2):
        """Find the rotation matrix that aligns vec1 to vec2
        How to create random orthonormal matrix in python numpy
        https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy/55289807
        :param vec1: source vector
        :param vec2: destination vector
        :return mat: vec2 = mat @ vec1
        """
        res = np.eye(3)
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        if v.any():
            c = a @ b
            s = np.linalg.norm(v)
            kmat = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            res += kmat + kmat @ kmat * (1 - c) / (s ** 2)
        return res

    def get_ppotential(self, n_vectors: np.ndarray) -> np.ndarray:
        """Value φ(r)
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        # FIXME: self.pp[0].shape[0]-1) = lmax
        res = nb.typed.List.empty_list(ppotential_type)
        for atom in range(n_vectors.shape[0]):
            atom_pp = self.ppotential_list[atom]
            ppotential = np.zeros(shape=(self.neu + self.ned, atom_pp.shape[0]-1))
            for i in range(self.neu + self.ned):
                r = np.linalg.norm(n_vectors[atom, i])
                # atom_pp[0, i-1] < r <= atom_pp[0, i]
                idx = np.searchsorted(atom_pp[0], r)
                di_dx = (r - atom_pp[0, idx-1]) / (atom_pp[0, idx] - atom_pp[0, idx-1])
                ppotential[i] = (atom_pp[1:, idx-1] + (atom_pp[1:, idx] - atom_pp[1:, idx-1]) * di_dx) / r
            res.append(ppotential)
        return res

    def integration_grid(self, n_vectors: np.ndarray) -> np.ndarray:
        """Nonlocal PP grid.
        :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
        """
        z = np.array([0.0, 0.0, 1.0])
        grid = np.zeros(shape=(n_vectors.shape[0], self.neu + self.ned, self.quadrature[0].shape[0], 3))
        for atom in range(n_vectors.shape[0]):
            for i in range(self.neu + self.ned):
                r = np.linalg.norm(n_vectors[atom, i])
                rotation_marix = self.rotation_marix(z, n_vectors[atom, i])
                grid[atom, i] = self.quadrature[atom] @ rotation_marix.T * r
        return grid

    def legendre(self, l, x):
        """Legendre polynomial (2 * l + 1) times"""
        res = (2 * l + 1)
        if l == 0:
            res *= 1
        elif l == 1:
            res *= x
        elif l == 2:
            res *= (3 * x ** 2 - 1) / 2
        elif l == 3:
            res *= (5 * x ** 2 - 3) * x / 2
        elif l == 4:
            res *= (35 * x ** 4 - 30 * x ** 2 + 3) / 8
        return res
