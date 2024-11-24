import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method


@structref.register
class PPotential_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


ppotential_type = nb.float64[:, ::1]
weight_type = nb.float64[::1]
quadrature_type = nb.float64[:, ::1]


PPotential_t = PPotential_class_t(
    [
        ('neu', nb.int64),
        ('ned', nb.int64),
        ('lcutofftol', nb.float64),
        ('nlcutofftol', nb.float64),
        ('atom_charges', nb.float64[::1]),
        ('vmc_nonlocal_grid', nb.int64[::1]),
        ('dmc_nonlocal_grid', nb.int64[::1]),
        ('local_angular_momentum', nb.int64[::1]),
        ('ppotential', nb.types.ListType(ppotential_type)),
        ('is_pseudoatom', nb.boolean[::1]),
        ('weight', nb.types.ListType(weight_type)),
        ('quadrature', nb.types.ListType(quadrature_type)),
    ]
)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(PPotential_class_t, 'generate_quadratures')
def ppotential_generate_quadratures(self):
    """Formulae from "Nonlocal pseudopotentials and diffusion monte carlo"
    Lubos Mitas, Eric L. Shirley, David M. Ceperley J. Chem. Phys. 95, 3467 (1991).
    """

    def impl(self):
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
                # Octahedron symmetry quadratures.
                weight = [1 / 6] * 6
                quadrature = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
                if vmc_nonlocal_grid >= 5:
                    p = 1 / np.sqrt(3)
                    weight = [1 / 30] * 6 + [1 / 15] * 12
                    quadrature += [
                        [p, p, 0.0], [p, 0.0, p], [0.0, p, p], [-p, p, 0.0], [-p, 0.0, p], [0.0, -p, p],
                        [p, -p, 0.0], [p, 0.0, -p], [0.0, p, -p], [-p, -p, 0.0], [-p, 0.0, -p], [0.0, -p, -p]
                    ]  # fmt: skip
                if vmc_nonlocal_grid >= 6:
                    q = 1 / np.sqrt(3)
                    weight = [1 / 21] * 6 + [4 / 105] * 12 + [27 / 840] * 8
                    quadrature += [
                        [q, q, q], [q, q, -q], [q, -q, q], [q, -q, -q], [-q, q, q], [-q, q, -q], [-q, -q, q], [-q, -q, -q]
                    ]  # fmt: skip
                if vmc_nonlocal_grid == 7:
                    r = 1 / np.sqrt(11)
                    s = 3 / np.sqrt(11)
                    weight = [4 / 315] * 6 + [64 / 2835] * 12 + [27 / 1280] * 8 + [146411 / 725760] * 24
                    quadrature += [
                        [r, r, s], [r, r, -s], [r, -r, s], [r, -r, -s], [-r, r, s], [-r, r, -s], [-r, -r, s], [-r, -r, -s],
                        [r, s, r], [r, s, -r], [r, -s, r], [r, -s, -r], [-r, s, r], [-r, s, -r], [-r, -s, r], [-r, -s, -r],
                        [s, r, r], [s, r, -r], [s, -r, r], [s, -r, -r], [-s, r, r], [-s, r, -r], [-s, -r, r], [-s, -r, -r]
                    ]  # fmt: skip
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
                ]  # fmt: skip
            else:
                # No grid
                weight = [0.0]
                quadrature = [[0.0, 0.0, 0.0]]
            self.weight.append(np.array(weight))
            self.quadrature.append(np.array(quadrature))

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(PPotential_class_t, 'to_cartesian')
def ppotential_to_cartesian(self, theta: float, phi: float):
    """Converts a spherical coordinate (theta, phi) into a cartesian one (x, y, z)."""

    def impl(self, theta: float, phi: float) -> list:
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(theta)
        return [x, y, z]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(PPotential_class_t, 'random_rotation_matrix')
def ppotential_random_rotation_matrix(self):
    """Creates a random rotation matrix.
    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """

    def impl(self):
        # FIXME: use sp.stats.special_ortho_group(3)
        rand = np.random.uniform(0, 1, 3)
        theta = rand[0] * 2.0 * np.pi
        phi = rand[1] * 2.0 * np.pi
        st = np.sin(theta)
        ct = np.cos(theta)
        # simple 2D rotation matrix with a random angle
        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
        # Householder matrix
        r = np.sqrt(rand[2])
        v = np.array((np.sin(phi) * r, np.cos(phi) * r, np.sqrt(1 - rand[2])))
        return (2 * np.outer(v, v) - np.eye(3)) @ R

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(PPotential_class_t, 'get_ppotential')
def ppotential_get_ppotential(self, n_vectors: np.ndarray) -> np.ndarray:
    """Value φ(r)
    :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
    """

    def impl(self, n_vectors: np.ndarray):
        res = nb.typed.List.empty_list(ppotential_type)
        for atom in range(n_vectors.shape[0]):
            if self.is_pseudoatom[atom]:
                atom_pp = self.ppotential[atom]
                ppotential = np.zeros(shape=(self.neu + self.ned, atom_pp.shape[0] - 1))
                for i in range(self.neu + self.ned):
                    r = np.linalg.norm(n_vectors[atom, i])
                    # atom_pp[0, i-1] < r <= atom_pp[0, i]
                    idx = np.searchsorted(atom_pp[0], r)
                    if idx == 1:
                        ppotential[i] = atom_pp[1:, idx] / atom_pp[0, idx]
                    else:
                        di_dx = (r - atom_pp[0, idx - 1]) / (atom_pp[0, idx] - atom_pp[0, idx - 1])
                        ppotential[i] = (atom_pp[1:, idx - 1] + (atom_pp[1:, idx] - atom_pp[1:, idx - 1]) * di_dx) / r
                res.append(ppotential)
            else:
                res.append(np.zeros(shape=(self.neu + self.ned, 0)))
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(PPotential_class_t, 'integration_grid')
def ppotential_integration_grid(self, n_vectors: np.ndarray):
    """Nonlocal PP grid.
    :param n_vectors: electron-nuclei vectors shape = (natom, nelec, 3)
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        grid = np.zeros(shape=(n_vectors.shape[0], self.neu + self.ned, self.quadrature[0].shape[0], 3))
        for atom in range(n_vectors.shape[0]):
            if self.is_pseudoatom[atom]:
                for i in range(self.neu + self.ned):
                    r = np.linalg.norm(n_vectors[atom, i])
                    rotation_marix = self.random_rotation_matrix()
                    grid[atom, i] = self.quadrature[atom] @ rotation_marix.T * r
        return grid

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(PPotential_class_t, 'legendre')
def ppotential_legendre(self, l, x):
    """Legendre polynomial (2 * l + 1) times"""

    def impl(self, l, x):
        if l == 0:
            res = 1
        elif l == 1:
            res = 3 * x
        elif l == 2:
            res = 5 * (3 * x**2 - 1) / 2
        elif l == 3:
            res = 7 * (5 * x**2 - 3) * x / 2
        elif l == 4:
            res = 9 * (35 * x**4 - 30 * x**2 + 3) / 8
        else:
            res = 0
        return res

    return impl


class PPotential(structref.StructRefProxy):
    def __new__(cls, config):
        @nb.njit(nogil=True, parallel=False, cache=True)
        def ppotential_init(
            neu, ned, lcutofftol, nlcutofftol, atom_charges, vmc_nonlocal_grid, dmc_nonlocal_grid, local_angular_momentum, ppotential, is_pseudoatom
        ):
            self = structref.new(PPotential_t)
            self.neu = neu
            self.ned = ned
            self.lcutofftol = lcutofftol
            self.nlcutofftol = nlcutofftol
            self.atom_charges = atom_charges
            self.vmc_nonlocal_grid = vmc_nonlocal_grid
            self.dmc_nonlocal_grid = dmc_nonlocal_grid
            self.local_angular_momentum = local_angular_momentum
            self.ppotential = ppotential
            self.is_pseudoatom = is_pseudoatom
            self.weight = nb.typed.List.empty_list(weight_type)
            self.quadrature = nb.typed.List.empty_list(quadrature_type)
            self.generate_quadratures()
            # make s-d and p-d
            for atom in range(len(self.ppotential)):
                if not self.is_pseudoatom[atom]:
                    continue
                atom_pp = self.ppotential[atom]
                local_angular_momentum = self.local_angular_momentum[atom]
                atom_pp[1] -= atom_pp[local_angular_momentum + 1]
                atom_pp[2] -= atom_pp[local_angular_momentum + 1]
                r_nlcutoff_1 = atom_pp.shape[1] - np.argmax(np.abs(atom_pp[1, ::-1]) > self.nlcutofftol)
                r_nlcutoff_2 = atom_pp.shape[1] - np.argmax(np.abs(atom_pp[2, ::-1]) > self.nlcutofftol)
                atom_pp[1, r_nlcutoff_1:] = 0
                atom_pp[2, r_nlcutoff_2:] = 0
                atom_pp[local_angular_momentum + 1] += atom_charges[atom]
                r_lcutoff = atom_pp.shape[1] - np.argmax(np.abs(atom_pp[3, ::-1]) > self.lcutofftol)
                atom_pp[local_angular_momentum + 1, r_lcutoff:] = 0
            return self

        return ppotential_init(
            config.input.neu,
            config.input.ned,
            config.input.lcutofftol,
            config.input.nlcutofftol,
            config.wfn.atom_charges,
            config.wfn.vmc_nonlocal_grid,
            config.wfn.dmc_nonlocal_grid,
            config.wfn.local_angular_momentum,
            config.wfn.ppotential,
            config.wfn.is_pseudoatom,
        )


structref.define_boxing(PPotential_class_t, PPotential)
