from numpy_config import np
import numba as nb

from readers.numerical import rref
from readers.jastrow import construct_a_matrix
from overload import subtract_outer, random_step


labels_type = nb.int64[:]
u_mask_type = nb.boolean[:, :]
chi_mask_type = nb.boolean[:, :]
f_mask_type = nb.boolean[:, :, :, :]
u_parameters_type = nb.float64[:, :]
chi_parameters_type = nb.float64[:, :]
f_parameters_type = nb.float64[:, :, :, :]
u_parameters_optimizable_type = nb.boolean[:, :]
chi_parameters_optimizable_type = nb.boolean[:, :]
f_parameters_optimizable_type = nb.boolean[:, :, :, :]

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('trunc', nb.int64),
    ('u_mask', u_mask_type),
    ('chi_mask', nb.types.ListType(chi_mask_type)),
    ('f_mask', nb.types.ListType(f_mask_type)),
    ('u_parameters', u_parameters_type),
    ('chi_parameters', nb.types.ListType(chi_parameters_type)),
    ('f_parameters', nb.types.ListType(f_parameters_type)),
    ('u_parameters_optimizable', u_parameters_optimizable_type),
    ('chi_parameters_optimizable', nb.types.ListType(chi_parameters_optimizable_type)),
    ('f_parameters_optimizable', nb.types.ListType(f_parameters_optimizable_type)),
    ('u_cutoff', nb.float64),
    ('u_cutoff_optimizable', nb.boolean),
    ('chi_cutoff', nb.float64[:]),
    ('chi_cutoff_optimizable', nb.boolean[:]),
    ('f_cutoff', nb.float64[:]),
    ('f_cutoff_optimizable', nb.boolean[:]),
    ('chi_labels', nb.types.ListType(labels_type)),
    ('f_labels', nb.types.ListType(labels_type)),
    ('max_ee_order', nb.int64),
    ('max_en_order', nb.int64),
    ('chi_cusp', nb.boolean[:]),
    ('no_dup_u_term', nb.boolean[:]),
    ('no_dup_chi_term', nb.boolean[:]),
    ('u_cusp_const', nb.float64[:]),
]


@nb.experimental.jitclass(spec)
class Jastrow:

    def __init__(
        self, neu, ned, trunc, u_parameters, u_parameters_optimizable, u_mask, u_cutoff, u_cusp_const,
        chi_parameters, chi_parameters_optimizable, chi_mask, chi_cutoff, chi_labels, chi_cusp,
        f_parameters, f_parameters_optimizable, f_mask, f_cutoff, f_labels, no_dup_u_term, no_dup_chi_term
    ):
        self.neu = neu
        self.ned = ned
        self.trunc = trunc
        # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        self.u_cutoff = u_cutoff[0]['value']
        self.u_cutoff_optimizable = u_cutoff[0]['optimizable']
        self.u_mask = u_mask
        self.u_parameters = u_parameters
        self.u_parameters_optimizable = u_parameters_optimizable
        # spin dep (0->u=d; 1->u/=d)
        self.chi_labels = chi_labels
        self.chi_cutoff = chi_cutoff['value']
        self.chi_cutoff_optimizable = chi_cutoff['optimizable']
        self.chi_mask = chi_mask
        self.chi_parameters = chi_parameters
        self.chi_parameters_optimizable = chi_parameters_optimizable
        # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        self.f_labels = f_labels
        self.f_cutoff = f_cutoff['value']
        self.f_cutoff_optimizable = f_cutoff['optimizable']
        self.f_mask = f_mask
        self.f_parameters = f_parameters
        self.f_parameters_optimizable = f_parameters_optimizable

        self.max_ee_order = max((
            self.u_parameters.shape[0],
            max([p.shape[2] for p in self.f_parameters]) if self.f_parameters else 0,
        ))
        self.max_en_order = max((
            max([p.shape[0] for p in self.chi_parameters]) if self.chi_parameters else 0,
            max([p.shape[0] for p in self.f_parameters]) if self.f_parameters else 0,
        ))
        self.u_cusp_const = u_cusp_const
        self.chi_cusp = chi_cusp
        self.no_dup_u_term = no_dup_u_term
        self.no_dup_chi_term = no_dup_chi_term
        self.fix_optimizable()

    def fix_optimizable(self):
        """Set parameter fixed if there is no corresponded spin-pairs"""
        if self.u_parameters.shape[1] == 2:
            if self.neu == 1 and self.ned == 1:
                self.u_parameters_optimizable[:, 0] = False
        elif self.u_parameters.shape[1] == 3:
            if self.neu == 1:
                self.u_parameters_optimizable[:, 0] = False
            if self.ned == 1:
                self.u_parameters_optimizable[:, 2] = False

        for chi_parameters, chi_parameters_optimizable in zip(self.chi_parameters, self.chi_parameters_optimizable):
            if chi_parameters.shape[1] == 2:
                if self.neu == 1 and self.ned == 1:
                    chi_parameters_optimizable[:, 0] = False
            elif chi_parameters.shape[1] == 3:
                if self.neu == 1:
                    chi_parameters_optimizable[:, 0] = False
                if self.ned == 1:
                    chi_parameters_optimizable[:, 2] = False

        for f_parameters, f_parameters_optimizable in zip(self.f_parameters, self.f_parameters_optimizable):
            if f_parameters.shape[3] == 2:
                if self.neu == 1 and self.ned == 1:
                    f_parameters_optimizable[:, :, :, 0] = False
            elif f_parameters.shape[3] == 3:
                if self.neu == 1:
                    f_parameters_optimizable[:, :, :, 0] = False
                if self.ned == 1:
                    f_parameters_optimizable[:, :, :, 2] = False

    def ee_powers(self, e_vectors) -> np.ndarray:
        """Powers of e-e distances
        :param e_vectors: e-e vectors - array(nelec, nelec, 3)
        :return: powers of e-e distances - array(nelec, nelec, max_ee_order)
        """
        res = np.ones(shape=(e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(1, self.max_ee_order):
                    res[i, j, k] = res[j, i, k] = r_ee ** k
        return res

    def en_powers(self, n_vectors) -> np.ndarray:
        """Powers of e-n distances
        :param n_vectors: e-n vectors - array(natom, nelec, 3)
        :return: powers of e-n distances - array(natom, nelec, max_en_order)
        """
        res = np.ones(shape=(n_vectors.shape[0], n_vectors.shape[1], self.max_en_order))
        for i in range(n_vectors.shape[0]):
            for j in range(n_vectors.shape[1]):
                r_eI = np.linalg.norm(n_vectors[i, j])
                for k in range(1, self.max_en_order):
                    res[i, j, k] = r_eI ** k
        return res

    def u_term(self, e_powers) -> float:
        """Jastrow u-term
        :param e_powers: powers of e-e distances
        :return:
        """
        res = 0.0
        if not self.u_cutoff:
            return res

        C = self.trunc
        parameters = self.u_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r = e_powers[i, j, 1]
                if r < self.u_cutoff:
                    cusp_set = int(i >= self.neu) + int(j >= self.neu)
                    u_set = cusp_set % parameters.shape[1]
                    poly = 0.0
                    for k in range(parameters.shape[0]):
                        if k == 1:
                            p = self.u_cusp_const[cusp_set]
                        else:
                            p = parameters[k, u_set]
                        poly += p * e_powers[i, j, k]
                    res += poly * (r - self.u_cutoff) ** C
        return res

    def chi_term(self, n_powers) -> float:
        """Jastrow chi-term
        :param n_powers: powers of e-e distances
        :return:
        """
        res = 0.0
        if not self.chi_cutoff.any():
            return res

        C = self.trunc
        for parameters, L, chi_labels in zip(self.chi_parameters, self.chi_cutoff, self.chi_labels):
            for i in chi_labels:
                for j in range(self.neu + self.ned):
                    r = n_powers[i, j, 1]
                    if r < L:
                        chi_set = int(j >= self.neu) % parameters.shape[1]
                        # FIXME: maybe in next numba
                        # from numpy.polynomial.polynomial import polyval, polyval3d
                        # res += polyval(r, parameters[:, chi_set]) * (r - L) ** C
                        poly = 0.0
                        for k in range(parameters.shape[0]):
                            poly += parameters[k, chi_set] * n_powers[i, j, k]
                        res += poly * (r - L) ** C
        return res

    def f_term(self, e_powers, n_powers) -> float:
        """Jastrow f-term
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :return:
        """
        res = 0.0
        if not self.f_cutoff.any():
            return res

        C = self.trunc
        for parameters, L, f_labels in zip(self.f_parameters, self.f_cutoff, self.f_labels):
            for i in f_labels:
                for j in range(1, self.neu + self.ned):
                    for k in range(j):
                        r_e1I = n_powers[i, j, 1]
                        r_e2I = n_powers[i, k, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(j >= self.neu) + int(k >= self.neu)) % parameters.shape[3]
                            poly = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        poly += parameters[l, m, n, f_set] * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n]
                            res += poly * (r_e1I - L) ** C * (r_e2I - L) ** C
        return res

    def u_term_gradient(self, e_powers, e_vectors) -> np.ndarray:
        """Jastrow u-term gradient with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :param e_vectors: e-e vectors
        :return:
        """
        res = np.zeros(shape=(self.neu + self.ned, 3))

        if not self.u_cutoff:
            return res.ravel()

        C = self.trunc
        L = self.u_cutoff
        parameters = self.u_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_vec = e_vectors[i, j]
                r = e_powers[i, j, 1]
                if r < L:
                    cusp_set = (int(i >= self.neu) + int(j >= self.neu))
                    u_set = cusp_set % parameters.shape[1]
                    poly = poly_diff = 0.0
                    for k in range(parameters.shape[0]):
                        if k == 1:
                            p = self.u_cusp_const[cusp_set]
                        else:
                            p = parameters[k, u_set]
                        poly += p * e_powers[i, j, k]
                        if k > 0:
                            poly_diff += p * k * e_powers[i, j, k-1]

                    gradient = r_vec/r * (r-L) ** C * (C/(r-L) * poly + poly_diff)
                    res[i, :] += gradient
                    res[j, :] -= gradient
        return res.ravel()

    def chi_term_gradient(self, n_powers, n_vectors) -> np.ndarray:
        """Jastrow chi-term gradient with respect to a e-coordinates
        :param n_powers: powers of e-n distances
        :param n_vectors: e-n vectors
        :return:
        """
        res = np.zeros(shape=(self.neu + self.ned, 3))

        if not self.chi_cutoff.any():
            return res.ravel()

        C = self.trunc
        for parameters, L, chi_labels in zip(self.chi_parameters, self.chi_cutoff, self.chi_labels):
            for i in chi_labels:
                for j in range(self.neu + self.ned):
                    r_vec = n_vectors[i, j]
                    r = n_powers[i, j, 1]
                    if r < L:
                        chi_set = int(j >= self.neu) % parameters.shape[1]
                        poly = poly_diff = 0.0
                        for k in range(parameters.shape[0]):
                            p = parameters[k, chi_set]
                            poly += p * n_powers[i, j, k]
                            if k > 0:
                                poly_diff += p * k * n_powers[i, j, k-1]

                        res[j, :] += r_vec/r * (r-L) ** C * (C/(r-L) * poly + poly_diff)
        return res.ravel()

    def f_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors) -> np.ndarray:
        """Jastrow f-term gradient with respect to a e-coordinates
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        res = np.zeros(shape=(self.neu + self.ned, 3))

        if not self.f_cutoff.any():
            return res.ravel()

        C = self.trunc
        for parameters, L, f_labels in zip(self.f_parameters, self.f_cutoff, self.f_labels):
            for i in f_labels:
                for j in range(1, self.neu + self.ned):
                    for k in range(j):
                        r_e1I_vec = n_vectors[i, j]
                        r_e2I_vec = n_vectors[i, k]
                        r_ee_vec = e_vectors[j, k]
                        r_e1I = n_powers[i, j, 1]
                        r_e2I = n_powers[i, k, 1]
                        r_ee = e_powers[j, k, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(j >= self.neu) + int(k >= self.neu)) % parameters.shape[3]
                            poly = poly_diff_e1I = poly_diff_e2I = poly_diff_ee = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        p = parameters[l, m, n, f_set]
                                        poly += n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n] * p
                                        if l > 0:
                                            poly_diff_e1I += l * n_powers[i, j, l-1] * n_powers[i, k, m] * e_powers[j, k, n] * p
                                        if m > 0:
                                            poly_diff_e2I += m * n_powers[i, j, l] * n_powers[i, k, m-1] * e_powers[j, k, n] * p
                                        if n > 0:
                                            poly_diff_ee += n * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n-1] * p

                            e1_gradient = r_e1I_vec/r_e1I * (C/(r_e1I - L) * poly + poly_diff_e1I)
                            e2_gradient = r_e2I_vec/r_e2I * (C/(r_e2I - L) * poly + poly_diff_e2I)
                            ee_gradient = r_ee_vec/r_ee * poly_diff_ee
                            res[j, :] += (r_e1I - L) ** C * (r_e2I - L) ** C * (e1_gradient + ee_gradient)
                            res[k, :] += (r_e1I - L) ** C * (r_e2I - L) ** C * (e2_gradient - ee_gradient)
        return res.ravel()

    def u_term_laplacian(self, e_powers) -> float:
        """Jastrow u-term laplacian with respect to e-coordinates
        :param e_powers: powers of e-e distances
        :return:
        """
        res = 0.0
        if not self.u_cutoff:
            return res

        C = self.trunc
        L = self.u_cutoff
        parameters = self.u_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r = e_powers[i, j, 1]
                if r < L:
                    cusp_set = (int(i >= self.neu) + int(j >= self.neu))
                    u_set = cusp_set % parameters.shape[1]
                    poly = poly_diff = poly_diff_2 = 0.0
                    for k in range(parameters.shape[0]):
                        if k == 1:
                            p = self.u_cusp_const[cusp_set]
                        else:
                            p = parameters[k, u_set]
                        poly += p * e_powers[i, j, k]
                        if k > 0:
                            poly_diff += k * p * e_powers[i, j, k-1]
                        if k > 1:
                            poly_diff_2 += k * (k-1) * p * e_powers[i, j, k-2]

                    res += (r-L)**C * (
                        C*(C - 1)/(r-L)**2 * poly + 2 * C/(r-L) * poly_diff + poly_diff_2 +
                        2 * (C/(r-L) * poly + poly_diff) / r
                    )
        return 2 * res

    def chi_term_laplacian(self, n_powers) -> float:
        """Jastrow chi-term laplacian with respect to e-coordinates
        :param n_powers: powers of e-n distances
        :return:
        """
        res = 0.0
        if not self.chi_cutoff.any():
            return res

        C = self.trunc
        for parameters, L, chi_labels in zip(self.chi_parameters, self.chi_cutoff, self.chi_labels):
            for i in chi_labels:
                for j in range(self.neu + self.ned):
                    r = n_powers[i, j, 1]
                    if r < L:
                        chi_set = int(j >= self.neu) % parameters.shape[1]
                        poly = poly_diff = poly_diff_2 = 0.0
                        for k in range(parameters.shape[0]):
                            p = parameters[k, chi_set]
                            poly += p * n_powers[i, j, k]
                            if k > 0:
                                poly_diff += k * p * n_powers[i, j, k-1]
                            if k > 1:
                                poly_diff_2 += k * (k-1) * p * n_powers[i, j, k-2]

                        res += (r-L)**C * (
                            C*(C - 1)/(r-L)**2 * poly + 2 * C/(r-L) * poly_diff + poly_diff_2 +
                            2 * (C/(r-L) * poly + poly_diff) / r
                        )
        return res

    def f_term_laplacian(self, e_powers, n_powers, e_vectors, n_vectors) -> float:
        """Jastrow f-term laplacian with respect to e-coordinates
        f-term is a product of two spherically symmetric functions f(r_eI) and g(r_ee) so using
            ∇²(f*g) = ∇²(f)*g + 2*∇(f)*∇(g) + f*∇²(g)
        then Laplace operator of spherically symmetric function (in 3-D space) is
            ∇²(f) = d²f/dr² + 2/r * df/dr
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        res = 0.0
        if not self.f_cutoff.any():
            return res

        C = self.trunc
        for parameters, L, f_labels in zip(self.f_parameters, self.f_cutoff, self.f_labels):
            for i in f_labels:
                for j in range(1, self.neu + self.ned):
                    for k in range(j):
                        r_e1I_vec = n_vectors[i, j]
                        r_e2I_vec = n_vectors[i, k]
                        r_ee_vec = e_vectors[j, k]
                        r_e1I = n_powers[i, j, 1]
                        r_e2I = n_powers[i, k, 1]
                        r_ee = e_powers[j, k, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(j >= self.neu) + int(k >= self.neu)) % parameters.shape[3]
                            poly = poly_diff_e1I = poly_diff_e2I = 0.0
                            poly_diff_ee = poly_diff_e1I_2 = poly_diff_e2I_2 = 0.0
                            poly_diff_ee_2 = poly_diff_e1I_ee = poly_diff_e2I_ee = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        p = parameters[l, m, n, f_set]
                                        poly += n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n] * p
                                        if l > 0:
                                            poly_diff_e1I += l * n_powers[i, j, l-1] * n_powers[i, k, m] * e_powers[j, k, n] * p
                                        if m > 0:
                                            poly_diff_e2I += m * n_powers[i, j, l] * n_powers[i, k, m-1] * e_powers[j, k, n] * p
                                        if n > 0:
                                            poly_diff_ee += n * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n-1] * p
                                        if l > 1:
                                            poly_diff_e1I_2 += l * (l-1) * n_powers[i, j, l-2] * n_powers[i, k, m] * e_powers[j, k, n] * p
                                        if m > 1:
                                            poly_diff_e2I_2 += m * (m-1) * n_powers[i, j, l] * n_powers[i, k, m-2] * e_powers[j, k, n] * p
                                        if n > 1:
                                            poly_diff_ee_2 += n * (n-1) * n_powers[i, j, l] * n_powers[i, k, m] * e_powers[j, k, n-2] * p
                                        if l > 0 and n > 0:
                                            poly_diff_e1I_ee += l * n * n_powers[i, j, l-1] * n_powers[i, k, m] * e_powers[j, k, n-1] * p
                                        if m > 0 and n > 0:
                                            poly_diff_e2I_ee += m * n * n_powers[i, j, l] * n_powers[i, k, m-1] * e_powers[j, k, n-1] * p

                            diff_1 = (
                                (C/(r_e1I - L) * poly + poly_diff_e1I) / r_e1I +
                                (C/(r_e2I - L) * poly + poly_diff_e2I) / r_e2I +
                                2 * poly_diff_ee / r_ee
                            )
                            diff_2 = (
                                C * (C - 1) / (r_e1I - L) ** 2 * poly +
                                C * (C - 1) / (r_e2I - L) ** 2 * poly +
                                (poly_diff_e1I_2 + poly_diff_e2I_2 + 2 * poly_diff_ee_2) +
                                2 * C/(r_e1I - L) * poly_diff_e1I +
                                2 * C/(r_e2I - L) * poly_diff_e2I
                            )
                            dot_product = (
                                np.sum(r_e1I_vec * r_ee_vec) * (C/(r_e1I - L) * poly_diff_ee + poly_diff_e1I_ee) / r_e1I / r_ee -
                                np.sum(r_e2I_vec * r_ee_vec) * (C/(r_e2I - L) * poly_diff_ee + poly_diff_e2I_ee) / r_e2I / r_ee
                            )
                            res += (r_e1I - L) ** C * (r_e2I - L) ** C * (diff_2 + 2 * diff_1 + 2 * dot_product)
        return res

    def value(self, e_vectors, n_vectors) -> float:
        """Jastrow with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param neu: number of up electrons
        :return:
        """

        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term(e_powers) +
            self.chi_term(n_powers) +
            self.f_term(e_powers, n_powers)
        )

    def gradient(self, e_vectors, n_vectors) -> np.ndarray:
        """Gradient with respect to e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term_gradient(e_powers, e_vectors) +
            self.chi_term_gradient(n_powers, n_vectors) +
            self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
        )

    def laplacian(self, e_vectors, n_vectors) -> float:
        """Laplacian with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term_laplacian(e_powers) +
            self.chi_term_laplacian(n_powers) +
            self.f_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)
        )

    def numerical_gradient(self, e_vectors, n_vectors) -> np.ndarray:
        """Numerical gradient with respect to an e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        delta = 0.00001

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

    def numerical_laplacian(self, e_vectors, n_vectors) -> float:
        """Numerical laplacian with respect to an e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        delta = 0.00001

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

    def fix_u_parameters(self):
        """Fix u-term parameters"""
        C = self.trunc
        L = self.u_cutoff
        for i in range(3):
            self.u_cusp_const[i] = 1 / np.array([4, 2, 4])[i] / (-L) ** C + self.u_parameters[0, i % self.u_parameters.shape[1]] * C / L

    def fix_chi_parameters(self):
        """Fix chi-term parameters"""
        C = self.trunc
        for chi_parameters, L, chi_cusp in zip(self.chi_parameters, self.chi_cutoff, self.chi_cusp):
            chi_parameters[1] = chi_parameters[0] * C / L
            if chi_cusp:
                pass
                # FIXME: chi cusp not implemented
                # chi_parameters[1] -= charge / (-L) ** C

    def fix_f_parameters(self):
        """To find the dependent coefficients of f-term it is necessary to solve
        the system of linear equations:  A*x=b
        A-matrix has the following rows:
        (2 * f_en_order + 1) constraints imposed to satisfy electron–electron no-cusp condition.
        (f_en_order + f_ee_order + 1) constraints imposed to satisfy electron–nucleus no-cusp condition.
        (f_ee_order + 1) constraints imposed to prevent duplication of u-term
        (f_en_order + 1) constraints imposed to prevent duplication of chi-term
        b-column has the sum of independent coefficients for each condition.
        """
        for f_parameters, L, no_dup_u_term, no_dup_chi_term in zip(self.f_parameters, self.f_cutoff, self.no_dup_u_term, self.no_dup_chi_term):
            f_en_order = f_parameters.shape[0] - 1
            f_ee_order = f_parameters.shape[2] - 1
            f_spin_dep = f_parameters.shape[3] - 1

            a = construct_a_matrix(self.trunc, f_en_order, f_ee_order, L, no_dup_u_term, no_dup_chi_term)
            a, pivot_positions = rref(a)
            # remove zero-rows
            a = a[:pivot_positions.size, :]
            b = np.zeros(shape=(f_spin_dep+1, a.shape[0]))
            p = 0
            for n in range(f_ee_order + 1):
                for m in range(f_en_order + 1):
                    for l in range(m, f_en_order + 1):
                        if p not in pivot_positions:
                            for temp in range(pivot_positions.size):
                                b[:, temp] -= a[temp, p] * f_parameters[l, m, n, :]
                        p += 1

            x = np.empty(shape=(f_spin_dep + 1, pivot_positions.size))
            for i in range(f_spin_dep + 1):
                x[i, :] = np.linalg.solve(a[:, pivot_positions], b[i])

            p = 0
            temp = 0
            for n in range(f_ee_order + 1):
                for m in range(f_en_order + 1):
                    for l in range(m, f_en_order + 1):
                        if temp in pivot_positions:
                            f_parameters[l, m, n, :] = f_parameters[m, l, n, :] = x[:, p]
                            p += 1
                        temp += 1

    def get_parameters_bounds(self):
        """Bonds constraints fluctuation of Jastrow parameters
        and thus increases robustness of the energy minimization procedure.
        :return:
        """
        scale = np.concatenate(self.get_parameters_scale())
        parameters = self.get_parameters()
        return parameters - scale, parameters + scale

    def get_parameters_scale(self):
        """Characteristic scale of each variable. Setting x_scale is equivalent
        to reformulating the problem in scaled variables xs = x / x_scale.
        An alternative view is that the size of a trust region along j-th
        dimension is proportional to x_scale[j].
        The purpose of this method is to reformulate the optimization problem
        with dimensionless variables having only one dimensional parameter - scale.
        """
        scale_u = []
        if self.u_cutoff:
            if self.u_cutoff_optimizable:
                scale_u.append(self.u_cutoff)
            for j1 in range(self.u_parameters.shape[0]):
                for j2 in range(self.u_parameters.shape[1]):
                    if self.u_mask[j1, j2] and self.u_parameters_optimizable[j1, j2]:
                        scale_u.append(1 / self.u_cutoff ** (j1 + self.trunc - 1))

        scale_chi = []
        if self.chi_cutoff.any():
            for chi_parameters, chi_parameters_optimizable, chi_mask, chi_cutoff, chi_cutoff_optimizable in zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_mask, self.chi_cutoff, self.chi_cutoff_optimizable):
                if chi_cutoff_optimizable:
                    scale_chi.append(chi_cutoff)
                for j1 in range(chi_parameters.shape[0]):
                    for j2 in range(chi_parameters.shape[1]):
                        if chi_mask[j1, j2] and chi_parameters_optimizable[j1, j2]:
                            scale_chi.append(1 / chi_cutoff ** (j1 + self.trunc - 1))

        scale_f = []
        if self.f_cutoff.any():
            for f_parameters, f_parameters_optimizable, f_mask, f_cutoff, f_cutoff_optimizable in zip(self.f_parameters, self.f_parameters_optimizable, self.f_mask, self.f_cutoff, self.f_cutoff_optimizable):
                if f_cutoff_optimizable:
                    scale_f.append(f_cutoff)
                for j1 in range(f_parameters.shape[0]):
                    for j2 in range(f_parameters.shape[1]):
                        for j3 in range(f_parameters.shape[2]):
                            for j4 in range(f_parameters.shape[3]):
                                if f_mask[j1, j2, j3, j4] and f_parameters_optimizable[j1, j2, j3, j4]:
                                    scale_f.append(1 / f_cutoff ** (j1 + j2 + j3 + 2 * self.trunc - 1))

        return np.array(scale_u), np.array(scale_chi), np.array(scale_f)

    def get_parameters(self):
        """Returns parameters in the following order:
        u-cutoff, u-linear parameters,
        for every chi-set: chi-cutoff, chi-linear parameters,
        for every f-set: f-cutoff, f-linear parameters.
        :return:
        """
        res = []
        if self.u_cutoff:
            if self.u_cutoff_optimizable:
                res.append(self.u_cutoff)
            res += list(self.u_parameters.ravel()[(self.u_mask & self.u_parameters_optimizable).ravel()])

        if self.chi_cutoff.any():
            for chi_parameters, chi_parameters_optimizable, chi_mask, chi_cutoff, chi_cutoff_optimizable in zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_mask, self.chi_cutoff, self.chi_cutoff_optimizable):
                if chi_cutoff_optimizable:
                    res.append(chi_cutoff)
                res += list(chi_parameters.ravel()[(chi_mask & chi_parameters_optimizable).ravel()])

        if self.f_cutoff.any():
            for f_parameters, f_parameters_optimizable, f_mask, f_cutoff, f_cutoff_optimizable in zip(self.f_parameters, self.f_parameters_optimizable, self.f_mask, self.f_cutoff, self.f_cutoff_optimizable):
                if f_cutoff_optimizable:
                    res.append(f_cutoff)
                res += list(f_parameters.ravel()[(f_mask & f_parameters_optimizable).ravel()])

        return np.array(res)

    def set_parameters(self, parameters):
        """Set parameters in the following order:
        u-cutoff, u-linear parameters,
        for every chi-set: chi-cutoff, chi-linear parameters,
        for every f-set: f-cutoff, f-linear parameters.
        :param parameters:
        :return:
        """
        n = 0
        if self.u_cutoff:
            if self.u_cutoff_optimizable:
                self.u_cutoff = parameters[n]
                n += 1
            for j1 in range(self.u_parameters.shape[0]):
                for j2 in range(self.u_parameters.shape[1]):
                    if self.u_mask[j1, j2] and self.u_parameters_optimizable[j1, j2]:
                        self.u_parameters[j1, j2] = parameters[n]
                        n += 1
            self.fix_u_parameters()

        if self.chi_cutoff.any():
            for i, (chi_parameters, chi_parameters_optimizable, chi_mask, chi_cutoff_optimizable) in enumerate(zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_mask, self.chi_cutoff_optimizable)):
                if chi_cutoff_optimizable:
                    # Sequence type is a pointer, but numeric type is not.
                    self.chi_cutoff[i] = parameters[n]
                    n += 1
                for j1 in range(chi_parameters.shape[0]):
                    for j2 in range(chi_parameters.shape[1]):
                        if chi_mask[j1, j2] and chi_parameters_optimizable[j1, j2]:
                            chi_parameters[j1, j2] = parameters[n]
                            n += 1
            self.fix_chi_parameters()

        if self.f_cutoff.any():
            for i, (f_parameters, f_parameters_optimizable, f_mask, f_cutoff_optimizable) in enumerate(zip(self.f_parameters, self.f_parameters_optimizable, self.f_mask, self.f_cutoff_optimizable)):
                if f_cutoff_optimizable:
                    # Sequence types is a pointer, but numeric types is not.
                    self.f_cutoff[i] = parameters[n]
                    n += 1
                for j1 in range(f_parameters.shape[0]):
                    for j2 in range(f_parameters.shape[1]):
                        for j3 in range(f_parameters.shape[2]):
                            for j4 in range(f_parameters.shape[3]):
                                if f_mask[j1, j2, j3, j4] and f_parameters_optimizable[j1, j2, j3, j4]:
                                    f_parameters[j1, j2, j3, j4] = f_parameters[j2, j1, j3, j4] = parameters[n]
                                    n += 1
            self.fix_f_parameters()

        return parameters[n:]

    def u_term_numerical_d1(self, e_powers):
        """Numerical first derivatives of logarithm wfn with respect to u-term parameters
        :param e_powers: powers of e-e distances
        """
        if not self.u_cutoff:
            return np.zeros((0,))

        delta = 0.000001
        scale = self.get_parameters_scale()[0]
        size = (self.u_mask & self.u_parameters_optimizable).sum() + self.u_cutoff_optimizable
        res = np.zeros(shape=(size,))

        n = -1
        if self.u_cutoff_optimizable:
            n += 1
            self.u_cutoff -= delta * scale[n]
            self.fix_u_parameters()
            res[n] -= self.u_term(e_powers) / scale[n]
            self.u_cutoff += 2 * delta * scale[n]
            self.fix_u_parameters()
            res[n] += self.u_term(e_powers) / scale[n]
            self.u_cutoff -= delta * scale[n]

        for j1 in range(self.u_parameters.shape[0]):
            for j2 in range(self.u_parameters.shape[1]):
                if self.u_mask[j1, j2] and self.u_parameters_optimizable[j1, j2]:
                    n += 1
                    self.u_parameters[j1, j2] -= delta * scale[n]
                    self.fix_u_parameters()
                    res[n] -= self.u_term(e_powers) / scale[n]
                    self.u_parameters[j1, j2] += 2 * delta * scale[n]
                    self.fix_u_parameters()
                    res[n] += self.u_term(e_powers) / scale[n]
                    self.u_parameters[j1, j2] -= delta * scale[n]

        self.fix_u_parameters()
        return res / delta / 2

    def chi_term_numerical_d1(self, n_powers):
        """Numerical first derivatives of logarithm wfn with respect to chi-term parameters
        :param e_powers: powers of e-e distances
        """
        if not self.chi_cutoff.any():
            return np.zeros((0,))

        delta = 0.000001
        scale = self.get_parameters_scale()[1]
        size = sum([
            (chi_mask & chi_parameters_optimizable).sum() + chi_cutoff_optimizable
            for chi_parameters_optimizable, chi_mask, chi_cutoff_optimizable
            in zip(self.chi_parameters_optimizable, self.chi_mask, self.chi_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size,))

        n = -1
        for i, (chi_parameters, chi_parameters_optimizable, chi_mask) in enumerate(zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_mask)):
            if self.chi_cutoff_optimizable[i]:
                n += 1
                self.chi_cutoff[i] -= delta * scale[n]
                self.fix_chi_parameters()
                res[n] -= self.chi_term(n_powers)
                self.chi_cutoff[i] += 2 * delta * scale[n]
                self.fix_chi_parameters()
                res[n] += self.chi_term(n_powers)
                self.chi_cutoff[i] -= delta * scale[n]

            for j1 in range(chi_parameters.shape[0]):
                for j2 in range(chi_parameters.shape[1]):
                    if chi_mask[j1, j2] and chi_parameters_optimizable[j1, j2]:
                        n += 1
                        chi_parameters[j1, j2] -= delta * scale[n]
                        self.fix_chi_parameters()
                        res[n] -= self.chi_term(n_powers) / scale[n]
                        chi_parameters[j1, j2] += 2 * delta * scale[n]
                        self.fix_chi_parameters()
                        res[n] += self.chi_term(n_powers) / scale[n]
                        chi_parameters[j1, j2] -= delta * scale[n]

        self.fix_chi_parameters()
        return res / delta / 2

    def f_term_numerical_d1(self, e_powers, n_powers):
        """Numerical first derivatives of logarithm wfn with respect to f-term parameters
        :param e_powers: powers of e-e distances
        """
        if not self.f_cutoff.any():
            return np.zeros((0,))

        delta = 0.000001
        scale = self.get_parameters_scale()[2]
        size = sum([
            (f_mask & f_parameters_optimizable).sum() + f_cutoff_optimizable
            for f_parameters_optimizable, f_mask, f_cutoff_optimizable
            in zip(self.f_parameters_optimizable, self.f_mask, self.f_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size,))

        n = -1
        for i, (f_parameters, f_parameters_optimizable, f_mask) in enumerate(zip(self.f_parameters, self.f_parameters_optimizable, self.f_mask)):
            if self.f_cutoff_optimizable[i]:
                n += 1
                self.f_cutoff[i] -= delta * scale[n]
                self.fix_f_parameters()
                res[n] -= self.f_term(e_powers, n_powers) / scale[n]
                self.f_cutoff[i] += 2 * delta * scale[n]
                self.fix_f_parameters()
                res[n] += self.f_term(e_powers, n_powers) / scale[n]
                self.f_cutoff[i] -= delta * scale[n]

            for j1 in range(f_parameters.shape[0]):
                for j2 in range(f_parameters.shape[1]):
                    for j3 in range(f_parameters.shape[2]):
                        for j4 in range(f_parameters.shape[3]):
                            if f_mask[j1, j2, j3, j4] and f_parameters_optimizable[j1, j2, j3, j4]:
                                n += 1
                                f_parameters[j1, j2, j3, j4] -= delta * scale[n]
                                if j1 != j2:
                                    f_parameters[j2, j1, j3, j4] -= delta * scale[n]
                                self.fix_f_parameters()
                                res[n] -= self.f_term(e_powers, n_powers) / scale[n]
                                f_parameters[j1, j2, j3, j4] += 2 * delta * scale[n]
                                if j1 != j2:
                                    f_parameters[j2, j1, j3, j4] += 2 * delta * scale[n]
                                self.fix_f_parameters()
                                res[n] += self.f_term(e_powers, n_powers) / scale[n]
                                f_parameters[j1, j2, j3, j4] -= delta * scale[n]
                                if j1 != j2:
                                    f_parameters[j2, j1, j3, j4] -= delta * scale[n]

        self.fix_f_parameters()
        return res / delta / 2

    def parameters_numerical_d1(self, e_vectors, n_vectors):
        """Numerical first derivatives logarithm Jastrow with respect to the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return np.concatenate((
            self.u_term_numerical_d1(e_powers),
            self.chi_term_numerical_d1(n_powers),
            self.f_term_numerical_d1(e_powers, n_powers),
        ))

    def u_term_numerical_d2(self, e_powers):
        """Numerical second derivatives of logarithm wfn with respect to u-term parameters
        :param e_powers: powers of e-e distances
        """

        delta = 0.000001
        scale = self.get_parameters_scale()[0]
        size = (self.u_mask & self.u_parameters_optimizable).sum() + self.u_cutoff_optimizable
        res = -2 * self.u_term(e_powers) * np.eye(size)
        for i in range(size):
            res[i, i] /= scale[i] * scale[i]

        if self.u_cutoff_optimizable:
            n = m = 0
            # derivatives of cutoff
            self.u_cutoff -= 2 * delta * scale[n]
            self.fix_u_parameters()
            res[n, m] += self.u_term(e_powers) / scale[n] / scale[n]
            self.u_cutoff += 4 * delta * scale[n]
            self.fix_u_parameters()
            res[n, m] += self.u_term(e_powers) / scale[n] / scale[n]
            self.u_cutoff -= 2 * delta * scale[n]
            for i1 in range(self.u_parameters.shape[0]):
                for j1 in range(self.u_parameters.shape[1]):
                    if self.u_mask[i1, j1] and self.u_parameters_optimizable[i1, j1]:
                        # derivatives of cutoff and linear parameters
                        n += 1
                        self.u_parameters[i1, j1] -= delta * scale[n]
                        self.u_cutoff -= delta * scale[m]
                        self.fix_u_parameters()
                        res[n, m] += self.u_term(e_powers) / scale[n] / scale[m]
                        self.u_parameters[i1, j1] += 2 * delta * scale[n]
                        self.fix_u_parameters()
                        res[n, m] -= self.u_term(e_powers) / scale[n] / scale[m]
                        self.u_cutoff += 2 * delta * scale[m]
                        self.fix_u_parameters()
                        res[n, m] += self.u_term(e_powers) / scale[n] / scale[m]
                        self.u_parameters[i1, j1] -= 2 * delta * scale[n]
                        self.fix_u_parameters()
                        res[n, m] -= self.u_term(e_powers) / scale[n] / scale[m]
                        self.u_parameters[i1, j1] += delta * scale[n]
                        self.u_cutoff -= delta * scale[m]
                        res[m, n] = res[n, m]

        n = self.u_cutoff_optimizable - 1
        for i1 in range(self.u_parameters.shape[0]):
            for j1 in range(self.u_parameters.shape[1]):
                if self.u_mask[i1, j1] and self.u_parameters_optimizable[i1, j1]:
                    n += 1
                    m = self.u_cutoff_optimizable - 1
                    for i2 in range(self.u_parameters.shape[0]):
                        for j2 in range(self.u_parameters.shape[1]):
                            if self.u_mask[i2, j2] and self.u_parameters_optimizable[i2, j2]:
                                m += 1
                                # diagonal derivatives of linear parameters
                                if n == m:
                                    self.u_parameters[i1, j1] -= 2 * delta * scale[n]
                                    self.fix_u_parameters()
                                    res[n, m] += self.u_term(e_powers) / scale[n] / scale[m]
                                    self.u_parameters[i1, j1] += 4 * delta * scale[n]
                                    self.fix_u_parameters()
                                    res[n, m] += self.u_term(e_powers) / scale[n] / scale[m]
                                    self.u_parameters[i1, j1] -= 2 * delta * scale[n]
                                # off-diagonal derivatives of linear parameters
                                elif n > m:
                                    self.u_parameters[i1, j1] -= delta * scale[n]
                                    self.u_parameters[i2, j2] -= delta * scale[m]
                                    self.fix_u_parameters()
                                    res[n, m] += self.u_term(e_powers) / scale[n] / scale[m]
                                    self.u_parameters[i1, j1] += 2 * delta * scale[n]
                                    self.fix_u_parameters()
                                    res[n, m] -= self.u_term(e_powers) / scale[n] / scale[m]
                                    self.u_parameters[i2, j2] += 2 * delta * scale[m]
                                    self.fix_u_parameters()
                                    res[n, m] += self.u_term(e_powers) / scale[n] / scale[m]
                                    self.u_parameters[i1, j1] -= 2 * delta * scale[n]
                                    self.fix_u_parameters()
                                    res[n, m] -= self.u_term(e_powers) / scale[n] / scale[m]
                                    self.u_parameters[i1, j1] += delta * scale[n]
                                    self.u_parameters[i2, j2] -= delta * scale[m]
                                    res[m, n] = res[n, m]

        self.fix_u_parameters()
        return res / delta / delta / 4

    def chi_term_numerical_d2(self, n_powers):
        """Numerical second derivatives of logarithm wfn with respect to chi-term parameters
        :param n_powers: powers of e-n distances
        """
        if not self.chi_cutoff.any():
            return np.zeros((0, 0))

        delta = 0.000001
        scale = self.get_parameters_scale()[1]
        size = sum([
            (chi_mask & chi_parameters_optimizable).sum() + chi_cutoff_optimizable
            for chi_parameters_optimizable, chi_mask, chi_cutoff_optimizable
            in zip(self.chi_parameters_optimizable, self.chi_mask, self.chi_cutoff_optimizable)
        ])
        res = -2 * self.chi_term(n_powers) * np.eye(size)
        for i in range(size):
            res[i, i] /= scale[i] * scale[i]

        for i, (chi_parameters, chi_parameters_optimizable, chi_mask) in enumerate(zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_mask)):
            if self.chi_cutoff_optimizable[i]:
                n = m = 0
                # derivatives of cutoff
                self.chi_cutoff[i] -= 2 * delta * scale[n]
                self.fix_chi_parameters()
                res[n, m] += self.chi_term(n_powers) / scale[n]
                self.chi_cutoff[i] += 4 * delta * scale[n]
                self.fix_chi_parameters()
                res[n, m] += self.chi_term(n_powers) / scale[n]
                self.chi_cutoff[i] -= 2 * delta * scale[n]
                for i1 in range(chi_parameters.shape[0]):
                    for j1 in range(chi_parameters.shape[1]):
                        if chi_mask[i1, j1] and chi_parameters_optimizable[i1, j1]:
                            # derivatives of cutoff and linear parameters
                            n += 1
                            chi_parameters[i1, j1] -= delta * scale[n]
                            self.chi_cutoff[i] -= delta * scale[m]
                            self.fix_chi_parameters()
                            res[n, m] += self.chi_term(n_powers) / scale[n] / scale[m]
                            chi_parameters[i1, j1] += 2 * delta * scale[n]
                            self.fix_chi_parameters()
                            res[n, m] -= self.chi_term(n_powers) / scale[n] / scale[m]
                            self.chi_cutoff[i] += 2 * delta * scale[m]
                            self.fix_chi_parameters()
                            res[n, m] += self.chi_term(n_powers) / scale[n] / scale[m]
                            chi_parameters[i1, j1] -= 2 * delta * scale[n]
                            self.fix_chi_parameters()
                            res[n, m] -= self.chi_term(n_powers) / scale[n] / scale[m]
                            chi_parameters[i1, j1] += delta * scale[n]
                            self.chi_cutoff[i] -= delta * scale[m]
                            res[m, n] = res[n, m]

            n = self.chi_cutoff_optimizable[i] - 1
            for i1 in range(chi_parameters.shape[0]):
                for j1 in range(chi_parameters.shape[1]):
                    if chi_mask[i1, j1] and chi_parameters_optimizable[i1, j1]:
                        n += 1
                        m = self.chi_cutoff_optimizable[i] - 1
                        for i2 in range(chi_parameters.shape[0]):
                            for j2 in range(chi_parameters.shape[1]):
                                if chi_mask[i2, j2] and chi_parameters_optimizable[i2, j2]:
                                    m += 1
                                    # diagonal derivatives of linear parameters
                                    if n == m:
                                        chi_parameters[i1, j1] -= 2 * delta * scale[n]
                                        self.fix_chi_parameters()
                                        res[n, m] += self.chi_term(n_powers) / scale[n] / scale[n]
                                        chi_parameters[i1, j1] += 4 * delta * scale[n]
                                        self.fix_chi_parameters()
                                        res[n, m] += self.chi_term(n_powers) / scale[n] / scale[n]
                                        chi_parameters[i1, j1] -= 2 * delta * scale[n]
                                    # off-diagonal derivatives of linear parameters
                                    elif n > m:
                                        chi_parameters[i1, j1] -= delta * scale[n]
                                        chi_parameters[i2, j2] -= delta * scale[m]
                                        self.fix_chi_parameters()
                                        res[n, m] += self.chi_term(n_powers) / scale[n] / scale[m]
                                        chi_parameters[i1, j1] += 2 * delta * scale[n]
                                        self.fix_chi_parameters()
                                        res[n, m] -= self.chi_term(n_powers) / scale[n] / scale[m]
                                        chi_parameters[i2, j2] += 2 * delta * scale[m]
                                        self.fix_chi_parameters()
                                        res[n, m] += self.chi_term(n_powers) / scale[n] / scale[m]
                                        chi_parameters[i1, j1] -= 2 * delta * scale[n]
                                        self.fix_chi_parameters()
                                        res[n, m] -= self.chi_term(n_powers) / scale[n] / scale[m]
                                        chi_parameters[i1, j1] += delta * scale[n]
                                        chi_parameters[i2, j2] -= delta * scale[m]
                                        res[m, n] = res[n, m]

        self.fix_chi_parameters()
        return res / delta / delta / 4

    def f_term_numerical_d2(self, e_powers, n_powers):
        """Numerical second derivatives of logarithm wfn with respect to f-term parameters
        :param n_powers: powers of e-n distances
        """
        if not self.f_cutoff.any():
            return np.zeros((0, 0))

        delta = 0.000001
        scale = self.get_parameters_scale()[2]
        size = sum([
            (f_mask & f_parameters_optimizable).sum() + f_cutoff_optimizable
            for f_parameters_optimizable, f_mask, f_cutoff_optimizable
            in zip(self.f_parameters_optimizable, self.f_mask, self.f_cutoff_optimizable)
        ])
        res = -2 * self.f_term(e_powers, n_powers) * np.eye(size)
        for i in range(size):
            res[i, i] /= scale[i] * scale[i]

        for i, (f_parameters, f_parameters_optimizable, f_mask) in enumerate(zip(self.f_parameters, self.f_parameters_optimizable, self.f_mask)):
            if self.f_cutoff_optimizable[i]:
                n = m = 0
                # derivatives of cutoff
                self.f_cutoff[i] -= 2 * delta * scale[n]
                self.fix_f_parameters()
                res[n, m] += self.f_term(e_powers, n_powers) / scale[n] / scale[n]
                self.f_cutoff[i] += 4 * delta * scale[n]
                self.fix_f_parameters()
                res[n, m] += self.f_term(e_powers, n_powers) / scale[n] / scale[n]
                self.f_cutoff[i] -= 2 * delta * scale[n]
                for j1 in range(f_parameters.shape[0]):
                    for j2 in range(f_parameters.shape[1]):
                        for j3 in range(f_parameters.shape[2]):
                            for j4 in range(f_parameters.shape[3]):
                                if f_mask[j1, j2, j3, j4] and f_parameters_optimizable[j1, j2, j3, j4]:
                                    # derivatives on cutoff and linear parameters
                                    n += 1
                                    f_parameters[j1, j2, j3, j4] -= delta * scale[n]
                                    if j1 != j2:
                                        f_parameters[j2, j1, j3, j4] -= delta * scale[n]
                                    self.f_cutoff[i] -= delta * scale[m]
                                    self.fix_f_parameters()
                                    res[n, m] += self.f_term(e_powers, n_powers) / scale[n] / scale[m]
                                    f_parameters[j1, j2, j3, j4] += 2 * delta * scale[n]
                                    if j1 != j2:
                                        f_parameters[j2, j1, j3, j4] += 2 * delta * scale[n]
                                    self.fix_f_parameters()
                                    res[n, m] -= self.f_term(e_powers, n_powers) / scale[n] / scale[m]
                                    self.f_cutoff[i] += 2 * delta * scale[m]
                                    self.fix_f_parameters()
                                    res[n, m] += self.f_term(e_powers, n_powers) / scale[n] / scale[m]
                                    f_parameters[j1, j2, j3, j4] -= 2 * delta * scale[n]
                                    if j1 != j2:
                                        f_parameters[j2, j1, j3, j4] -= 2 * delta * scale[n]
                                    self.fix_f_parameters()
                                    res[n, m] -= self.f_term(e_powers, n_powers) / scale[n] / scale[m]
                                    f_parameters[j1, j2, j3, j4] += delta * scale[n]
                                    if j1 != j2:
                                        f_parameters[j2, j1, j3, j4] += delta * scale[n]
                                    self.f_cutoff[i] -= delta * scale[m]
                                    res[m, n] = res[n, m]

            n = self.f_cutoff_optimizable[i] - 1
            for i1 in range(f_parameters.shape[0]):
                for j1 in range(f_parameters.shape[1]):
                    for k1 in range(f_parameters.shape[2]):
                        for l1 in range(f_parameters.shape[3]):
                            if f_mask[i1, j1, k1, l1] and f_parameters_optimizable[i1, j1, k1, l1]:
                                n += 1
                                m = self.f_cutoff_optimizable[i] - 1
                                for i2 in range(f_parameters.shape[0]):
                                    for j2 in range(f_parameters.shape[1]):
                                        for k2 in range(f_parameters.shape[2]):
                                            for l2 in range(f_parameters.shape[3]):
                                                if f_mask[i2, j2, k2, l2] and f_parameters_optimizable[i2, j2, k2, l2]:
                                                    m += 1
                                                    # diagonal terms of linear parameters
                                                    if n == m:
                                                        f_parameters[i1, j1, k1, l1] -= 2 * delta * scale[n]
                                                        if i1 != j1:
                                                            f_parameters[j1, i1, k1, l1] -= 2 * delta * scale[n]
                                                        self.fix_f_parameters()
                                                        res[n, m] += self.f_term(e_powers, n_powers) / scale[n] / scale[n]
                                                        f_parameters[i1, j1, k1, l1] += 4 * delta * scale[n]
                                                        if i1 != j1:
                                                            f_parameters[j1, i1, k1, l1] += 4 * delta * scale[n]
                                                        self.fix_f_parameters()
                                                        res[n, m] += self.f_term(e_powers, n_powers) / scale[n] / scale[n]
                                                        f_parameters[i1, j1, k1, l1] -= 2 * delta * scale[n]
                                                        if i1 != j1:
                                                            f_parameters[j1, i1, k1, l1] -= 2 * delta * scale[n]
                                                    # off-diagonal derivatives of linear parameters
                                                    elif n > m:
                                                        f_parameters[i1, j1, k1, l1] -= delta * scale[n]
                                                        if i1 != j1:
                                                            f_parameters[j1, i1, k1, l1] -= delta * scale[n]
                                                        f_parameters[i2, j2, k2, l2] -= delta * scale[m]
                                                        if i2 != j2:
                                                            f_parameters[j2, i2, k2, l2] -= delta * scale[m]
                                                        self.fix_f_parameters()
                                                        res[n, m] += self.f_term(e_powers, n_powers) / scale[n] / scale[m]
                                                        f_parameters[i1, j1, k1, l1] += 2 * delta * scale[n]
                                                        if i1 != j1:
                                                            f_parameters[j1, i1, k1, l1] += 2 * delta * scale[n]
                                                        self.fix_f_parameters()
                                                        res[n, m] -= self.f_term(e_powers, n_powers) / scale[n] / scale[m]
                                                        f_parameters[i2, j2, k2, l2] += 2 * delta * scale[m]
                                                        if i2 != j2:
                                                            f_parameters[j2, i2, k2, l2] += 2 * delta * scale[m]
                                                        self.fix_f_parameters()
                                                        res[n, m] += self.f_term(e_powers, n_powers) / scale[n] / scale[m]
                                                        f_parameters[i1, j1, k1, l1] -= 2 * delta * scale[n]
                                                        if i1 != j1:
                                                            f_parameters[j1, i1, k1, l1] -= 2 * delta * scale[n]
                                                        self.fix_f_parameters()
                                                        res[n, m] -= self.f_term(e_powers, n_powers) / scale[n] / scale[m]
                                                        f_parameters[i1, j1, k1, l1] += delta * scale[n]
                                                        if i1 != j1:
                                                            f_parameters[j1, i1, k1, l1] += delta * scale[n]
                                                        f_parameters[i2, j2, k2, l2] -= delta * scale[m]
                                                        if i2 != j2:
                                                            f_parameters[j2, i2, k2, l2] -= delta * scale[m]
                                                        res[m, n] = res[n, m]

        self.fix_f_parameters()
        return res / delta / delta / 4

    def parameters_numerical_d2(self, e_vectors, n_vectors):
        """Numerical second derivatives with respect to the Jastrow parameters (diagonal terms)
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        u_term = self.u_term_numerical_d2(e_powers)
        chi_term = self.chi_term_numerical_d2(n_powers)
        f_term = self.f_term_numerical_d2(e_powers, n_powers)

        # not supported by numba
        # res = np.block((
        #     (u_term, np.zeros((u_term.shape[0], chi_term.shape[0])), np.zeros((u_term.shape[0], f_term.shape[0]))),
        #     (np.zeros((chi_term.shape[0], u_term.shape[0])), chi_term, np.zeros((chi_term.shape[0], f_term.shape[0]))),
        #     (np.zeros((f_term.shape[0], u_term.shape[0])), np.zeros((f_term.shape[0], chi_term.shape[0])), f_term)
        # ))
        b = np.cumsum(np.array([0, u_term.shape[0], chi_term.shape[0], f_term.shape[0]]))
        res = np.zeros((b[3], b[3]))
        res[b[0]:b[1], b[0]:b[1]] = u_term
        res[b[1]:b[2], b[1]:b[2]] = chi_term
        res[b[2]:b[3], b[2]:b[3]] = f_term
        return res

    def profile_value(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = subtract_outer(r_e, r_e)
            n_vectors = subtract_outer(atom_positions, r_e)
            self.value(e_vectors, n_vectors)

    def profile_gradient(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = subtract_outer(r_e, r_e)
            n_vectors = subtract_outer(atom_positions, r_e)
            self.gradient(e_vectors, n_vectors)

    def profile_laplacian(self, dr, steps, atom_positions, r_initial) -> None:
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = subtract_outer(r_e, r_e)
            n_vectors = subtract_outer(atom_positions, r_e)
            self.laplacian(e_vectors, n_vectors)
