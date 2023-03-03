from numpy_config import np
import numba as nb

from readers.numerical import rref
from readers.jastrow import construct_a_matrix
from overload import subtract_outer, random_step


labels_type = nb.int64[:]
u_parameters_type = nb.float64[:, :]
chi_parameters_type = nb.float64[:, :]
f_parameters_type = nb.float64[:, :, :, :]
u_parameters_mask_type = nb.boolean[:, :]
chi_parameters_mask_type = nb.boolean[:, :]
f_parameters_mask_type = nb.boolean[:, :, :, :]

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('trunc', nb.int64),
    ('u_parameters', u_parameters_type),
    ('chi_parameters', nb.types.ListType(chi_parameters_type)),
    ('f_parameters', nb.types.ListType(f_parameters_type)),
    ('u_parameters_optimizable', u_parameters_mask_type),
    ('chi_parameters_optimizable', nb.types.ListType(chi_parameters_mask_type)),
    ('f_parameters_optimizable', nb.types.ListType(f_parameters_mask_type)),
    ('u_parameters_available', u_parameters_mask_type),
    ('chi_parameters_available', nb.types.ListType(chi_parameters_mask_type)),
    ('f_parameters_available', nb.types.ListType(f_parameters_mask_type)),
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
        self, neu, ned, trunc, u_parameters, u_parameters_optimizable, u_cutoff, u_cusp_const,
        chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_labels, chi_cusp,
        f_parameters, f_parameters_optimizable, f_cutoff, f_labels, no_dup_u_term, no_dup_chi_term
    ):
        self.neu = neu
        self.ned = ned
        self.trunc = trunc
        # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        self.u_cutoff = u_cutoff[0]['value']
        self.u_cutoff_optimizable = u_cutoff[0]['optimizable']
        self.u_parameters = u_parameters
        self.u_parameters_optimizable = u_parameters_optimizable
        self.u_parameters_available = np.ones_like(u_parameters_optimizable)
        # spin dep (0->u=d; 1->u/=d)
        self.chi_labels = chi_labels
        self.chi_cutoff = chi_cutoff['value']
        self.chi_cutoff_optimizable = chi_cutoff['optimizable']
        self.chi_parameters = chi_parameters
        self.chi_parameters_optimizable = chi_parameters_optimizable
        self.chi_parameters_available = nb.typed.List.empty_list(chi_parameters_mask_type)
        # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        self.f_labels = f_labels
        self.f_cutoff = f_cutoff['value']
        self.f_cutoff_optimizable = f_cutoff['optimizable']
        self.f_parameters = f_parameters
        self.f_parameters_optimizable = f_parameters_optimizable
        self.f_parameters_available = nb.typed.List.empty_list(f_parameters_mask_type)

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

    def check_constraint(self):
        """"""
        for f_parameters, L, no_dup_u_term, no_dup_chi_term in zip(self.f_parameters, self.f_cutoff, self.no_dup_u_term, self.no_dup_chi_term):
            f_en_order = f_parameters.shape[0] - 1
            f_ee_order = f_parameters.shape[2] - 1

            a = construct_a_matrix(self.trunc, f_en_order, f_ee_order, L, no_dup_u_term, no_dup_chi_term)
            for k in range(f_parameters.shape[3]):
                x = []
                for n in range(f_parameters.shape[2]):
                    for m in range(f_parameters.shape[1]):
                        for l in range(m, f_parameters.shape[0]):
                            x.append(f_parameters[l, m, n, k])
                print(a @ np.array(x))

    def fix_optimizable(self):
        """Set parameter fixed if there is no corresponded spin-pairs"""
        ee_order = 2
        if self.u_parameters.shape[1] == 2:
            if self.neu < ee_order and self.ned < ee_order:
                self.u_parameters_available[:, 0] = False
            if self.neu + self.ned < ee_order:
                self.u_parameters_available[:, 1] = False
        elif self.u_parameters.shape[1] == 3:
            if self.neu < ee_order:
                self.u_parameters_available[:, 0] = False
            if self.neu + self.ned < ee_order:
                self.u_parameters_available[:, 1] = False
            if self.ned < ee_order:
                self.u_parameters_available[:, 2] = False

        ee_order = 1
        for chi_parameters_optimizable in self.chi_parameters_optimizable:
            chi_parameters_available = np.ones_like(chi_parameters_optimizable)
            if chi_parameters_optimizable.shape[1] == 2:
                if self.neu < ee_order:
                    chi_parameters_available[:, 0] = False
                if self.ned < ee_order:
                    chi_parameters_available[:, 1] = False
            self.chi_parameters_available.append(chi_parameters_available)

        ee_order = 2
        for f_parameters_optimizable in self.f_parameters_optimizable:
            f_parameters_available = np.ones_like(f_parameters_optimizable)
            for j2 in range(f_parameters_optimizable.shape[1]):
                for j1 in range(f_parameters_optimizable.shape[0]):
                    if j1 < j2:
                        f_parameters_available[j1, j2, :, :] = False
            if f_parameters_optimizable.shape[3] == 2:
                if self.neu < ee_order and self.ned < ee_order:
                    f_parameters_available[:, :, :, 0] = False
                if self.neu + self.ned < ee_order:
                    f_parameters_available[:, :, :, 1] = False
            elif f_parameters_optimizable.shape[3] == 3:
                if self.neu < ee_order:
                    f_parameters_available[:, :, :, 0] = False
                if self.neu + self.ned < ee_order:
                    f_parameters_available[:, :, :, 1] = False
                if self.ned < ee_order:
                    f_parameters_available[:, :, :, 2] = False
            self.f_parameters_available.append(f_parameters_available)

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
        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                r = e_powers[e1, e2, 1]
                if r < self.u_cutoff:
                    cusp_set = int(e1 >= self.neu) + int(e2 >= self.neu)
                    u_set = cusp_set % parameters.shape[1]
                    poly = 0.0
                    for k in range(parameters.shape[0]):
                        if k == 1:
                            p = self.u_cusp_const[cusp_set]
                        else:
                            p = parameters[k, u_set]
                        poly += p * e_powers[e1, e2, k]
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
            for label in chi_labels:
                for e1 in range(self.neu + self.ned):
                    r = n_powers[label, e1, 1]
                    if r < L:
                        chi_set = int(e1 >= self.neu) % parameters.shape[1]
                        # FIXME: maybe in next numba
                        # from numpy.polynomial.polynomial import polyval, polyval3d
                        # res += polyval(r, parameters[:, chi_set]) * (r - L) ** C
                        poly = 0.0
                        for k in range(parameters.shape[0]):
                            poly += parameters[k, chi_set] * n_powers[label, e1, k]
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
            for label in f_labels:
                for e1 in range(1, self.neu + self.ned):
                    for e2 in range(e1):
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % parameters.shape[3]
                            poly = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        poly += parameters[l, m, n, f_set] * n_powers[label, e1, l] * n_powers[label, e2, m] * e_powers[e1, e2, n]
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
        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                r_vec = e_vectors[e1, e2]
                r = e_powers[e1, e2, 1]
                if r < L:
                    cusp_set = (int(e1 >= self.neu) + int(e2 >= self.neu))
                    u_set = cusp_set % parameters.shape[1]
                    poly = poly_diff = 0.0
                    for k in range(parameters.shape[0]):
                        if k == 1:
                            p = self.u_cusp_const[cusp_set]
                        else:
                            p = parameters[k, u_set]
                        poly += p * e_powers[e1, e2, k]
                        if k > 0:
                            poly_diff += p * k * e_powers[e1, e2, k-1]

                    gradient = r_vec/r * (r-L) ** C * (C/(r-L) * poly + poly_diff)
                    res[e1, :] += gradient
                    res[e2, :] -= gradient
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
            for label in chi_labels:
                for e1 in range(self.neu + self.ned):
                    r_vec = n_vectors[label, e1]
                    r = n_powers[label, e1, 1]
                    if r < L:
                        chi_set = int(e1 >= self.neu) % parameters.shape[1]
                        poly = poly_diff = 0.0
                        for k in range(parameters.shape[0]):
                            p = parameters[k, chi_set]
                            poly += p * n_powers[label, e1, k]
                            if k > 0:
                                poly_diff += p * k * n_powers[label, e1, k-1]

                        res[e1, :] += r_vec/r * (r-L) ** C * (C/(r-L) * poly + poly_diff)
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
            for label in f_labels:
                for e1 in range(1, self.neu + self.ned):
                    for e2 in range(e1):
                        r_e1I_vec = n_vectors[label, e1]
                        r_e2I_vec = n_vectors[label, e2]
                        r_ee_vec = e_vectors[e1, e2]
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        r_ee = e_powers[e1, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % parameters.shape[3]
                            poly = poly_diff_e1I = poly_diff_e2I = poly_diff_ee = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        p = parameters[l, m, n, f_set]
                                        poly += n_powers[label, e1, l] * n_powers[label, e2, m] * e_powers[e1, e2, n] * p
                                        if l > 0:
                                            poly_diff_e1I += l * n_powers[label, e1, l-1] * n_powers[label, e2, m] * e_powers[e1, e2, n] * p
                                        if m > 0:
                                            poly_diff_e2I += m * n_powers[label, e1, l] * n_powers[label, e2, m-1] * e_powers[e1, e2, n] * p
                                        if n > 0:
                                            poly_diff_ee += n * n_powers[label, e1, l] * n_powers[label, e2, m] * e_powers[e1, e2, n-1] * p

                            e1_gradient = r_e1I_vec/r_e1I * (C/(r_e1I - L) * poly + poly_diff_e1I)
                            e2_gradient = r_e2I_vec/r_e2I * (C/(r_e2I - L) * poly + poly_diff_e2I)
                            ee_gradient = r_ee_vec/r_ee * poly_diff_ee
                            res[e1, :] += (r_e1I - L) ** C * (r_e2I - L) ** C * (e1_gradient + ee_gradient)
                            res[e2, :] += (r_e1I - L) ** C * (r_e2I - L) ** C * (e2_gradient - ee_gradient)
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
        for e1 in range(1, self.neu + self.ned):
            for e2 in range(e1):
                r = e_powers[e1, e2, 1]
                if r < L:
                    cusp_set = (int(e1 >= self.neu) + int(e2 >= self.neu))
                    u_set = cusp_set % parameters.shape[1]
                    poly = poly_diff = poly_diff_2 = 0.0
                    for k in range(parameters.shape[0]):
                        if k == 1:
                            p = self.u_cusp_const[cusp_set]
                        else:
                            p = parameters[k, u_set]
                        poly += p * e_powers[e1, e2, k]
                        if k > 0:
                            poly_diff += k * p * e_powers[e1, e2, k-1]
                        if k > 1:
                            poly_diff_2 += k * (k-1) * p * e_powers[e1, e2, k-2]

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
            for label in chi_labels:
                for e1 in range(self.neu + self.ned):
                    r = n_powers[label, e1, 1]
                    if r < L:
                        chi_set = int(e1 >= self.neu) % parameters.shape[1]
                        poly = poly_diff = poly_diff_2 = 0.0
                        for k in range(parameters.shape[0]):
                            p = parameters[k, chi_set]
                            poly += p * n_powers[label, e1, k]
                            if k > 0:
                                poly_diff += k * p * n_powers[label, e1, k-1]
                            if k > 1:
                                poly_diff_2 += k * (k-1) * p * n_powers[label, e1, k-2]

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
            for label in f_labels:
                for e1 in range(1, self.neu + self.ned):
                    for e2 in range(e1):
                        r_e1I_vec = n_vectors[label, e1]
                        r_e2I_vec = n_vectors[label, e2]
                        r_ee_vec = e_vectors[e1, e2]
                        r_e1I = n_powers[label, e1, 1]
                        r_e2I = n_powers[label, e2, 1]
                        r_ee = e_powers[e1, e2, 1]
                        if r_e1I < L and r_e2I < L:
                            f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % parameters.shape[3]
                            poly = poly_diff_e1I = poly_diff_e2I = 0.0
                            poly_diff_ee = poly_diff_e1I_2 = poly_diff_e2I_2 = 0.0
                            poly_diff_ee_2 = poly_diff_e1I_ee = poly_diff_e2I_ee = 0.0
                            for l in range(parameters.shape[0]):
                                for m in range(parameters.shape[1]):
                                    for n in range(parameters.shape[2]):
                                        p = parameters[l, m, n, f_set]
                                        poly += n_powers[label, e1, l] * n_powers[label, e2, m] * e_powers[e1, e2, n] * p
                                        if l > 0:
                                            poly_diff_e1I += l * n_powers[label, e1, l-1] * n_powers[label, e2, m] * e_powers[e1, e2, n] * p
                                        if m > 0:
                                            poly_diff_e2I += m * n_powers[label, e1, l] * n_powers[label, e2, m-1] * e_powers[e1, e2, n] * p
                                        if n > 0:
                                            poly_diff_ee += n * n_powers[label, e1, l] * n_powers[label, e2, m] * e_powers[e1, e2, n-1] * p
                                        if l > 1:
                                            poly_diff_e1I_2 += l * (l-1) * n_powers[label, e1, l-2] * n_powers[label, e2, m] * e_powers[e1, e2, n] * p
                                        if m > 1:
                                            poly_diff_e2I_2 += m * (m-1) * n_powers[label, e1, l] * n_powers[label, e2, m-2] * e_powers[e1, e2, n] * p
                                        if n > 1:
                                            poly_diff_ee_2 += n * (n-1) * n_powers[label, e1, l] * n_powers[label, e2, m] * e_powers[e1, e2, n-2] * p
                                        if l > 0 and n > 0:
                                            poly_diff_e1I_ee += l * n * n_powers[label, e1, l-1] * n_powers[label, e2, m] * e_powers[e1, e2, n-1] * p
                                        if m > 0 and n > 0:
                                            poly_diff_e2I_ee += m * n * n_powers[label, e1, l] * n_powers[label, e2, m-1] * e_powers[e1, e2, n-1] * p

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

    def get_parameters_mask(self):
        """Mask of each variable. Setting x_scale is equivalent
        to reformulating the problem in scaled variables xs = x / x_scale.
        An alternative view is that the size of a trust region along j-th
        dimension is proportional to x_scale[j].
        The purpose of this method is to reformulate the optimization problem
        with dimensionless variables having only one dimensional parameter - scale.
        """
        scale = []
        if self.u_cutoff:
            if self.u_cutoff_optimizable:
                scale.append(self.u_cutoff)
            for j2 in range(self.u_parameters.shape[1]):
                for j1 in range(self.u_parameters.shape[0]):
                    if self.u_parameters_available[j1, j2]:
                        scale.append(self.u_parameters_optimizable[j1, j2])

        if self.chi_cutoff.any():
            for chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_cutoff_optimizable, chi_parameters_available in zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_cutoff, self.chi_cutoff_optimizable, self.chi_parameters_available):
                if chi_cutoff_optimizable:
                    scale.append(chi_cutoff)
                for j2 in range(chi_parameters.shape[1]):
                    for j1 in range(chi_parameters.shape[0]):
                        if chi_parameters_available[j1, j2]:
                            scale.append(chi_parameters_optimizable[j1, j2])

        if self.f_cutoff.any():
            for f_parameters, f_parameters_optimizable, f_cutoff, f_cutoff_optimizable, f_parameters_available in zip(self.f_parameters, self.f_parameters_optimizable, self.f_cutoff, self.f_cutoff_optimizable, self.f_parameters_available):
                if f_cutoff_optimizable:
                    scale.append(f_cutoff)
                for j4 in range(f_parameters.shape[3]):
                    for j3 in range(f_parameters.shape[2]):
                        for j2 in range(f_parameters.shape[1]):
                            for j1 in range(j2, f_parameters.shape[0]):
                                if f_parameters_available[j1, j2, j3, j4]:
                                    scale.append(f_parameters_optimizable[j1, j2, j3, j4])

        return np.array(scale)

    def get_parameters_scale(self):
        """Characteristic scale of each variable. Setting x_scale is equivalent
        to reformulating the problem in scaled variables xs = x / x_scale.
        An alternative view is that the size of a trust region along j-th
        dimension is proportional to x_scale[j].
        The purpose of this method is to reformulate the optimization problem
        with dimensionless variables having only one dimensional parameter - scale.
        """
        scale = []
        if self.u_cutoff:
            if self.u_cutoff_optimizable:
                scale.append(self.u_cutoff)
            for j2 in range(self.u_parameters.shape[1]):
                for j1 in range(self.u_parameters.shape[0]):
                    if self.u_parameters_optimizable[j1, j2] and self.u_parameters_available[j1, j2]:
                        scale.append(1 / self.u_cutoff ** (j1 + self.trunc - 1))

        if self.chi_cutoff.any():
            for chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_cutoff_optimizable, chi_parameters_available in zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_cutoff, self.chi_cutoff_optimizable, self.chi_parameters_available):
                if chi_cutoff_optimizable:
                    scale.append(chi_cutoff)
                for j2 in range(chi_parameters.shape[1]):
                    for j1 in range(chi_parameters.shape[0]):
                        if chi_parameters_optimizable[j1, j2] and chi_parameters_available[j1, j2]:
                            scale.append(1 / chi_cutoff ** (j1 + self.trunc - 1))

        if self.f_cutoff.any():
            for f_parameters, f_parameters_optimizable, f_cutoff, f_cutoff_optimizable, f_parameters_available in zip(self.f_parameters, self.f_parameters_optimizable, self.f_cutoff, self.f_cutoff_optimizable, self.f_parameters_available):
                if f_cutoff_optimizable:
                    scale.append(f_cutoff)
                for j4 in range(f_parameters.shape[3]):
                    for j3 in range(f_parameters.shape[2]):
                        for j2 in range(f_parameters.shape[1]):
                            for j1 in range(j2, f_parameters.shape[0]):
                                if f_parameters_optimizable[j1, j2, j3, j4] and f_parameters_available[j1, j2, j3, j4]:
                                    scale.append(1 / f_cutoff ** (j1 + j2 + j3 + 2 * self.trunc - 2))

        return np.array(scale)

    def get_parameters_constraints(self):
        """Returns parameters in the following order
        u-cutoff, u-linear parameters,
        for every chi-set: chi-cutoff, chi-linear parameters,
        for every f-set: f-cutoff, f-linear parameters.
        :return:
        """
        a_list = []
        b_list = []

        u_parameters_size = self.u_parameters.shape[0] + self.u_cutoff_optimizable
        u_matrix = np.zeros(shape=(1, u_parameters_size))
        u_matrix[0, 0] = self.trunc
        u_matrix[0, 1] = -self.u_cutoff

        u_spin_deps = self.u_parameters.shape[1]
        if u_spin_deps == 2:
            u_b = [1/4, 1/2]
            if self.neu < 2 and self.ned < 2:
                u_spin_deps -= 1
                u_b = [1/2]
            if self.neu + self.ned < 2:
                u_spin_deps -= 1
                u_b = [1/4]
        elif u_spin_deps == 3:
            u_b = [1/4, 1/2, 1/4]
            if self.neu < 2:
                u_b = [1/2, 1/4]
                u_spin_deps -= 1
            if self.neu + self.ned < 2:
                u_b = [1/4, 1/4]
                u_spin_deps -= 1
            if self.ned < 2:
                u_b = [1/4, 1/2]
                u_spin_deps -= 1
        else:
            # FIXME: u_spin_deps == 1
            u_b = [1/4]

        for spin_dep in range(u_spin_deps):
            a_list.append(u_matrix)
            b_list.append(u_b[spin_dep] / (-self.u_cutoff) ** (self.trunc - 1))

        for chi_parameters, chi_cutoff, chi_cutoff_optimizable in zip(self.chi_parameters, self.chi_cutoff, self.chi_cutoff_optimizable):
            chi_parameters_size = chi_parameters.shape[0] + chi_cutoff_optimizable
            chi_matrix = np.zeros(shape=(1, chi_parameters_size))
            chi_matrix[0, 0] = self.trunc
            chi_matrix[0, 1] = -chi_cutoff

            chi_spin_deps = chi_parameters.shape[1]
            if chi_spin_deps == 2:
                if self.neu < 1:
                    chi_spin_deps -= 1
                if self.ned < 1:
                    chi_spin_deps -= 1

            for spin_dep in range(chi_spin_deps):
                a_list.append(chi_matrix)
                b_list.append(0)

        for f_parameters, f_cutoff, no_dup_u_term, no_dup_chi_term in zip(self.f_parameters, self.f_cutoff, self.no_dup_u_term, self.no_dup_chi_term):
            f_en_order = f_parameters.shape[0] - 1
            f_ee_order = f_parameters.shape[2] - 1
            f_matrix = construct_a_matrix(self.trunc, f_en_order, f_ee_order, f_cutoff, no_dup_u_term, no_dup_chi_term)
            f_constrains_size, f_parameters_size = f_matrix.shape

            f_spin_deps = f_parameters.shape[3]
            if f_spin_deps == 2:
                if self.neu < 2 and self.ned < 2:
                    f_spin_deps -= 1
                if self.neu + self.ned < 2:
                    f_spin_deps -= 1
            elif f_spin_deps == 3:
                if self.neu < 2:
                    f_spin_deps -= 1
                if self.neu + self.ned < 2:
                    f_spin_deps -= 1
                if self.ned < 2:
                    f_spin_deps -= 1

            for spin_dep in range(f_spin_deps):
                a_list.append(f_matrix)
                b_list += [0] * f_constrains_size

        # FIXME: create blockdiagonal matrix from list of matrix like scipy.linalg.block_diag
        # a = sp.linalg.block_diag(a_list)
        shape_0_list = np.cumsum(np.array([a.shape[0] for a in a_list]))
        shape_1_list = np.cumsum(np.array([a.shape[1] for a in a_list]))
        a = np.zeros(shape=(shape_0_list[-1], shape_1_list[-1]))
        for a_part, p0, p1 in zip(a_list, shape_0_list, shape_1_list):
            a[p0 - a_part.shape[0]:p0, p1 - a_part.shape[1]:p1] = a_part
        b = np.array(b_list)
        return a, b

    def get_parameters(self, all_parameters):
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
            for j2 in range(self.u_parameters.shape[1]):
                for j1 in range(self.u_parameters.shape[0]):
                    if (self.u_parameters_optimizable[j1, j2] or all_parameters) and self.u_parameters_available[j1, j2]:
                        if j1 == 1:
                            res.append(self.u_cusp_const[j2])
                        else:
                            res.append(self.u_parameters[j1, j2])

        if self.chi_cutoff.any():
            for chi_parameters, chi_parameters_optimizable, chi_cutoff, chi_cutoff_optimizable, chi_parameters_available in zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_cutoff, self.chi_cutoff_optimizable, self.chi_parameters_available):
                if chi_cutoff_optimizable:
                    res.append(chi_cutoff)
                for j2 in range(chi_parameters.shape[1]):
                    for j1 in range(chi_parameters.shape[0]):
                        if (chi_parameters_optimizable[j1, j2] or all_parameters) and chi_parameters_available[j1, j2]:
                            res.append(chi_parameters[j1, j2])

        if self.f_cutoff.any():
            for f_parameters, f_parameters_optimizable, f_cutoff, f_cutoff_optimizable, f_parameters_available in zip(self.f_parameters, self.f_parameters_optimizable, self.f_cutoff, self.f_cutoff_optimizable, self.f_parameters_available):
                if f_cutoff_optimizable:
                    res.append(f_cutoff)
                for j4 in range(f_parameters.shape[3]):
                    for j3 in range(f_parameters.shape[2]):
                        for j2 in range(f_parameters.shape[1]):
                            for j1 in range(j2, f_parameters.shape[0]):
                                if (f_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and f_parameters_available[j1, j2, j3, j4]:
                                    res.append(f_parameters[j1, j2, j3, j4])

        return np.array(res)

    def set_parameters(self, parameters, all_parameters):
        """Set parameters in the following order:
        u-cutoff, u-linear parameters,
        for every chi-set: chi-cutoff, chi-linear parameters,
        for every f-set: f-cutoff, f-linear parameters.
        :param parameters:
        :param all_parameters:
        :return:
        """
        n = 0
        if self.u_cutoff:
            if self.u_cutoff_optimizable:
                self.u_cutoff = parameters[n]
                n += 1
            for j2 in range(self.u_parameters.shape[1]):
                for j1 in range(self.u_parameters.shape[0]):
                    if (self.u_parameters_optimizable[j1, j2] or all_parameters) and self.u_parameters_available[j1, j2]:
                        if j1 == 1:
                            for cup_set in range(j2, 3, self.u_parameters.shape[1]):
                                self.u_cusp_const[cup_set] = parameters[n]
                        else:
                            self.u_parameters[j1, j2] = parameters[n]
                        n += 1
            if not all_parameters:
                self.fix_u_parameters()

        if self.chi_cutoff.any():
            for i, (chi_parameters, chi_parameters_optimizable, chi_cutoff_optimizable, chi_parameters_available) in enumerate(zip(self.chi_parameters, self.chi_parameters_optimizable, self.chi_cutoff_optimizable, self.chi_parameters_available)):
                if chi_cutoff_optimizable:
                    # Sequence type is a pointer, but numeric type is not.
                    self.chi_cutoff[i] = parameters[n]
                    n += 1
                for j2 in range(chi_parameters.shape[1]):
                    for j1 in range(chi_parameters.shape[0]):
                        if (chi_parameters_optimizable[j1, j2] or all_parameters) and chi_parameters_available[j1, j2]:
                            chi_parameters[j1, j2] = parameters[n]
                            n += 1
            if not all_parameters:
                self.fix_chi_parameters()

        if self.f_cutoff.any():
            for i, (f_parameters, f_parameters_optimizable, f_cutoff_optimizable, f_parameters_available) in enumerate(zip(self.f_parameters, self.f_parameters_optimizable, self.f_cutoff_optimizable, self.f_parameters_available)):
                if f_cutoff_optimizable:
                    # Sequence types is a pointer, but numeric types is not.
                    self.f_cutoff[i] = parameters[n]
                    n += 1
                for j4 in range(f_parameters.shape[3]):
                    for j3 in range(f_parameters.shape[2]):
                        for j2 in range(f_parameters.shape[1]):
                            for j1 in range(j2, f_parameters.shape[0]):
                                if (f_parameters_optimizable[j1, j2, j3, j4] or all_parameters) and f_parameters_available[j1, j2, j3, j4]:
                                    f_parameters[j1, j2, j3, j4] = f_parameters[j2, j1, j3, j4] = parameters[n]
                                    n += 1
            if not all_parameters:
                self.fix_f_parameters()

        return parameters[n:]

    def u_term_parameters_d1(self, e_powers):
        """First derivatives of logarithm wfn with respect to u-term parameters
        :param e_powers: powers of e-e distances
        """
        if not self.u_cutoff:
            return np.zeros((0,))

        delta = 0.000001
        size = self.u_parameters_available.sum() + self.u_cutoff_optimizable
        res = np.zeros(shape=(size,))

        n = -1
        if self.u_cutoff_optimizable:
            n += 1
            self.u_cutoff -= delta
            res[n] -= self.u_term(e_powers) / delta / 2
            self.u_cutoff += 2 * delta
            res[n] += self.u_term(e_powers) / delta / 2
            self.u_cutoff -= delta

        for j2 in range(self.u_parameters.shape[1]):
            for j1 in range(self.u_parameters.shape[0]):
                if self.u_parameters_available[j1, j2]:
                    n += 1
                    for e1 in range(1, self.neu + self.ned):
                        for e2 in range(e1):
                            r = e_powers[e1, e2, 1]
                            if r < self.u_cutoff:
                                u_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % self.u_parameters.shape[1]
                                if u_set == j2:
                                    res[n] += e_powers[e1, e2, j1] * (r - self.u_cutoff) ** self.trunc

        return res

    def chi_term_parameters_d1(self, n_powers):
        """First derivatives of logarithm wfn with respect to chi-term parameters
        :param n_powers: powers of e-n distances
        """
        if not self.chi_cutoff.any():
            return np.zeros((0,))

        delta = 0.000001
        size = sum([
            chi_parameters_available.sum() + chi_cutoff_optimizable
            for chi_parameters_available, chi_cutoff_optimizable
            in zip(self.chi_parameters_available, self.chi_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size,))

        n = -1
        for i, (chi_parameters, chi_parameters_available, chi_labels) in enumerate(zip(self.chi_parameters, self.chi_parameters_available, self.chi_labels)):
            if self.chi_cutoff_optimizable[i]:
                n += 1
                self.chi_cutoff[i] -= delta
                res[n] -= self.chi_term(n_powers) / delta / 2
                self.chi_cutoff[i] += 2 * delta
                res[n] += self.chi_term(n_powers) / delta / 2
                self.chi_cutoff[i] -= delta

            L = self.chi_cutoff[i]
            for j2 in range(chi_parameters.shape[1]):
                for j1 in range(chi_parameters.shape[0]):
                    if chi_parameters_available[j1, j2]:
                        n += 1
                        for label in chi_labels:
                            for e1 in range(self.neu + self.ned):
                                r = n_powers[label, e1, 1]
                                if r < L:
                                    chi_set = int(e1 >= self.neu) % chi_parameters.shape[1]
                                    if chi_set == j2:
                                        res[n] += n_powers[label, e1, j1] * (r - L) ** self.trunc

        return res

    def f_term_parameters_d1(self, e_powers, n_powers):
        """Numerical first derivatives of logarithm wfn with respect to f-term parameters
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        """
        if not self.f_cutoff.any():
            return np.zeros((0,))

        delta = 0.000001
        C = self.trunc
        size = sum([
            f_parameters_available.sum() + f_cutoff_optimizable
            for f_parameters_available, f_cutoff_optimizable
            in zip(self.f_parameters_available, self.f_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size,))

        n = -1

        for i, (f_parameters, f_parameters_available, f_labels) in enumerate(zip(self.f_parameters, self.f_parameters_available, self.f_labels)):
            if self.f_cutoff_optimizable[i]:
                n += 1
                self.f_cutoff[i] -= delta
                res[n] -= self.f_term(e_powers, n_powers) / delta / 2
                self.f_cutoff[i] += 2 * delta
                res[n] += self.f_term(e_powers, n_powers) / delta / 2
                self.f_cutoff[i] -= delta

            L = self.f_cutoff[i]
            for j4 in range(f_parameters.shape[3]):
                for j3 in range(f_parameters.shape[2]):
                    for j2 in range(f_parameters.shape[1]):
                        for j1 in range(j2, f_parameters.shape[0]):
                            if f_parameters_available[j1, j2, j3, j4]:
                                n += 1
                                for label in f_labels:
                                    for e1 in range(1, self.neu + self.ned):
                                        for e2 in range(e1):
                                            r_e1I = n_powers[label, e1, 1]
                                            r_e2I = n_powers[label, e2, 1]
                                            if r_e1I < L and r_e2I < L:
                                                f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % f_parameters.shape[3]
                                                if f_set == j4:
                                                    res[n] += (
                                                        n_powers[label, e1, j1] * n_powers[label, e2, j2] * e_powers[e1, e2, j3] *
                                                        (r_e1I - L) ** C * (r_e2I - L) ** C
                                                    )
                                                    if j1 != j2:
                                                        res[n] += (
                                                            n_powers[label, e1, j2] * n_powers[label, e2, j1] * e_powers[e1, e2, j3] *
                                                            (r_e1I - L) ** C * (r_e2I - L) ** C
                                                        )

        return res

    def u_term_gradient_parameters_d1(self, e_powers, e_vectors):
        """Gradient with respect to parameters
        :param e_vectors: e-e vectors
        :param e_powers: e-e powers
        :return:
        """
        if not self.u_cutoff:
            return np.zeros((0, (self.neu + self.ned) * 3))

        delta = 0.000001
        size = self.u_parameters_available.sum() + self.u_cutoff_optimizable
        res = np.zeros(shape=(size, (self.neu + self.ned), 3))

        n = -1
        if self.u_cutoff_optimizable:
            n += 1
            self.u_cutoff -= delta
            res[n] -= self.u_term_gradient(e_powers, e_vectors) / delta / 2
            self.u_cutoff += 2 * delta
            res[n] += self.u_term_gradient(e_powers, e_vectors) / delta / 2
            self.u_cutoff -= delta

        for j2 in range(self.u_parameters.shape[1]):
            for j1 in range(self.u_parameters.shape[0]):
                if self.u_parameters_available[j1, j2]:
                    n += 1
                    for e1 in range(1, self.neu + self.ned):
                        for e2 in range(e1):
                            r_vec = e_vectors[e1, e2]
                            r = e_powers[e1, e2, 1]
                            if r < self.u_cutoff:
                                u_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % self.u_parameters.shape[1]
                                if u_set == j2:
                                    poly = e_powers[e1, e2, j1]
                                    poly_diff = 0
                                    if j1 > 0:
                                        poly_diff = j1 * e_powers[e1, e2, j1 - 1]
                                    gradient = r_vec / r * (r - self.u_cutoff) ** self.trunc * (self.trunc / (r - self.u_cutoff) * poly + poly_diff)
                                    res[n, e1, :] += gradient
                                    res[n, e2, :] -= gradient

        return res.reshape(size, (self.neu + self.ned) * 3)

    def chi_term_gradient_parameters_d1(self, n_powers, n_vectors):
        """Gradient with respect to parameters
        :param n_vectors: e-n vectors
        :param n_powers: e-n powers
        :return:
        """
        if not self.chi_cutoff.any():
            return np.zeros((0, (self.neu + self.ned) * 3))

        delta = 0.000001
        size = sum([
            chi_parameters_available.sum() + chi_cutoff_optimizable
            for chi_parameters_available, chi_cutoff_optimizable
            in zip(self.chi_parameters_available, self.chi_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size, (self.neu + self.ned), 3))

        n = -1
        for i, (chi_parameters, chi_parameters_available, chi_labels) in enumerate(zip(self.chi_parameters, self.chi_parameters_available, self.chi_labels)):
            if self.chi_cutoff_optimizable[i]:
                n += 1
                self.chi_cutoff[i] -= delta
                res[n] -= self.chi_term_gradient(n_powers, n_vectors) / delta / 2
                self.chi_cutoff[i] += 2 * delta
                res[n] += self.chi_term_gradient(n_powers, n_vectors) / delta / 2
                self.chi_cutoff[i] -= delta

            L = self.chi_cutoff[i]
            for j2 in range(chi_parameters.shape[1]):
                for j1 in range(chi_parameters.shape[0]):
                    if chi_parameters_available[j1, j2]:
                        n += 1
                        for label in chi_labels:
                            for e1 in range(self.neu + self.ned):
                                r_vec = n_vectors[label, e1]
                                r = n_powers[label, e1, 1]
                                if r < L:
                                    chi_set = int(e1 >= self.neu) % chi_parameters.shape[1]
                                    if chi_set == j2:
                                        poly = n_powers[label, e1, j1]
                                        poly_diff = 0
                                        if j1 > 0:
                                            poly_diff = j1 * n_powers[label, e1, j1 - 1]
                                        res[n, e1, :] += r_vec / r * (r - L) ** self.trunc * (self.trunc / (r - L) * poly + poly_diff)

        return res.reshape(size, (self.neu + self.ned) * 3)

    def f_term_gradient_parameters_d1(self, e_powers, n_powers, e_vectors, n_vectors):
        """Gradient with respect to parameters
        :param n_vectors: e-n vectors
        :param n_powers: e-n powers
        :return:
        """
        if not self.f_cutoff.any():
            return np.zeros((0, (self.neu + self.ned) * 3))

        delta = 0.000001
        size = sum([
            f_parameters_available.sum() + f_cutoff_optimizable
            for f_parameters_available, f_cutoff_optimizable
            in zip(self.f_parameters_available, self.f_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size, (self.neu + self.ned), 3))

        n = -1
        for i, (f_parameters, f_parameters_available, f_labels) in enumerate(zip(self.f_parameters, self.f_parameters_available, self.f_labels)):
            if self.f_cutoff_optimizable[i]:
                n += 1
                self.f_cutoff[i] -= delta
                res[n] -= self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors) / delta / 2
                self.f_cutoff[i] += 2 * delta
                res[n] += self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors) / delta / 2
                self.f_cutoff[i] -= delta

            L = self.f_cutoff[i]
            for j4 in range(f_parameters.shape[3]):
                for j3 in range(f_parameters.shape[2]):
                    for j2 in range(f_parameters.shape[1]):
                        for j1 in range(j2, f_parameters.shape[0]):
                            if f_parameters_available[j1, j2, j3, j4]:
                                n += 1
                                for label in f_labels:
                                    for e1 in range(1, self.neu + self.ned):
                                        for e2 in range(e1):
                                            r_e1I_vec = n_vectors[label, e1]
                                            r_e2I_vec = n_vectors[label, e2]
                                            r_ee_vec = e_vectors[e1, e2]
                                            r_e1I = n_powers[label, e1, 1]
                                            r_e2I = n_powers[label, e2, 1]
                                            r_ee = e_powers[e1, e2, 1]
                                            if r_e1I < L and r_e2I < L:
                                                f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % f_parameters.shape[3]
                                                if f_set == j4:
                                                    poly = n_powers[label, e1, j1] * n_powers[label, e2, j2] * e_powers[e1, e2, j3]
                                                    poly_diff_e1I = poly_diff_e2I = poly_diff_ee = 0
                                                    if j1 > 0:
                                                        poly_diff_e1I = j1 * n_powers[label, e1, j1 - 1] * n_powers[label, e2, j2] * e_powers[e1, e2, j3]
                                                    if j2 > 0:
                                                        poly_diff_e2I = j2 * n_powers[label, e1, j1] * n_powers[label, e2, j2 - 1] * e_powers[e1, e2, j3]
                                                    if j3 > 0:
                                                        poly_diff_ee = j3 * n_powers[label, e1, j1] * n_powers[label, e2, j2] * e_powers[e1, e2, j3 - 1]
                                                    e1_gradient = r_e1I_vec / r_e1I * (self.trunc / (r_e1I - L) * poly + poly_diff_e1I)
                                                    e2_gradient = r_e2I_vec / r_e2I * (self.trunc / (r_e2I - L) * poly + poly_diff_e2I)
                                                    ee_gradient = r_ee_vec / r_ee * poly_diff_ee
                                                    res[n, e1, :] += (r_e1I - L) ** self.trunc * (r_e2I - L) ** self.trunc * (e1_gradient + ee_gradient)
                                                    res[n, e2, :] += (r_e1I - L) ** self.trunc * (r_e2I - L) ** self.trunc * (e2_gradient - ee_gradient)

                                                    if j1 != j2:
                                                        poly = n_powers[label, e1, j2] * n_powers[label, e2, j1] * e_powers[e1, e2, j3]
                                                        poly_diff_e1I = poly_diff_e2I = poly_diff_ee = 0
                                                        if j2 > 0:
                                                            poly_diff_e1I = j2 * n_powers[label, e1, j2 - 1] * n_powers[label, e2, j1] * e_powers[e1, e2, j3]
                                                        if j1 > 0:
                                                            poly_diff_e2I = j1 * n_powers[label, e1, j2] * n_powers[label, e2, j1 - 1] * e_powers[e1, e2, j3]
                                                        if j3 > 0:
                                                            poly_diff_ee = j3 * n_powers[label, e1, j2] * n_powers[label, e2, j1] * e_powers[e1, e2, j3 - 1]
                                                        e1_gradient = r_e1I_vec / r_e1I * (self.trunc / (r_e1I - L) * poly + poly_diff_e1I)
                                                        e2_gradient = r_e2I_vec / r_e2I * (self.trunc / (r_e2I - L) * poly + poly_diff_e2I)
                                                        ee_gradient = r_ee_vec / r_ee * poly_diff_ee
                                                        res[n, e1, :] += (r_e1I - L) ** self.trunc * (r_e2I - L) ** self.trunc * (e1_gradient + ee_gradient)
                                                        res[n, e2, :] += (r_e1I - L) ** self.trunc * (r_e2I - L) ** self.trunc * (e2_gradient - ee_gradient)

        return res.reshape(size, (self.neu + self.ned) * 3)

    def u_term_laplacian_parameters_d1(self, e_powers):
        """Laplacian with respect to parameters
        :param e_powers: e-e powers
        :return:
        """
        if not self.u_cutoff:
            return np.zeros((0, ))

        delta = 0.000001
        size = self.u_parameters_available.sum() + self.u_cutoff_optimizable
        res = np.zeros(shape=(size, ))

        n = -1
        if self.u_cutoff_optimizable:
            n += 1
            self.u_cutoff -= delta
            res[n] -= self.u_term_laplacian(e_powers) / delta / 2
            self.u_cutoff += 2 * delta
            res[n] += self.u_term_laplacian(e_powers) / delta / 2
            self.u_cutoff -= delta

        for j2 in range(self.u_parameters.shape[1]):
            for j1 in range(self.u_parameters.shape[0]):
                if self.u_parameters_available[j1, j2]:
                    n += 1
                    for e1 in range(1, self.neu + self.ned):
                        for e2 in range(e1):
                            r = e_powers[e1, e2, 1]
                            if r < self.u_cutoff:
                                u_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % self.u_parameters.shape[1]
                                if u_set == j2:
                                    poly = e_powers[e1, e2, j1]
                                    poly_diff = poly_diff_2 = 0
                                    if j1 > 0:
                                        poly_diff = j1 * e_powers[e1, e2, j1 - 1]
                                    if j1 > 1:
                                        poly_diff_2 = j1 * (j1 - 1) * e_powers[e1, e2, j1 - 2]
                                    res[n] += (r - self.u_cutoff)**self.trunc * (
                                            self.trunc*(self.trunc - 1)/(r-self.u_cutoff)**2 * poly + 2 * self.trunc/(r-self.u_cutoff) * poly_diff + poly_diff_2 +
                                            2 * (self.trunc/(r-self.u_cutoff) * poly + poly_diff) / r
                                    )

        return 2 * res

    def chi_term_laplacian_parameters_d1(self, n_powers):
        """Laplacian with respect to parameters
        :param n_powers: e-n powers
        :return:
        """
        if not self.chi_cutoff.any():
            return np.zeros((0, ))

        delta = 0.000001
        size = sum([
            chi_parameters_available.sum() + chi_cutoff_optimizable
            for chi_parameters_available, chi_cutoff_optimizable
            in zip(self.chi_parameters_available, self.chi_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size, ))

        n = -1
        for i, (chi_parameters, chi_parameters_available, chi_labels) in enumerate(zip(self.chi_parameters, self.chi_parameters_available, self.chi_labels)):
            if self.chi_cutoff_optimizable[i]:
                n += 1
                self.chi_cutoff[i] -= delta
                res[n] -= self.chi_term_laplacian(n_powers) / delta / 2
                self.chi_cutoff[i] += 2 * delta
                res[n] += self.chi_term_laplacian(n_powers) / delta / 2
                self.chi_cutoff[i] -= delta

            L = self.chi_cutoff[i]
            for j2 in range(chi_parameters.shape[1]):
                for j1 in range(chi_parameters.shape[0]):
                    if chi_parameters_available[j1, j2]:
                        n += 1
                        for label in chi_labels:
                            for e1 in range(self.neu + self.ned):
                                r = n_powers[label, e1, 1]
                                if r < L:
                                    chi_set = int(e1 >= self.neu) % chi_parameters.shape[1]
                                    if chi_set == j2:
                                        poly = n_powers[label, e1, j1]
                                        poly_diff = poly_diff_2 = 0
                                        if j1 > 0:
                                            poly_diff = j1 * n_powers[label, e1, j1 - 1]
                                        if j1 > 1:
                                            poly_diff_2 = j1 * (j1 - 1) * n_powers[label, e1, j1 - 2]
                                        res[n] += (r-L)**self.trunc * (
                                                self.trunc*(self.trunc - 1)/(r-L)**2 * poly + 2 * self.trunc/(r-L) * poly_diff + poly_diff_2 +
                                                2 * (self.trunc/(r-L) * poly + poly_diff) / r
                                        )

        return res

    def f_term_laplacian_parameters_d1(self, e_powers, n_powers, e_vectors, n_vectors):
        """Laplacian with respect to parameters
        :param n_powers: e-n powers
        :return:
        """
        if not self.f_cutoff.any():
            return np.zeros((0, ))

        delta = 0.000001
        size = sum([
            f_parameters_available.sum() + f_cutoff_optimizable
            for f_parameters_available, f_cutoff_optimizable
            in zip(self.f_parameters_available, self.f_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size, ))

        n = -1
        C = self.trunc
        for i, (f_parameters, f_parameters_available, f_labels) in enumerate(zip(self.f_parameters, self.f_parameters_available, self.f_labels)):
            if self.f_cutoff_optimizable[i]:
                n += 1
                self.f_cutoff[i] -= delta
                res[n] -= self.f_term_laplacian(e_powers, n_powers, e_vectors, n_vectors) / delta / 2
                self.f_cutoff[i] += 2 * delta
                res[n] += self.f_term_laplacian(e_powers, n_powers, e_vectors, n_vectors) / delta / 2
                self.f_cutoff[i] -= delta

            L = self.f_cutoff[i]
            for j4 in range(f_parameters.shape[3]):
                for j3 in range(f_parameters.shape[2]):
                    for j2 in range(f_parameters.shape[1]):
                        for j1 in range(j2, f_parameters.shape[0]):
                            if f_parameters_available[j1, j2, j3, j4]:
                                n += 1
                                for label in f_labels:
                                    for e1 in range(1, self.neu + self.ned):
                                        for e2 in range(e1):
                                            r_e1I_vec = n_vectors[label, e1]
                                            r_e2I_vec = n_vectors[label, e2]
                                            r_ee_vec = e_vectors[e1, e2]
                                            r_e1I = n_powers[label, e1, 1]
                                            r_e2I = n_powers[label, e2, 1]
                                            r_ee = e_powers[e1, e2, 1]
                                            if r_e1I < L and r_e2I < L:
                                                f_set = (int(e1 >= self.neu) + int(e2 >= self.neu)) % f_parameters.shape[3]
                                                if f_set == j4:
                                                    poly = n_powers[label, e1, j1] * n_powers[label, e2, j2] * e_powers[e1, e2, j3]
                                                    poly_diff_e1I = poly_diff_e2I = 0
                                                    poly_diff_ee = poly_diff_e1I_2 = poly_diff_e2I_2 = 0.0
                                                    poly_diff_ee_2 = poly_diff_e1I_ee = poly_diff_e2I_ee = 0.0
                                                    if j1 > 0:
                                                        poly_diff_e1I = j1 * n_powers[label, e1, j1 - 1] * n_powers[label, e2, j2] * e_powers[e1, e2, j3]
                                                    if j2 > 0:
                                                        poly_diff_e2I = j2 * n_powers[label, e1, j1] * n_powers[label, e2, j2 - 1] * e_powers[e1, e2, j3]
                                                    if j3 > 0:
                                                        poly_diff_ee = j3 * n_powers[label, e1, j1] * n_powers[label, e2, j2] * e_powers[e1, e2, j3 - 1]
                                                    if j1 > 1:
                                                        poly_diff_e1I_2 = j1 * (j1-1) * n_powers[label, e1, j1 - 2] * n_powers[label, e2, j2] * e_powers[e1, e2, j3]
                                                    if j2 > 1:
                                                        poly_diff_e2I_2 = j2 * (j2-1) * n_powers[label, e1, j1] * n_powers[label, e2, j2 - 2] * e_powers[e1, e2, j3]
                                                    if j3 > 1:
                                                        poly_diff_ee_2 = j3 * (j3-1) * n_powers[label, e1, j1] * n_powers[label, e2, j2] * e_powers[e1, e2, j3 - 2]
                                                    if j1 > 0 and j3 > 0:
                                                        poly_diff_e1I_ee = j1 * j3 * n_powers[label, e1, j1 - 1] * n_powers[label, e2, j2] * e_powers[e1, e2, j3 - 1]
                                                    if j2 > 0 and j3 > 0:
                                                        poly_diff_e2I_ee = j2 * j3 * n_powers[label, e1, j1] * n_powers[label, e2, j2 - 1] * e_powers[e1, e2, j3 - 1]
                                                    diff_1 = (
                                                            (C / (r_e1I - L) * poly + poly_diff_e1I) / r_e1I +
                                                            (C / (r_e2I - L) * poly + poly_diff_e2I) / r_e2I +
                                                            2 * poly_diff_ee / r_ee
                                                    )
                                                    diff_2 = (
                                                            C * (C - 1) / (r_e1I - L) ** 2 * poly +
                                                            C * (C - 1) / (r_e2I - L) ** 2 * poly +
                                                            (poly_diff_e1I_2 + poly_diff_e2I_2 + 2 * poly_diff_ee_2) +
                                                            2 * C / (r_e1I - L) * poly_diff_e1I +
                                                            2 * C / (r_e2I - L) * poly_diff_e2I
                                                    )
                                                    dot_product = (
                                                            np.sum(r_e1I_vec * r_ee_vec) * (C / (r_e1I - L) * poly_diff_ee + poly_diff_e1I_ee) / r_e1I / r_ee -
                                                            np.sum(r_e2I_vec * r_ee_vec) * (C / (r_e2I - L) * poly_diff_ee + poly_diff_e2I_ee) / r_e2I / r_ee
                                                    )
                                                    res[n] += (r_e1I - L) ** C * (r_e2I - L) ** C * (diff_2 + 2 * diff_1 + 2 * dot_product)
                                                    if j1 != j2:
                                                        poly = n_powers[label, e1, j2] * n_powers[label, e2, j1] * e_powers[e1, e2, j3]
                                                        poly_diff_e1I = poly_diff_e2I = 0
                                                        poly_diff_ee = poly_diff_e1I_2 = poly_diff_e2I_2 = 0.0
                                                        poly_diff_ee_2 = poly_diff_e1I_ee = poly_diff_e2I_ee = 0.0
                                                        if j2 > 0:
                                                            poly_diff_e1I = j2 * n_powers[label, e1, j2 - 1] * n_powers[label, e2, j1] * e_powers[e1, e2, j3]
                                                        if j1 > 0:
                                                            poly_diff_e2I = j1 * n_powers[label, e1, j2] * n_powers[label, e2, j1 - 1] * e_powers[e1, e2, j3]
                                                        if j3 > 0:
                                                            poly_diff_ee = j3 * n_powers[label, e1, j2] * n_powers[label, e2, j1] * e_powers[e1, e2, j3 - 1]
                                                        if j2 > 1:
                                                            poly_diff_e1I_2 = j2 * (j2 - 1) * n_powers[label, e1, j2 - 2] * n_powers[label, e2, j1] * e_powers[e1, e2, j3]
                                                        if j1 > 1:
                                                            poly_diff_e2I_2 = j1 * (j1 - 1) * n_powers[label, e1, j2] * n_powers[label, e2, j1 - 2] * e_powers[e1, e2, j3]
                                                        if j3 > 1:
                                                            poly_diff_ee_2 = j3 * (j3 - 1) * n_powers[label, e1, j2] * n_powers[label, e2, j1] * e_powers[e1, e2, j3 - 2]
                                                        if j2 > 0 and j3 > 0:
                                                            poly_diff_e1I_ee = j2 * j3 * n_powers[label, e1, j2 - 1] * n_powers[label, e2, j1] * e_powers[e1, e2, j3 - 1]
                                                        if j1 > 0 and j3 > 0:
                                                            poly_diff_e2I_ee = j1 * j3 * n_powers[label, e1, j2] * n_powers[label, e2, j1 - 1] * e_powers[e1, e2, j3 - 1]
                                                        diff_1 = (
                                                                (C / (r_e1I - L) * poly + poly_diff_e1I) / r_e1I +
                                                                (C / (r_e2I - L) * poly + poly_diff_e2I) / r_e2I +
                                                                2 * poly_diff_ee / r_ee
                                                        )
                                                        diff_2 = (
                                                                C * (C - 1) / (r_e1I - L) ** 2 * poly +
                                                                C * (C - 1) / (r_e2I - L) ** 2 * poly +
                                                                (poly_diff_e1I_2 + poly_diff_e2I_2 + 2 * poly_diff_ee_2) +
                                                                2 * C / (r_e1I - L) * poly_diff_e1I +
                                                                2 * C / (r_e2I - L) * poly_diff_e2I
                                                        )
                                                        dot_product = (
                                                                np.sum(r_e1I_vec * r_ee_vec) * (C / (r_e1I - L) * poly_diff_ee + poly_diff_e1I_ee) / r_e1I / r_ee -
                                                                np.sum(r_e2I_vec * r_ee_vec) * (C / (r_e2I - L) * poly_diff_ee + poly_diff_e2I_ee) / r_e2I / r_ee
                                                        )
                                                        res[n] += (r_e1I - L) ** C * (r_e2I - L) ** C * (diff_2 + 2 * diff_1 + 2 * dot_product)
        return res

    def parameters_d1(self, e_vectors, n_vectors):
        """First derivatives logarithm Jastrow with respect to the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return np.concatenate((
            self.u_term_parameters_d1(e_powers),
            self.chi_term_parameters_d1(n_powers),
            self.f_term_parameters_d1(e_powers, n_powers),
        ))

    def gradient_parameters_d1(self, e_vectors, n_vectors):
        """First derivatives of Jastrow gradient with respect to parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return np.concatenate((
            self.u_term_gradient_parameters_d1(e_powers, e_vectors),
            self.chi_term_gradient_parameters_d1(n_powers, n_vectors),
            self.f_term_gradient_parameters_d1(e_powers, n_powers, e_vectors, n_vectors),
        ))

    def laplacian_parameters_d1(self, e_vectors, n_vectors):
        """First derivatives of Jastrow laplacian with respect to parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return np.concatenate((
            self.u_term_laplacian_parameters_d1(e_powers),
            self.chi_term_laplacian_parameters_d1(n_powers),
            self.f_term_laplacian_parameters_d1(e_powers, n_powers, e_vectors, n_vectors),
        ))

    def gradient_parameters_numerical_d1(self, e_vectors, n_vectors):
        """Numerical first derivatives of Jastrow gradient with respect to parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        delta = 0.00001
        parameters = self.get_parameters(True)
        res = np.zeros(shape=(parameters.size, (self.neu + self.ned) * 3))
        for i in range(parameters.size):
            parameters[i] -= delta
            self.set_parameters(parameters, True)
            res[i] -= self.gradient(e_vectors, n_vectors)
            parameters[i] += 2 * delta
            self.set_parameters(parameters, True)
            res[i] += self.gradient(e_vectors, n_vectors)
            parameters[i] -= delta

        self.set_parameters(parameters, True)
        return res / delta / 2

    def laplacian_parameters_numerical_d1(self, e_vectors, n_vectors):
        """Numerical first derivatives of Jastrow laplacian with respect to parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        delta = 0.00001
        parameters = self.get_parameters(True)
        res = np.zeros(shape=parameters.shape)
        for i in range(parameters.size):
            parameters[i] -= delta
            self.set_parameters(parameters, True)
            res[i] -= self.laplacian(e_vectors, n_vectors)
            parameters[i] += 2 * delta
            self.set_parameters(parameters, True)
            res[i] += self.laplacian(e_vectors, n_vectors)
            parameters[i] -= delta

        self.set_parameters(parameters, True)
        return res / delta / 2

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
