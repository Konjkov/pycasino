from numpy_config import np
import numba as nb

from readers.numerical import rref
from readers.backflow import construct_c_matrix
from overload import subtract_outer, random_step


labels_type = nb.int64[:]
eta_mask_type = nb.boolean[:, :]
mu_mask_type = nb.boolean[:, :]
phi_mask_type = nb.boolean[:, :, :, :]
theta_mask_type = nb.boolean[:, :, :, :]
eta_parameters_type = nb.float64[:, :]
mu_parameters_type = nb.float64[:, :]
phi_parameters_type = nb.float64[:, :, :, :]
theta_parameters_type = nb.float64[:, :, :, :]
eta_parameters_optimizable_type = nb.boolean[:, :]
mu_parameters_optimizable_type = nb.boolean[:, :]
phi_parameters_optimizable_type = nb.boolean[:, :, :, :]
theta_parameters_optimizable_type = nb.boolean[:, :, :, :]


spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('trunc', nb.int64),
    ('eta_mask', eta_mask_type),
    ('mu_mask', nb.types.ListType(mu_mask_type)),
    ('phi_mask', nb.types.ListType(phi_mask_type)),
    ('theta_mask', nb.types.ListType(theta_mask_type)),
    ('eta_parameters', eta_parameters_type),
    ('mu_parameters', nb.types.ListType(mu_parameters_type)),
    ('phi_parameters', nb.types.ListType(phi_parameters_type)),
    ('theta_parameters', nb.types.ListType(theta_parameters_type)),
    ('eta_parameters_optimizable', eta_parameters_optimizable_type),
    ('mu_parameters_optimizable', nb.types.ListType(mu_parameters_optimizable_type)),
    ('phi_parameters_optimizable', nb.types.ListType(phi_parameters_optimizable_type)),
    ('theta_parameters_optimizable', nb.types.ListType(theta_parameters_optimizable_type)),
    ('eta_cutoff', nb.float64[:]),
    ('eta_cutoff_optimizable', nb.boolean[:]),
    ('mu_cutoff', nb.float64[:]),
    ('mu_cutoff_optimizable', nb.boolean[:]),
    ('phi_cutoff', nb.float64[:]),
    ('phi_cutoff_optimizable', nb.boolean[:]),
    ('mu_labels', nb.types.ListType(labels_type)),
    ('phi_labels', nb.types.ListType(labels_type)),
    ('max_ee_order', nb.int64),
    ('max_en_order', nb.int64),
    ('mu_cusp', nb.boolean[:]),
    ('phi_cusp', nb.boolean[:]),
    ('phi_irrotational', nb.boolean[:]),
    ('ae_cutoff', nb.float64[:]),
]


@nb.experimental.jitclass(spec)
class Backflow:

    def __init__(
        self, neu, ned, trunc, eta_parameters, eta_parameters_optimizable, eta_mask, eta_cutoff,
        mu_parameters, mu_parameters_optimizable, mu_mask, mu_cutoff, mu_cusp, mu_labels,
        phi_parameters, phi_parameters_optimizable, phi_mask, theta_parameters, theta_parameters_optimizable, theta_mask,
        phi_cutoff, phi_cusp, phi_labels, phi_irrotational, ae_cutoff
    ):
        self.neu = neu
        self.ned = ned
        self.trunc = trunc
        # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        self.eta_cutoff = eta_cutoff['value']
        self.eta_cutoff_optimizable = eta_cutoff['optimizable']
        self.eta_mask = eta_mask
        self.eta_parameters = eta_parameters
        self.eta_parameters_optimizable = eta_parameters_optimizable
        # spin dep (0->u=d; 1->u/=d)
        self.mu_cusp = mu_cusp
        self.mu_labels = mu_labels
        self.mu_cutoff = mu_cutoff['value']
        self.mu_cutoff_optimizable = mu_cutoff['optimizable']
        self.mu_mask = mu_mask
        self.mu_parameters = mu_parameters
        self.mu_parameters_optimizable = mu_parameters_optimizable
        # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
        self.phi_irrotational = phi_irrotational
        self.phi_cusp = phi_cusp
        self.phi_labels = phi_labels
        self.phi_cutoff = phi_cutoff['value']
        self.phi_cutoff_optimizable = phi_cutoff['optimizable']
        self.phi_mask = phi_mask
        self.theta_mask = theta_mask
        self.phi_parameters = phi_parameters
        self.theta_parameters = theta_parameters
        self.phi_parameters_optimizable = phi_parameters_optimizable
        self.theta_parameters_optimizable = theta_parameters_optimizable

        self.max_ee_order = max((
            self.eta_parameters.shape[0],
            max([p.shape[2] for p in self.phi_parameters]) if self.phi_parameters else 0,
        ))
        self.max_en_order = max((
            max([p.shape[0] for p in self.mu_parameters]) if self.mu_parameters else 0,
            max([p.shape[0] for p in self.phi_parameters]) if self.phi_parameters else 0,
            2
        ))
        self.ae_cutoff = ae_cutoff
        self.fix_optimizable()

    def fix_optimizable(self):
        """Set parameter fixed if there is no corresponded spin-pairs"""
        if self.eta_parameters.shape[1] == 2:
            if self.neu == 1 and self.ned == 1:
                self.eta_parameters_optimizable[:, 0] = False
        elif self.eta_parameters.shape[1] == 3:
            if self.neu == 1:
                self.eta_parameters_optimizable[:, 0] = False
            if self.ned == 1:
                self.eta_parameters_optimizable[:, 2] = False

        for mu_parameters, mu_parameters_optimizable in zip(self.mu_parameters, self.mu_parameters_optimizable):
            if mu_parameters.shape[1] == 2:
                if self.neu == 1 and self.ned == 1:
                    mu_parameters_optimizable[:, 0] = False
            elif mu_parameters.shape[1] == 3:
                if self.neu == 1:
                    mu_parameters_optimizable[:, 0] = False
                if self.ned == 1:
                    mu_parameters_optimizable[:, 2] = False

        for phi_parameters, phi_parameters_optimizable in zip(self.phi_parameters, self.phi_parameters_optimizable):
            if phi_parameters.shape[3] == 2:
                if self.neu == 1 and self.ned == 1:
                    phi_parameters_optimizable[:, :, :, 0] = False
            elif phi_parameters.shape[3] == 3:
                if self.neu == 1:
                    phi_parameters_optimizable[:, :, :, 0] = False
                if self.ned == 1:
                    phi_parameters_optimizable[:, :, :, 2] = False

        for theta_parameters, theta_parameters_optimizable in zip(self.theta_parameters, self.theta_parameters_optimizable):
            if theta_parameters.shape[3] == 2:
                if self.neu == 1 and self.ned == 1:
                    theta_parameters_optimizable[:, :, :, 0] = False
            elif theta_parameters.shape[3] == 3:
                if self.neu == 1:
                    theta_parameters_optimizable[:, :, :, 0] = False
                if self.ned == 1:
                    theta_parameters_optimizable[:, :, :, 2] = False

    def ee_powers(self, e_vectors):
        """Powers of e-e distances
        :param e_vectors: e-e vectors - array(nelec, nelec, 3)
        :return:
        """
        res = np.ones(shape=(e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(1, e_vectors.shape[0]):
            for j in range(i):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(1, self.max_ee_order):
                    res[i, j, k] = res[j, i, k] = r_ee ** k
        return res

    def en_powers(self, n_vectors):
        """Powers of e-n distances
        :param n_vectors: e-n vectors - array(natom, nelec, 3)
        :return:
        """
        res = np.ones(shape=(n_vectors.shape[0], n_vectors.shape[1], self.max_en_order))
        for i in range(n_vectors.shape[0]):
            for j in range(n_vectors.shape[1]):
                r_eI = np.linalg.norm(n_vectors[i, j])
                for k in range(1, self.max_en_order):
                    res[i, j, k] = r_eI ** k
        return res

    def ae_multiplier(self, n_vectors, n_powers):
        """Zeroing the backflow displacement at AE atoms."""
        res = np.ones(shape=(self.neu + self.ned, 3))
        for i in range(n_vectors.shape[0]):
            Lg = self.ae_cutoff[i]
            for j in range(self.neu + self.ned):
                r = n_powers[i, j, 1]
                if r < Lg:
                    res[j] = (r/Lg)**2 * (6 - 8 * (r/Lg) + 3 * (r/Lg)**2)
        return res

    def ae_multiplier_gradient(self, n_vectors, n_powers):
        """Zeroing the backflow displacement at AE atoms.
        Gradient of spherically symmetric function (in 3-D space) is:
            ∇(f) = df/dr * r_vec/r
        """
        res = np.zeros(shape=(self.neu + self.ned, 3, self.neu + self.ned, 3))
        for i in range(n_vectors.shape[0]):
            Lg = self.ae_cutoff[i]
            for j in range(self.neu + self.ned):
                r_vec = n_vectors[i, j]
                r = n_powers[i, j, 1]
                if r < Lg:
                    res[j, :, j, :] = 12*r_vec/Lg**2 * (1 - r/Lg)**2
        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    def ae_multiplier_laplacian(self, n_vectors, n_powers):
        """Zeroing the backflow displacement at AE atoms.
        Laplace operator of spherically symmetric function (in 3-D space) is:
            ∇²(f) = d²f/dr² + 2/r * df/dr
        """
        res = np.zeros(shape=(self.neu + self.ned, 3))
        for i in range(n_vectors.shape[0]):
            Lg = self.ae_cutoff[i]
            for j in range(self.neu + self.ned):
                r = n_powers[i, j, 1]
                if r < Lg:
                    res[j] = 12/Lg**2 * (3 - 8 * (r/Lg) + 5 * (r/Lg)**2)
        return res.ravel()

    def eta_term(self, e_vectors, e_powers):
        """
        :param e_vectors: e-e vectors
        :param e_powers: powers of e-e distances
        :return: displacements of electrons - array(nelec, 3)
        """
        res = np.zeros(shape=(self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res

        C = self.trunc
        parameters = self.eta_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_vec = e_vectors[i, j]
                r = e_powers[i, j, 1]
                eta_set = (int(i >= self.neu) + int(j >= self.neu)) % parameters.shape[1]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    poly = 0
                    for k in range(parameters.shape[0]):
                        poly += parameters[k, eta_set] * e_powers[i, j, k]
                    bf = (1 - r/L) ** C * poly * r_vec
                    res[i] += bf
                    res[j] -= bf
        return res

    def mu_term(self, n_vectors, n_powers):
        """
        :param n_vectors: e-n vectors
        :param n_powers: powers of e-n distances
        :return: displacements of electrons - array(2, nelec, 3)
        """
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        if not self.mu_cutoff.any():
            return res

        C = self.trunc
        for parameters, L, mu_labels in zip(self.mu_parameters, self.mu_cutoff, self.mu_labels):
            for i in mu_labels:
                for j in range(self.neu + self.ned):
                    r_vec = n_vectors[i, j]
                    r = n_powers[i, j, 1]
                    if r < L:
                        mu_set = int(j >= self.neu) % parameters.shape[1]
                        poly = 0.0
                        for k in range(parameters.shape[0]):
                            poly += parameters[k, mu_set] * n_powers[i, j, k]
                        # cutoff_condition
                        # 0: AE cutoff definitely not applied
                        # 1: AE cutoff maybe applied
                        ae_cutoff_condition = int(r > self.ae_cutoff[i])
                        res[ae_cutoff_condition, j] += poly * (1 - r / L) ** C * r_vec
        return res

    def phi_term(self, e_powers, n_powers, e_vectors, n_vectors):
        """
        :param e_vectors:
        :param n_vectors:
        :param e_powers:
        :param n_powers:
        :return: displacements of electrons - array(2, nelec, 3)
        """
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        if not self.phi_cutoff.any():
            return res

        C = self.trunc
        for phi_parameters, theta_parameters, L, phi_labels in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_labels):
            for i in phi_labels:
                for j1 in range(self.neu + self.ned):
                    for j2 in range(self.neu + self.ned):
                        if j1 == j2:
                            continue
                        r_e1I_vec = n_vectors[i, j1]
                        r_ee_vec = e_vectors[j1, j2]
                        r_e1I = n_powers[i, j1, 1]
                        r_e2I = n_powers[i, j2, 1]
                        if r_e1I < L and r_e2I < L:
                            phi_set = (int(j1 >= self.neu) + int(j2 >= self.neu)) % phi_parameters.shape[3]
                            phi_poly = theta_poly = 0.0
                            for k in range(phi_parameters.shape[0]):
                                for l in range(phi_parameters.shape[1]):
                                    for m in range(phi_parameters.shape[2]):
                                        poly = n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m]
                                        phi_poly += phi_parameters[k, l, m, phi_set] * poly
                                        theta_poly += theta_parameters[k, l, m, phi_set] * poly
                            # cutoff_condition
                            # 0: AE cutoff definitely not applied
                            # 1: AE cutoff maybe applied
                            ae_cutoff_condition = int(r_e1I > self.ae_cutoff[i])
                            res[ae_cutoff_condition, j1] += (1-r_e1I/L) ** C * (1-r_e2I/L) ** C * (phi_poly * r_ee_vec + theta_poly * r_e1I_vec)
        return res

    def eta_term_gradient(self, e_powers, e_vectors):
        """
        :param e_powers:
        :param e_vectors:
        Gradient of spherically symmetric function (in 3-D space) is df/dr * (x, y, z)
        :return: partial derivatives of displacements of electrons - array(nelec * 3, nelec * 3)
        """
        res = np.zeros(shape=(self.neu + self.ned, 3, self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

        C = self.trunc
        parameters = self.eta_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_vec = e_vectors[i, j]
                r = e_powers[i, j, 1]
                eta_set = (int(i >= self.neu) + int(j >= self.neu)) % parameters.shape[1]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    poly = poly_diff = 0
                    for k in range(parameters.shape[0]):
                        p = parameters[k, eta_set]
                        poly += p * e_powers[i, j, k]
                        if k > 0:
                            poly_diff += p * k * e_powers[i, j, k - 1]

                    bf = (1 - r/L)**C * (
                        (poly_diff - C/(L - r)*poly) * np.outer(r_vec, r_vec)/r + poly * np.eye(3)
                    )
                    res[i, :, i, :] += bf
                    res[i, :, j, :] -= bf
                    res[j, :, i, :] -= bf
                    res[j, :, j, :] += bf

        return res.reshape((self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    def mu_term_gradient(self, n_powers, n_vectors):
        """
        :param n_powers:
        :param n_vectors:
        :return: partial derivatives of displacements of electrons - array(2, nelec * 3, nelec * 3)
        """
        res = np.zeros(shape=(2, self.neu + self.ned, 3, self.neu + self.ned, 3))
        if not self.mu_cutoff.any():
            return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

        C = self.trunc
        for parameters, L, mu_labels in zip(self.mu_parameters, self.mu_cutoff, self.mu_labels):
            for i in mu_labels:
                for j in range(self.neu + self.ned):
                    r_vec = n_vectors[i, j]
                    r = n_powers[i, j, 1]
                    if r < L:
                        mu_set = int(j >= self.neu) % parameters.shape[1]
                        poly = poly_diff = 0.0
                        for k in range(parameters.shape[0]):
                            p = parameters[k, mu_set]
                            poly += p * n_powers[i, j, k]
                            if k > 0:
                                poly_diff += k * p * n_powers[i, j, k-1]

                        # cutoff_condition
                        # 0: AE cutoff definitely not applied
                        # 1: AE cutoff maybe applied
                        ae_cutoff_condition = int(r > self.ae_cutoff[i])
                        res[ae_cutoff_condition, j, :, j, :] += (1 - r/L)**C * (
                            (poly_diff - C/(L - r)*poly) * np.outer(r_vec, r_vec)/r + poly * np.eye(3)
                        )

        return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    def phi_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors):
        """
        :param e_powers:
        :param e_vectors:
        :return: partial derivatives of displacements of electrons - array(2, nelec * 3, nelec * 3)
        """
        res = np.zeros(shape=(2, self.neu + self.ned, 3, self.neu + self.ned, 3))
        if not self.mu_cutoff.any():
            return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

        C = self.trunc
        for phi_parameters, theta_parameters, L, phi_labels in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_labels):
            for i in phi_labels:
                for j1 in range(self.neu + self.ned):
                    for j2 in range(self.neu + self.ned):
                        if j1 == j2:
                            continue
                        r_e1I_vec = n_vectors[i, j1]
                        r_e2I_vec = n_vectors[i, j2]
                        r_ee_vec = e_vectors[j1, j2]
                        r_e1I = n_powers[i, j1, 1]
                        r_e2I = n_powers[i, j2, 1]
                        r_ee = e_powers[j1, j2, 1]
                        if r_e1I < L and r_e2I < L:
                            phi_set = (int(j1 >= self.neu) + int(j2 >= self.neu)) % phi_parameters.shape[3]
                            phi_poly = phi_poly_diff_e1I = phi_poly_diff_e2I = phi_poly_diff_ee = 0.0
                            theta_poly = theta_poly_diff_e1I = theta_poly_diff_e2I = theta_poly_diff_ee = 0.0
                            for k in range(phi_parameters.shape[0]):
                                for l in range(phi_parameters.shape[1]):
                                    for m in range(phi_parameters.shape[2]):
                                        phi_p = phi_parameters[k, l, m, phi_set]
                                        theta_p = theta_parameters[k, l, m, phi_set]
                                        phi_poly += n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m] * phi_p
                                        theta_poly += n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m] * theta_p
                                        if k > 0:
                                            poly_diff_e1I = k * n_powers[i, j1, k-1] * n_powers[i, j2, l] * e_powers[j1, j2, m]
                                            phi_poly_diff_e1I += poly_diff_e1I * phi_p
                                            theta_poly_diff_e1I += poly_diff_e1I * theta_p
                                        if l > 0:
                                            poly_diff_e2I = l * n_powers[i, j1, k] * n_powers[i, j2, l-1] * e_powers[j1, j2, m]
                                            phi_poly_diff_e2I += poly_diff_e2I * phi_p
                                            theta_poly_diff_e2I += poly_diff_e2I * theta_p
                                        if m > 0:
                                            poly_diff_ee = m * n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m-1]
                                            phi_poly_diff_ee += poly_diff_ee * phi_p
                                            theta_poly_diff_ee += poly_diff_ee * theta_p

                            # cutoff_condition
                            # 0: AE cutoff definitely not applied
                            # 1: AE cutoff maybe applied
                            ae_cutoff_condition = int(r_e1I > self.ae_cutoff[i])
                            res[ae_cutoff_condition, j1, :, j1, :] += (1-r_e1I/L) ** C * (1-r_e2I/L) ** C * (
                                (phi_poly_diff_e1I - C/(L - r_e1I)*phi_poly) * np.outer(r_ee_vec, r_e1I_vec)/r_e1I +
                                phi_poly_diff_ee * np.outer(r_ee_vec, r_ee_vec) / r_ee +
                                phi_poly * np.eye(3) +
                                (theta_poly_diff_e1I - C / (L - r_e1I) * theta_poly) * np.outer(r_e1I_vec, r_e1I_vec) / r_e1I +
                                theta_poly_diff_ee * np.outer(r_e1I_vec, r_ee_vec) / r_ee +
                                theta_poly * np.eye(3)
                            )
                            res[ae_cutoff_condition, j1, :, j2, :] += (1-r_e1I/L) ** C * (1-r_e2I/L) ** C * (
                                (phi_poly_diff_e2I - C/(L - r_e2I)*phi_poly) * np.outer(r_ee_vec, r_e2I_vec)/r_e2I -
                                phi_poly_diff_ee * np.outer(r_ee_vec, r_ee_vec) / r_ee -
                                phi_poly * np.eye(3) +
                                (theta_poly_diff_e2I - C / (L - r_e2I) * theta_poly) * np.outer(r_e1I_vec, r_e2I_vec) / r_e2I -
                                theta_poly_diff_ee * np.outer(r_e1I_vec, r_ee_vec) / r_ee
                            )

        return res.reshape(2, (self.neu + self.ned) * 3, (self.neu + self.ned) * 3)

    def eta_term_laplacian(self, e_powers, e_vectors):
        """
        :param e_powers:
        :param e_vectors:
        Laplace operator of spherically symmetric function (in 3-D space) is
            ∇²(f) = d²f/dr² + 2/r * df/dr
        :return: vector laplacian - array(nelec * 3)
        """
        res = np.zeros(shape=(self.neu + self.ned, 3))
        if not self.eta_cutoff.any():
            return res.ravel()

        C = self.trunc
        parameters = self.eta_parameters
        for i in range(1, self.neu + self.ned):
            for j in range(i):
                r_vec = e_vectors[i, j]
                r = e_powers[i, j, 1]
                eta_set = (int(i >= self.neu) + int(j >= self.neu)) % parameters.shape[1]
                L = self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
                if r < L:
                    poly = poly_diff = poly_diff_2 = 0
                    for k in range(parameters.shape[0]):
                        p = parameters[k, eta_set]
                        poly += p * e_powers[i, j, k]
                        if k > 0:
                            poly_diff += k * p * e_powers[i, j, k-1]
                        if k > 1:
                            poly_diff_2 += k * (k - 1) * p * e_powers[i, j, k-2]

                    bf = 2 * (1 - r/L)**C * (
                        4*(poly_diff - C/(L - r) * poly) +
                        r*(C*(C - 1)/(L - r)**2*poly - 2*C/(L - r)*poly_diff + poly_diff_2)
                    ) * r_vec/r
                    res[i] += bf
                    res[j] -= bf

        return res.ravel()

    def mu_term_laplacian(self, n_powers, n_vectors):
        """
        :param e_powers:
        :param e_vectors:
        Laplace operator of spherically symmetric function (in 3-D space) is
            ∇²(f) = d²f/dr² + 2/r * df/dr
        :return: vector laplacian - array(2, nelec * 3)
        """
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        if not self.mu_cutoff.any():
            return res.reshape(2, (self.neu + self.ned) * 3)

        C = self.trunc
        for parameters, L, mu_labels in zip(self.mu_parameters, self.mu_cutoff, self.mu_labels):
            for i in mu_labels:
                for j in range(self.neu + self.ned):
                    r_vec = n_vectors[i, j]
                    r = n_powers[i, j, 1]
                    if r < L:
                        mu_set = int(j >= self.neu) % parameters.shape[1]
                        poly = poly_diff = poly_diff_2 = 0.0
                        for k in range(parameters.shape[0]):
                            p = parameters[k, mu_set]
                            poly += p * n_powers[i, j, k]
                            if k > 0:
                                poly_diff += k * p * n_powers[i, j, k-1]
                            if k > 1:
                                poly_diff_2 += k * (k-1) * p * n_powers[i, j, k-2]

                        # cutoff_condition
                        # 0: AE cutoff definitely not applied
                        # 1: AE cutoff maybe applied
                        ae_cutoff_condition = int(r > self.ae_cutoff[i])
                        res[ae_cutoff_condition, j] += (1 - r/L)**C * (
                            4*(poly_diff - C/(L - r) * poly) +
                            r*(C*(C - 1)/(L - r)**2*poly - 2*C/(L - r)*poly_diff + poly_diff_2)
                        ) * r_vec/r

        return res.reshape(2, (self.neu + self.ned) * 3)

    def phi_term_laplacian(self, e_powers, n_powers, e_vectors, n_vectors):
        """
        :param e_powers:
        :param e_vectors:
        phi-term is a product of two spherically symmetric functions f(r_eI) and g(r_ee) so using
            ∇²(f*g) = ∇²(f)*g + 2*∇(f)*∇(g) + f*∇²(g)
        Laplace operator of spherically symmetric function (in 3-D space) is
            ∇²(f) = d²f/dr² + 2/r * df/dr
        :return: vector laplacian - array(2, nelec * 3)
        """
        res = np.zeros(shape=(2, self.neu + self.ned, 3))
        if not self.phi_cutoff.any():
            return res.reshape(2, (self.neu + self.ned) * 3)

        C = self.trunc
        for phi_parameters, theta_parameters, L, phi_labels in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_labels):
            for i in phi_labels:
                for j1 in range(self.neu + self.ned):
                    for j2 in range(self.neu + self.ned):
                        if j1 == j2:
                            continue
                        r_e1I_vec = n_vectors[i, j1]
                        r_e2I_vec = n_vectors[i, j2]
                        r_ee_vec = e_vectors[j1, j2]
                        r_e1I = n_powers[i, j1, 1]
                        r_e2I = n_powers[i, j2, 1]
                        r_ee = e_powers[j1, j2, 1]
                        if r_e1I < L and r_e2I < L:
                            phi_set = (int(j1 >= self.neu) + int(j2 >= self.neu)) % phi_parameters.shape[3]
                            phi_poly = phi_poly_diff_e1I = phi_poly_diff_e2I = phi_poly_diff_ee = 0.0
                            phi_poly_diff_e1I_2 = phi_poly_diff_e2I_2 = phi_poly_diff_ee_2 = 0.0
                            phi_poly_diff_e1I_ee = phi_poly_diff_e2I_ee = 0.0
                            theta_poly = theta_poly_diff_e1I = theta_poly_diff_e2I = theta_poly_diff_ee = 0.0
                            theta_poly_diff_e1I_2 = theta_poly_diff_e2I_2 = theta_poly_diff_ee_2 = 0.0
                            theta_poly_diff_e1I_ee = theta_poly_diff_e2I_ee = 0.0
                            for k in range(phi_parameters.shape[0]):
                                for l in range(phi_parameters.shape[1]):
                                    for m in range(phi_parameters.shape[2]):
                                        phi_p = phi_parameters[k, l, m, phi_set]
                                        theta_p = theta_parameters[k, l, m, phi_set]
                                        poly = n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m]
                                        phi_poly += poly * phi_p
                                        theta_poly += poly * theta_p
                                        if k > 0:
                                            poly_diff_e1I = k * n_powers[i, j1, k-1] * n_powers[i, j2, l] * e_powers[j1, j2, m]
                                            phi_poly_diff_e1I += poly_diff_e1I * phi_p
                                            theta_poly_diff_e1I += poly_diff_e1I * theta_p
                                        if l > 0:
                                            poly_diff_e2I = l * n_powers[i, j1, k] * n_powers[i, j2, l-1] * e_powers[j1, j2, m]
                                            phi_poly_diff_e2I += poly_diff_e2I * phi_p
                                            theta_poly_diff_e2I += poly_diff_e2I * theta_p
                                        if m > 0:
                                            poly_diff_ee = m * n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m-1]
                                            phi_poly_diff_ee += poly_diff_ee * phi_p
                                            theta_poly_diff_ee += poly_diff_ee * theta_p
                                        if k > 1:
                                            poly_diff_e1I_2 = k * (k-1) * n_powers[i, j1, k-2] * n_powers[i, j2, l] * e_powers[j1, j2, m]
                                            phi_poly_diff_e1I_2 += poly_diff_e1I_2 * phi_p
                                            theta_poly_diff_e1I_2 += poly_diff_e1I_2 * theta_p
                                        if l > 1:
                                            poly_diff_e2I_2 = l * (l-1) * n_powers[i, j1, k] * n_powers[i, j2, l-2] * e_powers[j1, j2, m]
                                            phi_poly_diff_e2I_2 += poly_diff_e2I_2 * phi_p
                                            theta_poly_diff_e2I_2 += poly_diff_e2I_2 * theta_p
                                        if m > 1:
                                            poly_diff_ee_2 = m * (m-1) * n_powers[i, j1, k] * n_powers[i, j2, l] * e_powers[j1, j2, m-2]
                                            phi_poly_diff_ee_2 += poly_diff_ee_2 * phi_p
                                            theta_poly_diff_ee_2 += poly_diff_ee_2 * theta_p
                                        if k > 0 and m > 0:
                                            poly_diff_e1I_ee = k * m * n_powers[i, j1, k-1] * n_powers[i, j2, l] * e_powers[j1, j2, m-1]
                                            phi_poly_diff_e1I_ee += poly_diff_e1I_ee * phi_p
                                            theta_poly_diff_e1I_ee += poly_diff_e1I_ee * theta_p
                                        if l > 0 and m > 0:
                                            poly_diff_e2I_ee = l * m * n_powers[i, j1, k] * n_powers[i, j2, l-1] * e_powers[j1, j2, m-1]
                                            phi_poly_diff_e2I_ee += poly_diff_e2I_ee * phi_p
                                            theta_poly_diff_e2I_ee += poly_diff_e2I_ee * theta_p

                            phi_diff_1 = (
                                (phi_poly_diff_e1I - C*phi_poly/(L - r_e1I))/r_e1I +
                                (phi_poly_diff_e2I - C*phi_poly/(L - r_e2I))/r_e2I +
                                4 * phi_poly_diff_ee/r_ee
                            )
                            phi_diff_2 = (
                                (C*(C - 1)*phi_poly/(L - r_e1I)**2 - 2*C*phi_poly_diff_e1I/(L - r_e1I) + phi_poly_diff_e1I_2) +
                                (C*(C - 1)*phi_poly/(L - r_e2I)**2 - 2*C*phi_poly_diff_e2I/(L - r_e2I) + phi_poly_diff_e2I_2) +
                                2 * phi_poly_diff_ee_2
                            )
                            phi_dot_product = (
                                (phi_poly_diff_e1I - C*phi_poly/(L - r_e1I)) * np.eye(3) @ r_e1I_vec/r_e1I -
                                (phi_poly_diff_e2I - C*phi_poly/(L - r_e2I)) * np.eye(3) @ r_e2I_vec/r_e2I +
                                (phi_poly_diff_e1I_ee - C*phi_poly_diff_ee/(L - r_e1I)) * np.outer(r_ee_vec, r_ee_vec)/r_ee @ r_e1I_vec/r_e1I -
                                (phi_poly_diff_e2I_ee - C*phi_poly_diff_ee/(L - r_e2I)) * np.outer(r_ee_vec, r_ee_vec)/r_ee @ r_e2I_vec/r_e2I
                            )
                            theta_diff_1 = (
                                2 * (theta_poly_diff_e1I - C*theta_poly/(L - r_e1I))/r_e1I +
                                (theta_poly_diff_e2I - C*theta_poly/(L - r_e2I))/r_e2I +
                                2 * theta_poly_diff_ee/r_ee
                            )
                            theta_diff_2 = (
                                (C*(C - 1)*theta_poly/(L - r_e1I)**2 - 2*C*theta_poly_diff_e1I/(L - r_e1I) + theta_poly_diff_e1I_2) +
                                (C*(C - 1)*theta_poly/(L - r_e2I)**2 - 2*C*theta_poly_diff_e2I/(L - r_e2I) + theta_poly_diff_e2I_2) +
                                2 * theta_poly_diff_ee_2
                            )
                            theta_dot_product = (
                                (theta_poly_diff_e1I_ee - C*theta_poly_diff_ee/(L - r_e1I)) * np.outer(r_e1I_vec, r_e1I_vec)/r_e1I @ r_ee_vec/r_ee -
                                (theta_poly_diff_e2I_ee - C*theta_poly_diff_ee/(L - r_e2I)) * np.outer(r_e1I_vec, r_e2I_vec)/r_e2I @ r_ee_vec/r_ee +
                                theta_poly_diff_ee * np.eye(3) @ r_ee_vec/r_ee
                            )
                            # cutoff_condition
                            # 0: AE cutoff definitely not applied
                            # 1: AE cutoff maybe applied
                            ae_cutoff_condition = int(r_e1I > self.ae_cutoff[i])
                            res[ae_cutoff_condition, j1] += (1-r_e1I/L)**C * (1-r_e2I/L) ** C * (
                                (phi_diff_2 + 2 * phi_diff_1) * r_ee_vec + 2 * phi_dot_product +
                                (theta_diff_2 + 2 * theta_diff_1) * r_e1I_vec + 2 * theta_dot_product
                            )

        return res.reshape(2, (self.neu + self.ned) * 3)

    def value(self, e_vectors, n_vectors):
        """Backflow displacemets
        :param e_vectors:
        :param n_vectors:
        :return: backflow displacement array(nelec * 3)
        """

        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        eta_term = self.eta_term(e_vectors, e_powers)
        mu_term = self.mu_term(n_vectors, n_powers)
        phi_term = self.phi_term(e_powers, n_powers, e_vectors, n_vectors)

        ae_value = eta_term + mu_term[1] + phi_term[1]

        return (
            ae_value * self.ae_multiplier(n_vectors, n_powers) +
            mu_term[0] + phi_term[0]
        ) + n_vectors

    def gradient(self, e_vectors, n_vectors):
        """Gradient with respect to e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        eta_term = self.eta_term(e_vectors, e_powers)
        mu_term = self.mu_term(n_vectors, n_powers)
        phi_term = self.phi_term(e_powers, n_powers, e_vectors, n_vectors)

        eta_term_gradient = self.eta_term_gradient(e_powers, e_vectors)
        mu_term_gradient = self.mu_term_gradient(n_powers, n_vectors)
        phi_term_gradient = self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)

        ae_value = eta_term + mu_term[1] + phi_term[1]
        ae_gradient = eta_term_gradient + mu_term_gradient[1] + phi_term_gradient[1]

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)
        ae_multiplier_gradient = self.ae_multiplier_gradient(n_vectors, n_powers)

        value = (
            ae_value * ae_multiplier +
            mu_term[0] + phi_term[0]
        ) + n_vectors

        gradient = (
            ae_multiplier_gradient * ae_value.reshape((-1, 1)) +
            ae_gradient * ae_multiplier.reshape((-1, 1)) +
            mu_term_gradient[0] + phi_term_gradient[0]
        ) + np.eye((self.neu + self.ned) * 3)

        return gradient, value

    def laplacian(self, e_vectors, n_vectors):
        """Backflow laplacian, gradient, value
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        eta_term = self.eta_term(e_vectors, e_powers)
        mu_term = self.mu_term(n_vectors, n_powers)
        phi_term = self.phi_term(e_powers, n_powers, e_vectors, n_vectors)

        eta_term_gradient = self.eta_term_gradient(e_powers, e_vectors)
        mu_term_gradient = self.mu_term_gradient(n_powers, n_vectors)
        phi_term_gradient = self.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)

        eta_term_laplacian = self.eta_term_laplacian(e_powers, e_vectors)
        mu_term_laplacian = self.mu_term_laplacian(n_powers, n_vectors)
        phi_term_laplacian = self.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)

        ae_value = eta_term + mu_term[1] + phi_term[1]
        ae_gradient = eta_term_gradient + mu_term_gradient[1] + phi_term_gradient[1]
        ae_laplacian = eta_term_laplacian + mu_term_laplacian[1] + phi_term_laplacian[1]

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)
        ae_multiplier_gradient = self.ae_multiplier_gradient(n_vectors, n_powers)
        ae_multiplier_laplacian = self.ae_multiplier_laplacian(n_vectors, n_powers)

        value = (
            ae_value * ae_multiplier +
            mu_term[0] + phi_term[0]
        ) + n_vectors

        gradient = (
            ae_multiplier_gradient * ae_value.reshape((-1, 1)) +
            ae_gradient * ae_multiplier.reshape((-1, 1)) +
            mu_term_gradient[0] + phi_term_gradient[0]
        ) + np.eye((self.neu + self.ned) * 3)

        laplacian = (
            ae_multiplier_laplacian * ae_value.ravel() +
            2 * (ae_gradient * ae_multiplier_gradient).sum(axis=1) +
            ae_laplacian * ae_multiplier.ravel() +
            mu_term_laplacian[0] + phi_term_laplacian[0]
        )

        return laplacian, gradient, value

    def numerical_gradient(self, e_vectors, n_vectors):
        """Numerical gradient with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return: partial derivatives of displacements of electrons - array(nelec * 3, nelec * 3)
        """
        delta = 0.00001

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

    def numerical_laplacian(self, e_vectors, n_vectors):
        """Numerical laplacian with respect to a e-coordinates
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :return: vector laplacian - array(nelec * 3)
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

        return res.ravel() / delta / delta

    def fix_eta_parameters(self):
        """Fix eta-term parameters"""
        C = self.trunc
        L = self.eta_cutoff[0]
        self.eta_parameters[1, 0] = C * self.eta_parameters[0, 0] / L
        if self.eta_parameters.shape[1] == 3:
            L = self.eta_cutoff[2] or self.eta_cutoff[0]
            self.eta_parameters[1, 2] = C * self.eta_parameters[0, 2] / L

    def fix_mu_parameters(self):
        """Fix mu-term parameters"""

    def fix_phi_parameters(self):
        """Fix phi-term parameters"""
        for phi_parameters, theta_parameters, phi_cutoff, phi_cusp, phi_irrotational in zip(self.phi_parameters, self.theta_parameters, self.phi_cutoff, self.phi_cusp, self.phi_irrotational):
            for spin_dep in range(phi_parameters.shape[3]):
                c = construct_c_matrix(self.trunc, phi_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational)
                c, pivot_positions = rref(c)
                c = c[:pivot_positions.size, :]

                b = np.zeros((c.shape[0], ))
                p = 0
                for m in range(phi_parameters.shape[2]):
                    for l in range(phi_parameters.shape[1]):
                        for k in range(phi_parameters.shape[0]):
                            if p not in pivot_positions:
                                for temp in range(c.shape[0]):
                                    b[temp] -= c[temp, p] * phi_parameters[k, l, m, spin_dep]
                            p += 1

                for m in range(phi_parameters.shape[2]):
                    for l in range(phi_parameters.shape[1]):
                        for k in range(phi_parameters.shape[0]):
                            if p not in pivot_positions:
                                for temp in range(c.shape[0]):
                                    b[temp] -= c[temp, p] * theta_parameters[k, l, m, spin_dep]
                            p += 1

                x = np.linalg.solve(c[:, pivot_positions], b)

                p = 0
                temp = 0
                for m in range(phi_parameters.shape[2]):
                    for l in range(phi_parameters.shape[1]):
                        for k in range(phi_parameters.shape[0]):
                            if temp in pivot_positions:
                                phi_parameters[k, l, m, spin_dep] = x[p]
                                p += 1
                            temp += 1

                for m in range(phi_parameters.shape[2]):
                    for l in range(phi_parameters.shape[1]):
                        for k in range(phi_parameters.shape[0]):
                            if temp in pivot_positions:
                                theta_parameters[k, l, m, spin_dep] = x[p]
                                p += 1
                            temp += 1

    def get_parameters_bounds(self):
        """Bonds constraints fluctuation of Backflow parameters
        and thus increases robustness of the energy minimization procedure.
        :return:
        """
        scale = self.get_parameters_scale()
        parameters = self.get_parameters()
        return parameters - scale, parameters + scale

    def get_parameters_scale(self):
        """Characteristic scale of each variable. Setting x_scale is equivalent
        to reformulating the problem in scaled variables xs = x / x_scale.
        An alternative view is that the size of a trust region along j-th
        dimension is proportional to x_scale[j].
        The purpose of this method is to reformulate the optimization problem
        with dimensionless variables having only one dimensional parameter - cutoff length.
        """
        res = []
        scale = 1.5  # AU
        if self.eta_cutoff.any():
            for eta_cutoff, eta_cutoff_optimizable in zip(self.eta_cutoff, self.eta_cutoff_optimizable):
                if eta_cutoff_optimizable:
                    res.append(1)
            for j1 in range(self.eta_parameters.shape[0]):
                for j2 in range(self.eta_parameters.shape[1]):
                    if self.eta_mask[j1, j2] and self.eta_parameters_optimizable[j1, j2]:
                        res.append(1 / self.eta_cutoff[j2 % self.eta_cutoff.shape[0]] ** j1)

        if self.mu_cutoff.any():
            for i, (mu_parameters, mu_parameters_optimizable, mu_mask, mu_cutoff, mu_cutoff_optimizable) in enumerate(zip(self.mu_parameters, self.mu_parameters_optimizable, self.mu_mask, self.mu_cutoff, self.mu_cutoff_optimizable)):
                if mu_cutoff_optimizable:
                    res.append(1)
                for j1 in range(mu_parameters.shape[0]):
                    for j2 in range(mu_parameters.shape[1]):
                        if mu_mask[j1, j2] and mu_parameters_optimizable[j1, j2]:
                            res.append(1 / mu_cutoff ** j1)

        if self.phi_cutoff.any():
            for i, (phi_parameters, phi_parameters_optimizable, theta_parameters_optimizable, phi_mask, theta_mask, phi_cutoff, phi_cutoff_optimizable) in enumerate(zip(self.phi_parameters, self.phi_parameters_optimizable, self.theta_parameters_optimizable, self.phi_mask, self.theta_mask, self.phi_cutoff, self.phi_cutoff_optimizable)):
                if phi_cutoff_optimizable:
                    res.append(1)
                for j1 in range(phi_parameters.shape[0]):
                    for j2 in range(phi_parameters.shape[1]):
                        for j3 in range(phi_parameters.shape[2]):
                            for j4 in range(phi_parameters.shape[3]):
                                if phi_mask[j1, j2, j3, j4] and phi_parameters_optimizable[j1, j2, j3, j4]:
                                    res.append(1 / phi_cutoff ** (j1 + j2 + j3))

                for j1 in range(phi_parameters.shape[0]):
                    for j2 in range(phi_parameters.shape[1]):
                        for j3 in range(phi_parameters.shape[2]):
                            for j4 in range(phi_parameters.shape[3]):
                                if theta_mask[j1, j2, j3, j4] and theta_parameters_optimizable[j1, j2, j3, j4]:
                                    res.append(1 / phi_cutoff ** (j1 + j2 + j3))

        return np.array(res)

    def get_parameters(self):
        """Returns parameters in the following order:
        eta-cutoff(s), eta-linear parameters,
        for every mu-set: mu-cutoff, mu-linear parameters,
        for every phi/theta-set: phi-cutoff, phi-linear parameters, theta-linear parameters.
        :return:
        """
        res = []
        if self.eta_cutoff.any():
            for eta_cutoff, eta_cutoff_optimizable in zip(self.eta_cutoff, self.eta_cutoff_optimizable):
                if eta_cutoff_optimizable:
                    res.append(eta_cutoff)
            res += list(self.eta_parameters.ravel()[(self.eta_mask & self.eta_parameters_optimizable).ravel()])
        if self.mu_cutoff.any():
            for mu_parameters, mu_parameters_optimizable, mu_mask, mu_cutoff, mu_cutoff_optimizable in zip(self.mu_parameters, self.mu_parameters_optimizable, self.mu_mask, self.mu_cutoff, self.mu_cutoff_optimizable):
                if mu_cutoff_optimizable:
                    res.append(mu_cutoff)
                res += list(mu_parameters.ravel()[(mu_mask & mu_parameters_optimizable).ravel()])
        if self.phi_cutoff.any():
            for phi_parameters, phi_parameters_optimizable, theta_parameters, theta_parameters_optimizable, phi_mask, theta_mask, phi_cutoff, phi_cutoff_optimizable in zip(self.phi_parameters, self.phi_parameters_optimizable, self.theta_parameters, self.theta_parameters_optimizable, self.phi_mask, self.theta_mask, self.phi_cutoff, self.phi_cutoff_optimizable):
                if phi_cutoff_optimizable:
                    res.append(phi_cutoff)
                res += list(phi_parameters.ravel()[(phi_mask & phi_parameters_optimizable).ravel()])
                res += list(theta_parameters.ravel()[(theta_mask & theta_parameters_optimizable).ravel()])

        return np.array(res)

    def set_parameters(self, parameters):
        """Set parameters in the following order:
        eta-cutoff(s), eta-linear parameters,
        for every mu-set: mu-cutoff, mu-linear parameters,
        for every phi/theta-set: phi-cutoff, phi-linear parameters, theta-linear parameters.
        :param parameters:
        :return:
        """
        n = 0
        if self.eta_cutoff.any():
            for j1 in range(self.eta_cutoff.shape[0]):
                if self.eta_cutoff_optimizable[j1]:
                    self.eta_cutoff[j1] = parameters[n]
                    n += 1
            for j1 in range(self.eta_parameters.shape[0]):
                for j2 in range(self.eta_parameters.shape[1]):
                    if self.eta_mask[j1, j2] and self.eta_parameters_optimizable[j1, j2]:
                        self.eta_parameters[j1, j2] = parameters[n]
                        n += 1
            self.fix_eta_parameters()

        if self.mu_cutoff.any():
            for i, (mu_parameters, mu_parameters_optimizable, mu_mask, mu_cutoff_optimizable) in enumerate(zip(self.mu_parameters, self.mu_parameters_optimizable, self.mu_mask, self.mu_cutoff_optimizable)):
                if mu_cutoff_optimizable:
                    self.mu_cutoff[i] = parameters[n]
                    n += 1
                for j1 in range(mu_parameters.shape[0]):
                    for j2 in range(mu_parameters.shape[1]):
                        if mu_mask[j1, j2] and mu_parameters_optimizable[j1, j2]:
                            mu_parameters[j1, j2] = parameters[n]
                            n += 1
            self.fix_mu_parameters()

        if self.phi_cutoff.any():
            for i, (phi_parameters, phi_parameters_optimizable, theta_parameters, theta_parameters_optimizable, phi_mask, theta_mask, phi_cutoff_optimizable) in enumerate(zip(self.phi_parameters, self.phi_parameters_optimizable, self.theta_parameters, self.theta_parameters_optimizable, self.phi_mask, self.theta_mask, self.phi_cutoff_optimizable)):
                if phi_cutoff_optimizable:
                    self.phi_cutoff[i] = parameters[n]
                    n += 1
                for j1 in range(phi_parameters.shape[0]):
                    for j2 in range(phi_parameters.shape[1]):
                        for j3 in range(phi_parameters.shape[2]):
                            for j4 in range(phi_parameters.shape[3]):
                                if phi_mask[j1, j2, j3, j4] and phi_parameters_optimizable[j1, j2, j3, j4]:
                                    phi_parameters[j1, j2, j3, j4] = parameters[n]
                                    n += 1

                for j1 in range(phi_parameters.shape[0]):
                    for j2 in range(phi_parameters.shape[1]):
                        for j3 in range(phi_parameters.shape[2]):
                            for j4 in range(phi_parameters.shape[3]):
                                if theta_mask[j1, j2, j3, j4] and theta_parameters_optimizable[j1, j2, j3, j4]:
                                    theta_parameters[j1, j2, j3, j4] = parameters[n]
                                    n += 1
            self.fix_phi_parameters()

        return parameters[n:]

    def eta_term_numerical_d1(self, e_vectors, e_powers):
        """Numerical first derivatives of logarithm wfn with respect to eta-term parameters
        :param e_vectors: e-e vectors
        :param e_powers: powers of e-e distances
        """
        if not self.eta_cutoff.any():
            return np.zeros(shape=(0, (self.neu + self.ned) * 3))

        delta = 0.00001
        size = (self.eta_mask & self.eta_parameters_optimizable).sum() + self.eta_cutoff_optimizable.sum()
        res = np.zeros(shape=(size, self.neu + self.ned, 3))

        n = -1
        for i in range(self.eta_cutoff.shape[0]):
            if self.eta_cutoff_optimizable[i]:
                n += 1
                self.eta_cutoff -= delta
                self.fix_eta_parameters()
                res[n] -= self.eta_term(e_vectors, e_powers)
                self.eta_cutoff += 2 * delta
                self.fix_eta_parameters()
                res[n] += self.eta_term(e_vectors, e_powers)
                self.eta_cutoff -= delta

        for j1 in range(self.eta_parameters.shape[0]):
            for j2 in range(self.eta_parameters.shape[1]):
                if self.eta_mask[j1, j2] and self.eta_parameters_optimizable[j1, j2]:
                    n += 1
                    self.eta_parameters[j1, j2] -= delta
                    self.fix_eta_parameters()
                    res[n] -= self.eta_term(e_vectors, e_powers)
                    self.eta_parameters[j1, j2] += 2 * delta
                    self.fix_eta_parameters()
                    res[n] += self.eta_term(e_vectors, e_powers)
                    self.eta_parameters[j1, j2] -= delta

        self.fix_eta_parameters()
        return res.reshape(size, (self.neu + self.ned) * 3) / delta / 2

    def mu_term_numerical_d1(self, n_vectors, n_powers):
        """Numerical first derivatives of logarithm wfn with respect to mu-term parameters
        :param n_vectors: e-n vectors
        :param n_powers: powers of e-n distances
        """
        if not self.mu_cutoff.any():
            return np.zeros(shape=(0, (self.neu + self.ned) * 3))

        delta = 0.00001
        size = sum([
            (mu_mask & mu_parameters_optimizable).sum() + mu_cutoff_optimizable
            for mu_parameters_optimizable, mu_mask, mu_cutoff_optimizable
            in zip(self.mu_parameters_optimizable, self.mu_mask, self.mu_cutoff_optimizable)
        ])
        res = np.zeros(shape=(size, self.neu + self.ned, 3))

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)
        n = -1
        for i, (mu_parameters, mu_parameters_optimizable, mu_mask) in enumerate(zip(self.mu_parameters, self.mu_parameters_optimizable, self.mu_mask)):
            if self.mu_cutoff_optimizable[i]:
                n += 1
                self.mu_cutoff[i] -= delta
                self.fix_mu_parameters()
                res[n] -= self.mu_term(n_vectors, n_powers)[0]
                self.mu_cutoff[i] += 2 * delta
                self.fix_mu_parameters()
                res[n] += self.mu_term(n_vectors, n_powers)[0]
                self.mu_cutoff[i] -= delta

            for j1 in range(mu_parameters.shape[0]):
                for j2 in range(mu_parameters.shape[1]):
                    if mu_mask[j1, j2] and mu_parameters_optimizable[j1, j2]:
                        n += 1
                        mu_parameters[j1, j2] -= delta
                        self.fix_mu_parameters()
                        res[n] -= self.mu_term(n_vectors, n_powers)[0]
                        mu_parameters[j1, j2] += 2 * delta
                        self.fix_mu_parameters()
                        res[n] += self.mu_term(n_vectors, n_powers)[0]
                        mu_parameters[j1, j2] -= delta

        return res.reshape(size, (self.neu + self.ned) * 3) / delta / 2

    def phi_term_numerical_d1(self, e_powers, n_powers, e_vectors, n_vectors):
        """Numerical first derivatives of logarithm wfn with respect to phi-term parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        :param e_powers: powers of e-e distances
        :param n_powers: powers of e-n distances
        """
        if not self.phi_cutoff.any():
            return np.zeros(shape=(0, (self.neu + self.ned) * 3))

        delta = 0.00001
        size = sum([
            (phi_mask & phi_parameters_optimizable).sum() + phi_cutoff_optimizable
            for phi_parameters_optimizable, phi_mask, phi_cutoff_optimizable
            in zip(self.phi_parameters_optimizable, self.phi_mask, self.phi_cutoff_optimizable)
        ])
        size += sum([
            (theta_mask & theta_parameters_optimizable).sum()
            for theta_parameters_optimizable, theta_mask
            in zip(self.theta_parameters_optimizable, self.theta_mask)
        ])
        res = np.zeros(shape=(size, self.neu + self.ned, 3))

        ae_multiplier = self.ae_multiplier(n_vectors, n_powers)
        n = -1
        for i, (phi_parameters, phi_parameters_optimizable, theta_parameters, theta_parameters_optimizable, phi_mask, theta_mask) in enumerate(zip(self.phi_parameters, self.phi_parameters_optimizable, self.theta_parameters, self.theta_parameters_optimizable, self.phi_mask, self.theta_mask)):
            if self.phi_cutoff_optimizable[i]:
                n += 1
                self.phi_cutoff[i] -= delta
                self.fix_phi_parameters()
                # FIXME: ae_multiplier
                res[n] -= self.phi_term(e_powers, n_powers, e_vectors, n_vectors)[0]
                self.phi_cutoff[i] += 2 * delta
                self.fix_phi_parameters()
                # FIXME: ae_multiplier
                res[n] += self.phi_term(e_powers, n_powers, e_vectors, n_vectors)[0]
                self.phi_cutoff[i] -= delta

            for j1 in range(phi_parameters.shape[0]):
                for j2 in range(phi_parameters.shape[1]):
                    for j3 in range(phi_parameters.shape[2]):
                        for j4 in range(phi_parameters.shape[3]):
                            if phi_mask[j1, j2, j3, j4] and phi_parameters_optimizable[j1, j2, j3, j4]:
                                n += 1
                                phi_parameters[j1, j2, j3, j4] -= delta
                                self.fix_phi_parameters()
                                res[n] -= self.phi_term(e_powers, n_powers, e_vectors, n_vectors)[0]
                                phi_parameters[j1, j2, j3, j4] += 2 * delta
                                self.fix_phi_parameters()
                                res[n] += self.phi_term(e_powers, n_powers, e_vectors, n_vectors)[0]
                                phi_parameters[j1, j2, j3, j4] -= delta

            for j1 in range(phi_parameters.shape[0]):
                for j2 in range(phi_parameters.shape[1]):
                    for j3 in range(phi_parameters.shape[2]):
                        for j4 in range(phi_parameters.shape[3]):
                            if theta_mask[j1, j2, j3, j4] and theta_parameters_optimizable[j1, j2, j3, j4]:
                                n += 1
                                theta_parameters[j1, j2, j3, j4] -= delta
                                self.fix_phi_parameters()
                                res[n] -= self.phi_term(e_powers, n_powers, e_vectors, n_vectors)[0]
                                theta_parameters[j1, j2, j3, j4] += 2 * delta
                                self.fix_phi_parameters()
                                res[n] += self.phi_term(e_powers, n_powers, e_vectors, n_vectors)[0]
                                theta_parameters[j1, j2, j3, j4] -= delta

        return res.reshape(size, (self.neu + self.ned) * 3) / delta / 2

    def parameters_numerical_d1(self, e_vectors, n_vectors):
        """Numerical first derivatives logarithm Backflow with respect to the parameters
        :param e_vectors: e-e vectors
        :param n_vectors: e-n vectors
        """
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return np.concatenate((
            self.eta_term_numerical_d1(e_vectors, e_powers),
            self.mu_term_numerical_d1(n_vectors, n_powers),
            self.phi_term_numerical_d1(e_powers, n_powers, e_vectors, n_vectors),
        ))

    def profile_value(self, dr, steps, atom_positions, r_initial):
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = subtract_outer(r_e, r_e)
            n_vectors = subtract_outer(atom_positions, r_e)
            self.value(e_vectors, n_vectors)

    def profile_gradient(self, dr, steps, atom_positions, r_initial):
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = subtract_outer(r_e, r_e)
            n_vectors = subtract_outer(atom_positions, r_e)
            self.gradient(e_vectors, n_vectors)

    def profile_laplacian(self, dr, steps, atom_positions, r_initial):
        """auxiliary code"""
        for _ in range(steps):
            r_e = r_initial + random_step(dr, self.neu + self.ned)
            e_vectors = subtract_outer(r_e, r_e)
            n_vectors = subtract_outer(atom_positions, r_e)
            self.laplacian(e_vectors, n_vectors)

