import numpy as np
import numba as nb
from numpy.polynomial.polynomial import polyval
from numba.core import types
from numba.experimental import structref
from numba.core.extending import overload_method


@nb.njit(nogil=True, parallel=False, cache=True)
def construct_c_matrix(trunc, phi_parameters, theta_parameters, phi_cutoff, spin_dep, phi_cusp, phi_irrotational):
    """C-matrix has the following rows:
    6 * (phi_en_order + phi_ee_order + 1) - 2 constraints imposed to satisfy phi-term conditions.
    ... constraints imposed to satisfy theta-term conditions.
    copy-paste from /CASINO/src/pbackflow.f90 SUBROUTINE construct_C
    """
    phi_ee_order = phi_parameters.shape[1] - 1
    phi_en_order = phi_parameters.shape[2] - 1

    ee_constrains = 2 * phi_en_order + 1
    en_constrains = phi_en_order + phi_ee_order + 1
    offset = 0
    if phi_cusp:
        # AE cusps: 6 * (ee + eN) - 8
        phi_constraints = 6 * en_constrains - 2
    else:
        # PP cusps: 2 * (ee + eN) - 2
        phi_constraints = 2 * en_constrains
    if spin_dep in (0, 2):
        # e-e cusps: 2 * eN - 1 from Theta, and same from some spin-deps of Phi
        phi_constraints += ee_constrains
        offset += ee_constrains
    if phi_cusp:
        # AE cusps: 5 * (ee + eN) - 7
        theta_constraints = 5 * en_constrains + ee_constrains - 2
    else:
        # PP cusps: ee + eN - 1
        theta_constraints = en_constrains + ee_constrains
    n_constraints = phi_constraints + theta_constraints
    if phi_irrotational:
        # ee * eN * eN + 2 * eN * (ee-1) + eN**2 constraints from Phi and Theta
        n_constraints += ((phi_en_order + 3) * (phi_ee_order + 2) - 4) * (phi_en_order + 1)
        if trunc == 0:
            n_constraints -= (phi_en_order + 1) * (phi_ee_order + 1)

    parameters_size = 2 * (phi_parameters.shape[1] * phi_parameters.shape[2] * phi_parameters.shape[3])
    c = np.zeros((n_constraints, parameters_size))
    cutoff_constraints = np.zeros(shape=(n_constraints, ))
    p = 0
    # Do Phi bit of the constraint matrix.
    for m in range(phi_parameters.shape[1]):
        for l in range(phi_parameters.shape[2]):
            for k in range(phi_parameters.shape[3]):
                if spin_dep in (0, 2):  # e-e cusp
                    if m == 1:
                        c[k + l, p] = 1
                if phi_cusp:
                    if l == 0:  # 1b
                        c[k + m + offset + en_constrains, p] = 1
                        if m > 0:  # 3b
                            c[k + m - 1 + offset + 5 * en_constrains - 1, p] = m
                    elif l == 1:  # b2
                        c[k + m + offset + 3 * en_constrains, p] = 1
                    if k == 0:  # 1a
                        c[l + m + offset, p] = 1
                        if m > 0:  # 3a
                            c[l + m - 1 + offset + 4 * en_constrains, p] = m
                    elif k == 1:  # 2a
                        c[l + m + offset + 2 * en_constrains, p] = 1
                else:
                    if l == 0:  # 1b
                        c[k + m + offset + en_constrains, p] = -trunc / phi_cutoff
                        cutoff_constraints[k + m + offset + en_constrains] += trunc * phi_parameters[spin_dep, m, l, k] / phi_cutoff ** 2
                    elif l == 1:  # 1b
                        c[k + m + offset + en_constrains, p] = 1
                    if k == 0:  # 1a
                        c[l + m + offset, p] = -trunc / phi_cutoff
                        cutoff_constraints[l + m + offset] += trunc * phi_parameters[spin_dep, m, l, k] / phi_cutoff ** 2
                    elif k == 1:  # 1a
                        c[l + m + offset, p] = 1
                p += 1
    # Do Theta bit of the constraint matrix.
    offset = phi_constraints
    for m in range(phi_parameters.shape[1]):
        for l in range(phi_parameters.shape[2]):
            for k in range(phi_parameters.shape[3]):
                if m == 1:
                    c[k + l + offset, p] = 1
                if phi_cusp:
                    if l == 0:  # 2b
                        c[k + m + offset + ee_constrains + 2 * en_constrains, p] = -trunc / phi_cutoff
                        cutoff_constraints[k + m + offset + ee_constrains + 2 * en_constrains] += trunc * theta_parameters[spin_dep, m, l, k] / phi_cutoff ** 2
                        if m > 0:  # 3b
                            c[k + m - 1 + offset + ee_constrains + 4 * en_constrains - 1, p] = m
                    elif l == 1:  # 2b
                        c[k + m + offset + ee_constrains + 2 * en_constrains, p] = 1
                    if k == 0:  # 1a
                        c[l + m + offset + ee_constrains, p] = 1
                        if m > 0:  # 3a
                            c[l + m - 1 + offset + ee_constrains + 3 * en_constrains, p] = m
                    elif k == 1:  # 2a
                        c[l + m + offset + ee_constrains + en_constrains, p] = 1
                else:
                    if l == 0:  # 1a
                        c[k + m + offset + ee_constrains, p] = -trunc / phi_cutoff
                        cutoff_constraints[l + m + offset + ee_constrains] += trunc * theta_parameters[spin_dep, m, l, k] / phi_cutoff ** 2
                    elif l == 1:  # 1a
                        c[k + m + offset + ee_constrains, p] = 1
                p += 1
    # Do irrotational bit of the constraint matrix.
    n = phi_constraints + theta_constraints
    if phi_irrotational:
        p = 0
        inc_k = 1
        inc_l = inc_k * (phi_en_order + 1)
        inc_m = inc_l * (phi_en_order + 1)
        nphi = inc_m * (phi_ee_order + 1)
        for m in range(phi_parameters.shape[1]):
            for l in range(phi_parameters.shape[2]):
                for k in range(phi_parameters.shape[3]):
                    if trunc > 0:
                        if m > 0:
                            c[n, p - inc_m] = trunc + k
                            if k < phi_en_order:
                                c[n, p + inc_k - inc_m] = -phi_cutoff * (k + 1)
                        if m < phi_ee_order:
                            if k > 1:
                                c[n, p + nphi - 2 * inc_k + inc_m] = -(m + 1)
                            if k > 0:
                                c[n, p + nphi - inc_k + inc_m] = phi_cutoff * (m + 1)
                                cutoff_constraints[n] += (m + 1) * phi_parameters[spin_dep, m, l, k]
                    else:
                        if m > 0 and k < phi_en_order:
                            c[n, p + inc_k - inc_m] = k + 1
                        if k > 0 and m < phi_ee_order:
                            c[n, p + nphi - inc_k + inc_m] = -(m + 1)
                    p += 1
                    n += 1
        if trunc > 0:
            # Same as above, for m=N_ee+1...
            p = phi_ee_order * (phi_en_order + 1) ** 2
            for l in range(phi_parameters.shape[2]):
                for k in range(phi_parameters.shape[3]):
                    c[n, p] = trunc + k
                    if k < phi_en_order:
                        c[n, p + inc_k] = -phi_cutoff * (k + 1)
                        cutoff_constraints[n] -= (k + 1) * phi_parameters[spin_dep, :, l, k+1].sum()
                    p += 1
                    n += 1
            # ...for k=N_eN+1...
            p = phi_en_order - 1
            for m in range(phi_parameters.shape[1] - 1):
                for l in range(phi_parameters.shape[2]):
                    c[n, p + nphi + inc_m] = -(m + 1)
                    c[n, p + nphi + inc_k + inc_m] = phi_cutoff * (m + 1)
                    cutoff_constraints[n] += (m + 1) * theta_parameters[spin_dep, m + 1, l, :].sum()
                    p += inc_l
                    n += 1
            # ...and for k=N_eN+2.
            p = phi_en_order
            for m in range(phi_parameters.shape[1] - 1):
                for l in range(phi_parameters.shape[2]):
                    c[n, p + nphi + inc_m] = -(m + 1)
                    p += inc_l
                    n += 1
        else:
            # Same as above, for m=N_ee+1...
            p = phi_ee_order * (phi_en_order + 1) ** 2
            for l in range(phi_parameters.shape[2]):
                for k in range(phi_parameters.shape[3] - 1):
                    c[n, p + inc_k] = 1  # just zeroes the corresponding param
                    p += 1
                    n += 1
            # ...and for k=N_eN+1.
            p = phi_en_order - 1
            for m in range(phi_parameters.shape[1] - 1):
                for l in range(phi_parameters.shape[2]):
                    c[n, p + nphi + inc_m] = 1  # just zeroes the corresponding param
                    p += inc_l
                    n += 1

    assert n == n_constraints
    return c, cutoff_constraints


@structref.register
class Backflow_class_t(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


labels_type = nb.int64[:]
eta_parameters_type = nb.float64[:, :]
mu_parameters_type = nb.float64[:, :]
phi_parameters_type = nb.float64[:, :, :, :]
theta_parameters_type = nb.float64[:, :, :, :]
eta_parameters_mask_type = nb.boolean[:, :]
mu_parameters_mask_type = nb.boolean[:, :]
phi_parameters_mask_type = nb.boolean[:, :, :, :]
theta_parameters_mask_type = nb.boolean[:, :, :, :]

Backflow_t = Backflow_class_t([
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('trunc', nb.int64),
    ('eta_parameters', eta_parameters_type),
    ('mu_parameters', nb.types.ListType(mu_parameters_type)),
    ('phi_parameters', nb.types.ListType(phi_parameters_type)),
    ('theta_parameters', nb.types.ListType(theta_parameters_type)),
    ('eta_parameters_optimizable', eta_parameters_mask_type),
    ('mu_parameters_optimizable', nb.types.ListType(mu_parameters_mask_type)),
    ('mu_parameters_available', nb.types.ListType(mu_parameters_mask_type)),
    ('phi_parameters_optimizable', nb.types.ListType(phi_parameters_mask_type)),
    ('theta_parameters_optimizable', nb.types.ListType(theta_parameters_mask_type)),
    ('eta_parameters_available', eta_parameters_mask_type),
    ('phi_parameters_available', nb.types.ListType(phi_parameters_mask_type)),
    ('theta_parameters_available', nb.types.ListType(theta_parameters_mask_type)),
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
    ('ae_cutoff_optimizable', nb.boolean[:]),
    ('parameters_projector', nb.float64[:, :]),
    ('cutoffs_optimizable', nb.boolean),
])


class Backflow(structref.StructRefProxy):

    def __new__(self, neu, trunc, eta_parameters, eta_parameters_optimizable, eta_cutoff,
        mu_parameters, mu_parameters_optimizable, mu_cutoff, mu_cusp, mu_labels,
        phi_parameters, phi_parameters_optimizable, theta_parameters, theta_parameters_optimizable,
        phi_cutoff, phi_cusp, phi_labels, phi_irrotational, ae_cutoff, ae_cutoff_optimizable):
        return backflow_new(neu, ned, trunc, eta_parameters, eta_parameters_optimizable, eta_cutoff,
        mu_parameters, mu_parameters_optimizable, mu_cutoff, mu_cusp, mu_labels,
        phi_parameters, phi_parameters_optimizable, theta_parameters, theta_parameters_optimizable,
        phi_cutoff, phi_cusp, phi_labels, phi_irrotational, ae_cutoff, ae_cutoff_optimizable)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'fix_optimizable')
def backflow_fix_optimizable(self):
    """Set parameter fixed if there is no corresponded spin-pairs"""
    def impl(self):
        if self.neu + self.ned == 1:
            # H-atom
            for i in range(len(self.eta_cutoff_optimizable)):
                self.eta_cutoff_optimizable[i] = False
            for i in range(len(self.phi_cutoff_optimizable)):
                self.phi_cutoff_optimizable[i] = False

        ee_order = 2
        if self.eta_parameters.shape[0] == 2:
            if self.neu < ee_order and self.ned < ee_order:
                self.eta_parameters_available[0] = False
            if self.neu + self.ned < ee_order:
                self.eta_parameters_available[1] = False
        elif self.eta_parameters.shape[0] == 3:
            if self.neu < ee_order:
                self.eta_parameters_available[0] = False
            if self.neu + self.ned < ee_order:
                self.eta_parameters_available[1] = False
            if self.ned < ee_order:
                self.eta_parameters_available[2] = False

        ee_order = 1
        for mu_parameters_optimizable in self.mu_parameters_optimizable:
            mu_parameters_available = np.ones_like(mu_parameters_optimizable)
            if mu_parameters_optimizable.shape[0] == 2:
                if self.neu < ee_order:
                    mu_parameters_available[0] = False
                if self.ned < ee_order:
                    mu_parameters_available[1] = False
            self.mu_parameters_available.append(mu_parameters_available)

        ee_order = 2
        for phi_parameters_optimizable in self.phi_parameters_optimizable:
            phi_parameters_available = np.ones_like(phi_parameters_optimizable)
            if phi_parameters_optimizable.shape[0] == 2:
                if self.neu < ee_order and self.ned < ee_order:
                    phi_parameters_available[0] = False
                if self.neu + self.ned < ee_order:
                    phi_parameters_available[1] = False
            elif phi_parameters_optimizable.shape[0] == 3:
                if self.neu < ee_order:
                    phi_parameters_available[0] = False
                if self.neu + self.ned < ee_order:
                    phi_parameters_available[1] = False
                if self.ned < ee_order:
                    phi_parameters_available[2] = False
            self.phi_parameters_available.append(phi_parameters_available)

        for theta_parameters_optimizable in self.theta_parameters_optimizable:
            theta_parameters_available = np.ones_like(theta_parameters_optimizable)
            if theta_parameters_optimizable.shape[0] == 2:
                if self.neu < ee_order and self.ned < ee_order:
                    theta_parameters_available[0] = False
                if self.neu + self.ned < ee_order:
                    theta_parameters_available[1] = False
            elif theta_parameters_optimizable.shape[0] == 3:
                if self.neu < ee_order:
                    theta_parameters_available[0] = False
                if self.neu + self.ned < ee_order:
                    theta_parameters_available[1] = False
                if self.ned < ee_order:
                    theta_parameters_available[2] = False
            self.theta_parameters_available.append(theta_parameters_available)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'ee_powers')
def backflow_ee_powers(self, e_vectors: np.ndarray):
    """Powers of e-e distances
    :param e_vectors: e-e vectors - array(nelec, nelec, 3)
    :return:
    """
    def impl(self, e_vectors: np.ndarray) -> np.ndarray:
        res = np.ones(shape=(e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        for i in range(1, e_vectors.shape[0]):
            for j in range(i):
                r_ee = np.linalg.norm(e_vectors[i, j])
                for k in range(1, self.max_ee_order):
                    res[i, j, k] = res[j, i, k] = r_ee ** k
        return res
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Backflow_class_t, 'en_powers')
def backflow_en_powers(self, n_vectors: np.ndarray):
    """Powers of e-n distances
    :param n_vectors: e-n vectors - array(natom, nelec, 3)
    :return:
    """
    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        res = np.ones(shape=(n_vectors.shape[0], n_vectors.shape[1], self.max_en_order))
        for i in range(n_vectors.shape[0]):
            for j in range(n_vectors.shape[1]):
                r_eI = np.linalg.norm(n_vectors[i, j])
                for k in range(1, self.max_en_order):
                    res[i, j, k] = r_eI ** k
        return res
    return impl


# This associates the proxy with MyStruct_t for the given set of fields.
# Notice how we are not constraining the type of each field.
# Field types remain generic.
structref.define_proxy(Backflow, Backflow_class_t, ['neu', 'ned',
    'trunc', 'eta_parameters', 'mu_parameters', 'phi_parameters', 'theta_parameters',
    'eta_parameters_optimizable', 'mu_parameters_optimizable', 'mu_parameters_available',
    'phi_parameters_optimizable', 'theta_parameters_optimizable', 'eta_parameters_available',
    'phi_parameters_available', 'theta_parameters_available', 'eta_cutoff',
    'eta_cutoff_optimizable', 'mu_cutoff', 'mu_cutoff_optimizable', 'phi_cutoff',
    'phi_cutoff_optimizable', 'mu_labels', 'phi_labels', 'max_ee_order', 'max_en_order',
    'mu_cusp', 'phi_cusp', 'phi_irrotational', 'ae_cutoff', 'ae_cutoff_optimizable',
    'parameters_projector', 'cutoffs_optimizable'])

@nb.njit(nogil=True, parallel=False, cache=True)
def backflow_new(neu, ned, trunc, eta_parameters, eta_parameters_optimizable, eta_cutoff,
        mu_parameters, mu_parameters_optimizable, mu_cutoff, mu_cusp, mu_labels,
        phi_parameters, phi_parameters_optimizable, theta_parameters, theta_parameters_optimizable,
        phi_cutoff, phi_cusp, phi_labels, phi_irrotational, ae_cutoff, ae_cutoff_optimizable
    ):
    self = structref.new(Jastrow_t)
    self.neu = neu
    self.ned = ned
    self.trunc = trunc
    # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
    self.eta_cutoff = eta_cutoff['value']
    self.eta_cutoff_optimizable = eta_cutoff['optimizable']
    self.eta_parameters = eta_parameters
    self.eta_parameters_optimizable = eta_parameters_optimizable
    self.eta_parameters_available = np.ones_like(eta_parameters_optimizable)
    # spin dep (0->u=d; 1->u/=d)
    self.mu_cusp = mu_cusp
    self.mu_labels = mu_labels
    self.mu_cutoff = mu_cutoff['value']
    self.mu_cutoff_optimizable = mu_cutoff['optimizable']
    self.mu_parameters = mu_parameters
    self.mu_parameters_optimizable = mu_parameters_optimizable
    self.mu_parameters_available = nb.typed.List.empty_list(mu_parameters_mask_type)
    # spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
    self.phi_irrotational = phi_irrotational
    self.phi_cusp = phi_cusp
    self.phi_labels = phi_labels
    self.phi_cutoff = phi_cutoff['value']
    self.phi_cutoff_optimizable = phi_cutoff['optimizable']
    self.phi_parameters = phi_parameters
    self.theta_parameters = theta_parameters
    self.phi_parameters_optimizable = phi_parameters_optimizable
    self.theta_parameters_optimizable = theta_parameters_optimizable
    self.phi_parameters_available = nb.typed.List.empty_list(phi_parameters_mask_type)
    self.theta_parameters_available = nb.typed.List.empty_list(theta_parameters_mask_type)

    self.max_ee_order = max((
        self.eta_parameters.shape[1],
        max([p.shape[1] for p in self.phi_parameters]) if self.phi_parameters else 0,
    ))
    self.max_en_order = max((
        max([p.shape[1] for p in self.mu_parameters]) if self.mu_parameters else 0,
        max([p.shape[2] for p in self.phi_parameters]) if self.phi_parameters else 0,
        2
    ))
    self.ae_cutoff = ae_cutoff
    self.ae_cutoff_optimizable = ae_cutoff_optimizable
    self.cutoffs_optimizable = True
    self.fix_optimizable()
    return self
