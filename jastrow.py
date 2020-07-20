#!/usr/bin/env python3
from math import sqrt

import numpy as np
import numba as nb


@nb.jit(nopython=True)
def u_term(C, u_parameters, L, r_u, r_d):
    """Jastrow u-term
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :return:
    """
    res = 0.0
    if not L:
        return res

    for i in range(r_u.shape[0] - 1):
        for j in range(i + 1, r_u.shape[0]):
            r = np.linalg.norm(r_u[i] - r_u[j])  # FIXME to slow
            if r <= L:
                a = 0.0
                for k in range(u_parameters.shape[0]):
                    a += u_parameters[k, 0]*r**k
                res += a * (r - L)**C

    for i in range(r_u.shape[0]):
        for j in range(r_d.shape[0]):
            r = np.linalg.norm(r_u[i] - r_d[j])  # FIXME to slow
            if r <= L:
                a = 0.0
                for k in range(u_parameters.shape[0]):
                    a += u_parameters[k, 1]*r**k
                res += a * (r - L) ** C

    for i in range(r_d.shape[0] - 1):
        for j in range(i + 1, r_d.shape[0]):
            r = np.linalg.norm(r_d[i] - r_d[j])  # FIXME to slow
            if r <= L:
                a = 0.0
                for k in range(u_parameters.shape[0]):
                    a += u_parameters[k, 2]*r**k
                res += a * (r - L) ** C

    return res


@nb.jit(nopython=True)
def chi_term(C, chi_parameters, L, r_u, r_d, atoms):
    """Jastrow chi-term
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :param atoms:
    :return:
    """
    res = 0.0
    if not L:
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_u.shape[0]):
            r = np.linalg.norm(atoms[i]['position'] - r_u[j])  # FIXME to slow
            if r <= L:
                a = 0.0
                for k in range(chi_parameters.shape[1]):
                    a += chi_parameters[i, k, 0] * r ** k
                res += a * (r - L) ** C

    for i in range(atoms.shape[0]):
        for j in range(r_d.shape[0]):
            r = np.linalg.norm(atoms[i]['position'] - r_d[j])  # FIXME to slow
            if r <= L:
                a = 0.0
                for k in range(chi_parameters.shape[1]):
                    a += chi_parameters[i, k, 1] * r ** k
                res += a * (r - L) ** C

    return res


@nb.jit(nopython=True)
def f_term(C, f_parameters, L, r_u, r_d, atoms):
    """Jastrow f-term
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :param atoms:
    :return:
    """
    res = 0.0
    if not L:
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_u.shape[0] - 1):
            for k in range(j+1, r_u.shape[0]):
                r_ee = np.linalg.norm(r_u[j] - r_u[k])  # FIXME to slow
                r_e1I = np.linalg.norm(atoms[i]['position'] - r_u[j])  # FIXME to slow
                r_e2I = np.linalg.norm(atoms[i]['position'] - r_u[k])  # FIXME to slow
                if r_e1I <= L and r_e2I <= L:
                    a = 0.0
                    for l1 in range(f_parameters.shape[1]):
                        for l2 in range(f_parameters.shape[1]):
                            for l3 in range(f_parameters.shape[1]):
                                a += f_parameters[i, l1, l2, l3, 0] * r_e1I ** l1 * r_e2I ** l2 * r_ee * l3
                    res += a * (r_e1I - L) ** C * (r_e2I - L) ** C

    for i in range(atoms.shape[0]):
        for j in range(r_u.shape[0]):
            for k in range(r_d.shape[0]):
                r_ee = np.linalg.norm(r_u[j] - r_u[k])  # FIXME to slow
                r_e1I = np.linalg.norm(atoms[i]['position'] - r_u[j])  # FIXME to slow
                r_e2I = np.linalg.norm(atoms[i]['position'] - r_u[k])  # FIXME to slow
                if r_e1I <= L and r_e2I <= L:
                    a = 0.0
                    for l1 in range(f_parameters.shape[1]):
                        for l2 in range(f_parameters.shape[1]):
                            for l3 in range(f_parameters.shape[1]):
                                a += f_parameters[i, l1, l2, l3, 1] * r_e1I ** l1 * r_e2I ** l2 * r_ee * l3
                    res += a * (r_e1I - L) ** C * (r_e2I - L) ** C

    for i in range(atoms.shape[0]):
        for j in range(r_d.shape[0] - 1):
            for k in range(j+1, r_d.shape[0]):
                r_ee = np.linalg.norm(r_u[j] - r_u[k])  # FIXME to slow
                r_e1I = np.linalg.norm(atoms[i]['position'] - r_u[j])  # FIXME to slow
                r_e2I = np.linalg.norm(atoms[i]['position'] - r_u[k])  # FIXME to slow
                if r_e1I <= L and r_e2I <= L:
                    a = 0.0
                    for l1 in range(f_parameters.shape[1]):
                        for l2 in range(f_parameters.shape[1]):
                            for l3 in range(f_parameters.shape[1]):
                                a += f_parameters[i, l1, l2, l3, 2] * r_e1I ** l1 * r_e2I ** l2 * r_ee * l3
                    res += a * (r_e1I - L) ** C * (r_e2I - L) ** C

    return res


@nb.jit(nopython=True)
def u_term_gradient(C, u_parameters, L, r_u, r_d):
    """Jastrow u-term gradient
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :return:
    """
    gradient_u = np.zeros((r_u.shape[0], 3))
    gradient_d = np.zeros((r_d.shape[0], 3))

    if not L:
        return gradient_u, gradient_d

    for i in range(r_u.shape[0] - 1):
        for j in range(i + 1, r_u.shape[0]):
            r = np.linalg.norm(r_u[i] - r_u[j])  # FIXME to slow
            x, y, z = r_u[i] - r_u[j]  # FIXME to slow
            if r <= L:
                poly = 0.0
                for k in range(u_parameters.shape[0]):
                    poly += u_parameters[k, 0]*r**k

                poly_diff = 0.0
                for k in range(1, u_parameters.shape[0]):
                    poly_diff += k * u_parameters[k, 0]*r**(k-1)

                gradient = (r-L)**(C-1) * (C*poly + (r-L)*poly_diff) / r
                gradient_u[i, 0] += x * gradient
                gradient_u[i, 1] += y * gradient
                gradient_u[i, 2] += z * gradient
                gradient_u[j, 0] -= x * gradient
                gradient_u[j, 1] -= y * gradient
                gradient_u[j, 2] -= z * gradient

    for i in range(r_u.shape[0]):
        for j in range(r_d.shape[0]):
            r = np.linalg.norm(r_u[i] - r_d[j])  # FIXME to slow
            x, y, z = r_u[i] - r_d[j]  # FIXME to slow
            if r <= L:
                poly = 0.0
                for k in range(u_parameters.shape[0]):
                    poly += u_parameters[k, 1]*r**k

                poly_diff = 0.0
                for k in range(1, u_parameters.shape[0]):
                    poly_diff += k * u_parameters[k, 1]*r**(k-1)

                gradient = (r-L)**(C-1) * (C*poly + (r-L)*poly_diff) / r
                gradient_u[i, 0] += x * gradient
                gradient_u[i, 1] += y * gradient
                gradient_u[i, 2] += z * gradient
                gradient_d[j, 0] -= x * gradient
                gradient_d[j, 1] -= y * gradient
                gradient_d[j, 2] -= z * gradient

    for i in range(r_d.shape[0] - 1):
        for j in range(i + 1, r_d.shape[0]):
            r = np.linalg.norm(r_d[i] - r_d[j])  # FIXME to slow
            x, y, z = r_d[i] - r_d[j]  # FIXME to slow
            if r <= L:
                poly = 0.0
                for k in range(u_parameters.shape[0]):
                    poly += u_parameters[k, 2]*r**k

                poly_diff = 0.0
                for k in range(1, u_parameters.shape[0]):
                    poly_diff += k * u_parameters[k, 2]*r**(k-1)

                gradient = (r-L)**(C-1) * (C*poly + (r-L)*poly_diff) / r
                gradient_d[i, 0] += x * gradient
                gradient_d[i, 1] += y * gradient
                gradient_d[i, 2] += z * gradient
                gradient_d[j, 0] -= x * gradient
                gradient_d[j, 1] -= y * gradient
                gradient_d[j, 2] -= z * gradient

    return gradient_u, gradient_d


@nb.jit(nopython=True)
def chi_term_gradient(trunc, chi_parameters, chi_cutoff, r_u, r_d):
    """Jastrow chi-term gradient
    :param chi_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :return:
    """
    gradient_u = np.zeros((r_u.shape[0], 3))
    gradient_d = np.zeros((r_u.shape[0], 3))

    return gradient_u, gradient_d


@nb.jit(nopython=True)
def f_term_gradient(trunc, f_parameters, f_cutoff, r_u, r_d):
    """Jastrow f-term gradient
    :param f_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :return:
    """
    gradient_u = np.zeros((r_u.shape[0], 3))
    gradient_d = np.zeros((r_u.shape[0], 3))

    return gradient_u, gradient_d


@nb.jit(nopython=True)
def u_term_laplacian(C, u_parameters, L, r_u, r_d):
    """Jastrow u-term laplacian
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :return:
    """
    res = 0.0
    if not L:
        return res

    for i in range(r_u.shape[0] - 1):
        for j in range(i + 1, r_u.shape[0]):
            r = np.linalg.norm(r_u[i] - r_u[j])  # FIXME to slow
            if r <= L:
                poly = 0.0
                for k in range(u_parameters.shape[0]):
                    poly += u_parameters[k, 0]*r**k

                poly_diff = 0.0
                for k in range(1, u_parameters.shape[0]):
                    poly_diff += k * u_parameters[k, 0]*r**(k-1)

                poly_diff_2 = 0.0
                for k in range(2, u_parameters.shape[0]):
                    poly_diff_2 += k * (k-1) * u_parameters[k, 0]*r**(k-2)
                res += 2*(
                        r*(C*(C - 1)*(r-L)**(C + 1)*poly + 2*C*(r-L)**(C + 2)*poly_diff + (r-L)**(C + 3)*poly_diff_2)
                        + 2*(r-L)**2*(C*(r-L)**C*poly + (r-L)**(C + 1)*poly_diff)
                )/(r*(r-L)**3)

    for i in range(r_u.shape[0]):
        for j in range(r_d.shape[0]):
            r = np.linalg.norm(r_u[i] - r_d[j])  # FIXME to slow
            if r <= L:
                poly = 0.0
                for k in range(u_parameters.shape[0]):
                    poly += u_parameters[k, 1]*r**k

                poly_diff = 0.0
                for k in range(1, u_parameters.shape[0]):
                    poly_diff += k * u_parameters[k, 1]*r**(k-1)

                poly_diff_2 = 0.0
                for k in range(2, u_parameters.shape[0]):
                    poly_diff_2 += k * (k-1) * u_parameters[k, 1]*r**(k-2)
                res += 2*(
                        r*(C*(C - 1)*(r-L)**(C + 1)*poly + 2*C*(r-L)**(C + 2)*poly_diff + (r-L)**(C + 3)*poly_diff_2)
                        + 2*(r-L)**2*(C*(r-L)**C*poly + (r-L)**(C + 1)*poly_diff)
                )/(r*(r-L)**3)

    for i in range(r_d.shape[0] - 1):
        for j in range(i + 1, r_d.shape[0]):
            r = np.linalg.norm(r_d[i] - r_d[j])  # FIXME to slow
            if r <= L:
                poly = 0.0
                for k in range(u_parameters.shape[0]):
                    poly += u_parameters[k, 2]*r**k

                poly_diff = 0.0
                for k in range(1, u_parameters.shape[0]):
                    poly_diff += k * u_parameters[k, 2]*r**(k-1)

                poly_diff_2 = 0.0
                for k in range(2, u_parameters.shape[0]):
                    poly_diff_2 += k * (k-1) * u_parameters[k, 2]*r**(k-2)
                res += 2*(
                        r*(C*(C - 1)*(r-L)**(C + 1)*poly + 2*C*(r-L)**(C + 2)*poly_diff + (r-L)**(C + 3)*poly_diff_2)
                        + 2*(r-L)**2*(C*(r-L)**C*poly + (r-L)**(C + 1)*poly_diff)
                )/(r*(r-L)**3)

    return res


@nb.jit(nopython=True)
def chi_term_laplacian(trunc, chi_parameters, chi_cutoff, r_u, r_d, atoms):
    """Jastrow chi-term laplacian
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :return:
    """
    return 0


@nb.jit(nopython=True)
def f_term_laplacian(trunc, f_parameters, f_cutoff, r_u, r_d, atoms):
    """Jastrow f-term laplacian
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :return:
    """
    return 0


@nb.jit(nopython=True)
def jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_u, r_d, atoms):
    """Jastrow
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :param atoms:
    :return:
    """
    return np.exp(
        u_term(trunc, u_parameters, u_cutoff, r_u, r_d) +
        chi_term(trunc, chi_parameters, chi_cutoff, r_u, r_d, atoms) +
        f_term(trunc, f_parameters, f_cutoff, r_u, r_d, atoms)
    )


@nb.jit(nopython=True)
def jastrow_gradient(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_u, r_d, atoms):
    return u_term_gradient(trunc, u_parameters, u_cutoff, r_u, r_d)


@nb.jit(nopython=True)
def jastrow_laplacian(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_u, r_d, atoms):
    return (
        u_term_laplacian(trunc, u_parameters, u_cutoff, r_u, r_d) +
        chi_term_laplacian(trunc, chi_parameters, chi_cutoff, r_u, r_d, atoms) +
        f_term_laplacian(trunc, f_parameters, f_cutoff, r_u, r_d, atoms)
    )
