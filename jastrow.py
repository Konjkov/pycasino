#!/usr/bin/env python3
from math import sqrt

import numpy as np
import numba as nb


@nb.jit(nopython=True)
def u_term(C, u_parameters, L, r_e, neu):
    """Jastrow u-term
    :param u_parameters:
    :param r_e: electrons coordinates
    :param neu: number of up electrons
    :return:
    """
    res = 0.0
    if not L:
        return res

    for i in range(r_e.shape[0] - 1):
        for j in range(i + 1, r_e.shape[0]):
            r = np.linalg.norm(r_e[i] - r_e[j])  # FIXME to slow
            if r <= L:
                u_set = int(i >= neu) + int(j >= neu)
                poly = 0.0
                for k in range(u_parameters.shape[0]):
                    poly += u_parameters[k, u_set]*r**k
                res += poly * (r - L)**C
    return res


@nb.jit(nopython=True)
def chi_term(C, chi_parameters, L, r_e, neu, atoms):
    """Jastrow chi-term
    :param chi_parameters:
    :param r_e: electrons coordinates
    :param neu: number of up electrons
    :param atoms:
    :return:
    """
    res = 0.0
    if not L.any():
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_e.shape[0]):
            r = np.linalg.norm(r_e[j] - atoms[i]['position'])  # FIXME to slow
            if r <= L[i]:
                chi_set = int(j >= neu)
                poly = 0.0
                for k in range(chi_parameters.shape[1]):
                    poly += chi_parameters[i, k, chi_set] * r ** k
                res += poly * (r - L[i]) ** C
    return res


@nb.jit(nopython=True)
def f_term(C, f_parameters, L, r_e, neu, atoms):
    """Jastrow f-term
    :param u_parameters:
    :param r_e: electrons coordinates
    :param ned: number of up electrons
    :param atoms:
    :return:
    """
    res = 0.0
    if not L.any():
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_e.shape[0] - 1):
            for k in range(j+1, r_e.shape[0]):
                r_ee = np.linalg.norm(r_e[j] - r_e[k])  # FIXME to slow
                r_e1I = np.linalg.norm(r_e[j] - atoms[i]['position'])  # FIXME to slow
                r_e2I = np.linalg.norm(r_e[k] - atoms[i]['position'])  # FIXME to slow
                if r_e1I <= L and r_e2I <= L:
                    f_set = int(j >= neu) + int(k >= neu)
                    poly = 0.0
                    for l1 in range(f_parameters.shape[1]):
                        for l2 in range(f_parameters.shape[2]):
                            for l3 in range(f_parameters.shape[3]):
                                poly += f_parameters[i, l1, l2, l3, f_set] * r_e1I ** l1 * r_e2I ** l2 * r_ee ** l3
                    res += 2 * poly * (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C
    return res


@nb.jit(nopython=True)
def u_term_gradient(C, u_parameters, L, r_e, neu):
    """Jastrow u-term gradient
    :param u_parameters:
    :param r_e: electrons coordinates
    :param neu: number of up electrons
    :return:
    """
    res = np.zeros(r_e.shape)

    if not L:
        return res

    for i in range(r_e.shape[0] - 1):
        for j in range(i + 1, r_e.shape[0]):
            r = np.linalg.norm(r_e[i] - r_e[j])  # FIXME to slow
            if r <= L:
                u_set = int(i >= neu) + int(j >= neu)
                x, y, z = r_e[i] - r_e[j]  # FIXME to slow
                poly = 0.0
                for k in range(u_parameters.shape[0]):
                    poly += u_parameters[k, u_set]*r**k

                poly_diff = 0.0
                for k in range(1, u_parameters.shape[0]):
                    poly_diff += k * u_parameters[k, u_set]*r**(k-1)

                gradient = (r-L)**(C-1) * (C*poly + (r-L)*poly_diff) / r
                res[i, 0] += x * gradient
                res[i, 1] += y * gradient
                res[i, 2] += z * gradient
                res[j, 0] -= x * gradient
                res[j, 1] -= y * gradient
                res[j, 2] -= z * gradient
    return res


@nb.jit(nopython=True)
def chi_term_gradient(C, chi_parameters, L, r_e, neu, atoms):
    """Jastrow chi-term gradient
    :param chi_parameters:
    :param r_e: electrons coordinates
    :param neu: number of up electrons
    :return:
    """
    res = np.zeros(r_e.shape)

    if not L.any():
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_e.shape[0]):
            r = np.linalg.norm(r_e[j] - atoms[i]['position'])  # FIXME to slow
            if r <= L[i]:
                chi_set = int(j >= neu)
                x, y, z = r_e[j] - atoms[i]['position']  # FIXME to slow
                poly = 0.0
                for k in range(chi_parameters.shape[1]):
                    poly += chi_parameters[i, k, chi_set]*r**k

                poly_diff = 0.0
                for k in range(1, chi_parameters.shape[1]):
                    poly_diff += k * chi_parameters[i, k, chi_set]*r**(k-1)

                gradient = (r-L[i])**(C-1) * (C*poly + (r-L[i])*poly_diff) / r
                res[j, 0] += x * gradient
                res[j, 1] += y * gradient
                res[j, 2] += z * gradient
    return res


@nb.jit(nopython=True)
def f_term_gradient(C, f_parameters, L, r_e, neu, atoms):
    """Jastrow f-term gradient
    :param f_parameters:
    :param r_e: electrons coordinates
    :param neu: number of up electrons
    :return:
    """
    res = np.zeros(r_e.shape)

    if not L.any():
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_e.shape[0] - 1):
            for k in range(j+1, r_e.shape[0]):
                r_ee = np.linalg.norm(r_e[j] - r_e[k])  # FIXME to slow
                r_e1I = np.linalg.norm(r_e[j] - atoms[i]['position'])  # FIXME to slow
                r_e2I = np.linalg.norm(r_e[k] - atoms[i]['position'])  # FIXME to slow
                if r_e1I <= L[i] and r_e2I <= L[i]:
                    f_set = int(j >= neu) + int(k >= neu)
                    x, y, z = r_e[i] - r_e[j]  # FIXME to slow
                    poly = 0.0
                    for l1 in range(f_parameters.shape[1]):
                        for l2 in range(f_parameters.shape[2]):
                            for l3 in range(f_parameters.shape[3]):
                                poly += f_parameters[i, l1, l2, l3, f_set] * r_e1I ** l1 * r_e2I ** l2 * r_ee ** l3
                    gradient = poly * (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C
                    res[j, 0] += x * gradient
                    res[j, 1] += y * gradient
                    res[j, 2] += z * gradient
    return res


@nb.jit(nopython=True)
def u_term_laplacian(C, u_parameters, L, r_e, neu):
    """Jastrow u-term laplacian
    :param u_parameters:
    :param r_e: electrons coordinates
    :param neu: number of up electrons
    :return:
    """
    res = 0.0
    if not L:
        return res

    for i in range(r_e.shape[0] - 1):
        for j in range(i + 1, r_e.shape[0]):
            r = np.linalg.norm(r_e[i] - r_e[j])  # FIXME to slow
            if r <= L:
                u_set = int(i >= neu) + int(j >= neu)
                poly = 0.0
                for k in range(u_parameters.shape[0]):
                    poly += u_parameters[k, u_set]*r**k

                poly_diff = 0.0
                for k in range(1, u_parameters.shape[0]):
                    poly_diff += k * u_parameters[k, u_set]*r**(k-1)

                poly_diff_2 = 0.0
                for k in range(2, u_parameters.shape[0]):
                    poly_diff_2 += k * (k-1) * u_parameters[k, u_set]*r**(k-2)
                res += 2*(
                        r*(C*(C - 1)*(r-L)**(C + 1)*poly + 2*C*(r-L)**(C + 2)*poly_diff + (r-L)**(C + 3)*poly_diff_2)
                        + 2*(r-L)**2*(C*(r-L)**C*poly + (r-L)**(C + 1)*poly_diff)
                )/(r*(r-L)**3)
    return res


@nb.jit(nopython=True)
def chi_term_laplacian(C, chi_parameters, L, r_e, neu, atoms):
    """Jastrow chi-term laplacian
    :param chi_parameters:
    :param r_e: electrons coordinates
    :param neu: number of up electrons
    :return:
    """
    res = 0.0
    if not L.any():
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_e.shape[0]):
            r = np.linalg.norm(r_e[j] - atoms[i]['position'])  # FIXME to slow
            if r <= L[i]:
                chi_set = int(j >= neu)
                poly = 0.0
                for k in range(chi_parameters.shape[1]):
                    poly += chi_parameters[i, k, chi_set]*r**k

                poly_diff = 0.0
                for k in range(1, chi_parameters.shape[1]):
                    poly_diff += k * chi_parameters[i, k, chi_set]*r**(k-1)

                poly_diff_2 = 0.0
                for k in range(2, chi_parameters.shape[1]):
                    poly_diff_2 += k * (k-1) * chi_parameters[i, k, chi_set]*r**(k-2)
                res += (
                        r*(C*(C - 1)*(r-L[i])**(C + 1)*poly + 2*C*(r-L[i])**(C + 2)*poly_diff + (r-L[i])**(C + 3)*poly_diff_2)
                        + 2*(r-L[i])**2*(C*(r-L[i])**C*poly + (r-L[i])**(C + 1)*poly_diff)
                )/(r*(r-L[i])**3)
    return res


@nb.jit(nopython=True)
def f_term_laplacian(C, f_parameters, L, r_e, neu, atoms):
    """Jastrow f-term laplacian
    :param f_parameters:
    :param r_e: electrons coordinates
    :param neu: number of up electrons
    :return:
    """
    res = 0.0
    if not L.any():
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_e.shape[0] - 1):
            for k in range(j + 1, r_e.shape[0]):
                r_ee = np.linalg.norm(r_e[j] - r_e[k])  # FIXME to slow
                r_e1I = np.linalg.norm(r_e[j] - atoms[i]['position'])  # FIXME to slow
                r_e2I = np.linalg.norm(r_e[k] - atoms[i]['position'])  # FIXME to slow
                if r_e1I <= L[i] and r_e2I <= L[i]:
                    f_set = int(j >= neu) + int(k >= neu)
                    poly = 0.0
                    for l1 in range(f_parameters.shape[1]):
                        for l2 in range(f_parameters.shape[2]):
                            for l3 in range(f_parameters.shape[3]):
                                poly += f_parameters[i, l1, l2, l3, f_set] * r_e1I ** l1 * r_e2I ** l2 * r_ee ** l3
    return res


@nb.jit(nopython=True)
def jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms):
    """Jastrow
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :param atoms:
    :return:
    """
    return (
        u_term(trunc, u_parameters, u_cutoff, r_e, neu) +
        chi_term(trunc, chi_parameters, chi_cutoff, r_e, neu, atoms) +
        f_term(trunc, f_parameters, f_cutoff, r_e, neu, atoms)
    )


@nb.jit(nopython=True)
def jastrow_numerical_gradient(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms):
    delta = 0.00001

    gradient = np.zeros(r_e.shape)

    for i in range(r_e.shape[0]):
        for j in range(r_e.shape[1]):
            r_e[i, j] -= delta
            gradient[i, j] -= jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)
            r_e[i, j] += 2 * delta
            gradient[i, j] += jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)
            r_e[i, j] -= delta

    return gradient / delta / 2


@nb.jit(nopython=True)
def jastrow_numerical_laplacian(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms):
    delta = 0.00001

    j_00 = jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)

    res = -2 * (r_e.shape[0] * r_e.shape[1]) * j_00
    for i in range(r_e.shape[0]):
        for j in range(r_e.shape[1]):
            r_e[i, j] -= delta
            res += jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)
            r_e[i, j] += 2 * delta
            res += jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)
            r_e[i, j] -= delta

    return res / delta / delta


@nb.jit(nopython=True)
def jastrow_gradient(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms):
    return (
            u_term_gradient(trunc, u_parameters, u_cutoff, r_e, neu) +
            chi_term_gradient(trunc, chi_parameters, chi_cutoff, r_e, neu, atoms) +
            f_term_gradient(trunc, f_parameters, f_cutoff, r_e, neu, atoms)
    )


@nb.jit(nopython=True)
def jastrow_laplacian(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms):
    return (
        u_term_laplacian(trunc, u_parameters, u_cutoff, r_e, neu) +
        chi_term_laplacian(trunc, chi_parameters, chi_cutoff, r_e, neu, atoms) +
        f_term_laplacian(trunc, f_parameters, f_cutoff, r_e, neu, atoms)
    )
