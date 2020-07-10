#!/usr/bin/env python3
from math import sqrt

import numpy as np
import numba as nb


@nb.jit(nopython=True)
def u_term(trunc, u_parameters, u_cutoff, r_u, r_d):
    """Jastrow u-term
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :return:
    """
    res = 0.0
    if not u_cutoff:
        return res

    for i in range(r_u.shape[0]):
        for j in range(i + 1, r_u.shape[0]):
            r = np.linalg.norm(r_u[i] - r_u[j])  # FIXME to slow
            if r <= u_cutoff:
                a = 0.0
                for k in range(u_parameters.shape[0]):
                    a += u_parameters[k, 0]*r**k
                res += a * (r - u_cutoff)**trunc

    for i in range(r_u.shape[0]):
        for j in range(r_d.shape[0]):
            r = np.linalg.norm(r_u[i] - r_d[j])  # FIXME to slow
            if r <= u_cutoff:
                a = 0.0
                for k in range(u_parameters.shape[0]):
                    a += u_parameters[k, 1]*r**k
                res += a * (r - u_cutoff) ** trunc

    for i in range(r_d.shape[0]):
        for j in range(i + 1, r_d.shape[0]):
            r = np.linalg.norm(r_d[i] - r_d[j])  # FIXME to slow
            if r <= u_cutoff:
                a = 0.0
                for k in range(u_parameters.shape[0]):
                    a += u_parameters[k, 2]*r**k
                res += a * (r - u_cutoff) ** trunc

    return res


@nb.jit(nopython=True)
def chi_term(trunc, chi_parameters, chi_cutoff, r_u, r_d, atoms):
    """Jastrow chi-term
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :param atoms:
    :return:
    """
    res = 0.0
    if not chi_cutoff:
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_u.shape[0]):
            r = np.linalg.norm(atoms[i]['position'] - r_u[j])  # FIXME to slow
            if r <= chi_cutoff:
                a = 0.0
                for k in range(chi_parameters.shape[1]):
                    a += chi_parameters[i, k, 0] * r ** k
                res += a * (r - chi_cutoff) ** trunc

    for i in range(atoms.shape[0]):
        for j in range(r_d.shape[0]):
            r = np.linalg.norm(atoms[i]['position'] - r_d[j])  # FIXME to slow
            if r <= chi_cutoff:
                a = 0.0
                for k in range(chi_parameters.shape[1]):
                    a += chi_parameters[i, k, 1] * r ** k
                res += a * (r - chi_cutoff) ** trunc

    return res


@nb.jit(nopython=True)
def f_term(trunc, f_parameters, f_cutoff, r_u, r_d, atoms):
    """Jastrow f-term
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :param atoms:
    :return:
    """
    res = 0.0
    if not f_cutoff:
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_u.shape[0]):
            for k in range(j+1, r_u.shape[0]):
                r_ee = np.linalg.norm(r_u[j] - r_u[k])  # FIXME to slow
                r_e1I = np.linalg.norm(atoms[i]['position'] - r_u[j])  # FIXME to slow
                r_e2I = np.linalg.norm(atoms[i]['position'] - r_u[k])  # FIXME to slow
                if r_e1I <= f_cutoff and r_e2I <= f_cutoff:
                    a = 0.0
                    for l1 in range(f_parameters.shape[1]):
                        for l2 in range(f_parameters.shape[1]):
                            for l3 in range(f_parameters.shape[1]):
                                a += f_parameters[i, l1, l2, l3, 0] * r_e1I ** l1 * r_e2I ** l2 * r_ee * l3
                    res += a * (r_e1I - f_cutoff) ** trunc * (r_e2I - f_cutoff) ** trunc

    for i in range(atoms.shape[0]):
        for j in range(r_u.shape[0]):
            for k in range(r_d.shape[0]):
                r_ee = np.linalg.norm(r_u[j] - r_u[k])  # FIXME to slow
                r_e1I = np.linalg.norm(atoms[i]['position'] - r_u[j])  # FIXME to slow
                r_e2I = np.linalg.norm(atoms[i]['position'] - r_u[k])  # FIXME to slow
                if r_e1I <= f_cutoff and r_e2I <= f_cutoff:
                    a = 0.0
                    for l1 in range(f_parameters.shape[1]):
                        for l2 in range(f_parameters.shape[1]):
                            for l3 in range(f_parameters.shape[1]):
                                a += f_parameters[i, l1, l2, l3, 1] * r_e1I ** l1 * r_e2I ** l2 * r_ee * l3
                    res += a * (r_e1I - f_cutoff) ** trunc * (r_e2I - f_cutoff) ** trunc

    for i in range(atoms.shape[0]):
        for j in range(r_d.shape[0]):
            for k in range(j+1, r_d.shape[0]):
                r_ee = np.linalg.norm(r_u[j] - r_u[k])  # FIXME to slow
                r_e1I = np.linalg.norm(atoms[i]['position'] - r_u[j])  # FIXME to slow
                r_e2I = np.linalg.norm(atoms[i]['position'] - r_u[k])  # FIXME to slow
                if r_e1I <= f_cutoff and r_e2I <= f_cutoff:
                    a = 0.0
                    for l1 in range(f_parameters.shape[1]):
                        for l2 in range(f_parameters.shape[1]):
                            for l3 in range(f_parameters.shape[1]):
                                a += f_parameters[i, l1, l2, l3, 2] * r_e1I ** l1 * r_e2I ** l2 * r_ee * l3
                    res += a * (r_e1I - f_cutoff) ** trunc * (r_e2I - f_cutoff) ** trunc

    return res


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
