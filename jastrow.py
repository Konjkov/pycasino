#!/usr/bin/env python3

import numpy as np
import numba as nb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from readers.casino import Casino


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
                    poly += u_parameters[k, u_set] * r ** k
                res += poly * (r - L) ** C
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
    :param f_parameters:
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
                if r_e1I <= L[i] and r_e2I <= L[i]:
                    f_set = int(j >= neu) + int(k >= neu)
                    poly = 0.0
                    for l in range(f_parameters.shape[1]):
                        for m in range(f_parameters.shape[2]):
                            for n in range(f_parameters.shape[3]):
                                poly += f_parameters[i, l, m, n, f_set] * r_e1I ** l * r_e2I ** m * r_ee ** n
                    res += poly * (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C
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
            r_vec = r_e[i] - r_e[j]  # FIXME to slow
            r = np.linalg.norm(r_vec)
            if r <= L:
                u_set = int(i >= neu) + int(j >= neu)
                poly = 0.0
                for k in range(u_parameters.shape[0]):
                    poly += u_parameters[k, u_set] * r ** k

                poly_diff = 0.0
                for k in range(1, u_parameters.shape[0]):
                    poly_diff += k * u_parameters[k, u_set] * r ** (k-1)

                gradient = (C * (r-L) ** (C-1) * poly + (r-L) ** C * poly_diff) / r
                res[i, :] += r_vec * gradient
                res[j, :] -= r_vec * gradient
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
            r_vec = r_e[j] - atoms[i]['position']  # FIXME to slow
            r = np.linalg.norm(r_vec)
            if r <= L[i]:
                chi_set = int(j >= neu)
                poly = 0.0
                for k in range(chi_parameters.shape[1]):
                    poly += chi_parameters[i, k, chi_set] * r ** k

                poly_diff = 0.0
                for k in range(1, chi_parameters.shape[1]):
                    poly_diff += k * chi_parameters[i, k, chi_set] * r ** (k-1)

                gradient = (C * (r-L[i]) ** (C-1) * poly + (r-L[i]) ** C * poly_diff) / r
                res[j, :] += r_vec * gradient
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
                r_e1I_vec = r_e[j] - atoms[i]['position']  # FIXME to slow
                r_e2I_vec = r_e[k] - atoms[i]['position']  # FIXME to slow
                r_ee_vec = r_e[j] - r_e[k]  # FIXME to slow
                r_e1I = np.linalg.norm(r_e1I_vec)
                r_e2I = np.linalg.norm(r_e2I_vec)
                r_ee = np.linalg.norm(r_ee_vec)
                if r_e1I <= L[i] and r_e2I <= L[i]:
                    f_set = int(j >= neu) + int(k >= neu)
                    poly = 0.0
                    for l in range(f_parameters.shape[1]):
                        for m in range(f_parameters.shape[2]):
                            for n in range(f_parameters.shape[3]):
                                poly += f_parameters[i, l, m, n, f_set] * r_e1I ** l * r_e2I ** m * r_ee ** n

                    poly_diff_e1I = 0.0
                    for l in range(1, f_parameters.shape[1]):
                        for m in range(f_parameters.shape[2]):
                            for n in range(f_parameters.shape[3]):
                                poly_diff_e1I += f_parameters[i, l, m, n, f_set] * l * r_e1I ** (l-1) * r_e2I ** m * r_ee ** n

                    poly_diff_e2I = 0.0
                    for l in range(f_parameters.shape[1]):
                        for m in range(1, f_parameters.shape[2]):
                            for n in range(f_parameters.shape[3]):
                                poly_diff_e2I += f_parameters[i, l, m, n, f_set] * r_e1I ** l * m * r_e2I ** (m-1) * r_ee ** n

                    poly_diff_ee = 0.0
                    for l in range(f_parameters.shape[1]):
                        for m in range(f_parameters.shape[2]):
                            for n in range(1, f_parameters.shape[3]):
                                poly_diff_ee += f_parameters[i, l, m, n, f_set] * r_e1I ** l * r_e2I ** m * n * r_ee ** (n-1)

                    gradient = (C * (r_e1I - L[i]) ** (C-1) * (r_e2I - L[i]) ** C * poly + (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_e1I) / r_e1I
                    res[j, :] += r_e1I_vec * gradient

                    gradient = ((r_e1I - L[i]) ** C * C * (r_e2I - L[i]) ** (C-1) * poly + (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_e2I) / r_e2I
                    res[k, :] += r_e2I_vec * gradient

                    gradient = (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_ee / r_ee
                    res[j, :] += r_ee_vec * gradient
                    res[k, :] -= r_ee_vec * gradient
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
                    poly += u_parameters[k, u_set] * r ** k

                poly_diff = 0.0
                for k in range(1, u_parameters.shape[0]):
                    poly_diff += k * u_parameters[k, u_set] * r ** (k-1)

                poly_diff_2 = 0.0
                for k in range(2, u_parameters.shape[0]):
                    poly_diff_2 += k * (k-1) * u_parameters[k, u_set] * r ** (k-2)

                res += (
                    C*(C - 1)*(r-L)**(C - 2) * poly + 2*C*(r-L)**(C - 1) * poly_diff + (r-L)**C * poly_diff_2 +
                    2 * (C * (r-L)**(C-1) * poly + (r-L)**C * poly_diff) / r
                )
    return 2 * res


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
                    poly += chi_parameters[i, k, chi_set] * r ** k

                poly_diff = 0.0
                for k in range(1, chi_parameters.shape[1]):
                    poly_diff += k * chi_parameters[i, k, chi_set] * r ** (k-1)

                poly_diff_2 = 0.0
                for k in range(2, chi_parameters.shape[1]):
                    poly_diff_2 += k * (k-1) * chi_parameters[i, k, chi_set] * r ** (k-2)

                res += (
                    C*(C - 1)*(r-L[i])**(C - 2) * poly + 2*C*(r-L[i])**(C - 1) * poly_diff + (r-L[i])**C * poly_diff_2 +
                    2 * (C * (r-L[i])**(C-1) * poly + (r-L[i])**C * poly_diff) / r
                )
    return res


@nb.jit(nopython=True)
def f_term_laplacian(C, f_parameters, L, r_e, neu, atoms):
    """Jastrow f-term laplacian
    f-term is a product of two spherically symmetric functions f(r_eI) and g(r_ee) so using
        ∇²(f*g) = ∇²(f)*g + 2*∇(f)*∇(g) + f*∇²(g)
    then Laplace operator of spherically symmetric function (in 3-D space) is
        ∇²(f) = d²f/dr² + 2/r * df/dr
    :param C: truncation order
    :param f_parameters:
    :param L: cutoff length
    :param r_e: electrons coordinates
    :param neu: number of up electrons
    :param atoms: atomic coordinates
    :return:
    """
    res = 0.0
    if not L.any():
        return res

    for i in range(atoms.shape[0]):
        for j in range(r_e.shape[0] - 1):
            for k in range(j + 1, r_e.shape[0]):
                r_e1I_vec = r_e[j] - atoms[i]['position']  # FIXME to slow
                r_e2I_vec = r_e[k] - atoms[i]['position']  # FIXME to slow
                r_ee_vec = r_e[j] - r_e[k]  # FIXME to slow
                r_e1I = np.linalg.norm(r_e1I_vec)
                r_e2I = np.linalg.norm(r_e2I_vec)
                r_ee = np.linalg.norm(r_ee_vec)
                if r_e1I <= L[i] and r_e2I <= L[i]:
                    f_set = int(j >= neu) + int(k >= neu)
                    poly = 0.0
                    for l in range(f_parameters.shape[1]):
                        for m in range(f_parameters.shape[2]):
                            for n in range(f_parameters.shape[3]):
                                poly += f_parameters[i, l, m, n, f_set] * r_e1I ** l * r_e2I ** m * r_ee ** n

                    poly_diff_e1I = 0.0
                    for l in range(1, f_parameters.shape[1]):
                        for m in range(f_parameters.shape[2]):
                            for n in range(f_parameters.shape[3]):
                                poly_diff_e1I += f_parameters[i, l, m, n, f_set] * l * r_e1I ** (l-1) * r_e2I ** m * r_ee ** n

                    poly_diff_e2I = 0.0
                    for l in range(f_parameters.shape[1]):
                        for m in range(1, f_parameters.shape[2]):
                            for n in range(f_parameters.shape[3]):
                                poly_diff_e2I += f_parameters[i, l, m, n, f_set] * r_e1I ** l * m * r_e2I ** (m-1) * r_ee ** n

                    poly_diff_ee = 0.0
                    for l in range(f_parameters.shape[1]):
                        for m in range(f_parameters.shape[2]):
                            for n in range(1, f_parameters.shape[3]):
                                poly_diff_ee += f_parameters[i, l, m, n, f_set] * r_e1I ** l * r_e2I ** m * n * r_ee ** (n-1)

                    poly_diff_e1I_2 = 0.0
                    for l in range(2, f_parameters.shape[1]):
                        for m in range(f_parameters.shape[2]):
                            for n in range(f_parameters.shape[3]):
                                poly_diff_e1I_2 += f_parameters[i, l, m, n, f_set] * l * (l-1) * r_e1I ** (l-2) * r_e2I ** m * r_ee ** n

                    poly_diff_e2I_2 = 0.0
                    for l in range(f_parameters.shape[1]):
                        for m in range(2, f_parameters.shape[2]):
                            for n in range(f_parameters.shape[3]):
                                poly_diff_e2I_2 += f_parameters[i, l, m, n, f_set] * r_e1I ** l * m * (m-1) * r_e2I ** (m-2) * r_ee ** n

                    poly_diff_ee_2 = 0.0
                    for l in range(f_parameters.shape[1]):
                        for m in range(f_parameters.shape[2]):
                            for n in range(2, f_parameters.shape[3]):
                                poly_diff_ee_2 += f_parameters[i, l, m, n, f_set] * r_e1I ** l * r_e2I ** m * n * (n-1) * r_ee ** (n-2)

                    poly_diff_e1I_ee = 0.0
                    for l in range(1, f_parameters.shape[1]):
                        for m in range(f_parameters.shape[2]):
                            for n in range(1, f_parameters.shape[3]):
                                poly_diff_e1I_ee += f_parameters[i, l, m, n, f_set] * l * r_e1I ** (l-1) * r_e2I ** m * n * r_ee ** (n-1)

                    poly_diff_e2I_ee = 0.0
                    for l in range(f_parameters.shape[1]):
                        for m in range(1, f_parameters.shape[2]):
                            for n in range(1, f_parameters.shape[3]):
                                poly_diff_e2I_ee += f_parameters[i, l, m, n, f_set] * r_e1I ** l * m * r_e2I ** (m-1) * n * r_ee ** (n-1)

                    gradient = (
                        (C * (r_e1I - L[i]) ** (C-1) * (r_e2I - L[i]) ** C * poly + (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_e1I) / r_e1I +
                        ((r_e1I - L[i]) ** C * C * (r_e2I - L[i]) ** (C-1) * poly + (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_e2I) / r_e2I +
                        2 * (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_ee / r_ee
                    )

                    laplacian = (
                            C * (C - 1) * (r_e1I - L[i]) ** (C - 2) * (r_e2I - L[i]) ** C * poly +
                            (r_e1I - L[i]) ** C * C * (C - 1) * (r_e2I - L[i]) ** (C - 2) * poly +
                            (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * (poly_diff_e1I_2 + poly_diff_e2I_2 + 2 * poly_diff_ee_2) +
                            2 * C * (r_e1I - L[i]) ** (C - 1) * (r_e2I - L[i]) ** C * poly_diff_e1I +
                            2 * (r_e1I - L[i]) ** C * C * (r_e2I - L[i]) ** (C - 1) * poly_diff_e2I
                    )

                    dot_product = (
                            np.sum(r_e1I_vec * r_ee_vec) * (
                                    C * (r_e1I - L[i]) ** (C-1) * (r_e2I - L[i]) ** C * poly_diff_ee +
                                    (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_e1I_ee
                            ) / r_e1I / r_ee -
                            np.sum(r_e2I_vec * r_ee_vec) * (
                                    (r_e1I - L[i]) ** C * C * (r_e2I - L[i]) ** (C-1) * poly_diff_ee +
                                    (r_e1I - L[i]) ** C * (r_e2I - L[i]) ** C * poly_diff_e2I_ee
                            ) / r_e2I / r_ee
                    )

                    res += laplacian + 2 * gradient + 2 * dot_product
    return res


@nb.jit(nopython=True)
def jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms):
    """Jastrow
    :param u_parameters:
    :param r_e: electrons coordinates
    :param neu: number of up electrons
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

    res = np.zeros(r_e.shape)

    for i in range(r_e.shape[0]):
        for j in range(r_e.shape[1]):
            r_e[i, j] -= delta
            res[i, j] -= jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)
            r_e[i, j] += 2 * delta
            res[i, j] += jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)
            r_e[i, j] -= delta

    return res / delta / 2


@nb.jit(nopython=True)
def jastrow_numerical_laplacian(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms):
    delta = 0.00001

    res = -2 * r_e.size * jastrow(trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff, r_e, neu, atoms)
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


if __name__ == '__main__':
    """
    """

    term = 'f'

    # path = 'test/gwfn/he/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/be/HF/cc-pVQZ/VMC_OPT/emin/casl/8_8_44/10000/'
    # path = 'test/gwfn/be2/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/al/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'
    # path = 'test/gwfn/acetaldehyde/HF/cc-pVQZ/VMC_OPT/emin/legacy/f_term/'

    casino = Casino(path)
    trunc, u_parameters, u_cutoff, chi_parameters, chi_cutoff, f_parameters, f_cutoff = casino.jastrow.trunc, casino.jastrow.u_parameters, casino.jastrow.u_cutoff, casino.jastrow.chi_parameters, casino.jastrow.chi_cutoff, casino.jastrow.f_parameters, casino.jastrow.f_cutoff

    steps = 100

    if term == 'u':
        x_min, x_max = 0, u_cutoff
        x_grid = np.linspace(x_min, x_max, steps)
        for spin_dep in range(3):
            y_grid = np.zeros(steps)
            for i in range(100):
                r_e = np.array([[0.0, 0.0, 0.0], [x_grid[i], 0.0, 0.0]])
                y_grid[i] = u_term(trunc, u_parameters, u_cutoff, r_e, 2-spin_dep)
                if spin_dep == 1:
                    y_grid[i] /= 2.0
            plt.plot(x_grid, y_grid, label=['uu', 'ud/2', 'dd'][spin_dep])
        plt.xlabel('r_ee (au)')
        plt.ylabel('polynomial part')
        plt.title('JASTROW u-term')
    elif term == 'chi':
        for atom in range(casino.wfn.atoms.shape[0]):
            x_min, x_max = 0, chi_cutoff[atom]
            x_grid = np.linspace(x_min, x_max, steps)
            for spin_dep in range(2):
                y_grid = np.zeros(steps)
                for i in range(100):
                    r_e = np.array([[x_grid[i], 0.0, 0.0]]) + casino.wfn.atoms[atom]['position']
                    sl = slice(atom, atom+1)
                    y_grid[i] = chi_term(trunc, chi_parameters[sl], chi_cutoff[sl], r_e, 1-spin_dep, casino.wfn.atoms[sl])
                plt.plot(x_grid, y_grid, label=f'atom {atom} ' + ['u', 'd'][spin_dep])
        plt.xlabel('r_eN (au)')
        plt.ylabel('polynomial part')
        plt.title('JASTROW chi-term')
    elif term == 'f':
        figure = plt.figure()
        axis = figure.add_subplot(111, projection='3d')
        for atom in range(casino.wfn.atoms.shape[0]):
            x_min, x_max = -f_cutoff[atom], f_cutoff[atom]
            y_min, y_max = 0.0, np.pi
            x = np.linspace(x_min, x_max, steps)
            y = np.linspace(y_min, y_max, steps)
            x_grid, y_grid = np.meshgrid(x, y)
            for spin_dep in range(3):
                z_grid = np.zeros((steps, steps))
                for i in range(100):
                    for j in range(100):
                        r_e = np.array([
                            [x_grid[i, j] * np.cos(y_grid[i, j]), x_grid[i, j] * np.sin(y_grid[i, j]), 0.0],
                            [x_grid[i, j], 0.0, 0.0]
                        ]) + casino.wfn.atoms[atom]['position']
                        sl = slice(atom, atom + 1)
                        z_grid[i, j] = f_term(trunc, f_parameters[sl], f_cutoff[sl], r_e, 2-spin_dep, casino.wfn.atoms[sl])
                axis.plot_wireframe(x_grid, y_grid, z_grid, label=f'atom {atom} ' + ['uu', 'ud', 'dd'][spin_dep])
        axis.set_xlabel('r_e1N (au)')
        axis.set_ylabel('r_e2N (au)')
        axis.set_zlabel('polynomial part')
        plt.title('JASTROW f-term')

    plt.grid(True)
    plt.legend()
    plt.show()
