#!/usr/bin/env python3
from math import sqrt

import numpy as np
import numba as nb


@nb.jit(nopython=True)
def jastrow(trunc, u_parameters, u_cutoff, r_u, r_d, atoms):
    """Jastrow
    :param u_parameters:
    :param r_u: up-electrons coordinates
    :param r_d: down-electrons coordinates
    :param atoms:
    :return:
    """
    return 1.0
    res = 0.0

    for i in range(r_u.shape[0]):
        for j in range(i + 1, r_u.shape[0]):
            x = r_d[i][0] - r_d[j][0]
            y = r_d[i][1] - r_d[j][1]
            z = r_d[i][2] - r_d[j][2]
            r = sqrt(x * x + y * y + z * z)
            if r <= u_cutoff:
                a = 0.0
                for k in range(u_parameters.shape[0]):
                    a += u_parameters[k, 0]*r**k
                res += a * (r - u_cutoff)**trunc

    for i in range(r_u.shape[0]):
        for j in range(r_d.shape[0]):
            x = r_u[i][0] - r_d[j][0]
            y = r_u[i][1] - r_d[j][1]
            z = r_u[i][2] - r_d[j][2]
            r = sqrt(x * x + y * y + z * z)
            if r <= u_cutoff:
                a = 0.0
                for k in range(u_parameters.shape[0]):
                    a += u_parameters[k, 1]*r**k
                res += a * (r - u_cutoff) ** trunc

    for i in range(r_d.shape[0]):
        for j in range(i + 1, r_d.shape[0]):
            x = r_d[i][0] - r_d[j][0]
            y = r_d[i][1] - r_d[j][1]
            z = r_d[i][2] - r_d[j][2]
            r = sqrt(x * x + y * y + z * z)
            if r <= u_cutoff:
                a = 0.0
                for k in range(u_parameters.shape[0]):
                    a += u_parameters[k, 2]*r**k
                res += a * (r - u_cutoff) ** trunc

    return np.exp(res)
