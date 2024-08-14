import numpy as np
import numba as nb


pi22 = 3
pi33 = 15
pi44 = 105


@nb.njit(nogil=True, parallel=False, cache=True)
def value_angular_part(x, y, z) -> np.ndarray:
    """Angular part of WFN.
    https://en.wikipedia.org/wiki/Solid_harmonics
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    a2 = x2 - y2
    b2 = 2 * x * y
    a3 = a2 * x - b2 * y
    b3 = a2 * y + b2 * x
    a4 = a3 * x - b3 * y
    b4 = a3 * y + b3 * x
    r2 = x2 + y2 + z2
    pi20 = (3 * z2 - r2) / 2
    pi21 = 3 * z
    pi30 = z * (5 * z2 - 3 * r2) / 2
    pi31 = 3 * (5 * z2 - r2) / 2
    pi32 = 15 * z
    pi40 = (35 * z2 ** 2 - 30 * z2 * r2 + 3 * r2 ** 2) / 8
    pi41 = 5 * z * (7 * z2 - 3 * r2) / 2
    pi42 = 15 * (7 * z2 - r2) / 2
    pi43 = 105 * z
    return np.array([
        1,
        x, y, z,
        pi20, x * pi21, y * pi21, a2 * pi22, b2 * pi22,
        pi30, x * pi31, y * pi31, a2 * pi31, b2 * pi31, a3 * pi33, b3 * pi33,
        pi40, x * pi41, y * pi41, a2 * pi42, b2 * pi42, a3 * pi43, b3 * pi43, a4 * pi44, b4 * pi44
    ])


@nb.njit(nogil=True, parallel=False, cache=True)
def gradient_angular_part(x, y, z) -> np.ndarray:
    """Angular part of WFN gradient.
    order: dx, dy, dz

    https://en.wikipedia.org/wiki/Solid_harmonics
    dA(m)/dx = dB(m)/dy = m * A(m-1)
    -dA(m)/dy = dB(m)/dx = m * B(m-1)
    dA(m)/dz = dB(m)/dz = 0
    dΠ(l, m)/dx = -x * Π(l-1, m+1)
    dΠ(l, m)/dy = -y * Π(l-1, m+1)
    dΠ(l, m)/dz = (l + m) * Π(l-1, m)

    d(Π(l, m) * A(m))/dx = Π(l, m) * m * A(m-1) - x * Π(l-1, m+1) * A(m)
    d(Π(l, m) * A(m))/dz = (l + m) * Π(l-1, m) * A(m)
    d(Π(l, m) * B(m))/dz = (l + m) * Π(l-1, m) * B(m)

    d(pi42 * a2)/dx =  pi42 * 2x - x * pi33 * a2 = x * ( 2 * pi42 - pi33 * a2)
    d(pi42 * b2)/dx =  pi42 * 2y - x * pi33 * b2 = y * ( 2 * pi42 - x * pi33 * b2 / y)
    d(pi42 * a2)/dy = -pi42 * 2y - y * pi33 * a2 = y * (-2 * pi42 - pi33 * a2)
    d(pi42 * b2)/dy =  pi42 * 2x - y * pi33 * b2 = x * ( 2 * pi42 - y * pi33 * b2 / x)
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    a2 = x2 - y2
    b2 = 2 * x * y
    a3 = a2 * x - b2 * y
    b3 = a2 * y + b2 * x
    r2 = x2 + y2 + z2
    pi20 = (3 * z2 - r2) / 2
    pi21 = 3 * z
    pi30 = z * (5 * z2 - 3 * r2) / 2
    pi31 = 3 * (5 * z2 - r2) / 2
    pi32 = 15 * z
    pi41 = 5 * z * (7 * z2 - 3 * r2) / 2
    pi42 = 15 * (7 * z2 - r2) / 2
    pi43 = 105 * z
    return np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        -x, -y, 2*z,
        3*z, 0, 3*x,
        0, 3*z, 3*y,
        6*x, -6*y, 0,
        6*y, 6*x, 0,
        -x * pi21, -y * pi21, 3 * pi20,
        pi31 - 3 * x2, -x * y * pi22, 4 * x * pi21,
        -x * y * pi22, pi31 - 3 * y2, 4 * y * pi21,
        2 * x * pi32, -2 * y * pi32, 5 * a2 * pi22,
        2 * y * pi32, 2 * x * pi32, 5 * b2 * pi22,
        3 * a2 * pi33, -3 * b2 * pi33, 0,
        3 * b2 * pi33, 3 * a2 * pi33, 0,
        -x * pi31, -y * pi31, 4 * pi30,
        pi41 - x2*pi32, -x * y * pi32, 5 * x * pi31,
        -x * y * pi32, pi41 - y2*pi32, 5 * y * pi31,
        15*x*(4*z2 - 2*x2), -15*y*(6*z2 - 2*y2), 6 * a2 * pi32,
        15*y*(6*z2 - 3*x2 - y2), 15*x*(6*z2 - x2 - 3*y2), 6 * b2 * pi32,
        3 * a2 * pi43, -3 * b2 * pi43, 7 *  a3 * pi33,
        3 * b2 * pi43, 3 * a2 * pi43, 7 * b3 * pi33,
        4 * x * a3 * pi44, -4 * y * b3 * pi44, 0,
        4 * y * b3 * pi44, 4 * x * a3 * pi44, 0,
    ]).reshape(25, 3)


@nb.njit(nogil=True, parallel=False, cache=True)
def hessian_angular_part(x, y, z) -> np.ndarray:
    """Angular part of WFN hessian.
    order: dxdx, dxdy, dxdz,
                 dydy, dydz,
                       dzdz
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    a2 = x2 - y2
    b2 = 2 * x * y
    return np.array([
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        -1, 0, 0, -1, 0, 2,
        0, 0, 3, 0, 0, 0,
        0, 0, 0, 0, 3, 0,
        6, 0, 0, -6, 0, 0,
        0, 6, 0, 0, 0, 0,
        -3*z, 0, -3*x, -3*z, -3*y, 6*z,
        -9*x, -3*y, 12*z, -3*x, 0, 12*x,
        -3*y, -3*x, 0, -9*y, 12*z, 12*y,
        30*z, 0, 30*x, -30*z, -30*y, 0,
        0, 30*z, 30*y, 0, 30*x, 0,
        90*x, -90*y, 0, -90*x, 0, 0,
        90*y, 90*x, 0, -90*y, 0, 0,
        4.5*x2 + 1.5*y2 - 6.0*z2, 3.0*x*y, -12.0*x*z, 1.5*x2 + 4.5*y2 - 6.0*z2, -12.0*y*z, 12*z2 - 6*x2 - 6*y2,
        -45*x*z, -15*y*z, 30*z2 - 22.5*x2 - 7.5*y2, -15*x*z, -15*x*y, 60*x*z,
        -15*y*z, -15*x*z, -15*x*y, -45*y*z, 30*z2 -7.5*x2 - 22.5*y2, 60*y*z,
        -90*x2 + 90*z2, 0, 180*x*z, 90*y2 - 90*z2, -180*y*z, 90*a2,
        -90*x*y, 90*z2 - 45*x2 - 45*y2, 180*y*z, -90*x*y, 180*x*z, 90*b2,
        630*x*z, -630*y*z, 315*a2, -630*x*z, -315*b2, 0,
        630*y*z, 630*x*z, 315*b2, -630*y*z, 315*a2, 0,
        1260 * a2, -1260 * b2, 0, -1260 * a2, 0, 0,
        1260 * b2, 1260 * a2, 0, -1260 * b2, 0, 0,
    ]).reshape(25, 6)


@nb.njit(nogil=True, parallel=False, cache=True)
def tressian_angular_part(x, y, z) -> np.ndarray:
    """Angular part of WFN 3-rd derivatives.
    order: dxdxdx, dxdxdy, dxdxdz,
                   dxdydy, dxdydz,
                           dxdzdz,
           ----------------------
                   dydydy, dydydz,
                           dydzdz,
           ----------------------
                           dzdzdz
    """
    return np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -3, 0, 0, 0, 0, -3, 0, 6,
        -9, 0, 0, -3, 0, 12, 0, 0, 0, 0,
        0, -3, 0, 0, 0, 0, -9, 0, 12, 0,
        0, 0, 30, 0, 0, 0, 0, -30, 0, 0,
        0, 0, 0, 0, 30, 0, 0, 0, 0, 0,
        90, 0, 0, -90, 0, 0, 0, 0, 0, 0,
        0, 90, 0, 0, 0, 0, -90, 0, 0, 0,
        9*x, 3*y, -12*z, 3*x, 0, -12*x, 9*y, -12*z, -12*y, 24*z,
        -45*z, 0, -45*x, -15*z, -15*y, 60*z, 0, -15*x, 0, 60*x,
        0, -15*z, -15*y, 0, -15*x, 0, -45*z, -45*y, 60*z, 60*y,
        -180*x, 0, 180*z, 0, 0, 180*x, 180*y, -180*z, -180*y, 0,
        -90*y, -90*x, 0, -90*y, 180*z, 180*y, -90*x, 0, 180*x, 0,
        630*z, 0, 630*x, -630*z, -630*y, 0, 0, -630*x, 0, 0,
        0, 630*z, 630*y, 0, 630*x, 0, -630*z, -630*y, 0, 0,
        2520*x, -2520*y, 0, -2520*x, 0, 0, 2520*y, 0, 0, 0,
        2520*y, 2520*x, 0, -2520*y, 0, 0, -2520*x, 0, 0, 0,
    ]).reshape(25, 10)
