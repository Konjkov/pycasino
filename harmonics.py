from numpy_config import np
import numba as nb


@nb.jit(nopython=True, nogil=True, parallel=False)
def angular_part(x, y, z):
    """Angular part of WFN.
    :return:
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    r2 = x2 + y2 + z2
    return np.array([
        1.0,
        x,
        y,
        z,
        (3.0 * z2 - r2) / 2.0,
        3.0 * x*z,
        3.0 * y*z,
        3.0 * (x2 - y2),
        6.0 * x*y,
        z * (5.0 * z2 - 3.0 * r2) / 2.0,
        1.5 * x * (5.0 * z2 - r2),
        1.5 * y * (5.0 * z2 - r2),
        15.0 * z * (x2 - y2),
        30.0 * x * y*z,
        15.0 * x * (x2 - 3.0 * y2),
        15.0 * y * (3.0 * x2 - y2),
        (35.0 * z**4 - 30.0 * z2 * r2 + 3.0 * r2**2) / 8.0,
        2.5 * x*z * (7.0 * z2 - 3.0 * r2),
        2.5 * y*z * (7.0 * z2 - 3.0 * r2),
        7.5 * (x2 - y2) * (7.0 * z2 - r2),
        15.0 * x*y * (7.0 * z2 - r2),
        105.0 * x*z * (x2 - 3.0 * y2),
        105.0 * y*z * (3.0 * x2 - y2),
        105.0 * (x2**2 - 6.0 * x2 * y2 + y2**2),
        420.0 * x*y * (x2 - y2)
    ])


@nb.jit(nopython=True, nogil=True, parallel=False)
def gradient_angular_part(x, y, z):
    """Angular part of WFN gradient.
    order: dx, dy, dz
    :return:
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    return np.array([
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        -x, -y, 2.0*z,
        3.0*z, 0.0, 3.0*x,
        0.0, 3.0*z, 3.0*y,
        6.0*x, -6.0*y, 0.0,
        6.0*y, 6.0*x, 0.0,
        -3.0*x*z, -3.0*y*z, -1.5*x2 - 1.5*y2 + 3.0*z2,
        -4.5*x2 - 1.5*y2 + 6.0*z2, -3.0*x*y, 12.0*x*z,
        -3.0*x*y, -1.5*x2 - 4.5*y2 + 6.0*z2, 12.0*y*z,
        30.0*x*z, -30.0*y*z, 15.0*x2 - 15.0*y2,
        30.0*y*z, 30.0*x*z, 30.0*x*y,
        45.0*x2 - 45.0*y2, -90.0*x*y, 0,
        90.0*x*y, 45.0*x2 - 45.0*y2, 0,
        x*(1.5*x2 + 1.5*y2 - 6.0*z2), y*(1.5*x2 + 1.5*y2 - 6.0*z2), z*(-6.0*x2 - 6.0*y2 + 4.0*z2),
        z*(-22.5*x2 - 7.5*y2 + 10.0*z2), -15.0*x*y*z, x*(-7.5*x2 - 7.5*y2 + 30.0*z2),
        -15.0*x*y*z, z*(-7.5*x2 - 22.5*y2 + 10.0*z2), y*(-7.5*x2 - 7.5*y2 + 30.0*z2),
        x*(-30.0*x2 + 90.0*z2), y*(30.0*y2 - 90.0*z2), 90.0*z*(x2 - y2),
        y*(-45.0*x2 - 15.0*y2 + 90.0*z2), x*(-15.0*x2 - 45.0*y2 + 90.0*z2), 180.0*x*y*z,
        315.0*z*(x2 - y2), -630.0*x*y*z, x*(105.0*x2 - 315.0*y2),
        630.0*x*y*z, 315.0*z*(x2 - y2), y*(315.0*x2 - 105.0*y2),
        x*(420.0*x2 - 1260.0*y2), y*(-1260.0*x2 + 420.0*y2), 0,
        y*(1260.0*x2 - 420.0*y2), x*(420.0*x2 - 1260.0*y2), 0,
    ]).reshape(25, 3)


@nb.jit(nopython=True, nogil=True, parallel=False)
def hessian_angular_part(x, y, z):
    """Angular part of WFN hessian.
    order: dxdx, dxdy, dydy, dxdz, dydz, dzdz
    :return:
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    return np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -1.0, 0.0, -1.0, 0.0, 0.0, 2.0,
        0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 3.0, 0.0,
        6.0, 0.0, -6.0, 0.0, 0.0, 0.0,
        0.0, 6.0, 0.0, 0.0, 0.0, 0.0,
        -3.0*z, 0, -3.0*z, -3.0*x, -3.0*y, 6.0*z,
        -9.0*x, -3.0*y, -3.0*x, 12.0*z, 0, 12.0*x,
        -3.0*y, -3.0*x, -9.0*y, 0, 12.0*z, 12.0*y,
        30.0*z, 0, -30.0*z, 30.0*x, -30.0*y, 0,
        0, 30.0*z, 0, 30.0 * y, 30.0*x, 0,
        90.0*x, -90.0*y, -90.0*x, 0, 0, 0,
        90.0*y, 90.0*x, -90.0*y, 0, 0, 0,
        4.5*x2 + 1.5*y2 - 6.0*z2, 3.0*x*y, 1.5*x2 + 4.5*y2 - 6.0*z2, -12.0*x*z, -12.0*y*z, -6.0*x2 - 6.0*y2 + 12.0*z2,
        -45.0*x*z, -15.0*y*z, -15.0*x*z, -22.5*x2 - 7.5*y2 + 30.0*z2, -15.0*x*y, 60.0*x*z,
        -15.0*y*z, -15.0*x*z, -45.0*y*z, -15.0*x*y, -7.5*x2 - 22.5*y2 + 30.0*z2, 60.0*y*z,
        -90.0*x2 + 90.0*z2, 0, 90.0*y2 - 90.0*z2, 180.0*x*z, -180.0*y*z, 90.0*x2 - 90.0*y2,
        -90.0*x*y, -45.0*x2 - 45.0*y2 + 90.0*z2, -90.0*x*y, 180.0*y*z, 180.0*x*z, 180.0*x*y,
        630.0*x*z, -630.0*y*z, -630.0*x*z, 315.0*x2 - 315.0*y2, -630.0*x*y, 0,
        630.0*y*z, 630.0*x*z, -630.0*y*z, 630.0*x*y, 315.0*x2 - 315.0*y2, 0,
        1260.0*x2 - 1260.0*y2, -2520.0*x*y, -1260.0*x2 + 1260.0*y2, 0, 0, 0,
        2520.0*x*y, 1260.0*x2 - 1260.0*y2, -2520.0*x*y, 0, 0, 0,
    ]).reshape(25, 6)
