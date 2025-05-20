import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino.abstract import SimpleHarmonics


@structref.register
class Harmonics_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Harmonics_class_t, 'get_value')
def harmonics_get_value(self, x, y, z):
    """Angular part of WFN.
    Solid harmonics with factor √(2 - ẟm,0)(l - |n|)!/(l + |n|)!)
    https://www.vallico.net/casino-forum/viewtopic.php?p=481&sid=9235a407b02d192bef8b66a3ba52e62d#p481
    :return:
    """

    def impl(self, x, y, z) -> np.ndarray:
        x2 = x**2
        y2 = y**2
        z2 = z**2
        r2 = x2 + y2 + z2
        self.value[1] = x
        self.value[2] = y
        self.value[3] = z
        self.value[4] = (3 * z2 - r2) / 2
        self.value[5] = 3 * x * z
        self.value[6] = 3 * y * z
        self.value[7] = 3 * (x2 - y2)
        self.value[8] = 6 * x * y
        if self.l_max > 2:
            self.value[9] = z * (5 * z2 - 3 * r2) / 2
            self.value[10] = 1.5 * x * (5 * z2 - r2)
            self.value[11] = 1.5 * y * (5 * z2 - r2)
            self.value[12] = 15 * z * (x2 - y2)
            self.value[13] = 30 * x * y * z
            self.value[14] = 15 * x * (x2 - 3 * y2)
            self.value[15] = 15 * y * (3 * x2 - y2)
            if self.l_max > 3:
                self.value[16] = (35 * z**4 - 30 * z2 * r2 + 3 * r2**2) / 8
                self.value[17] = 2.5 * x * z * (7 * z2 - 3 * r2)
                self.value[18] = 2.5 * y * z * (7 * z2 - 3 * r2)
                self.value[19] = 7.5 * (x2 - y2) * (7 * z2 - r2)
                self.value[20] = 15 * x * y * (7 * z2 - r2)
                self.value[21] = 105 * x * z * (x2 - 3 * y2)
                self.value[22] = 105 * y * z * (3 * x2 - y2)
                self.value[23] = 105 * (x2**2 - 6 * x2 * y2 + y2**2)
                self.value[24] = 420 * x * y * (x2 - y2)
        return self.value

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Harmonics_class_t, 'get_gradient')
def harmonics_get_gradient(self, x, y, z):
    """Angular part of WFN gradient.
    order: dx, dy, dz
    :return:
    """

    def impl(self, x, y, z) -> np.ndarray:
        x2 = x**2
        y2 = y**2
        z2 = z**2
        self.gradient[1, 0] = 1
        self.gradient[2, 1] = 1
        self.gradient[3, 2] = 1
        self.gradient[4, 0] = -x
        self.gradient[4, 1] = -y
        self.gradient[4, 2] = 2 * z
        self.gradient[5, 0] = 3 * z
        self.gradient[5, 2] = 3 * x
        self.gradient[6, 1] = 3 * z
        self.gradient[6, 2] = 3 * y
        self.gradient[7, 0] = 6 * x
        self.gradient[7, 1] = -6 * y
        self.gradient[8, 0] = 6 * y
        self.gradient[8, 1] = 6 * x
        if self.l_max > 2:
            self.gradient[9, 0] = -3 * x * z
            self.gradient[9, 1] = -3 * y * z
            self.gradient[9, 2] = -1.5 * x2 - 1.5 * y2 + 3 * z2
            self.gradient[10, 0] = -4.5 * x2 - 1.5 * y2 + 6 * z2
            self.gradient[10, 1] = -3 * x * y
            self.gradient[10, 2] = 12 * x * z
            self.gradient[11, 0] = -3 * x * y
            self.gradient[11, 1] = -1.5 * x2 - 4.5 * y2 + 6 * z2
            self.gradient[11, 2] = 12 * y * z
            self.gradient[12, 0] = 30 * x * z
            self.gradient[12, 1] = -30 * y * z
            self.gradient[12, 2] = 15 * x2 - 15 * y2
            self.gradient[13, 0] = 30 * y * z
            self.gradient[13, 1] = 30 * x * z
            self.gradient[13, 2] = 30 * x * y
            self.gradient[14, 0] = 45 * x2 - 45 * y2
            self.gradient[14, 1] = -90 * x * y
            self.gradient[15, 0] = 90 * x * y
            self.gradient[15, 1] = 45 * x2 - 45 * y2
            if self.l_max > 3:
                self.gradient[16, 0] = x * (1.5 * x2 + 1.5 * y2 - 6.0 * z2)
                self.gradient[16, 1] = y * (1.5 * x2 + 1.5 * y2 - 6.0 * z2)
                self.gradient[16, 2] = z * (-6.0 * x2 - 6.0 * y2 + 4.0 * z2)
                self.gradient[17, 0] = z * (-22.5 * x2 - 7.5 * y2 + 10.0 * z2)
                self.gradient[17, 1] = -15.0 * x * y * z
                self.gradient[17, 2] = x * (-7.5 * x2 - 7.5 * y2 + 30.0 * z2)
                self.gradient[18, 0] = -15.0 * x * y * z
                self.gradient[18, 1] = z * (-7.5 * x2 - 22.5 * y2 + 10.0 * z2)
                self.gradient[18, 2] = y * (-7.5 * x2 - 7.5 * y2 + 30.0 * z2)
                self.gradient[19, 0] = x * (-30.0 * x2 + 90.0 * z2)
                self.gradient[19, 1] = y * (30.0 * y2 - 90.0 * z2)
                self.gradient[19, 2] = 90.0 * z * (x2 - y2)
                self.gradient[20, 0] = y * (-45.0 * x2 - 15.0 * y2 + 90.0 * z2)
                self.gradient[20, 1] = x * (-15.0 * x2 - 45.0 * y2 + 90.0 * z2)
                self.gradient[20, 2] = 180.0 * x * y * z
                self.gradient[21, 0] = 315.0 * z * (x2 - y2)
                self.gradient[21, 1] = -630.0 * x * y * z
                self.gradient[21, 2] = x * (105.0 * x2 - 315.0 * y2)
                self.gradient[22, 0] = 630.0 * x * y * z
                self.gradient[22, 1] = 315.0 * z * (x2 - y2)
                self.gradient[22, 2] = y * (315.0 * x2 - 105.0 * y2)
                self.gradient[23, 0] = x * (420.0 * x2 - 1260.0 * y2)
                self.gradient[23, 1] = y * (-1260.0 * x2 + 420.0 * y2)
                self.gradient[24, 0] = y * (1260.0 * x2 - 420.0 * y2)
                self.gradient[24, 1] = x * (420.0 * x2 - 1260.0 * y2)
        return self.gradient

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Harmonics_class_t, 'get_hessian')
def harmonics_get_hessian(self, x, y, z):
    """Angular part of WFN hessian.
    order: dxdx, dxdy, dxdz,
                 dydy, dydz,
                       dzdz
    :return:
    """

    def impl(self, x, y, z) -> np.ndarray:
        x2 = x**2
        y2 = y**2
        z2 = z**2
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
            -3.0*z, 0, -3.0*x, -3.0*z, -3.0*y, 6.0*z,
            -9.0*x, -3.0*y, 12.0*z, -3.0*x, 0, 12.0*x,
            -3.0*y, -3.0*x, 0, -9.0*y, 12.0*z, 12.0*y,
            30.0*z, 0, 30.0*x, -30.0*z, -30.0*y, 0,
            0, 30.0*z, 30.0*y, 0, 30.0*x, 0,
            90.0*x, -90.0*y, 0, -90.0*x, 0, 0,
            90.0*y, 90.0*x, 0, -90.0*y, 0, 0,
            4.5*x2 + 1.5*y2 - 6.0*z2, 3.0*x*y, -12.0*x*z, 1.5*x2 + 4.5*y2 - 6.0*z2, -12.0*y*z, -6.0*x2 - 6.0*y2 + 12.0*z2,
            -45.0*x*z, -15.0*y*z, -22.5*x2 - 7.5*y2 + 30.0*z2, -15.0*x*z, -15.0*x*y, 60.0*x*z,
            -15.0*y*z, -15.0*x*z, -15.0*x*y, -45.0*y*z, -7.5*x2 - 22.5*y2 + 30.0*z2, 60.0*y*z,
            -90.0*x2 + 90.0*z2, 0, 180.0*x*z, 90.0*y2 - 90.0*z2, -180.0*y*z, 90.0*x2 - 90.0*y2,
            -90.0*x*y, -45.0*x2 - 45.0*y2 + 90.0*z2, 180.0*y*z, -90.0*x*y, 180.0*x*z, 180.0*x*y,
            630.0*x*z, -630.0*y*z, 315.0*x2 - 315.0*y2, -630.0*x*z, -630.0*x*y, 0,
            630.0*y*z, 630.0*x*z, 630.0*x*y, -630.0*y*z, 315.0*x2 - 315.0*y2, 0,
            1260.0*x2 - 1260.0*y2, -2520.0*x*y, 0, -1260.0*x2 + 1260.0*y2, 0, 0,
            2520.0*x*y, 1260.0*x2 - 1260.0*y2, 0, -2520.0*x*y, 0, 0,
        ]).reshape(25, 6)  # fmt: skip

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Harmonics_class_t, 'get_tressian')
def harmonics_get_tressian(self, x, y, z):
    """Angular part of WFN 3-rd derivatives.
    order: dxdxdx, dxdxdy, dxdxdz,
                   dxdydy, dxdydz,
                           dxdzdz,
           ----------------------
                   dydydy, dydydz,
                           dydzdz,
           ----------------------
                           dzdzdz
    :return:
    """

    def impl(self, x, y, z) -> np.ndarray:
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
        ]).reshape(25, 10)  # fmt: skip

    return impl


Harmonics_t = Harmonics_class_t(
    [
        ('l_max', nb.int64),
        ('value', nb.float64[::1]),
        ('gradient', nb.float64[:, ::1]),
        ('hessian', nb.float64[:, ::1]),
        ('tressian', nb.float64[:, ::1]),
    ]
)


class Harmonics(structref.StructRefProxy, SimpleHarmonics):
    """Harmonics calculator."""

    def __new__(cls, l_max):
        """Init sphericalHarmonics."""

        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(l_max):
            self = structref.new(Harmonics_t)
            self.l_max = l_max
            self.value = np.ones(shape=((l_max + 1) ** 2,))
            self.gradient = np.zeros(shape=((l_max + 1) ** 2, 3))
            self.hessian = np.zeros(shape=((l_max + 1) ** 2, 6))
            self.tressian = np.zeros(shape=((l_max + 1) ** 2, 10))
            return self

        return init(l_max)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def get_value(self, x, y, z):
        return self.get_value(x, y, z)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def get_gradient(self, x, y, z):
        return self.get_gradient(x, y, z)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def get_hessian(self, x, y, z):
        return self.get_hessian(x, y, z)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def get_tressian(self, x, y, z):
        return self.get_tressian(x, y, z)


structref.define_boxing(Harmonics_class_t, Harmonics)
