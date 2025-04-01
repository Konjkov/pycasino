import ctypes

import numba as nb
import numpy as np
from numba.core import cgutils, types
from numba.experimental import structref
from numba.extending import overload_method
from sphericart._c_lib import _get_library

lib = _get_library()
# Spherical
sphericart_spherical_harmonics_new = lib.sphericart_spherical_harmonics_new
sphericart_spherical_harmonics_new.restype = ctypes.c_void_p
sphericart_spherical_harmonics_new.argtypes = [ctypes.c_size_t]
sphericart_spherical_harmonics_new_f = lib.sphericart_spherical_harmonics_new_f
sphericart_spherical_harmonics_new_f.restype = ctypes.c_void_p
sphericart_spherical_harmonics_new_f.argtypes = [ctypes.c_size_t]

sphericart_spherical_harmonics_compute_array = lib.sphericart_spherical_harmonics_compute_array
sphericart_spherical_harmonics_compute_array.restype = None
sphericart_spherical_harmonics_compute_array.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_int64,
]
sphericart_spherical_harmonics_compute_array_f = lib.sphericart_spherical_harmonics_compute_array_f
sphericart_spherical_harmonics_compute_array_f.restype = None
sphericart_spherical_harmonics_compute_array_f.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_int64,
]
# Solid
sphericart_solid_harmonics_new = lib.sphericart_solid_harmonics_new
sphericart_solid_harmonics_new.restype = ctypes.c_void_p
sphericart_solid_harmonics_new.argtypes = [ctypes.c_size_t]
sphericart_solid_harmonics_new_f = lib.sphericart_solid_harmonics_new_f
sphericart_solid_harmonics_new_f.restype = ctypes.c_void_p
sphericart_solid_harmonics_new_f.argtypes = [ctypes.c_size_t]

sphericart_solid_harmonics_compute_array = lib.sphericart_solid_harmonics_compute_array
sphericart_solid_harmonics_compute_array.restype = None
sphericart_solid_harmonics_compute_array.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_int64,
]
sphericart_solid_harmonics_compute_array_f = lib.sphericart_solid_harmonics_compute_array_f
sphericart_solid_harmonics_compute_array_f.restype = None
sphericart_solid_harmonics_compute_array_f.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int64,
]


@nb.extending.intrinsic
def address_as_void_ptr(typingctx, data):
    """Returns given memory address as a void pointer.
    :return: (void*) (long) addr.
    """

    def impl(context, builder, signature, args):
        val = builder.inttoptr(args[0], cgutils.voidptr_t)
        return val

    sig = types.voidptr(data)
    return sig, impl


@structref.register
class SphericalHarmonics_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(SphericalHarmonics_class_t, 'compute')
def spherical_harmonics_compute(self, xyz: np.ndarray):
    """Calculates the spherical harmonics for a set of 3D points.
    :param xyz:
        The Cartesian coordinates of the 3D points, as an array with
        shape ``(n_samples, 3)``
    :return:
        An array of shape ``(n_samples, (l_max+1)**2)`` containing all the
        spherical harmonics up to degree `l_max` in lexicographic order.
        For example, if ``l_max = 2``, The last axis will correspond to
        spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
        1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
    """

    def impl(self, xyz: np.ndarray) -> np.ndarray:
        if not (xyz.dtype == np.dtype('float32') or xyz.dtype == np.dtype('float64')):
            raise TypeError('xyz must be a numpy array of 32 or 64-bit floats')

        if len(xyz.shape) != 2 or xyz.shape[1] != 3:
            raise ValueError('xyz array must be a `N x 3` array')

        # make xyz contiguous before taking a pointer to it
        if not xyz.flags.c_contiguous:
            xyz = np.ascontiguousarray(xyz)

        n_samples = xyz.shape[0]
        sph = np.empty((n_samples, (self.l_max + 1) ** 2), dtype=xyz.dtype)

        if xyz.dtype == np.dtype('float64'):
            _calculator = self.sphericart_spherical_harmonics_new(self.l_max)
            self.sphericart_spherical_harmonics_compute_array(
                _calculator,
                xyz.ctypes.data,
                xyz.size,
                sph.ctypes.data,
                sph.size,
            )
        elif xyz.dtype == np.dtype('float32'):
            _calculator_f = self.sphericart_spherical_harmonics_new_f(self.l_max)
            self.sphericart_spherical_harmonics_compute_array_f(
                _calculator_f,
                xyz.ctypes.data,
                xyz.size,
                sph.ctypes.data,
                sph.size,
            )
        return sph

    return impl


SphericalHarmonics_t = SphericalHarmonics_class_t(
    [
        ('l_max', nb.int64),
        ('_calculator', nb.uint64),
        ('_calculator_f', nb.uint64),
        ('sphericart_spherical_harmonics_compute_array', nb.typeof(sphericart_spherical_harmonics_compute_array)),
        ('sphericart_spherical_harmonics_compute_array_f', nb.typeof(sphericart_spherical_harmonics_compute_array_f)),
        ('sphericart_spherical_harmonics_new', nb.typeof(sphericart_spherical_harmonics_new)),
        ('sphericart_spherical_harmonics_new_f', nb.typeof(sphericart_spherical_harmonics_new_f)),
    ]
)


class SphericalHarmonics(structref.StructRefProxy):
    """Spherical harmonics calculator."""

    def __new__(cls, l_max: int):
        """Init calculator, up to degree l_max"""

        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(*args):
            self = structref.new(SphericalHarmonics_t)
            (
                self.l_max,
                self.sphericart_spherical_harmonics_compute_array,
                self.sphericart_spherical_harmonics_compute_array_f,
                self.sphericart_spherical_harmonics_new,
                self.sphericart_spherical_harmonics_new_f,
            ) = args
            # self._calculator = _sphericart_spherical_harmonics_new(self.l_max)
            # self._calculator_f = _sphericart_spherical_harmonics_new_f(self.l_max)
            return self

        args = (
            l_max,
            lib.sphericart_spherical_harmonics_compute_array,
            lib.sphericart_spherical_harmonics_compute_array_f,
            lib.sphericart_spherical_harmonics_new,
            lib.sphericart_spherical_harmonics_new_f,
        )
        return init(*args)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def compute(self, xyz: np.ndarray) -> np.ndarray:
        return self.compute(xyz)


structref.define_boxing(SphericalHarmonics_class_t, SphericalHarmonics)


@structref.register
class SolidHarmonics_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(SolidHarmonics_class_t, 'compute')
def solid_harmonics_compute(self, xyz: np.ndarray):
    """Calculates the solid harmonics for a set of 3D points.
    :param xyz:
        The Cartesian coordinates of the 3D points, as an array with
        shape ``(n_samples, 3)``
    :return:
        An array of shape ``(n_samples, (l_max+1)**2)`` containing all the
        spherical harmonics up to degree `l_max` in lexicographic order.
        For example, if ``l_max = 2``, The last axis will correspond to
        spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
        1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
    """

    def impl(self, xyz: np.ndarray) -> np.ndarray:
        if not (xyz.dtype == np.dtype('float32') or xyz.dtype == np.dtype('float64')):
            raise TypeError('xyz must be a numpy array of 32 or 64-bit floats')

        if len(xyz.shape) != 2 or xyz.shape[1] != 3:
            raise ValueError('xyz array must be a `N x 3` array')

        # make xyz contiguous before taking a pointer to it
        if not xyz.flags.c_contiguous:
            xyz = np.ascontiguousarray(xyz)

        n_samples = xyz.shape[0]
        sph = np.empty((n_samples, (self.l_max + 1) ** 2), dtype=xyz.dtype)

        if xyz.dtype == np.dtype('float64'):
            _calculator = self.sphericart_spherical_harmonics_new(self.l_max)
            self.sphericart_spherical_harmonics_compute_array(
                _calculator,
                xyz.ctypes.data,
                xyz.size,
                sph.ctypes.data,
                sph.size,
            )
        elif xyz.dtype == np.dtype('float32'):
            _calculator_f = self.sphericart_spherical_harmonics_new_f(self.l_max)
            self.sphericart_spherical_harmonics_compute_array_f(
                _calculator_f,
                xyz.ctypes.data,
                xyz.size,
                sph.ctypes.data,
                sph.size,
            )
        return sph

    return impl


SolidHarmonics_t = SolidHarmonics_class_t(
    [
        ('l_max', nb.int64),
        ('_calculator', nb.types.voidptr),
        ('_calculator_f', nb.types.voidptr),
        ('sphericart_solid_harmonics_compute_array', nb.typeof(sphericart_solid_harmonics_compute_array)),
        ('sphericart_solid_harmonics_compute_array_f', nb.typeof(sphericart_solid_harmonics_compute_array_f)),
        ('sphericart_solid_harmonics_new', nb.typeof(sphericart_solid_harmonics_new)),
        ('sphericart_solid_harmonics_new_f', nb.typeof(sphericart_solid_harmonics_new_f)),
    ]
)


class SolidHarmonics(structref.StructRefProxy):
    """Solid harmonics calculator."""

    def __new__(cls, l_max: int):
        """Init calculator, up to degree l_max"""

        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(*args):
            self = structref.new(SphericalHarmonics_t)
            (
                self.l_max,
                self.sphericart_solid_harmonics_compute_array,
                self.sphericart_solid_harmonics_compute_array_f,
                self.sphericart_solid_harmonics_new,
                self.sphericart_solid_harmonics_new_f,
            ) = args
            # self._calculator = self.sphericart_solid_harmonics_new(self.l_max)
            # self._calculator_f = self.sphericart_solid_harmonics_new_f(self.l_max)
            return self

        args = (
            l_max,
            lib.sphericart_solid_harmonics_compute_array,
            lib.sphericart_solid_harmonics_compute_array_f,
            lib.sphericart_solid_harmonics_new,
            lib.sphericart_solid_harmonics_new_f,
        )
        return init(*args)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def compute(self, xyz: np.ndarray) -> np.ndarray:
        return self.compute(xyz)


structref.define_boxing(SolidHarmonics_class_t, SolidHarmonics)
