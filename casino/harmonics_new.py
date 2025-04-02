import ctypes
from typing import Tuple

import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method
from sphericart import _c_lib

_c_lib.sphericart_spherical_harmonics_calculator_t = ctypes.c_void_p
_c_lib.sphericart_spherical_harmonics_calculator_f_t = ctypes.c_void_p
_c_lib.sphericart_solid_harmonics_calculator_t = ctypes.c_void_p
_c_lib.sphericart_solid_harmonics_calculator_f_t = ctypes.c_void_p
lib = _c_lib._get_library()


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
            self.value_64(
                self.calculator_64,
                xyz.ctypes,
                xyz.size,
                sph.ctypes,
                sph.size,
            )
        # elif xyz.dtype == np.dtype('float32'):
        #     self.value_32(
        #         self.calculator_32,
        #         xyz.ctypes,
        #         xyz.size,
        #         sph.ctypes,
        #         sph.size,
        #     )
        return sph

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(SphericalHarmonics_class_t, 'compute_with_gradients')
def spherical_harmonics_compute_with_gradients(self, xyz: np.ndarray):
    """Calculates the spherical harmonics for a set of 3D points with gradients.
    :param xyz:
        The Cartesian coordinates of the 3D points, as an array with
        shape ``(n_samples, 3)``
    :return:
        - an array of shape ``(n_samples, (l_max+1)**2)`` containing all the
          spherical harmonics up to degree ``l_max`` in lexicographic order.
          For example, if ``l_max = 2``, The last axis will correspond to
          spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
          1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
        - an array of shape ``(n_samples, 3, (l_max+1)**2)`` containing all
          the spherical harmonics' derivatives up to degree ``l_max``. The
          last axis is organized in the same way as in the spherical
          harmonics return array, while the second-to-last axis refers to
          derivatives in the x, y, and z directions, respectively.
    """

    def impl(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not (xyz.dtype == np.dtype('float32') or xyz.dtype == np.dtype('float64')):
            raise TypeError('xyz must be a numpy array of 32 or 64-bit floats')

        if len(xyz.shape) != 2 or xyz.shape[1] != 3:
            raise ValueError('xyz array must be a `N x 3` array')

        # make xyz contiguous before taking a pointer to it
        if not xyz.flags.c_contiguous:
            xyz = np.ascontiguousarray(xyz)

        n_samples = xyz.shape[0]
        sph = np.empty((n_samples, (self.l_max + 1) ** 2), dtype=xyz.dtype)
        dsph = np.empty((n_samples, 3, (self._l_max + 1) ** 2), dtype=xyz.dtype)

        if xyz.dtype == np.dtype('float64'):
            self.gradient_64(
                self.calculator_64,
                xyz.ctypes,
                xyz.size,
                sph.ctypes,
                sph.size,
                dsph.ctypes,
                dsph.size,
            )
        elif xyz.dtype == np.dtype('float32'):
            self.gradient_32(
                self.calculator_32,
                xyz.ctypes,
                xyz.size,
                sph.ctypes,
                sph.size,
                dsph.ctypes,
                dsph.size,
            )
        return sph, dsph

    return impl


SphericalHarmonics_t = SphericalHarmonics_class_t(
    [
        ('l_max', nb.int64),
        ('calculator_64', nb.int64),
        ('calculator_32', nb.int64),
        ('value_64', nb.typeof(lib.sphericart_spherical_harmonics_compute_array)),
        ('value_32', nb.typeof(lib.sphericart_spherical_harmonics_compute_array_f)),
        ('gradient_64', nb.typeof(lib.sphericart_spherical_harmonics_compute_array_with_gradients)),
        ('gradient_32', nb.typeof(lib.sphericart_spherical_harmonics_compute_array_with_gradients_f)),
        ('hessian_64', nb.typeof(lib.sphericart_spherical_harmonics_compute_array_with_hessians)),
        ('hessian_32', nb.typeof(lib.sphericart_spherical_harmonics_compute_array_with_hessians_f)),
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
                self.calculator_64,
                self.calculator_32,
                self.value_64,
                self.value_32,
                self.gradient_64,
                self.gradient_32,
                self.hessian_64,
                self.hessian_32,
            ) = args
            return self

        args = (
            l_max,
            lib.sphericart_spherical_harmonics_new(l_max),
            lib.sphericart_spherical_harmonics_new_f(l_max),
            lib.sphericart_spherical_harmonics_compute_array,
            lib.sphericart_spherical_harmonics_compute_array_f,
            lib.sphericart_spherical_harmonics_compute_array_with_gradients,
            lib.sphericart_spherical_harmonics_compute_array_with_gradients_f,
            lib.sphericart_spherical_harmonics_compute_array_with_hessians,
            lib.sphericart_spherical_harmonics_compute_array_with_hessians_f,
        )
        return init(*args)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def compute(self, xyz: np.ndarray) -> np.ndarray:
        return self.compute(xyz)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def compute_with_gradients(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.compute_with_gradients(xyz)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def compute_with_hessian(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.compute_with_hessian(xyz)


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
        solid harmonics up to degree `l_max` in lexicographic order.
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
            self.value_64(
                self.calculator_64,
                xyz.ctypes,
                xyz.size,
                sph.ctypes,
                sph.size,
            )
        # elif xyz.dtype == np.dtype('float32'):
        #     self.value_32(
        #         self.calculator_32,
        #         xyz.ctypes,
        #         xyz.size,
        #         sph.ctypes,
        #         sph.size,
        #     )
        return sph

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(SolidHarmonics_class_t, 'compute_with_gradients')
def solid_harmonics_compute_with_gradients(self, xyz: np.ndarray):
    """Calculates the solid harmonics for a set of 3D points with gradients.
    :param xyz:
        The Cartesian coordinates of the 3D points, as an array with
        shape ``(n_samples, 3)``
    :return:
        - an array of shape ``(n_samples, (l_max+1)**2)`` containing all the
          solid harmonics up to degree ``l_max`` in lexicographic order.
          For example, if ``l_max = 2``, The last axis will correspond to
          spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
          1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
        - an array of shape ``(n_samples, 3, (l_max+1)**2)`` containing all
          the solid harmonics' derivatives up to degree ``l_max``. The
          last axis is organized in the same way as in the spherical
          harmonics return array, while the second-to-last axis refers to
          derivatives in the x, y, and z directions, respectively.
    """

    def impl(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not (xyz.dtype == np.dtype('float32') or xyz.dtype == np.dtype('float64')):
            raise TypeError('xyz must be a numpy array of 32 or 64-bit floats')

        if len(xyz.shape) != 2 or xyz.shape[1] != 3:
            raise ValueError('xyz array must be a `N x 3` array')

        # make xyz contiguous before taking a pointer to it
        if not xyz.flags.c_contiguous:
            xyz = np.ascontiguousarray(xyz)

        n_samples = xyz.shape[0]
        sph = np.empty((n_samples, (self.l_max + 1) ** 2), dtype=xyz.dtype)
        dsph = np.empty((n_samples, 3, (self._l_max + 1) ** 2), dtype=xyz.dtype)

        if xyz.dtype == np.dtype('float64'):
            self.gradient_64(
                self.calculator_64,
                xyz.ctypes,
                xyz.size,
                sph.ctypes,
                sph.size,
                dsph.ctypes,
                dsph.size,
            )
        # elif xyz.dtype == np.dtype('float32'):
        #     self.gradient_32(
        #         self.calculator_32,
        #         xyz.ctypes,
        #         xyz.size,
        #         sph.ctypes,
        #         sph.size,
        #         dsph.ctypes,
        #         dsph.size,
        #     )
        return sph, dsph

    return impl


SolidHarmonics_t = SolidHarmonics_class_t(
    [
        ('l_max', nb.int64),
        ('calculator_64', nb.types.int64),
        ('calculator_32', nb.types.int64),
        ('value_64', nb.typeof(lib.sphericart_solid_harmonics_compute_array)),
        ('value_32', nb.typeof(lib.sphericart_solid_harmonics_compute_array_f)),
        ('gradient_64', nb.typeof(lib.sphericart_solid_harmonics_compute_array_with_gradients)),
        ('gradient_32', nb.typeof(lib.sphericart_solid_harmonics_compute_array_with_gradients_f)),
        ('hessian_64', nb.typeof(lib.sphericart_solid_harmonics_compute_array_with_hessians)),
        ('hessian_32', nb.typeof(lib.sphericart_solid_harmonics_compute_array_with_hessians_f)),
    ]
)


class SolidHarmonics(structref.StructRefProxy):
    """Solid harmonics calculator."""

    def __new__(cls, l_max: int):
        """Init calculator, up to degree l_max"""

        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(*args):
            self = structref.new(SolidHarmonics_t)
            (
                self.l_max,
                self.calculator_64,
                self.calculator_32,
                self.value_64,
                self.value_32,
                self.gradient_64,
                self.gradient_32,
                self.hessian_64,
                self.hessian_32,
            ) = args
            return self

        args = (
            l_max,
            lib.sphericart_solid_harmonics_new(l_max),
            lib.sphericart_solid_harmonics_new_f(l_max),
            lib.sphericart_solid_harmonics_compute_array,
            lib.sphericart_solid_harmonics_compute_array_f,
            lib.sphericart_solid_harmonics_compute_array_with_gradients,
            lib.sphericart_solid_harmonics_compute_array_with_gradients_f,
            lib.sphericart_solid_harmonics_compute_array_with_hessians,
            lib.sphericart_solid_harmonics_compute_array_with_hessians_f,
        )
        return init(*args)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def compute(self, xyz: np.ndarray) -> np.ndarray:
        return self.compute(xyz)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def compute_with_gradients(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.compute_with_gradients(xyz)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def compute_with_hessian(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.compute_with_hessian(xyz)


structref.define_boxing(SolidHarmonics_class_t, SolidHarmonics)
