import numba as nb
import numpy as np
from mpi4py import MPI
from enum import IntEnum
from numba.core import cgutils, types
from numba.core.typing.ctypes_utils import get_pointer
from numba.extending import overload_method
from numba.experimental import structref
from .header import MPI_Initialized, MPI_Barrier, MPI_Comm_size, MPI_Comm_rank, MPI_Allreduce

# https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html

# Cannot cache compiled function "init" as it uses dynamic globals (such as ctypes pointers and large global arrays)
class MPI_Communicators(IntEnum):
    """Global Communicators."""
    COMM_NULL = MPI._addressof(MPI.COMM_NULL)
    COMM_SELF = MPI._addressof(MPI.COMM_SELF)
    COMM_WORLD = MPI._addressof(MPI.COMM_WORLD)


# Cannot cache compiled function "init" as it uses dynamic globals (such as ctypes pointers and large global arrays)
class MPI_Operator(IntEnum):
    """Global Reduction Operations."""
    MAX = MPI._addressof(MPI.MAX)
    MIN = MPI._addressof(MPI.MIN)
    SUM = MPI._addressof(MPI.SUM)
    PROD = MPI._addressof(MPI.PROD)
    LAND = MPI._addressof(MPI.LAND)
    BAND = MPI._addressof(MPI.BAND)


# Cannot cache compiled function "init" as it uses dynamic globals (such as ctypes pointers and large global arrays)
class MPI_DTYPES(IntEnum):
    """Global Data Types."""
    CHAR = MPI._addressof(MPI.CHAR)
    INT32_T = MPI._addressof(MPI.INT32_T)
    INT64_T = MPI._addressof(MPI.INT64_T)
    FLOAT = MPI._addressof(MPI.FLOAT)
    DOUBLE = MPI._addressof(MPI.DOUBLE)
    C_FLOAT_COMPLEX = MPI._addressof(MPI.C_FLOAT_COMPLEX)
    C_DOUBLE_COMPLEX = MPI._addressof(MPI.C_DOUBLE_COMPLEX)


# Calling C code from Numba
# https://numba.readthedocs.io/en/stable/user/cfunc.html#calling-c-code-from-numba
# https://github.com/numba/numba/issues/4115
# https://stackoverflow.com/questions/61509903/how-to-pass-array-pointer-to-numba-function
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


@nb.extending.intrinsic
def val_to_void_ptr(typingctx, data):
    """Returns void pointer to a variable.
    :return: void* val
    """
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr
    sig = types.voidptr(data)
    return sig, impl


# https://github.com/numba/numba/issues/7399
# https://stackoverflow.com/questions/51541302/how-to-wrap-a-cffi-function-in-numba-taking-pointers
@nb.extending.intrinsic
def val_to_ptr(typingctx, data):
    """Returns pointer to a variable.
    :return: * val
    """
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr
    sig = types.CPointer(data)(data)
    return sig, impl


@nb.extending.intrinsic
def val_from_ptr(typingctx, data):
    """Returns value by pointer.
    :return: & ptr
    """
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val
    sig = data.dtype(data)
    return sig, impl


# https://groups.google.com/g/mpi4py/c/jPqNrr_8UWY?pli=1
@structref.register
class Comm_class_t(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, '_mpi_addr')
def comm_mpi_addr(self, addr):
    """Return long value from given memory address as a void pointer.
    :return: (long) & (void*) (long) addr.
    """
    def impl(self, addr):
        return nb.carray(address_as_void_ptr(addr), shape=(1,), dtype=np.intp,)[0]
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'Initialized')
def comm_Initialized(self):
    """int MPI_Initialized(int *flag)."""
    def impl(self):
        flag_ptr = val_to_ptr(False)
        status = MPI_Initialized(flag_ptr)
        assert status == 0
        return val_from_ptr(flag_ptr)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'Barrier')
def comm_Barrier(self):
    """int MPI_Barrier(MPI_Comm comm)."""
    def impl(self):
        status = MPI_Barrier(self._mpi_addr(self.MPI_Comm_World_ptr))
        assert status == 0
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'Get_size')
def comm_Get_size(self):
    """int MPI_Comm_size(MPI_Comm comm, int *size)."""
    def impl(self):
        size_ptr = val_to_ptr(0)
        status = MPI_Comm_size(self._mpi_addr(self.MPI_Comm_World_ptr), size_ptr)
        assert status == 0
        return val_from_ptr(size_ptr)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'Get_rank')
def comm_Get_rank(self):
    """int MPI_Comm_rank(MPI_Comm comm, int *rank)."""
    def impl(self):
        size_ptr = val_to_ptr(0)
        status = MPI_Comm_rank(self._mpi_addr(self.MPI_Comm_World_ptr), size_ptr)
        assert status == 0
        return val_from_ptr(size_ptr)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'allreduce')
def comm_allreduce(self, sendobj, operator=MPI_Operator.SUM):
    """int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                     MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)."""
    def impl(self, sendobj, operator=MPI_Operator.SUM):
        sendobj_buf = np.array([sendobj])
        recvobj_buf = np.zeros_like(sendobj_buf)
        if sendobj_buf.dtype == np.dtype('uint8'):
            datatype = self.MPI_DTYPES.CHAR
        elif sendobj_buf.dtype == np.dtype('int32'):
            datatype = self.MPI_DTYPES.INT32_T
        elif sendobj_buf.dtype == np.dtype('int64'):
            datatype = self.MPI_DTYPES.INT64_T
        elif sendobj_buf.dtype == np.dtype('float32'):
            datatype = self.MPI_DTYPES.FLOAT
        elif sendobj_buf.dtype == np.dtype('float64'):
            datatype = self.MPI_DTYPES.DOUBLE
        elif sendobj_buf.dtype == np.dtype('complex64'):
            datatype = self.MPI_DTYPES.C_FLOAT_COMPLEX
        elif sendobj_buf.dtype == np.dtype('complex128'):
            datatype = self.MPI_DTYPES.C_DOUBLE_COMPLEX
        status = MPI_Allreduce(
            sendobj_buf.ctypes.data,
            recvobj_buf.ctypes.data,
            sendobj_buf.size,
            self._mpi_addr(datatype),
            self._mpi_addr(operator),
            self._mpi_addr(self.MPI_Comm_World_ptr),
        )
        return recvobj_buf[0]
    return impl


Comm_t = Comm_class_t([
    ('MPI_DTYPES', nb.types.IntEnumClass(MPI_DTYPES, nb.int64)),
    ('MPI_Operator', nb.types.IntEnumClass(MPI_Operator, nb.int64)),
    ('MPI_Comm_World_ptr', nb.int64),
])


class Comm(structref.StructRefProxy):

    def __new__(cls, *args, **kwargs):
        """MPI communicator."""
        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(MPI_Comm_World_ptr):
            self = structref.new(Comm_t)
            self.MPI_DTYPES = MPI_DTYPES
            self.MPI_Operator = MPI_Operator
            self.MPI_Comm_World_ptr = MPI_Comm_World_ptr
            return self
        MPI_Comm_World_ptr = MPI._addressof(MPI.COMM_WORLD)
        return init(MPI_Comm_World_ptr)


structref.define_boxing(Comm_class_t, Comm)
