import numba as nb
import numpy as np
from mpi4py import MPI
from numba.core import cgutils, types
from numba.extending import overload_method
from numba.experimental import structref
from .header import libmpi, MPI_Initialized, MPI_Barrier, MPI_Comm_size, MPI_Comm_rank, MPI_Allreduce, MPI_Send, MPI_Recv, MPI_Allgather

# https://mpi4py.readthedocs.io/en/stable/

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
@overload_method(Comm_class_t, '_mpi_dtype')
def comm_mpi_dtype(self, obj):
    """MPI data type."""
    def impl(self, obj):
        if obj.dtype == np.dtype('uint8'):
            datatype = self.MPI_CHAR
        elif obj.dtype == np.dtype('int32'):
            datatype = self.MPI_INT32_T
        elif obj.dtype == np.dtype('int64'):
            datatype = self.MPI_INT64_T
        elif obj.dtype == np.dtype('float32'):
            datatype = self.MPI_FLOAT
        elif obj.dtype == np.dtype('float64'):
            datatype = self.MPI_DOUBLE
        elif obj.dtype == np.dtype('complex64'):
            datatype = self.MPI_C_FLOAT_COMPLEX
        elif obj.dtype == np.dtype('complex128'):
            datatype = self.MPI_C_DOUBLE_COMPLEX
        return datatype
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'Initialized')
def comm_Initialized(self):
    """MPI.Initialized."""
    def impl(self):
        flag_ptr = val_to_ptr(False)
        status = self.MPI_Initialized(flag_ptr)
        assert status == 0
        return val_from_ptr(flag_ptr)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'Barrier')
def comm_Barrier(self):
    """MPI.Barrier."""
    def impl(self):
        status = self.MPI_Barrier(self._mpi_addr(self.MPI_COMM_WORLD))
        assert status == 0
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'Get_size')
def comm_Get_size(self):
    """MPI.Get_size."""
    def impl(self):
        size_ptr = val_to_ptr(0)
        status = self.MPI_Comm_size(self._mpi_addr(self.MPI_COMM_WORLD), size_ptr)
        assert status == 0
        return val_from_ptr(size_ptr)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'Get_rank')
def comm_Get_rank(self):
    """MPI.Get_rank."""
    def impl(self):
        size_ptr = val_to_ptr(0)
        status = self.MPI_Comm_rank(self._mpi_addr(self.MPI_COMM_WORLD), size_ptr)
        assert status == 0
        return val_from_ptr(size_ptr)
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'Send')
def comm_Send(self, data, dest, tag=0):
    """MPI.Send."""
    def impl(self, data, dest, tag=0):
        status = self.MPI_Send(
            data.ctypes.data,
            data.size,
            self._mpi_addr(self._mpi_dtype(data)),
            dest,
            tag,
            self._mpi_addr(self.MPI_COMM_WORLD),
        )
        assert status == 0
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'Recv')
def comm_Recv(self, data, dest, tag=0):
    """MPI.Recv."""
    def impl(self, data, dest, tag=0):
        status_buffer = np.empty(5, dtype=np.intc)
        status = self.MPI_Recv(
            data.ctypes.data,
            data.size,
            self._mpi_addr(self._mpi_dtype(data)),
            dest,
            tag,
            self._mpi_addr(self.MPI_COMM_WORLD),
            status_buffer.ctypes.data,
        )
        assert status == 0
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'allreduce')
def comm_allreduce(self, sendobj, operator=0):
    """MPI.allreduce."""
    def impl(self, sendobj, operator=0):
        if operator == 0:
            operator = self.MPI_SUM
        sendobj_buf = np.array([sendobj])
        recvobj_buf = np.zeros_like(sendobj_buf)
        status = self.MPI_Allreduce(
            sendobj_buf.ctypes.data,
            recvobj_buf.ctypes.data,
            sendobj_buf.size,
            self._mpi_addr(self._mpi_dtype(sendobj_buf)),
            self._mpi_addr(operator),
            self._mpi_addr(self.MPI_COMM_WORLD),
        )
        assert status == 0
        return recvobj_buf[0]
    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Comm_class_t, 'allgather')
def comm_allgather(self, send_data, recv_data, count):
    """MPI.allgather."""
    def impl(self, send_data, recv_data, count):
        status = self.MPI_Allgather(
            send_data.ctypes.data,
            send_data.size,
            self._mpi_addr(self._mpi_dtype(send_data)),
            recv_data.ctypes.data,
            count,
            self._mpi_addr(self._mpi_dtype(recv_data)),
            self._mpi_addr(self.MPI_COMM_WORLD),
        )
        assert status == 0
    return impl


Comm_t = Comm_class_t([
    # communictors
    ('MPI_COMM_WORLD', nb.typeof(MPI._addressof(MPI.COMM_WORLD))),
    # datatypes
    ('MPI_CHAR', nb.typeof(MPI._addressof(MPI.CHAR))),
    ('MPI_INT32_T', nb.typeof(MPI._addressof(MPI.INT32_T))),
    ('MPI_INT64_T', nb.typeof(MPI._addressof(MPI.INT64_T))),
    ('MPI_FLOAT', nb.typeof(MPI._addressof(MPI.FLOAT))),
    ('MPI_DOUBLE', nb.typeof(MPI._addressof(MPI.DOUBLE))),
    ('MPI_C_FLOAT_COMPLEX', nb.typeof(MPI._addressof(MPI.C_FLOAT_COMPLEX))),
    ('MPI_C_DOUBLE_COMPLEX', nb.typeof(MPI._addressof(MPI.C_DOUBLE_COMPLEX))),
    # operators
    ('MPI_MAX', nb.typeof(MPI._addressof(MPI.MAX))),
    ('MPI_MIN', nb.typeof(MPI._addressof(MPI.MIN))),
    ('MPI_SUM', nb.typeof(MPI._addressof(MPI.SUM))),
    ('MPI_PROD', nb.typeof(MPI._addressof(MPI.PROD))),
    ('MPI_LAND', nb.typeof(MPI._addressof(MPI.LAND))),
    ('MPI_BAND', nb.typeof(MPI._addressof(MPI.BAND))),
    # functions
    ('MPI_Initialized', nb.typeof(MPI_Initialized)),
    ('MPI_Barrier', nb.typeof(MPI_Barrier)),
    ('MPI_Comm_size', nb.typeof(MPI_Comm_size)),
    ('MPI_Comm_rank', nb.typeof(MPI_Comm_rank)),
    ('MPI_Send', nb.typeof(MPI_Send)),
    ('MPI_Recv', nb.typeof(MPI_Recv)),
    ('MPI_Allgather', nb.typeof(MPI_Allgather)),
    ('MPI_Allreduce', nb.typeof(MPI_Allreduce)),
])


class Comm(structref.StructRefProxy):

    def __new__(cls):
        """Communication context."""
        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(*args):
            self = structref.new(Comm_t)
            (
                # communictors
                self.MPI_COMM_WORLD,
                # datatypes
                self.MPI_CHAR,
                self.MPI_INT32_T,
                self.MPI_INT64_T,
                self.MPI_FLOAT,
                self.MPI_DOUBLE,
                self.MPI_C_FLOAT_COMPLEX,
                self.MPI_C_DOUBLE_COMPLEX,
                # operators
                self.MPI_MAX,
                self.MPI_MIN,
                self.MPI_SUM,
                self.MPI_PROD,
                self.MPI_LAND,
                self.MPI_BAND,
                # functions
                self.MPI_Initialized,
                self.MPI_Barrier,
                self.MPI_Comm_size,
                self.MPI_Comm_rank,
                self.MPI_Send,
                self.MPI_Recv,
                self.MPI_Allgather,
                self.MPI_Allreduce,
            ) = args
            return self
        args = (
            # communictors
            MPI._addressof(MPI.COMM_WORLD),
            # datatypes
            MPI._addressof(MPI.CHAR),
            MPI._addressof(MPI.INT32_T),
            MPI._addressof(MPI.INT64_T),
            MPI._addressof(MPI.FLOAT),
            MPI._addressof(MPI.DOUBLE),
            MPI._addressof(MPI.C_FLOAT_COMPLEX),
            MPI._addressof(MPI.C_DOUBLE_COMPLEX),
            # operators
            MPI._addressof(MPI.MAX),
            MPI._addressof(MPI.MIN),
            MPI._addressof(MPI.SUM),
            MPI._addressof(MPI.PROD),
            MPI._addressof(MPI.LAND),
            MPI._addressof(MPI.BAND),
            # functions
            libmpi.MPI_Initialized,
            libmpi.MPI_Barrier,
            libmpi.MPI_Comm_size,
            libmpi.MPI_Comm_rank,
            libmpi.MPI_Send,
            libmpi.MPI_Recv,
            libmpi.MPI_Allgather,
            libmpi.MPI_Allreduce,
        )
        return init(*args)


structref.define_boxing(Comm_class_t, Comm)
