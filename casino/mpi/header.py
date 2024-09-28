import ctypes
import numba as nb
from enum import IntEnum
from mpi4py import MPI


LIB = ctypes.util.find_library('mpi')
libmpi = ctypes.CDLL(LIB)

if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int32):
    _MpiComm = ctypes.c_int32
    _MpiDatatype = ctypes.c_int32
    _MpiOp = ctypes.c_int32
    _restype = ctypes.c_int32
    _c_int_p = ctypes.POINTER(ctypes.c_int32)
else:
    _MpiComm = ctypes.c_int64
    _MpiDatatype = ctypes.c_int64
    _MpiOp = ctypes.c_int64
    _restype = ctypes.c_int64
    _c_int_p = ctypes.POINTER(ctypes.c_int64)

_MpiStatusPtr = ctypes.c_void_p
_MpiRequestPtr = ctypes.c_void_p
# int MPI_Initialized(int *flag)
MPI_Initialized = libmpi.MPI_Initialized
MPI_Initialized.restype = _restype
MPI_Initialized.argtypes = []
# int MPI_Barrier(MPI_Comm comm)
MPI_Barrier = libmpi.MPI_Barrier
MPI_Barrier.restype = _restype
MPI_Barrier.argtypes = [_MpiComm]
# int MPI_Comm_size(MPI_Comm comm, int *size)
MPI_Comm_size = libmpi.MPI_Comm_size
MPI_Comm_size.restype = _restype
MPI_Comm_size.argtypes = [_MpiComm, _c_int_p]
# int MPI_Comm_rank(MPI_Comm comm, int *rank)
MPI_Comm_rank = libmpi.MPI_Comm_rank
MPI_Comm_rank.restype = _restype
MPI_Comm_rank.argtypes = [_MpiComm, _c_int_p]
# int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
MPI_Send = libmpi.MPI_Send
MPI_Send.restype = _restype
MPI_Send.argtypes = [
    ctypes.c_void_p,  # send data
    ctypes.c_int,  # send count
    _MpiDatatype,  # send data type
    ctypes.c_int,  # rank of destination
    ctypes.c_int,  # message tag
    _MpiComm,  # communicator
]
# int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
MPI_Recv = libmpi.MPI_Recv
MPI_Recv.restype = _restype
MPI_Recv.argtypes = [
    ctypes.c_void_p,  # send data
    ctypes.c_int,  # send count
    _MpiDatatype,  # send data type
    ctypes.c_int,  # rank of source
    ctypes.c_int,  # message tag
    _MpiComm,  # communicator
    _MpiStatusPtr,  # status object
]
# int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
MPI_Scatter = libmpi.MPI_Scatter
MPI_Scatter.restype = _restype
MPI_Scatter.argtypes = [
    ctypes.c_void_p,  # send data
    ctypes.c_int,  # send count
    _MpiDatatype,  # send data type
    ctypes.c_void_p,  # recv data
    ctypes.c_int,  # recv count
    _MpiDatatype,  # recv data type
    ctypes.c_int,  # root
    _MpiComm,  # communicator
]
# int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
MPI_Gather = libmpi.MPI_Gather
MPI_Gather.restype = _restype
MPI_Gather.argtypes = [
    ctypes.c_void_p,  # send data
    ctypes.c_int,  # send count
    _MpiDatatype,  # send data type
    ctypes.c_void_p,  # recv data
    ctypes.c_int,  # recv count
    _MpiDatatype,  # recv data type
    ctypes.c_int,  # root
    _MpiComm,  # communicator
]
# int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
MPI_Allgather = libmpi.MPI_Allgather
MPI_Allgather.restype = _restype
MPI_Allgather.argtypes = [
    ctypes.c_void_p,  # send data
    ctypes.c_int,  # send count
    _MpiDatatype,  # send data type
    ctypes.c_void_p,  # recv data
    ctypes.c_int,  # recv count
    _MpiDatatype,  # recv data type
    _MpiComm,  # communicator
]
# int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
MPI_Allreduce = libmpi.MPI_Allreduce
MPI_Allreduce.restype = _restype
MPI_Allreduce.argtypes = [
    ctypes.c_void_p,  # send data
    ctypes.c_void_p,  # recv data
    ctypes.c_int64,  # send count
    _MpiDatatype,  # send data type
    _MpiOp,  # operation
    _MpiComm,  # communicator
]
