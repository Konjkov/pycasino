import os
from timeit import default_timer
from slater import Slater
from jastrow import Jastrow
from backflow import Backflow
from coulomb import coulomb, nuclear_repulsion

os.environ["OMP_NUM_THREADS"] = "1"  # openmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas
os.environ["MKL_NUM_THREADS"] = "1"  # mkl
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr

import pyblock
import numpy as np
import numba as nb
import scipy as sp

from decorators import pool, thread
from readers.casino import Casino
from overload import subtract_outer
from logger import logging

logger = logging.getLogger('vmc')
numba_logger = logging.getLogger('numba')

spec = [
    ('neu', nb.int64),
    ('ned', nb.int64),
    ('r_e', nb.float64[:, :]),
    ('step', nb.float64),
    ('atom_positions', nb.float64[:, :]),
    ('atom_charges', nb.float64[:]),
    ('nuclear_repulsion', nb.float64),
    ('slater', Slater.class_type.instance_type),
    ('jastrow', nb.optional(Jastrow.class_type.instance_type)),
    ('backflow', nb.optional(Backflow.class_type.instance_type)),
]


@nb.experimental.jitclass(spec)
class Wfn:

    def __init__(self, neu, ned, atom_positions, atom_charges, slater, jastrow, backflow):
        """Wave function in general form.
        :param neu: number of up electrons
        :param ned: number of down electrons
        :param atom_positions: atomic positions
        :param atom_charges: atomic charges
        :param slater: instance of Slater class
        :param jastrow: instance of Jastrow class
        :param backflow: instance of Backflow class
        :return:
        """
        self.neu = neu
        self.ned = ned
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.nuclear_repulsion = nuclear_repulsion(atom_positions, atom_charges)
        self.slater = slater
        self.jastrow = jastrow
        self.backflow = backflow

    def value(self, e_vectors, n_vectors) -> float:
        """Value of wave function.
        :param e_vectors: e-e vectors - array(nelec, nelec, 3)
        :param n_vectors: e-n vectors- array(nelec, natom, 3)
        :return:
        """
        res = 1
        if self.jastrow is not None:
            res *= np.exp(self.jastrow.value(e_vectors, n_vectors, self.neu))
        if self.backflow is not None:
            n_vectors += self.backflow.value(e_vectors, n_vectors, self.neu)
        res *= self.slater.value(n_vectors, self.neu)
        return res

    def energy(self, r_e) -> float:
        """Local energy.
        :param r_e: electron coordinates - array(nelec, 3)
        :return:
        """
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)

        s = self.slater.value(n_vectors, self.neu)
        s_l = self.slater.laplacian(n_vectors, self.neu, self.ned) / s
        res = coulomb(e_vectors, n_vectors, self.atom_charges)
        if self.jastrow is not None:
            j_g = self.jastrow.gradient(e_vectors, n_vectors, self.neu)
            j_l = self.jastrow.laplacian(e_vectors, n_vectors, self.neu)
            if self.backflow is not None:
                b_g = self.backflow.gradient(e_vectors, n_vectors, self.neu)
                n_vectors += self.backflow.value(e_vectors, n_vectors, self.neu)
                s_g = self.slater.gradient(n_vectors, self.neu, self.ned) / s
                for j in range(s_g.shape[0]):
                    s_g[j] += np.dot(b_g[j], s_g[j])
            else:
                s_g = self.slater.gradient(n_vectors, self.neu, self.ned) / s
            F = np.sum((s_g + j_g) * (s_g + j_g)) / 2
            T = (np.sum(s_g * s_g) - s_l - j_l) / 4
            res += 2 * T - F
        else:
            res -= s_l / 2
        return res
