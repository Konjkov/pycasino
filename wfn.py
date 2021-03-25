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
            res *= np.exp(self.jastrow.value(e_vectors, n_vectors))
        if self.backflow is not None:
            n_vectors += self.backflow.value(e_vectors, n_vectors)
        res *= self.slater.value(n_vectors)
        return res

    def energy(self, r_e) -> float:
        """Local energy.
        :param r_e: electron coordinates - array(nelec, 3)

        if f is a scalar multivariable function and a is a vector multivariable function then:

        gradient composition rule:
        ∇(f ○ a) = (∇f ○ a) * ∇a
        where ∇a is a Jacobian matrix.

        laplacian composition rule
        Δ(f ○ a) = ∇((∇f ○ a) * ∇a) = ∇(∇f ○ a) * ∇a + (∇f ○ a) * ∇²a = tr((∇²f ○ a) * ∇a * ∇a) + (∇f ○ a) * Δa
        where ∇²f is a hessian

        :return: local energy
        """
        e_vectors = subtract_outer(r_e, r_e)
        n_vectors = -subtract_outer(self.atom_positions, r_e)

        if self.backflow is not None:
            b_v = self.backflow.value(e_vectors, n_vectors)
            b_g = self.backflow.numerical_gradient(e_vectors, n_vectors)
            b_l = self.backflow.numerical_laplacian(e_vectors, n_vectors)
            b_r_e = r_e + b_v
            slater_n_vectors = n_vectors + b_v
            slater_e_vectors = subtract_outer(b_r_e, b_r_e)
            s_v = self.slater.value(slater_n_vectors)
            s_g = self.slater.gradient(slater_n_vectors) / s_v
            s_h = self.slater.numerical_hessian(slater_n_vectors) / s_v
            b_g_i = np.eye((self.neu + self.ned) * 3) + b_g
            s_l = np.trace(s_h @ b_g_i @ b_g_i) + s_g @ b_l
            res = coulomb(slater_e_vectors, slater_n_vectors, self.atom_charges)
            if self.jastrow is not None:
                j_g = self.jastrow.gradient(e_vectors, n_vectors)
                j_l = self.jastrow.laplacian(e_vectors, n_vectors)
                s_g += b_g @ s_g
                F = np.sum((s_g + j_g) * (s_g + j_g)) / 2
                T = (np.sum(s_g * s_g) - s_l - j_l) / 4
                res += 2 * T - F
            else:
                res -= s_l / 2
        else:
            s_v = self.slater.value(n_vectors)
            s_l = self.slater.laplacian(n_vectors) / s_v
            res = coulomb(e_vectors, n_vectors, self.atom_charges)
            if self.jastrow is not None:
                j_g = self.jastrow.gradient(e_vectors, n_vectors)
                j_l = self.jastrow.laplacian(e_vectors, n_vectors)
                s_g = self.slater.gradient(n_vectors) / s_v
                F = np.sum((s_g + j_g) * (s_g + j_g)) / 2
                T = (np.sum(s_g * s_g) - s_l - j_l) / 4
                res += 2 * T - F
            else:
                res -= s_l / 2
        return res
