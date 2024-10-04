import numba as nb
import numpy as np
from numba.experimental import structref

from .abstract import AbstractJastrow


@structref.register
class Gjastrow_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


shape_type = nb.types.ListType(nb.int64)
linear_parameters_type = nb.float64[::1]
constants_type = nb.types.DictType(nb.types.unicode_type, nb.float64)
parameters_type = nb.types.ListType(nb.types.DictType(nb.types.unicode_type, nb.float64))

Gjastrow_t = Gjastrow_class_t(
    [
        ('neu', nb.int64),
        ('ned', nb.int64),
        ('rank', nb.int64[:, ::1]),
        ('cusp', nb.boolean[:, ::1]),
        ('ee_basis_type', nb.types.ListType(nb.types.unicode_type)),
        ('en_basis_type', nb.types.ListType(nb.types.unicode_type)),
        ('ee_cutoff_type', nb.types.ListType(nb.types.unicode_type)),
        ('en_cutoff_type', nb.types.ListType(nb.types.unicode_type)),
        ('ee_constants', nb.types.ListType(constants_type)),
        ('en_constants', nb.types.ListType(constants_type)),
        ('ee_basis_parameters', parameters_type),
        ('en_basis_parameters', parameters_type),
        ('ee_cutoff_parameters', parameters_type),
        ('en_cutoff_parameters', parameters_type),
        ('linear_parameters', nb.types.ListType(linear_parameters_type)),
        ('linear_parameters_shape', nb.types.ListType(shape_type)),
    ]
)


class Gjastrow(structref.StructRefProxy, AbstractJastrow):
    def __new__(cls, *args, **kwargs):
        def init(
            neu,
            ned,
            rank,
            cusp,
            ee_basis_type,
            en_basis_type,
            ee_cutoff_type,
            en_cutoff_type,
            ee_constants,
            en_constants,
            ee_basis_parameters,
            en_basis_parameters,
            ee_cutoff_parameters,
            en_cutoff_parameters,
            linear_parameters,
            linear_parameters_shape,
        ):
            self = structref.new(Gjastrow_t)
            self.neu = neu
            self.ned = ned
            self.rank = rank
            self.cusp = cusp
            self.ee_basis_type = ee_basis_type
            self.en_basis_type = en_basis_type
            self.ee_cutoff_type = ee_cutoff_type
            self.en_cutoff_type = en_cutoff_type
            self.ee_constants = ee_constants
            self.en_constants = en_constants
            self.ee_basis_parameters = ee_basis_parameters
            self.en_basis_parameters = en_basis_parameters
            self.ee_cutoff_parameters = ee_cutoff_parameters
            self.en_cutoff_parameters = en_cutoff_parameters
            self.linear_parameters = linear_parameters
            self.linear_parameters_shape = linear_parameters_shape
            return self

        return init(*args, **kwargs)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def ee_powers(self, e_vectors: np.ndarray):
        """Powers of e-e distances
        :param e_vectors: e-e vectors - array(nelec, nelec, 3)
        :return: powers of e-e distances - array(nelec, nelec, max_ee_order, channel)
        """
        shape = self.linear_parameters_shape[0]
        linear_parameters = np.reshape(self.linear_parameters[0], shape)
        res = np.zeros((e_vectors.shape[0], e_vectors.shape[1], linear_parameters.shape[1], linear_parameters.shape[0]))
        for i in range(e_vectors.shape[0] - 1):
            for j in range(i + 1, e_vectors.shape[1]):
                r = np.linalg.norm(e_vectors[i, j])
                for k in range(self.linear_parameters.shape[1]):
                    for channel in range(self.linear_parameters.shape[0]):
                        if self.ee_basis_parameters:
                            a = self.ee_basis_parameters[channel].get('a')
                            b = self.ee_basis_parameters[channel].get('b')
                        if self.ee_basis_type[0] == 'natural power':
                            res[i, j, k, channel] = r**k
                        elif self.ee_basis_type[0] == 'r/(r^b+a) power':
                            res[i, j, k, channel] = (r / (r**b + a)) ** k
                        elif self.ee_basis_type[0] == 'r/(r+a) power':
                            res[i, j, k, channel] = (r / (r + a)) ** k
                        elif self.ee_basis_type[0] == '1/(r+a) power':
                            res[i, j, k, channel] = (1 / (r + a)) ** k
        return res

    @nb.njit(nogil=True, parallel=False, cache=True)
    def en_powers(self, n_vectors: np.ndarray):
        """Powers of e-n distances
        :param n_vectors: e-n vectors - array(natom, nelec, 3)
        :return: powers of e-n distances - array(natom, nelec, max_en_order, channel)
        """
        shape = self.linear_parameters_shape[0]
        linear_parameters = np.reshape(self.linear_parameters[0], shape)
        res = np.zeros((n_vectors.shape[1], n_vectors.shape[0], linear_parameters.shape[1], linear_parameters.shape[0]))
        for i in range(n_vectors.shape[1]):
            for j in range(n_vectors.shape[0]):
                r = np.linalg.norm(n_vectors[j, i])
                for k in range(self.linear_parameters.shape[1]):
                    for channel in range(self.linear_parameters.shape[0]):
                        if self.ee_basis_parameters:
                            a = self.en_basis_parameters[channel].get('a')
                            b = self.en_basis_parameters[channel].get('b')
                        if self.en_basis_type[0] == 'natural power':
                            res[i, j, k, channel] = r**k
                        elif self.en_basis_type[0] == 'r/(r^b+a) power':
                            res[i, j, k, channel] = (r / (r**b + a)) ** k
                        elif self.en_basis_type[0] == 'r/(r+a) power':
                            res[i, j, k, channel] = (r / (r + a)) ** k
                        elif self.en_basis_type[0] == '1/(r+a) power':
                            res[i, j, k, channel] = (1 / (r + a)) ** k
        return res

    @nb.njit(nogil=True, parallel=False, cache=True)
    def term_2_0(self, e_powers: np.ndarray, e_vectors: np.ndarray) -> float:
        """Jastrow term rank [2, 0]
        :param e_powers: powers of e-e distances
        :param e_vectors: e-e vectors
        :return:
        """
        res = 0.0

        shape = self.linear_parameters_shape[0]
        p = np.reshape(self.linear_parameters[0], shape)
        C = self.ee_constants[0]['C']  # FIXME: first term hardcoded
        for i in range(e_powers.shape[0] - 1):
            for j in range(i + 1, e_powers.shape[1]):
                r = np.linalg.norm(e_vectors[i, j])
                # FIXME: it's not a channel
                channel = int(i >= self.neu) + int(j >= self.neu)
                channel = 0
                L = self.ee_cutoff_parameters[channel]['L']
                L_hard = self.ee_cutoff_parameters[channel].get('L_hard')

                poly = 0.0
                for k in range(p.shape[0]):
                    poly += p[channel, k] * e_powers[i, j, k, channel]

                if self.ee_cutoff_type == 'gaussian':
                    if r <= L_hard:
                        res += poly * np.exp(-((r / L) ** 2))
                elif r <= L:
                    if self.ee_cutoff_type[0] == 'polynomial':
                        res += poly * (1 - r / L) ** C
                    elif self.ee_cutoff_type == 'alt polynomial':
                        res += poly * (r - L) ** C
                    elif self.ee_cutoff_type == 'spline':
                        pass
                    elif self.ee_cutoff_type == 'anisotropic polynomial':
                        pass
        return res

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value(self, e_vectors, n_vectors) -> float:
        """Jastrow
        :param e_vectors: electrons coordinates
        :param n_vectors: nucleus coordinates
        :return:
        """
        e_powers = self.ee_powers(e_vectors)
        # n_powers = self.en_powers(n_vectors)

        return self.term_2_0(e_powers, e_vectors)


structref.define_boxing(Gjastrow_class_t, Gjastrow)
