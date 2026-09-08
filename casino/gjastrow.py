import collections

import numba as nb
import numpy as np
from numba.experimental import structref
from numba.extending import overload_method

from casino import delta
from casino.abstract import AbstractJastrow
from casino.readers.channels import signature
from casino.readers.gjastrow import BASIS_CODE, CUTOFF_CODE, DETERMINED, OPTIMIZABLE

NATURAL_POWER = BASIS_CODE['natural power']
POLYNOMIAL = CUTOFF_CODE['polynomial']
ALT_POLYNOMIAL = CUTOFF_CODE['alt polynomial']

# the ranks computed here, and the pair whose cutoff truncates each of them.
# numba has no array whose number of dimensions is decided at runtime, and a term
# of rank (rank_e, rank_n) is a function of rank_e (rank_e - 1) / 2 + rank_e rank_n
# pairs, so every rank needs an array of its own and only these three are written out
RANKS = {(2, 0): 'e-e', (1, 1): 'e-n', (2, 1): 'e-n'}

# characteristic size of a coefficient of every rank, as the numerator and the power
# of the electron count the standard jastrow scales its own u, chi and f terms by
SCALE = {(2, 0): (2, 2), (1, 1): (1, 1), (2, 1): (2, 3)}

Arrays = collections.namedtuple('Arrays', 'parameters cutoff channel cutoff_channel trunc alt')

# what the optimizer is given: the cutoff lengths that are free to move and the two
# constants of the cutoff function of each of them, the equations that determine
# coefficients, and which coefficients are free to move and how large they are
Optimization = collections.namedtuple('Optimization', 'cutoff_optimizable cutoff_trunc cutoff_alt value slope cutoff rhs pivot mask available scale')


@nb.njit(nogil=True, parallel=False, cache=True, inline='always')
def cutoff(r, L, C, alt):
    """Value and first two radial derivatives of the cutoff function, which is
    (1 - r/L)^C or, in its alternative form, (r - L)^C. The two are the same
    function up to a factor of (-L)^C, which the linear parameters absorb.
    The function is inlined so that a caller that only needs the value does not
    pay for the derivatives.
    :return: the function, its derivative and its second derivative at r
    """
    if alt:
        t = r - L
        dtdr = 1.0
    else:
        t = 1 - r / L
        dtdr = -1 / L
    f2 = t ** (C - 2)
    f1 = f2 * t
    return f1 * t, C * f1 * dtdr, C * (C - 1) * f2 * dtdr**2


@nb.njit(nogil=True, parallel=False, cache=True)
def cutoff_d1(r, L, C, alt):
    """Derivative of the cutoff function with respect to its length."""
    if alt:
        return -C * (r - L) ** (C - 1)
    return C * (1 - r / L) ** (C - 1) * r / L**2


@nb.njit(nogil=True, parallel=False, cache=True)
def cutoff_at_zero(L, C, alt, derivative):
    """Value and radial derivative of the cutoff function where the two particles of
    a pair meet, which is what the equations of a coalescence are built from, or the
    derivatives of the two with respect to the cutoff length.
    """
    if alt:
        if derivative:
            return -C * (-L) ** (C - 1), -C * (C - 1) * (-L) ** (C - 2)
        return (-L) ** C, C * (-L) ** (C - 1)
    if derivative:
        return 0.0, C / L**2
    return 1.0, -C / L


def cutoff_function(term):
    """The cutoff of a term, which has to be on the pair the rank of the term
    truncates and of a kind that is a plain power of the distance to the cutoff.
    """
    kind = RANKS[term.rank]
    function = term.ee_cutoff if kind == 'e-e' else term.en_cutoff
    other = term.en_cutoff if kind == 'e-e' else term.ee_cutoff
    if other is not None and other.code:
        raise NotImplementedError(f'{term.name} is cut off on its {"e-n" if kind == "e-e" else "e-e"} pairs as well')
    if function is None or function.code not in (POLYNOMIAL, ALT_POLYNOMIAL):
        raise NotImplementedError(f'{kind} cutoff of {term.name}: only a polynomial or an alt polynomial cutoff is computed')
    return kind, function


def check(term):
    """Refuse a term outside the subset computed here."""
    if term.rank not in RANKS:
        raise NotImplementedError(f'{term.name} of rank {term.rank}')
    for kind, basis, size in (('e-e', term.ee_basis, term.size_ee), ('e-n', term.en_basis, term.size_en)):
        if size and (basis is None or basis.code != NATURAL_POWER):
            raise NotImplementedError(f'{kind} basis of {term.name}: only a natural power basis is computed')
    return cutoff_function(term)


def channel_table(term):
    """The channel of every set of particles of the rank of the term, -1 for a set
    the term is not a function of.
    """
    nspin, natom = len(term.groups.nele), len(term.groups.atom_numbers)
    shape = (nspin,) * term.rank[0] + (natom,) * term.rank[1]
    table = np.full(shape=shape, fill_value=-1, dtype=np.int64)
    for index in np.ndindex(shape):
        spins, ions = index[: term.rank[0]], index[term.rank[0] :]
        if len(set(ions)) < len(ions):
            continue
        # a group of zero is a pair the system does not have, one of -1 a pair the
        # rules have taken out of the term, and either leaves the set out
        sig, _ = signature(term.groups.ee, term.groups.en, spins, ions)
        if all(group > 0 for group in sig):
            table[index] = term.groups.signatures.index(sig)
    return table


def cutoff_table(term, kind):
    """The channel of the cutoff of every pair of particles, -1 for a pair the term
    is not a function of. A cutoff is defined per group of pairs rather than per
    channel, so a rank (2, 1) term indexes it by one electron and one nucleus.
    """
    groups = term.groups.ee if kind == 'e-e' else term.groups.en.T
    return np.ascontiguousarray(np.where(groups > 0, groups - 1, -1), dtype=np.int64)


def term_arrays(term):
    """The arrays one term is computed from."""
    kind, function = check(term)
    return Arrays(
        np.ascontiguousarray(term.parameters),
        np.ascontiguousarray(function.parameters[:, 0]),
        channel_table(term),
        cutoff_table(term, kind),
        function.constants['C'],
        function.code == ALT_POLYNOMIAL,
    )


def empty_arrays(rank, nspin, natom):
    """What a rank the file carries no term of contributes: no channel at all."""
    size = rank[0] * (rank[0] - 1) // 2 + rank[0] * rank[1]
    table = np.full(shape=(nspin,) * rank[0] + (natom,) * rank[1], fill_value=-1, dtype=np.int64)
    pairs = np.full(shape=(nspin, nspin) if rank[1] == 0 else (nspin, natom), fill_value=-1, dtype=np.int64)
    return Arrays(np.zeros(shape=(0,) * (size + 1)), np.zeros(shape=0), table, pairs, 0, False)


def terms_by_rank(config):
    """The term of every computed rank, None for a rank the file carries no term of."""
    terms = {}
    for term in config.jastrow.terms:
        check(term)
        if term.rank in terms:
            raise NotImplementedError(f'{terms[term.rank].name} and {term.name} are both of rank {term.rank}')
        terms[term.rank] = term
    return [terms.get(rank) for rank in RANKS]


def term_by_rank(config):
    """The arrays of a generic Jastrow, one per rank, refusing what is not computed."""
    nspin, natom = 2, len(config.jastrow.atom_numbers)
    return [term_arrays(term) if term is not None else empty_arrays(rank, nspin, natom) for rank, term in zip(RANKS, terms_by_rank(config))]


def channel_cutoff(rank, arrays, channel):
    """The cutoff length a channel is truncated at, taken from the first set of
    particles in it and averaged where the two electrons of the set are cut off
    on e-n pairs of their own.
    """
    index = tuple(np.argwhere(arrays.channel == channel)[0])
    if rank == (2, 1):
        return (arrays.cutoff[arrays.cutoff_channel[index[0], index[2]]] + arrays.cutoff[arrays.cutoff_channel[index[1], index[2]]]) / 2
    return arrays.cutoff[arrays.cutoff_channel[index]]


def scale_array(rank, arrays, ne):
    """Characteristic size of every coefficient of a term, which is what it takes
    for the term to be of order one: the cutoff length to the power of the
    expansion index, over the number of sets of particles the term sums over.
    """
    numerator, power = SCALE[rank]
    scale = np.zeros(shape=arrays.parameters.shape)
    for channel in range(scale.shape[0]):
        L = channel_cutoff(rank, arrays, channel)
        for index in np.ndindex(scale.shape[1:]):
            scale[(channel,) + index] = numerator / L ** sum(index) / ne**power
    return scale.ravel()


def concatenate(blocks, dtype=float):
    """One array out of the blocks of the terms, empty where there is no term at all."""
    if not blocks:
        return np.zeros(shape=0, dtype=dtype)
    return np.concatenate(blocks)


def scatter(row, size, base):
    """One equation of one channel, over the coefficients of every term."""
    res = np.zeros(shape=size)
    res[base : base + row.size] = row
    return res


def linear_arrays(config, arrays):
    """What optimizing the parameters takes, over the cutoff lengths of every term of
    every rank followed by the coefficients of all of them, in the order they are
    stored in: which parameters are free to move, how large each of them is, and the
    equations that determine the coefficients the free ones leave, one equation per
    coefficient it determines.
    """
    ne = config.input.neu + config.input.ned
    size = sum(term_arrays.parameters.size for term_arrays in arrays)
    optimizable, trunc, alt, mask, available, scale = [], [], [], [], [], []
    value, slope, cutoff, rhs, pivot = [], [], [], [], []
    column, cut = 0, 0
    for rank, term, term_arrays in zip(RANKS, terms_by_rank(config), arrays):
        if term is None:
            continue
        function = cutoff_function(term)[1]
        optimizable.append(function.flags[:, 0] == OPTIMIZABLE)
        trunc.append(np.full(shape=term_arrays.cutoff.size, fill_value=term_arrays.trunc))
        alt.append(np.full(shape=term_arrays.cutoff.size, fill_value=term_arrays.alt))
        mask.append((term.flags == OPTIMIZABLE).ravel())
        available.append((term.flags != DETERMINED).ravel())
        scale.append(scale_array(rank, term_arrays, ne))
        channel = term.parameters[0].size
        for i, reduced in enumerate(term.constraints):
            base = column + i * channel
            for k, determines in enumerate(reduced.pivot):
                value.append(scatter(reduced.value[k], size, base))
                slope.append(scatter(reduced.slope[k], size, base))
                cutoff.append(cut + reduced.channel[k] if reduced.channel[k] >= 0 else -1)
                rhs.append(reduced.rhs[k])
                pivot.append(base + determines)
        column += term_arrays.parameters.size
        cut += term_arrays.cutoff.size
    return Optimization(
        concatenate(optimizable, bool),
        concatenate(trunc, int),
        concatenate(alt, bool),
        np.array(value).reshape(-1, size),
        np.array(slope).reshape(-1, size),
        np.array(cutoff, dtype=int),
        np.array(rhs),
        np.array(pivot, dtype=int),
        concatenate(mask, bool),
        concatenate(available, bool),
        concatenate(scale),
    )


@structref.register
class Gjastrow_class_t(nb.types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)


Gjastrow_t = Gjastrow_class_t(
    [
        ('neu', nb.int64),
        ('ned', nb.int64),
        ('max_ee_order', nb.int64),
        ('max_en_order', nb.int64),
        ('u_parameters', nb.float64[:, ::1]),
        ('u_cutoff', nb.float64[::1]),
        ('u_channel', nb.int64[:, ::1]),
        ('u_cutoff_channel', nb.int64[:, ::1]),
        ('u_trunc', nb.int64),
        ('u_alt', nb.boolean),
        ('chi_parameters', nb.float64[:, ::1]),
        ('chi_cutoff', nb.float64[::1]),
        ('chi_channel', nb.int64[:, ::1]),
        ('chi_cutoff_channel', nb.int64[:, ::1]),
        ('chi_trunc', nb.int64),
        ('chi_alt', nb.boolean),
        ('f_parameters', nb.float64[:, :, :, ::1]),
        ('f_cutoff', nb.float64[::1]),
        ('f_channel', nb.int64[:, :, ::1]),
        ('f_cutoff_channel', nb.int64[:, ::1]),
        ('f_trunc', nb.int64),
        ('f_alt', nb.boolean),
        ('cutoff_optimizable', nb.boolean[::1]),
        ('cutoff_trunc', nb.int64[::1]),
        ('cutoff_alt', nb.boolean[::1]),
        ('constraint_value', nb.float64[:, ::1]),
        ('constraint_slope', nb.float64[:, ::1]),
        ('constraint_cutoff', nb.int64[::1]),
        ('constraint_rhs', nb.float64[::1]),
        ('constraint_pivot', nb.int64[::1]),
        ('constraint_inverse', nb.float64[:, ::1]),
        ('parameters_jacobian', nb.float64[:, ::1]),
        ('parameters_offset', nb.float64[::1]),
        ('parameters_mask', nb.boolean[::1]),
        ('parameters_available', nb.boolean[::1]),
        ('parameters_scale', nb.float64[::1]),
        ('parameters_projector', nb.float64[:, ::1]),
        ('cutoffs_optimizable', nb.boolean),
    ]
)


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'ee_powers')
def gjastrow_ee_powers(self, e_vectors: np.ndarray):
    """Powers of e-e distances
    :param e_vectors: e-e vectors - array(nelec, nelec, 3)
    :return: powers of e-e distances - array(nelec, nelec, max_ee_order)
    """

    def impl(self, e_vectors: np.ndarray) -> np.ndarray:
        res = np.ones(shape=(e_vectors.shape[0], e_vectors.shape[1], self.max_ee_order))
        r_ee = np.sqrt((e_vectors * e_vectors).sum(axis=2))
        for k in range(1, self.max_ee_order):
            res[:, :, k] = r_ee**k
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'ee_powers_1e')
def gjastrow_ee_powers_1e(self, e_vectors_1e: np.ndarray):
    """Powers of the e-e distances of a single electron
    :param e_vectors_1e: e-e vectors of that electron - array(nelec, 3)
    :return: powers of e-e distances - array(nelec, max_ee_order)
    """

    def impl(self, e_vectors_1e: np.ndarray) -> np.ndarray:
        res = np.ones(shape=(e_vectors_1e.shape[0], self.max_ee_order))
        r_ee = np.sqrt((e_vectors_1e * e_vectors_1e).sum(axis=1))
        for k in range(1, self.max_ee_order):
            res[:, k] = r_ee**k
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'en_powers')
def gjastrow_en_powers(self, n_vectors: np.ndarray):
    """Powers of e-n distances
    :param n_vectors: e-n vectors - array(natom, nelec, 3)
    :return: powers of e-n distances - array(natom, nelec, max_en_order)
    """

    def impl(self, n_vectors: np.ndarray) -> np.ndarray:
        res = np.ones(shape=(n_vectors.shape[0], n_vectors.shape[1], self.max_en_order))
        r_eI = np.sqrt((n_vectors * n_vectors).sum(axis=2))
        for k in range(1, self.max_en_order):
            res[:, :, k] = r_eI**k
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'update_en_powers_1e')
def gjastrow_update_en_powers_1e(self, n_powers: np.ndarray, n_vector: np.ndarray, e: int):
    """Overwrite the powers of the e-n distances of a single electron in place. The rows of the
    electrons that did not move are the same at both ends of a single-electron move, so they are
    built once and only this one is replaced.
    :param n_powers: powers of e-n distances - array(natom, nelec, max_en_order)
    :param n_vector: e-n vectors of that electron - array(natom, 3)
    :param e: electron
    """

    def impl(self, n_powers: np.ndarray, n_vector: np.ndarray, e: int):
        for atom in range(n_vector.shape[0]):
            # written out rather than as a dot product: the caller may hand in a column of a
            # larger array, and numba only likes the contiguous case
            r_eI = np.sqrt(n_vector[atom, 0] ** 2 + n_vector[atom, 1] ** 2 + n_vector[atom, 2] ** 2)
            for k in range(1, self.max_en_order):
                n_powers[atom, e, k] = r_eI**k

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'u_term')
def gjastrow_u_term(self, e_powers: np.ndarray):
    """Term of rank (2, 0)
    :param e_powers: powers of e-e distances
    :return:
    """

    def impl(self, e_powers: np.ndarray) -> float:
        res = 0.0
        for e1 in range(1, self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for e2 in range(e1):
                s2 = int(e2 >= self.neu)
                channel = self.u_channel[s1, s2]
                if channel < 0:
                    continue
                L = self.u_cutoff[self.u_cutoff_channel[s1, s2]]
                r_ee = e_powers[e1, e2, 1]
                if r_ee >= L:
                    continue
                poly = 0.0
                for k in range(self.u_parameters.shape[1]):
                    poly += self.u_parameters[channel, k] * e_powers[e1, e2, k]
                res += poly * cutoff(r_ee, L, self.u_trunc, self.u_alt)[0]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'u_term_1e')
def gjastrow_u_term_1e(self, e_powers_1e: np.ndarray, e: int):
    """Term of rank (2, 0) of the pairs containing electron e
    :param e_powers_1e: powers of the e-e distances of electron e
    :param e: electron
    :return:
    """

    def impl(self, e_powers_1e: np.ndarray, e: int) -> float:
        res = 0.0
        s1 = int(e >= self.neu)
        for e2 in range(self.neu + self.ned):
            if e2 == e:
                continue
            s2 = int(e2 >= self.neu)
            channel = self.u_channel[s1, s2]
            if channel < 0:
                continue
            L = self.u_cutoff[self.u_cutoff_channel[s1, s2]]
            r_ee = e_powers_1e[e2, 1]
            if r_ee >= L:
                continue
            poly = 0.0
            for k in range(self.u_parameters.shape[1]):
                poly += self.u_parameters[channel, k] * e_powers_1e[e2, k]
            res += poly * cutoff(r_ee, L, self.u_trunc, self.u_alt)[0]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'chi_term')
def gjastrow_chi_term(self, n_powers: np.ndarray):
    """Term of rank (1, 1)
    :param n_powers: powers of e-n distances
    :return:
    """

    def impl(self, n_powers: np.ndarray) -> float:
        res = 0.0
        for e1 in range(self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for atom in range(self.chi_channel.shape[1]):
                channel = self.chi_channel[s1, atom]
                if channel < 0:
                    continue
                L = self.chi_cutoff[self.chi_cutoff_channel[s1, atom]]
                r_eI = n_powers[atom, e1, 1]
                if r_eI >= L:
                    continue
                poly = 0.0
                for k in range(self.chi_parameters.shape[1]):
                    poly += self.chi_parameters[channel, k] * n_powers[atom, e1, k]
                res += poly * cutoff(r_eI, L, self.chi_trunc, self.chi_alt)[0]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'chi_term_1e')
def gjastrow_chi_term_1e(self, n_powers: np.ndarray, e: int):
    """Term of rank (1, 1) of electron e
    :param n_powers: powers of e-n distances
    :param e: electron
    :return:
    """

    def impl(self, n_powers: np.ndarray, e: int) -> float:
        res = 0.0
        s1 = int(e >= self.neu)
        for atom in range(self.chi_channel.shape[1]):
            channel = self.chi_channel[s1, atom]
            if channel < 0:
                continue
            L = self.chi_cutoff[self.chi_cutoff_channel[s1, atom]]
            r_eI = n_powers[atom, e, 1]
            if r_eI >= L:
                continue
            poly = 0.0
            for k in range(self.chi_parameters.shape[1]):
                poly += self.chi_parameters[channel, k] * n_powers[atom, e, k]
            res += poly * cutoff(r_eI, L, self.chi_trunc, self.chi_alt)[0]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'f_term')
def gjastrow_f_term(self, e_powers: np.ndarray, n_powers: np.ndarray):
    """Term of rank (2, 1). The channel is named after the set of particles whose
    electrons are in the order of their spins, so the e-n index m belongs to the
    electron of the lower spin, which is the one of the lower number.
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :return:
    """

    def impl(self, e_powers: np.ndarray, n_powers: np.ndarray) -> float:
        res = 0.0
        for atom in range(self.f_channel.shape[2]):
            for e1 in range(1, self.neu + self.ned):
                s1 = int(e1 >= self.neu)
                for e2 in range(e1):
                    s2 = int(e2 >= self.neu)
                    channel = self.f_channel[s1, s2, atom]
                    if channel < 0:
                        continue
                    L1 = self.f_cutoff[self.f_cutoff_channel[s1, atom]]
                    L2 = self.f_cutoff[self.f_cutoff_channel[s2, atom]]
                    r_e1I = n_powers[atom, e1, 1]
                    r_e2I = n_powers[atom, e2, 1]
                    if r_e1I >= L1 or r_e2I >= L2:
                        continue
                    poly = 0.0
                    for n in range(self.f_parameters.shape[1]):
                        for m in range(self.f_parameters.shape[2]):
                            for l in range(self.f_parameters.shape[3]):
                                poly += self.f_parameters[channel, n, m, l] * e_powers[e1, e2, n] * n_powers[atom, e2, m] * n_powers[atom, e1, l]
                    res += poly * cutoff(r_e1I, L1, self.f_trunc, self.f_alt)[0] * cutoff(r_e2I, L2, self.f_trunc, self.f_alt)[0]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'f_term_1e')
def gjastrow_f_term_1e(self, e_powers_1e: np.ndarray, n_powers: np.ndarray, e: int):
    """Term of rank (2, 1) of the triplets containing electron e
    :param e_powers_1e: powers of the e-e distances of electron e
    :param n_powers: powers of e-n distances
    :param e: electron
    :return:
    """

    def impl(self, e_powers_1e: np.ndarray, n_powers: np.ndarray, e: int) -> float:
        res = 0.0
        s1 = int(e >= self.neu)
        for atom in range(self.f_channel.shape[2]):
            for e2 in range(self.neu + self.ned):
                if e2 == e:
                    continue
                s2 = int(e2 >= self.neu)
                channel = self.f_channel[s1, s2, atom]
                if channel < 0:
                    continue
                L1 = self.f_cutoff[self.f_cutoff_channel[s1, atom]]
                L2 = self.f_cutoff[self.f_cutoff_channel[s2, atom]]
                r_e1I = n_powers[atom, e, 1]
                r_e2I = n_powers[atom, e2, 1]
                if r_e1I >= L1 or r_e2I >= L2:
                    continue
                # the channel puts the electron of the lower spin first, which is the one of the lower number
                first, second = min(e, e2), max(e, e2)
                poly = 0.0
                for n in range(self.f_parameters.shape[1]):
                    for m in range(self.f_parameters.shape[2]):
                        for l in range(self.f_parameters.shape[3]):
                            poly += self.f_parameters[channel, n, m, l] * e_powers_1e[e2, n] * n_powers[atom, first, m] * n_powers[atom, second, l]
                res += poly * cutoff(r_e1I, L1, self.f_trunc, self.f_alt)[0] * cutoff(r_e2I, L2, self.f_trunc, self.f_alt)[0]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'u_term_gradient')
def gjastrow_u_term_gradient(self, e_powers: np.ndarray, e_vectors: np.ndarray):
    """Gradient of the term of rank (2, 0) w.r.t e-coordinates. A term is a
    product of the expansion P and the cutoff Q of every pair it is a function
    of, so its radial derivative is P'Q + PQ', and k-th power of the expansion
    contributes k times itself over r to P'.
    :param e_powers: powers of e-e distances
    :param e_vectors: e-e vectors
    :return:
    """

    def impl(self, e_powers: np.ndarray, e_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=(self.neu + self.ned, 3))
        for e1 in range(1, self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for e2 in range(e1):
                s2 = int(e2 >= self.neu)
                channel = self.u_channel[s1, s2]
                if channel < 0:
                    continue
                L = self.u_cutoff[self.u_cutoff_channel[s1, s2]]
                r_ee = e_powers[e1, e2, 1]
                if r_ee >= L:
                    continue
                poly = poly_diff = 0.0
                for k in range(self.u_parameters.shape[1]):
                    p = self.u_parameters[channel, k] * e_powers[e1, e2, k]
                    poly += p
                    poly_diff += k * p
                f, f_diff, _ = cutoff(r_ee, L, self.u_trunc, self.u_alt)
                gradient = (poly_diff / r_ee * f + poly * f_diff) / r_ee * e_vectors[e1, e2]
                res[e1] += gradient
                res[e2] -= gradient
        return res.ravel()

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'u_term_gradient_1e')
def gjastrow_u_term_gradient_1e(self, e_powers_1e: np.ndarray, e_vectors_1e: np.ndarray, e: int):
    """Gradient of the term of rank (2, 0) w.r.t the coordinates of electron e
    :param e_powers_1e: powers of the e-e distances of electron e
    :param e_vectors_1e: e-e vectors of electron e - array(nelec, 3)
    :param e: electron
    :return:
    """

    def impl(self, e_powers_1e: np.ndarray, e_vectors_1e: np.ndarray, e: int) -> np.ndarray:
        res = np.zeros(shape=3)
        s1 = int(e >= self.neu)
        for e2 in range(self.neu + self.ned):
            if e2 == e:
                continue
            s2 = int(e2 >= self.neu)
            channel = self.u_channel[s1, s2]
            if channel < 0:
                continue
            L = self.u_cutoff[self.u_cutoff_channel[s1, s2]]
            r_ee = e_powers_1e[e2, 1]
            if r_ee >= L:
                continue
            poly = poly_diff = 0.0
            for k in range(self.u_parameters.shape[1]):
                p = self.u_parameters[channel, k] * e_powers_1e[e2, k]
                poly += p
                poly_diff += k * p
            f, f_diff, _ = cutoff(r_ee, L, self.u_trunc, self.u_alt)
            res += (poly_diff / r_ee * f + poly * f_diff) / r_ee * e_vectors_1e[e2]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'chi_term_gradient')
def gjastrow_chi_term_gradient(self, n_powers: np.ndarray, n_vectors: np.ndarray):
    """Gradient of the term of rank (1, 1) w.r.t e-coordinates
    :param n_powers: powers of e-n distances
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, n_powers: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=(self.neu + self.ned, 3))
        for e1 in range(self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for atom in range(self.chi_channel.shape[1]):
                channel = self.chi_channel[s1, atom]
                if channel < 0:
                    continue
                L = self.chi_cutoff[self.chi_cutoff_channel[s1, atom]]
                r_eI = n_powers[atom, e1, 1]
                if r_eI >= L:
                    continue
                poly = poly_diff = 0.0
                for k in range(self.chi_parameters.shape[1]):
                    p = self.chi_parameters[channel, k] * n_powers[atom, e1, k]
                    poly += p
                    poly_diff += k * p
                f, f_diff, _ = cutoff(r_eI, L, self.chi_trunc, self.chi_alt)
                res[e1] += (poly_diff / r_eI * f + poly * f_diff) / r_eI * n_vectors[atom, e1]
        return res.ravel()

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'chi_term_gradient_1e')
def gjastrow_chi_term_gradient_1e(self, n_powers: np.ndarray, n_vector_1e: np.ndarray, e: int):
    """Gradient of the term of rank (1, 1) w.r.t the coordinates of electron e
    :param n_powers: powers of e-n distances
    :param n_vector_1e: e-n vectors of electron e - array(natom, 3)
    :param e: electron
    :return:
    """

    def impl(self, n_powers: np.ndarray, n_vector_1e: np.ndarray, e: int) -> np.ndarray:
        res = np.zeros(shape=3)
        s1 = int(e >= self.neu)
        for atom in range(self.chi_channel.shape[1]):
            channel = self.chi_channel[s1, atom]
            if channel < 0:
                continue
            L = self.chi_cutoff[self.chi_cutoff_channel[s1, atom]]
            r_eI = n_powers[atom, e, 1]
            if r_eI >= L:
                continue
            poly = poly_diff = 0.0
            for k in range(self.chi_parameters.shape[1]):
                p = self.chi_parameters[channel, k] * n_powers[atom, e, k]
                poly += p
                poly_diff += k * p
            f, f_diff, _ = cutoff(r_eI, L, self.chi_trunc, self.chi_alt)
            res += (poly_diff / r_eI * f + poly * f_diff) / r_eI * n_vector_1e[atom]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'f_term_gradient')
def gjastrow_f_term_gradient(self, e_powers: np.ndarray, n_powers: np.ndarray, e_vectors: np.ndarray, n_vectors: np.ndarray):
    """Gradient of the term of rank (2, 1) w.r.t e-coordinates. The e-e distance
    grows along the e-e vector for the first electron of the pair and shrinks
    along it for the second one.
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_powers: np.ndarray, n_powers: np.ndarray, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=(self.neu + self.ned, 3))
        for atom in range(self.f_channel.shape[2]):
            for e1 in range(1, self.neu + self.ned):
                s1 = int(e1 >= self.neu)
                for e2 in range(e1):
                    s2 = int(e2 >= self.neu)
                    channel = self.f_channel[s1, s2, atom]
                    if channel < 0:
                        continue
                    L1 = self.f_cutoff[self.f_cutoff_channel[s1, atom]]
                    L2 = self.f_cutoff[self.f_cutoff_channel[s2, atom]]
                    r_e1I = n_powers[atom, e1, 1]
                    r_e2I = n_powers[atom, e2, 1]
                    if r_e1I >= L1 or r_e2I >= L2:
                        continue
                    r_ee = e_powers[e1, e2, 1]
                    poly = poly_diff_ee = poly_diff_e1I = poly_diff_e2I = 0.0
                    for n in range(self.f_parameters.shape[1]):
                        for m in range(self.f_parameters.shape[2]):
                            for l in range(self.f_parameters.shape[3]):
                                p = self.f_parameters[channel, n, m, l] * e_powers[e1, e2, n] * n_powers[atom, e2, m] * n_powers[atom, e1, l]
                                poly += p
                                poly_diff_ee += n * p
                                poly_diff_e1I += l * p
                                poly_diff_e2I += m * p
                    f1, f1_diff, _ = cutoff(r_e1I, L1, self.f_trunc, self.f_alt)
                    f2, f2_diff, _ = cutoff(r_e2I, L2, self.f_trunc, self.f_alt)
                    diff_e1I = poly_diff_e1I / r_e1I * f1 * f2 + poly * f1_diff * f2
                    diff_e2I = poly_diff_e2I / r_e2I * f1 * f2 + poly * f1 * f2_diff
                    diff_ee = poly_diff_ee / r_ee * f1 * f2
                    res[e1] += diff_e1I / r_e1I * n_vectors[atom, e1] + diff_ee / r_ee * e_vectors[e1, e2]
                    res[e2] += diff_e2I / r_e2I * n_vectors[atom, e2] - diff_ee / r_ee * e_vectors[e1, e2]
        return res.ravel()

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'f_term_gradient_1e')
def gjastrow_f_term_gradient_1e(self, e_powers_1e: np.ndarray, n_powers: np.ndarray, e_vectors_1e: np.ndarray, n_vector_1e: np.ndarray, e: int):
    """Gradient of the term of rank (2, 1) w.r.t the coordinates of electron e,
    which moves the e-e distance of the triplet and its own e-n one.
    :param e_powers_1e: powers of the e-e distances of electron e
    :param n_powers: powers of e-n distances
    :param e_vectors_1e: e-e vectors of electron e - array(nelec, 3)
    :param n_vector_1e: e-n vectors of electron e - array(natom, 3)
    :param e: electron
    :return:
    """

    def impl(self, e_powers_1e: np.ndarray, n_powers: np.ndarray, e_vectors_1e: np.ndarray, n_vector_1e: np.ndarray, e: int) -> np.ndarray:
        res = np.zeros(shape=3)
        s1 = int(e >= self.neu)
        for atom in range(self.f_channel.shape[2]):
            for e2 in range(self.neu + self.ned):
                if e2 == e:
                    continue
                s2 = int(e2 >= self.neu)
                channel = self.f_channel[s1, s2, atom]
                if channel < 0:
                    continue
                L1 = self.f_cutoff[self.f_cutoff_channel[s1, atom]]
                L2 = self.f_cutoff[self.f_cutoff_channel[s2, atom]]
                r_e1I = n_powers[atom, e, 1]
                r_e2I = n_powers[atom, e2, 1]
                if r_e1I >= L1 or r_e2I >= L2:
                    continue
                r_ee = e_powers_1e[e2, 1]
                first, second = min(e, e2), max(e, e2)
                poly = poly_diff_ee = poly_diff_first = poly_diff_second = 0.0
                for n in range(self.f_parameters.shape[1]):
                    for m in range(self.f_parameters.shape[2]):
                        for l in range(self.f_parameters.shape[3]):
                            p = self.f_parameters[channel, n, m, l] * e_powers_1e[e2, n] * n_powers[atom, first, m] * n_powers[atom, second, l]
                            poly += p
                            poly_diff_ee += n * p
                            poly_diff_first += m * p
                            poly_diff_second += l * p
                f1, f1_diff, _ = cutoff(r_e1I, L1, self.f_trunc, self.f_alt)
                f2, _, _ = cutoff(r_e2I, L2, self.f_trunc, self.f_alt)
                if e == first:
                    poly_diff_e1I = poly_diff_first
                else:
                    poly_diff_e1I = poly_diff_second
                diff_e1I = poly_diff_e1I / r_e1I * f1 * f2 + poly * f1_diff * f2
                diff_ee = poly_diff_ee / r_ee * f1 * f2
                res += diff_e1I / r_e1I * n_vector_1e[atom] + diff_ee / r_ee * e_vectors_1e[e2]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'u_term_laplacian')
def gjastrow_u_term_laplacian(self, e_powers: np.ndarray):
    """Laplacian of the term of rank (2, 0) w.r.t e-coordinates. The term is a
    function of one distance alone, whose laplacian is d²/dr² + 2/r d/dr, and
    both electrons of the pair contribute it.
    :param e_powers: powers of e-e distances
    :return:
    """

    def impl(self, e_powers: np.ndarray) -> float:
        res = 0.0
        for e1 in range(1, self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for e2 in range(e1):
                s2 = int(e2 >= self.neu)
                channel = self.u_channel[s1, s2]
                if channel < 0:
                    continue
                L = self.u_cutoff[self.u_cutoff_channel[s1, s2]]
                r_ee = e_powers[e1, e2, 1]
                if r_ee >= L:
                    continue
                poly = poly_diff = poly_diff_2 = 0.0
                for k in range(self.u_parameters.shape[1]):
                    p = self.u_parameters[channel, k] * e_powers[e1, e2, k]
                    poly += p
                    poly_diff += k * p
                    poly_diff_2 += k * (k - 1) * p
                f, f_diff, f_diff_2 = cutoff(r_ee, L, self.u_trunc, self.u_alt)
                diff = poly_diff / r_ee * f + poly * f_diff
                diff_2 = poly_diff_2 / r_ee**2 * f + 2 * poly_diff / r_ee * f_diff + poly * f_diff_2
                res += diff_2 + 2 / r_ee * diff
        # sum by i-th and j-th electron coordinates
        return 2 * res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'chi_term_laplacian')
def gjastrow_chi_term_laplacian(self, n_powers: np.ndarray):
    """Laplacian of the term of rank (1, 1) w.r.t e-coordinates
    :param n_powers: powers of e-n distances
    :return:
    """

    def impl(self, n_powers: np.ndarray) -> float:
        res = 0.0
        for e1 in range(self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for atom in range(self.chi_channel.shape[1]):
                channel = self.chi_channel[s1, atom]
                if channel < 0:
                    continue
                L = self.chi_cutoff[self.chi_cutoff_channel[s1, atom]]
                r_eI = n_powers[atom, e1, 1]
                if r_eI >= L:
                    continue
                poly = poly_diff = poly_diff_2 = 0.0
                for k in range(self.chi_parameters.shape[1]):
                    p = self.chi_parameters[channel, k] * n_powers[atom, e1, k]
                    poly += p
                    poly_diff += k * p
                    poly_diff_2 += k * (k - 1) * p
                f, f_diff, f_diff_2 = cutoff(r_eI, L, self.chi_trunc, self.chi_alt)
                diff = poly_diff / r_eI * f + poly * f_diff
                diff_2 = poly_diff_2 / r_eI**2 * f + 2 * poly_diff / r_eI * f_diff + poly * f_diff_2
                res += diff_2 + 2 / r_eI * diff
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'f_term_laplacian')
def gjastrow_f_term_laplacian(self, e_powers: np.ndarray, n_powers: np.ndarray, n_vectors: np.ndarray):
    """Laplacian of the term of rank (2, 1) w.r.t e-coordinates. Every electron
    of the triplet moves two of the three distances the term is a function of,
    so its laplacian carries the cross derivative of the two, weighted by the
    cosine of the angle between the directions they grow in:
        ∇²F = ∂²F/∂r_eI² + 2/r_eI ∂F/∂r_eI + ∂²F/∂r_ee² + 2/r_ee ∂F/∂r_ee
              ± 2 (r_eI_vec r_ee_vec) / (r_eI r_ee) ∂²F/∂r_eI∂r_ee
    the sign being that of the electron in the e-e vector.
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_powers: np.ndarray, n_powers: np.ndarray, n_vectors: np.ndarray) -> float:
        res = 0.0
        for atom in range(self.f_channel.shape[2]):
            # r_ee_vec = r_e1I_vec - r_e2I_vec, so the two cosines are these dot products
            r_e1I_vec_dot_r_e2I_vec = n_vectors[atom] @ n_vectors[atom].T
            for e1 in range(1, self.neu + self.ned):
                s1 = int(e1 >= self.neu)
                for e2 in range(e1):
                    s2 = int(e2 >= self.neu)
                    channel = self.f_channel[s1, s2, atom]
                    if channel < 0:
                        continue
                    L1 = self.f_cutoff[self.f_cutoff_channel[s1, atom]]
                    L2 = self.f_cutoff[self.f_cutoff_channel[s2, atom]]
                    r_e1I = n_powers[atom, e1, 1]
                    r_e2I = n_powers[atom, e2, 1]
                    if r_e1I >= L1 or r_e2I >= L2:
                        continue
                    r_ee = e_powers[e1, e2, 1]
                    poly = poly_diff_ee = poly_diff_e1I = poly_diff_e2I = 0.0
                    poly_diff_ee_2 = poly_diff_e1I_2 = poly_diff_e2I_2 = 0.0
                    poly_diff_e1I_ee = poly_diff_e2I_ee = 0.0
                    for n in range(self.f_parameters.shape[1]):
                        for m in range(self.f_parameters.shape[2]):
                            for l in range(self.f_parameters.shape[3]):
                                p = self.f_parameters[channel, n, m, l] * e_powers[e1, e2, n] * n_powers[atom, e2, m] * n_powers[atom, e1, l]
                                poly += p
                                poly_diff_ee += n * p
                                poly_diff_e1I += l * p
                                poly_diff_e2I += m * p
                                poly_diff_ee_2 += n * (n - 1) * p
                                poly_diff_e1I_2 += l * (l - 1) * p
                                poly_diff_e2I_2 += m * (m - 1) * p
                                poly_diff_e1I_ee += l * n * p
                                poly_diff_e2I_ee += m * n * p
                    f1, f1_diff, f1_diff_2 = cutoff(r_e1I, L1, self.f_trunc, self.f_alt)
                    f2, f2_diff, f2_diff_2 = cutoff(r_e2I, L2, self.f_trunc, self.f_alt)
                    diff_e1I = poly_diff_e1I / r_e1I * f1 * f2 + poly * f1_diff * f2
                    diff_e2I = poly_diff_e2I / r_e2I * f1 * f2 + poly * f1 * f2_diff
                    diff_ee = poly_diff_ee / r_ee * f1 * f2
                    diff_e1I_2 = poly_diff_e1I_2 / r_e1I**2 * f1 * f2 + 2 * poly_diff_e1I / r_e1I * f1_diff * f2 + poly * f1_diff_2 * f2
                    diff_e2I_2 = poly_diff_e2I_2 / r_e2I**2 * f1 * f2 + 2 * poly_diff_e2I / r_e2I * f1 * f2_diff + poly * f1 * f2_diff_2
                    diff_ee_2 = poly_diff_ee_2 / r_ee**2 * f1 * f2
                    diff_e1I_ee = poly_diff_e1I_ee / (r_e1I * r_ee) * f1 * f2 + poly_diff_ee / r_ee * f1_diff * f2
                    diff_e2I_ee = poly_diff_e2I_ee / (r_e2I * r_ee) * f1 * f2 + poly_diff_ee / r_ee * f1 * f2_diff
                    cos_e1I = (r_e1I**2 - r_e1I_vec_dot_r_e2I_vec[e1, e2]) / (r_e1I * r_ee)
                    cos_e2I = (r_e2I**2 - r_e1I_vec_dot_r_e2I_vec[e1, e2]) / (r_e2I * r_ee)
                    res += (
                        diff_e1I_2
                        + 2 / r_e1I * diff_e1I
                        + diff_e2I_2
                        + 2 / r_e2I * diff_e2I
                        + 2 * (diff_ee_2 + 2 / r_ee * diff_ee)
                        + 2 * (cos_e1I * diff_e1I_ee + cos_e2I * diff_e2I_ee)
                    )
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'value')
def gjastrow_value(self, e_vectors: np.ndarray, n_vectors: np.ndarray):
    """Jastrow value
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> float:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.u_term(e_powers) + self.chi_term(n_powers) + self.f_term(e_powers, n_powers)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'value_1e')
def gjastrow_value_1e(self, e_vectors_1e: np.ndarray, n_powers: np.ndarray, e: int):
    """Part of the jastrow that contains electron e, CASINO's jas1_diff. Everything else is
    common to both ends of a single-electron move and cancels in the ratio.
    :param e_vectors_1e: e-e vectors of electron e - array(nelec, 3)
    :param n_powers: powers of e-n distances
    :param e: electron
    :return:
    """

    def impl(self, e_vectors_1e: np.ndarray, n_powers: np.ndarray, e: int) -> float:
        e_powers_1e = self.ee_powers_1e(e_vectors_1e)

        return self.u_term_1e(e_powers_1e, e) + self.chi_term_1e(n_powers, e) + self.f_term_1e(e_powers_1e, n_powers, e)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'gradient')
def gjastrow_gradient(self, e_vectors: np.ndarray, n_vectors: np.ndarray):
    """Jastrow gradient w.r.t. e-coordinates
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return (
            self.u_term_gradient(e_powers, e_vectors)
            + self.chi_term_gradient(n_powers, n_vectors)
            + self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
        )

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'gradient_1e')
def gjastrow_gradient_1e(self, e_vectors_1e: np.ndarray, n_vector_1e: np.ndarray, n_powers: np.ndarray, e: int):
    """Jastrow gradient w.r.t the coordinates of electron e, which is all a single-electron
    drift asks of the jastrow. Everything else it contains does not depend on that electron.
    :param e_vectors_1e: e-e vectors of electron e - array(nelec, 3)
    :param n_vector_1e: e-n vectors of electron e - array(natom, 3)
    :param n_powers: powers of e-n distances
    :param e: electron
    :return:
    """

    def impl(self, e_vectors_1e: np.ndarray, n_vector_1e: np.ndarray, n_powers: np.ndarray, e: int) -> np.ndarray:
        e_powers_1e = self.ee_powers_1e(e_vectors_1e)

        return (
            self.u_term_gradient_1e(e_powers_1e, e_vectors_1e, e)
            + self.chi_term_gradient_1e(n_powers, n_vector_1e, e)
            + self.f_term_gradient_1e(e_powers_1e, n_powers, e_vectors_1e, n_vector_1e, e)
        )

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'laplacian')
def gjastrow_laplacian(self, e_vectors: np.ndarray, n_vectors: np.ndarray):
    """Jastrow laplacian w.r.t. e-coordinates
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> tuple[float, np.ndarray]:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        grad = (
            self.u_term_gradient(e_powers, e_vectors)
            + self.chi_term_gradient(n_powers, n_vectors)
            + self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
        )

        lap = self.u_term_laplacian(e_powers) + self.chi_term_laplacian(n_powers) + self.f_term_laplacian(e_powers, n_powers, n_vectors)
        return lap, grad

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'u_term_parameters_d1')
def gjastrow_u_term_parameters_d1(self, e_powers: np.ndarray):
    """First derivatives of the term of rank (2, 0) w.r.t. its coefficients. The
    term is linear in them, so the derivative w.r.t. one of them is the basis
    function it multiplies, summed over the pairs of its channel.
    :param e_powers: powers of e-e distances
    :return:
    """

    def impl(self, e_powers: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.u_parameters.shape)
        for e1 in range(1, self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for e2 in range(e1):
                s2 = int(e2 >= self.neu)
                channel = self.u_channel[s1, s2]
                if channel < 0:
                    continue
                L = self.u_cutoff[self.u_cutoff_channel[s1, s2]]
                r_ee = e_powers[e1, e2, 1]
                if r_ee >= L:
                    continue
                f = cutoff(r_ee, L, self.u_trunc, self.u_alt)[0]
                for k in range(self.u_parameters.shape[1]):
                    res[channel, k] += e_powers[e1, e2, k] * f
        return res.ravel()

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'chi_term_parameters_d1')
def gjastrow_chi_term_parameters_d1(self, n_powers: np.ndarray):
    """First derivatives of the term of rank (1, 1) w.r.t. its coefficients
    :param n_powers: powers of e-n distances
    :return:
    """

    def impl(self, n_powers: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.chi_parameters.shape)
        for e1 in range(self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for atom in range(self.chi_channel.shape[1]):
                channel = self.chi_channel[s1, atom]
                if channel < 0:
                    continue
                L = self.chi_cutoff[self.chi_cutoff_channel[s1, atom]]
                r_eI = n_powers[atom, e1, 1]
                if r_eI >= L:
                    continue
                f = cutoff(r_eI, L, self.chi_trunc, self.chi_alt)[0]
                for k in range(self.chi_parameters.shape[1]):
                    res[channel, k] += n_powers[atom, e1, k] * f
        return res.ravel()

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'f_term_parameters_d1')
def gjastrow_f_term_parameters_d1(self, e_powers: np.ndarray, n_powers: np.ndarray):
    """First derivatives of the term of rank (2, 1) w.r.t. its coefficients
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :return:
    """

    def impl(self, e_powers: np.ndarray, n_powers: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.f_parameters.shape)
        for atom in range(self.f_channel.shape[2]):
            for e1 in range(1, self.neu + self.ned):
                s1 = int(e1 >= self.neu)
                for e2 in range(e1):
                    s2 = int(e2 >= self.neu)
                    channel = self.f_channel[s1, s2, atom]
                    if channel < 0:
                        continue
                    L1 = self.f_cutoff[self.f_cutoff_channel[s1, atom]]
                    L2 = self.f_cutoff[self.f_cutoff_channel[s2, atom]]
                    r_e1I = n_powers[atom, e1, 1]
                    r_e2I = n_powers[atom, e2, 1]
                    if r_e1I >= L1 or r_e2I >= L2:
                        continue
                    f = cutoff(r_e1I, L1, self.f_trunc, self.f_alt)[0] * cutoff(r_e2I, L2, self.f_trunc, self.f_alt)[0]
                    for n in range(self.f_parameters.shape[1]):
                        for m in range(self.f_parameters.shape[2]):
                            for l in range(self.f_parameters.shape[3]):
                                res[channel, n, m, l] += e_powers[e1, e2, n] * n_powers[atom, e2, m] * n_powers[atom, e1, l] * f
        return res.ravel()

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'u_term_gradient_parameters_d1')
def gjastrow_u_term_gradient_parameters_d1(self, e_powers: np.ndarray, e_vectors: np.ndarray):
    """First derivatives of the gradient of the term of rank (2, 0) w.r.t. its
    coefficients, which is the gradient of the basis function each of them multiplies.
    :param e_powers: powers of e-e distances
    :param e_vectors: e-e vectors
    :return:
    """

    def impl(self, e_powers: np.ndarray, e_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.u_parameters.shape + (self.neu + self.ned, 3))
        for e1 in range(1, self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for e2 in range(e1):
                s2 = int(e2 >= self.neu)
                channel = self.u_channel[s1, s2]
                if channel < 0:
                    continue
                L = self.u_cutoff[self.u_cutoff_channel[s1, s2]]
                r_ee = e_powers[e1, e2, 1]
                if r_ee >= L:
                    continue
                f, f_diff, _ = cutoff(r_ee, L, self.u_trunc, self.u_alt)
                for k in range(self.u_parameters.shape[1]):
                    p = e_powers[e1, e2, k]
                    gradient = (k * p / r_ee * f + p * f_diff) / r_ee * e_vectors[e1, e2]
                    res[channel, k, e1] += gradient
                    res[channel, k, e2] -= gradient
        return res.reshape(self.u_parameters.size, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'chi_term_gradient_parameters_d1')
def gjastrow_chi_term_gradient_parameters_d1(self, n_powers: np.ndarray, n_vectors: np.ndarray):
    """First derivatives of the gradient of the term of rank (1, 1) w.r.t. its coefficients
    :param n_powers: powers of e-n distances
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, n_powers: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.chi_parameters.shape + (self.neu + self.ned, 3))
        for e1 in range(self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for atom in range(self.chi_channel.shape[1]):
                channel = self.chi_channel[s1, atom]
                if channel < 0:
                    continue
                L = self.chi_cutoff[self.chi_cutoff_channel[s1, atom]]
                r_eI = n_powers[atom, e1, 1]
                if r_eI >= L:
                    continue
                f, f_diff, _ = cutoff(r_eI, L, self.chi_trunc, self.chi_alt)
                for k in range(self.chi_parameters.shape[1]):
                    p = n_powers[atom, e1, k]
                    res[channel, k, e1] += (k * p / r_eI * f + p * f_diff) / r_eI * n_vectors[atom, e1]
        return res.reshape(self.chi_parameters.size, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'f_term_gradient_parameters_d1')
def gjastrow_f_term_gradient_parameters_d1(self, e_powers: np.ndarray, n_powers: np.ndarray, e_vectors: np.ndarray, n_vectors: np.ndarray):
    """First derivatives of the gradient of the term of rank (2, 1) w.r.t. its coefficients
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_powers: np.ndarray, n_powers: np.ndarray, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.f_parameters.shape + (self.neu + self.ned, 3))
        for atom in range(self.f_channel.shape[2]):
            for e1 in range(1, self.neu + self.ned):
                s1 = int(e1 >= self.neu)
                for e2 in range(e1):
                    s2 = int(e2 >= self.neu)
                    channel = self.f_channel[s1, s2, atom]
                    if channel < 0:
                        continue
                    L1 = self.f_cutoff[self.f_cutoff_channel[s1, atom]]
                    L2 = self.f_cutoff[self.f_cutoff_channel[s2, atom]]
                    r_e1I = n_powers[atom, e1, 1]
                    r_e2I = n_powers[atom, e2, 1]
                    if r_e1I >= L1 or r_e2I >= L2:
                        continue
                    r_ee = e_powers[e1, e2, 1]
                    f1, f1_diff, _ = cutoff(r_e1I, L1, self.f_trunc, self.f_alt)
                    f2, f2_diff, _ = cutoff(r_e2I, L2, self.f_trunc, self.f_alt)
                    for n in range(self.f_parameters.shape[1]):
                        for m in range(self.f_parameters.shape[2]):
                            for l in range(self.f_parameters.shape[3]):
                                p = e_powers[e1, e2, n] * n_powers[atom, e2, m] * n_powers[atom, e1, l]
                                diff_e1I = l * p / r_e1I * f1 * f2 + p * f1_diff * f2
                                diff_e2I = m * p / r_e2I * f1 * f2 + p * f1 * f2_diff
                                diff_ee = n * p / r_ee * f1 * f2
                                res[channel, n, m, l, e1] += diff_e1I / r_e1I * n_vectors[atom, e1] + diff_ee / r_ee * e_vectors[e1, e2]
                                res[channel, n, m, l, e2] += diff_e2I / r_e2I * n_vectors[atom, e2] - diff_ee / r_ee * e_vectors[e1, e2]
        return res.reshape(self.f_parameters.size, (self.neu + self.ned) * 3)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'u_term_laplacian_parameters_d1')
def gjastrow_u_term_laplacian_parameters_d1(self, e_powers: np.ndarray):
    """First derivatives of the laplacian of the term of rank (2, 0) w.r.t. its coefficients
    :param e_powers: powers of e-e distances
    :return:
    """

    def impl(self, e_powers: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.u_parameters.shape)
        for e1 in range(1, self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for e2 in range(e1):
                s2 = int(e2 >= self.neu)
                channel = self.u_channel[s1, s2]
                if channel < 0:
                    continue
                L = self.u_cutoff[self.u_cutoff_channel[s1, s2]]
                r_ee = e_powers[e1, e2, 1]
                if r_ee >= L:
                    continue
                f, f_diff, f_diff_2 = cutoff(r_ee, L, self.u_trunc, self.u_alt)
                for k in range(self.u_parameters.shape[1]):
                    p = e_powers[e1, e2, k]
                    diff = k * p / r_ee * f + p * f_diff
                    diff_2 = k * (k - 1) * p / r_ee**2 * f + 2 * k * p / r_ee * f_diff + p * f_diff_2
                    # sum by i-th and j-th electron coordinates
                    res[channel, k] += 2 * (diff_2 + 2 / r_ee * diff)
        return res.ravel()

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'chi_term_laplacian_parameters_d1')
def gjastrow_chi_term_laplacian_parameters_d1(self, n_powers: np.ndarray):
    """First derivatives of the laplacian of the term of rank (1, 1) w.r.t. its coefficients
    :param n_powers: powers of e-n distances
    :return:
    """

    def impl(self, n_powers: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.chi_parameters.shape)
        for e1 in range(self.neu + self.ned):
            s1 = int(e1 >= self.neu)
            for atom in range(self.chi_channel.shape[1]):
                channel = self.chi_channel[s1, atom]
                if channel < 0:
                    continue
                L = self.chi_cutoff[self.chi_cutoff_channel[s1, atom]]
                r_eI = n_powers[atom, e1, 1]
                if r_eI >= L:
                    continue
                f, f_diff, f_diff_2 = cutoff(r_eI, L, self.chi_trunc, self.chi_alt)
                for k in range(self.chi_parameters.shape[1]):
                    p = n_powers[atom, e1, k]
                    diff = k * p / r_eI * f + p * f_diff
                    diff_2 = k * (k - 1) * p / r_eI**2 * f + 2 * k * p / r_eI * f_diff + p * f_diff_2
                    res[channel, k] += diff_2 + 2 / r_eI * diff
        return res.ravel()

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'f_term_laplacian_parameters_d1')
def gjastrow_f_term_laplacian_parameters_d1(self, e_powers: np.ndarray, n_powers: np.ndarray, n_vectors: np.ndarray):
    """First derivatives of the laplacian of the term of rank (2, 1) w.r.t. its coefficients
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_powers: np.ndarray, n_powers: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.f_parameters.shape)
        for atom in range(self.f_channel.shape[2]):
            r_e1I_vec_dot_r_e2I_vec = n_vectors[atom] @ n_vectors[atom].T
            for e1 in range(1, self.neu + self.ned):
                s1 = int(e1 >= self.neu)
                for e2 in range(e1):
                    s2 = int(e2 >= self.neu)
                    channel = self.f_channel[s1, s2, atom]
                    if channel < 0:
                        continue
                    L1 = self.f_cutoff[self.f_cutoff_channel[s1, atom]]
                    L2 = self.f_cutoff[self.f_cutoff_channel[s2, atom]]
                    r_e1I = n_powers[atom, e1, 1]
                    r_e2I = n_powers[atom, e2, 1]
                    if r_e1I >= L1 or r_e2I >= L2:
                        continue
                    r_ee = e_powers[e1, e2, 1]
                    f1, f1_diff, f1_diff_2 = cutoff(r_e1I, L1, self.f_trunc, self.f_alt)
                    f2, f2_diff, f2_diff_2 = cutoff(r_e2I, L2, self.f_trunc, self.f_alt)
                    cos_e1I = (r_e1I**2 - r_e1I_vec_dot_r_e2I_vec[e1, e2]) / (r_e1I * r_ee)
                    cos_e2I = (r_e2I**2 - r_e1I_vec_dot_r_e2I_vec[e1, e2]) / (r_e2I * r_ee)
                    for n in range(self.f_parameters.shape[1]):
                        for m in range(self.f_parameters.shape[2]):
                            for l in range(self.f_parameters.shape[3]):
                                p = e_powers[e1, e2, n] * n_powers[atom, e2, m] * n_powers[atom, e1, l]
                                diff_e1I = l * p / r_e1I * f1 * f2 + p * f1_diff * f2
                                diff_e2I = m * p / r_e2I * f1 * f2 + p * f1 * f2_diff
                                diff_ee = n * p / r_ee * f1 * f2
                                diff_e1I_2 = l * (l - 1) * p / r_e1I**2 * f1 * f2 + 2 * l * p / r_e1I * f1_diff * f2 + p * f1_diff_2 * f2
                                diff_e2I_2 = m * (m - 1) * p / r_e2I**2 * f1 * f2 + 2 * m * p / r_e2I * f1 * f2_diff + p * f1 * f2_diff_2
                                diff_ee_2 = n * (n - 1) * p / r_ee**2 * f1 * f2
                                diff_e1I_ee = l * n * p / (r_e1I * r_ee) * f1 * f2 + n * p / r_ee * f1_diff * f2
                                diff_e2I_ee = m * n * p / (r_e2I * r_ee) * f1 * f2 + n * p / r_ee * f1 * f2_diff
                                res[channel, n, m, l] += (
                                    diff_e1I_2
                                    + 2 / r_e1I * diff_e1I
                                    + diff_e2I_2
                                    + 2 / r_e2I * diff_e2I
                                    + 2 * (diff_ee_2 + 2 / r_ee * diff_ee)
                                    + 2 * (cos_e1I * diff_e1I_ee + cos_e2I * diff_e2I_ee)
                                )
        return res.ravel()

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'cutoff_parameters_d1')
def gjastrow_cutoff_parameters_d1(self, e_powers: np.ndarray, n_powers: np.ndarray):
    """First derivatives of the jastrow w.r.t. the cutoff lengths at fixed
    coefficients, taken numerically as the standard jastrow takes them. A cutoff
    length belongs to one term, so only that term is evaluated again.
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :return:
    """

    def impl(self, e_powers: np.ndarray, n_powers: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.u_cutoff.size + self.chi_cutoff.size + self.f_cutoff.size)
        n = 0
        for i in range(self.u_cutoff.size):
            self.u_cutoff[i] -= delta
            res[n] -= self.u_term(e_powers)
            self.u_cutoff[i] += 2 * delta
            res[n] += self.u_term(e_powers)
            self.u_cutoff[i] -= delta
            n += 1
        for i in range(self.chi_cutoff.size):
            self.chi_cutoff[i] -= delta
            res[n] -= self.chi_term(n_powers)
            self.chi_cutoff[i] += 2 * delta
            res[n] += self.chi_term(n_powers)
            self.chi_cutoff[i] -= delta
            n += 1
        for i in range(self.f_cutoff.size):
            self.f_cutoff[i] -= delta
            res[n] -= self.f_term(e_powers, n_powers)
            self.f_cutoff[i] += 2 * delta
            res[n] += self.f_term(e_powers, n_powers)
            self.f_cutoff[i] -= delta
            n += 1
        return res / delta / 2

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'cutoff_gradient_parameters_d1')
def gjastrow_cutoff_gradient_parameters_d1(self, e_powers: np.ndarray, n_powers: np.ndarray, e_vectors: np.ndarray, n_vectors: np.ndarray):
    """First derivatives of the jastrow gradient w.r.t. the cutoff lengths
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_powers: np.ndarray, n_powers: np.ndarray, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        size = self.u_cutoff.size + self.chi_cutoff.size + self.f_cutoff.size
        res = np.zeros(shape=(size, (self.neu + self.ned) * 3))
        n = 0
        for i in range(self.u_cutoff.size):
            self.u_cutoff[i] -= delta
            res[n] -= self.u_term_gradient(e_powers, e_vectors)
            self.u_cutoff[i] += 2 * delta
            res[n] += self.u_term_gradient(e_powers, e_vectors)
            self.u_cutoff[i] -= delta
            n += 1
        for i in range(self.chi_cutoff.size):
            self.chi_cutoff[i] -= delta
            res[n] -= self.chi_term_gradient(n_powers, n_vectors)
            self.chi_cutoff[i] += 2 * delta
            res[n] += self.chi_term_gradient(n_powers, n_vectors)
            self.chi_cutoff[i] -= delta
            n += 1
        for i in range(self.f_cutoff.size):
            self.f_cutoff[i] -= delta
            res[n] -= self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
            self.f_cutoff[i] += 2 * delta
            res[n] += self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors)
            self.f_cutoff[i] -= delta
            n += 1
        return res / delta / 2

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'cutoff_laplacian_parameters_d1')
def gjastrow_cutoff_laplacian_parameters_d1(self, e_powers: np.ndarray, n_powers: np.ndarray, n_vectors: np.ndarray):
    """First derivatives of the jastrow laplacian w.r.t. the cutoff lengths
    :param e_powers: powers of e-e distances
    :param n_powers: powers of e-n distances
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_powers: np.ndarray, n_powers: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=self.u_cutoff.size + self.chi_cutoff.size + self.f_cutoff.size)
        n = 0
        for i in range(self.u_cutoff.size):
            self.u_cutoff[i] -= delta
            res[n] -= self.u_term_laplacian(e_powers)
            self.u_cutoff[i] += 2 * delta
            res[n] += self.u_term_laplacian(e_powers)
            self.u_cutoff[i] -= delta
            n += 1
        for i in range(self.chi_cutoff.size):
            self.chi_cutoff[i] -= delta
            res[n] -= self.chi_term_laplacian(n_powers)
            self.chi_cutoff[i] += 2 * delta
            res[n] += self.chi_term_laplacian(n_powers)
            self.chi_cutoff[i] -= delta
            n += 1
        for i in range(self.f_cutoff.size):
            self.f_cutoff[i] -= delta
            res[n] -= self.f_term_laplacian(e_powers, n_powers, n_vectors)
            self.f_cutoff[i] += 2 * delta
            res[n] += self.f_term_laplacian(e_powers, n_powers, n_vectors)
            self.f_cutoff[i] -= delta
            n += 1
        return res / delta / 2

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'value_parameters_d1')
def gjastrow_value_parameters_d1(self, e_vectors: np.ndarray, n_vectors: np.ndarray):
    """First derivatives of the jastrow w.r.t. the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.parameters_projector.T @ np.concatenate(
            (
                self.cutoff_parameters_d1(e_powers, n_powers),
                self.u_term_parameters_d1(e_powers),
                self.chi_term_parameters_d1(n_powers),
                self.f_term_parameters_d1(e_powers, n_powers),
            )
        )

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'gradient_parameters_d1')
def gjastrow_gradient_parameters_d1(self, e_vectors: np.ndarray, n_vectors: np.ndarray):
    """First derivatives of the jastrow gradient w.r.t. the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.parameters_projector.T @ np.concatenate(
            (
                self.cutoff_gradient_parameters_d1(e_powers, n_powers, e_vectors, n_vectors),
                self.u_term_gradient_parameters_d1(e_powers, e_vectors),
                self.chi_term_gradient_parameters_d1(n_powers, n_vectors),
                self.f_term_gradient_parameters_d1(e_powers, n_powers, e_vectors, n_vectors),
            )
        )

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'laplacian_parameters_d1')
def gjastrow_laplacian_parameters_d1(self, e_vectors: np.ndarray, n_vectors: np.ndarray):
    """First derivatives of the jastrow laplacian w.r.t. the parameters
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        e_powers = self.ee_powers(e_vectors)
        n_powers = self.en_powers(n_vectors)

        return self.parameters_projector.T @ np.concatenate(
            (
                self.cutoff_laplacian_parameters_d1(e_powers, n_powers, n_vectors),
                self.u_term_laplacian_parameters_d1(e_powers),
                self.chi_term_laplacian_parameters_d1(n_powers),
                self.f_term_laplacian_parameters_d1(e_powers, n_powers, n_vectors),
            )
        )

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'value_parameters_d2')
def gjastrow_value_parameters_d2(self, e_vectors: np.ndarray, n_vectors: np.ndarray):
    """Second derivatives of the jastrow w.r.t. the parameters. The block of the
    coefficients is zero, the jastrow being linear in them and the constraints that
    determine the rest of them affine; the block of the cutoff lengths is not, and
    is not implemented, so newton emin optimizes them on a hessian that is short of
    their own second derivatives.
    :param e_vectors: e-e vectors
    :param n_vectors: e-n vectors
    :return:
    """

    def impl(self, e_vectors: np.ndarray, n_vectors: np.ndarray) -> np.ndarray:
        size = self.parameters_projector.shape[1]
        return np.zeros(shape=(size, size))

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'get_linear_parameters')
def gjastrow_get_linear_parameters(self):
    """Coefficients of every term of every rank, in the order they are stored in."""

    def impl(self) -> np.ndarray:
        return np.concatenate((self.u_parameters.ravel(), self.chi_parameters.ravel(), self.f_parameters.ravel()))

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'set_linear_parameters')
def gjastrow_set_linear_parameters(self, values: np.ndarray):
    """Set the coefficients of every term of every rank.
    :param values: coefficients in the order they are stored in
    """

    def impl(self, values: np.ndarray):
        u = self.u_parameters.size
        chi = u + self.chi_parameters.size
        self.u_parameters[:] = values[:u].reshape(self.u_parameters.shape)
        self.chi_parameters[:] = values[u:chi].reshape(self.chi_parameters.shape)
        self.f_parameters[:] = values[chi:].reshape(self.f_parameters.shape)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'get_cutoffs')
def gjastrow_get_cutoffs(self):
    """Cutoff lengths of every term of every rank, in the order they are stored in."""

    def impl(self) -> np.ndarray:
        return np.concatenate((self.u_cutoff, self.chi_cutoff, self.f_cutoff))

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'set_cutoffs')
def gjastrow_set_cutoffs(self, values: np.ndarray):
    """Set the cutoff lengths of every term of every rank.
    :param values: cutoff lengths in the order they are stored in
    """

    def impl(self, values: np.ndarray):
        u = self.u_cutoff.size
        chi = u + self.chi_cutoff.size
        self.u_cutoff[:] = values[:u]
        self.chi_cutoff[:] = values[u:chi]
        self.f_cutoff[:] = values[chi:]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'constraint_matrix')
def gjastrow_constraint_matrix(self, derivative):
    """The equations that determine coefficients, at the current cutoff lengths, or
    their derivatives w.r.t. one of those. A cutoff length enters an equation of a
    coalescence through the value and the radial derivative of the cutoff function
    where the two particles meet, which is why the equation is kept in the two parts
    those multiply rather than as it comes out of the elimination.
    :param derivative: the cutoff length to differentiate by, -1 for the equations
    :return:
    """

    def impl(self, derivative) -> np.ndarray:
        L = self.get_cutoffs()
        res = np.zeros(shape=self.constraint_value.shape)
        for i in range(res.shape[0]):
            k = self.constraint_cutoff[i]
            if derivative >= 0 and k != derivative:
                continue
            if k < 0:
                res[i] = self.constraint_value[i]
                continue
            f, f_diff = cutoff_at_zero(L[k], self.cutoff_trunc[k], self.cutoff_alt[k], derivative >= 0)
            res[i] = f_diff * self.constraint_value[i] + f * self.constraint_slope[i]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'fix_parameters')
def gjastrow_fix_parameters(self):
    """Solve the constraints at the current cutoff lengths: set every coefficient
    they determine and rebuild the affine map it is a function of the others by.
    Elimination has already chosen which equation determines which coefficient, and
    a cutoff length only scales the equations it is in, so what is left is one
    inversion of the block of the determined coefficients.
    """

    def impl(self):
        if self.constraint_pivot.size == 0:
            return
        a = self.constraint_matrix(-1)
        self.constraint_inverse = np.ascontiguousarray(np.linalg.inv(np.ascontiguousarray(a[:, self.constraint_pivot])))
        gradient = -self.constraint_inverse @ a
        offset = self.constraint_inverse @ self.constraint_rhs
        values = self.get_linear_parameters()
        self.parameters_jacobian = np.eye(a.shape[1])
        self.parameters_offset = np.zeros(shape=a.shape[1])
        for i in range(self.constraint_pivot.size):
            self.parameters_jacobian[self.constraint_pivot[i]] = gradient[i]
            self.parameters_offset[self.constraint_pivot[i]] = offset[i]
            values[self.constraint_pivot[i]] = 0.0
        # a determined coefficient is a function of the coefficients no equation
        # determines alone, and the elimination has taken the rest out of the row
        for i in range(self.constraint_pivot.size):
            for j in range(self.constraint_pivot.size):
                self.parameters_jacobian[self.constraint_pivot[i], self.constraint_pivot[j]] = 0.0
        self.set_linear_parameters(self.parameters_jacobian @ values + self.parameters_offset)

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'cutoff_parameters_jacobian')
def gjastrow_cutoff_parameters_jacobian(self, k):
    """Derivatives of the coefficients w.r.t. one cutoff length. The coefficients no
    equation determines do not move with it, and the ones that are determined move
    so that their equations keep holding: A(L) c = b differentiates into
    A dc/dL = -(dA/dL) c, whose only unknowns are the determined coefficients.
    :param k: the cutoff length
    :return:
    """

    def impl(self, k) -> np.ndarray:
        res = np.zeros(shape=self.parameters_offset.size)
        if self.constraint_pivot.size == 0:
            return res
        x = -self.constraint_inverse @ (self.constraint_matrix(k) @ self.get_linear_parameters())
        for i in range(self.constraint_pivot.size):
            res[self.constraint_pivot[i]] = x[i]
        return res

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'set_parameters_projector')
def gjastrow_set_parameters_projector(self):
    """Set the projector from the parameters that are optimized onto all of them,
    which is what the derivatives w.r.t. every parameter are contracted with: a
    cutoff length carries the determined coefficients with it, and so does a
    coefficient no equation determines.
    """

    def impl(self):
        cutoffs = self.get_cutoffs().size
        jacobian = np.eye(cutoffs + self.parameters_offset.size)
        jacobian[cutoffs:, cutoffs:] = self.parameters_jacobian
        for k in range(cutoffs):
            jacobian[cutoffs:, k] = self.cutoff_parameters_jacobian(k)
        self.parameters_projector = np.ascontiguousarray(jacobian[:, np.argwhere(self.get_parameters_mask(False)).ravel()])

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'get_parameters_mask')
def gjastrow_get_parameters_mask(self, all_parameters):
    """Which parameters are optimized, over the cutoff lengths followed by the
    coefficients. A cutoff length is only ever optimized when it is free to move,
    the coefficients it constrains being a function of it.
    :param all_parameters: every parameter no constraint determines, or only the optimizable ones
    :return:
    """

    def impl(self, all_parameters) -> np.ndarray:
        cutoffs = self.cutoff_optimizable.copy()
        if not self.cutoffs_optimizable:
            cutoffs[:] = False
        if all_parameters:
            return np.concatenate((cutoffs, self.parameters_available))
        return np.concatenate((cutoffs, self.parameters_mask))

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'get_parameters')
def gjastrow_get_parameters(self, all_parameters):
    """Parameters that are optimized, over the cutoff lengths of every term of every
    rank followed by the coefficients of all of them.
    :param all_parameters: every parameter no constraint determines, or only the optimizable ones
    :return:
    """

    def impl(self, all_parameters) -> np.ndarray:
        return np.concatenate((self.get_cutoffs(), self.get_linear_parameters()))[self.get_parameters_mask(all_parameters)]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'set_parameters')
def gjastrow_set_parameters(self, parameters, all_parameters):
    """Set the parameters that are optimized and solve the constraints at the cutoff
    lengths they leave, which is what keeps the cusp and the symmetry of a channel.
    :param parameters: parameters to set, followed by those of the rest of the wfn
    :param all_parameters: every parameter no constraint determines, or only the optimizable ones
    :return: the parameters that are left
    """

    def impl(self, parameters, all_parameters) -> np.ndarray:
        n = 0
        mask = self.get_parameters_mask(all_parameters)
        cutoffs = self.get_cutoffs()
        for i in range(cutoffs.size):
            if mask[i]:
                cutoffs[i] = parameters[n]
                n += 1
        self.set_cutoffs(cutoffs)
        values = self.get_linear_parameters()
        for i in range(values.size):
            if mask[cutoffs.size + i]:
                values[i] = parameters[n]
                n += 1
        self.set_linear_parameters(values)
        self.fix_parameters()
        return parameters[n:]

    return impl


@nb.njit(nogil=True, parallel=False, cache=True)
@overload_method(Gjastrow_class_t, 'get_parameters_scale')
def gjastrow_get_parameters_scale(self, all_parameters):
    """Characteristic scale of every optimized parameter, which reformulates the
    optimization in dimensionless variables. A cutoff length is of order one already.
    :param all_parameters: every parameter no constraint determines, or only the optimizable ones
    :return:
    """

    def impl(self, all_parameters) -> np.ndarray:
        return np.concatenate((np.ones(shape=self.get_cutoffs().size), self.parameters_scale))[self.get_parameters_mask(all_parameters)]

    return impl


class Gjastrow(structref.StructRefProxy, AbstractJastrow):
    """Generic Jastrow factor, restricted to the terms of rank (2, 0), (1, 1) and
    (2, 1) of a natural power basis with a polynomial cutoff.

    Framework for constructing generic Jastrow correlation factors
    P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
    Phys. Rev. E 86, 036703
    """

    def __new__(cls, config):
        @nb.njit(nogil=True, parallel=False, cache=True)
        def init(
            neu,
            ned,
            u_parameters,
            u_cutoff,
            u_channel,
            u_cutoff_channel,
            u_trunc,
            u_alt,
            chi_parameters,
            chi_cutoff,
            chi_channel,
            chi_cutoff_channel,
            chi_trunc,
            chi_alt,
            f_parameters,
            f_cutoff,
            f_channel,
            f_cutoff_channel,
            f_trunc,
            f_alt,
            cutoff_optimizable,
            cutoff_trunc,
            cutoff_alt,
            constraint_value,
            constraint_slope,
            constraint_cutoff,
            constraint_rhs,
            constraint_pivot,
            parameters_mask,
            parameters_available,
            parameters_scale,
        ):
            self = structref.new(Gjastrow_t)
            self.neu = neu
            self.ned = ned
            # the channel tables are indexed by the spin of every electron and the
            # number of every nucleus, and hold -1 where the term has no channel
            self.u_parameters = u_parameters
            self.u_cutoff = u_cutoff
            self.u_channel = u_channel
            self.u_cutoff_channel = u_cutoff_channel
            self.u_trunc = u_trunc
            self.u_alt = u_alt
            self.chi_parameters = chi_parameters
            self.chi_cutoff = chi_cutoff
            self.chi_channel = chi_channel
            self.chi_cutoff_channel = chi_cutoff_channel
            self.chi_trunc = chi_trunc
            self.chi_alt = chi_alt
            self.f_parameters = f_parameters
            self.f_cutoff = f_cutoff
            self.f_channel = f_channel
            self.f_cutoff_channel = f_cutoff_channel
            self.f_trunc = f_trunc
            self.f_alt = f_alt
            # the equations that determine coefficients, one per coefficient they
            # determine, kept in the parts a cutoff length scales so that they can be
            # solved again whenever one of those moves
            self.cutoff_optimizable = cutoff_optimizable
            self.cutoff_trunc = cutoff_trunc
            self.cutoff_alt = cutoff_alt
            self.constraint_value = constraint_value
            self.constraint_slope = constraint_slope
            self.constraint_cutoff = constraint_cutoff
            self.constraint_rhs = constraint_rhs
            self.constraint_pivot = constraint_pivot
            self.constraint_inverse = np.zeros(shape=(0, 0))
            self.parameters_mask = parameters_mask
            self.parameters_available = parameters_available
            self.parameters_scale = parameters_scale
            self.cutoffs_optimizable = True
            # a determined coefficient is an affine function of the coefficients no
            # equation determines, which is what the optimizer differentiates through
            self.parameters_jacobian = np.eye(parameters_mask.size)
            self.parameters_offset = np.zeros(shape=parameters_mask.size)
            self.fix_parameters()
            self.set_parameters_projector()
            # the distance itself is the first power, so it is there to be read
            # even for a rank the file carries no term of
            self.max_ee_order = max(2, u_parameters.shape[1], f_parameters.shape[1])
            self.max_en_order = max(2, chi_parameters.shape[1], f_parameters.shape[2])
            return self

        arrays = term_by_rank(config)
        u, chi, f = arrays
        return init(config.input.neu, config.input.ned, *u, *chi, *f, *linear_arrays(config, arrays))

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def max_ee_order(self):
        return self.max_ee_order

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def max_en_order(self):
        return self.max_en_order

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def u_parameters(self):
        return self.u_parameters

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def chi_parameters(self):
        return self.chi_parameters

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def f_parameters(self):
        return self.f_parameters

    @nb.njit(nogil=True, parallel=False, cache=True)
    def ee_powers(self, e_vectors):
        return self.ee_powers(e_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def ee_powers_1e(self, e_vectors_1e):
        return self.ee_powers_1e(e_vectors_1e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def en_powers(self, n_vectors):
        return self.en_powers(n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def update_en_powers_1e(self, n_powers, n_vector, e):
        return self.update_en_powers_1e(n_powers, n_vector, e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def u_term(self, e_powers):
        return self.u_term(e_powers)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def chi_term(self, n_powers):
        return self.chi_term(n_powers)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def f_term(self, e_powers, n_powers):
        return self.f_term(e_powers, n_powers)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def u_term_gradient(self, e_powers, e_vectors):
        return self.u_term_gradient(e_powers, e_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def chi_term_gradient(self, n_powers, n_vectors):
        return self.chi_term_gradient(n_powers, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def f_term_gradient(self, e_powers, n_powers, e_vectors, n_vectors):
        return self.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def u_term_laplacian(self, e_powers):
        return self.u_term_laplacian(e_powers)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def chi_term_laplacian(self, n_powers):
        return self.chi_term_laplacian(n_powers)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def f_term_laplacian(self, e_powers, n_powers, n_vectors):
        return self.f_term_laplacian(e_powers, n_powers, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def u_term_1e(self, e_powers_1e, e):
        return self.u_term_1e(e_powers_1e, e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def chi_term_1e(self, n_powers, e):
        return self.chi_term_1e(n_powers, e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def f_term_1e(self, e_powers_1e, n_powers, e):
        return self.f_term_1e(e_powers_1e, n_powers, e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value(self, e_vectors, n_vectors):
        return self.value(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value_1e(self, e_vectors_1e, n_powers, e):
        return self.value_1e(e_vectors_1e, n_powers, e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def gradient(self, e_vectors, n_vectors):
        return self.gradient(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def gradient_1e(self, e_vectors_1e, n_vector_1e, n_powers, e):
        return self.gradient_1e(e_vectors_1e, n_vector_1e, n_powers, e)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def laplacian(self, e_vectors, n_vectors):
        return self.laplacian(e_vectors, n_vectors)

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def parameters_projector(self):
        return self.parameters_projector

    @property
    @nb.njit(nogil=True, parallel=False, cache=True)
    def cutoffs_optimizable(self):
        return self.cutoffs_optimizable

    @cutoffs_optimizable.setter
    @nb.njit(nogil=True, parallel=False, cache=True)
    def cutoffs_optimizable(self, value):
        self.cutoffs_optimizable = value

    @nb.njit(nogil=True, parallel=False, cache=True)
    def set_parameters_projector(self):
        self.set_parameters_projector()

    @nb.njit(nogil=True, parallel=False, cache=True)
    def get_parameters(self, all_parameters=False):
        return self.get_parameters(all_parameters)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def set_parameters(self, parameters, all_parameters=False):
        return self.set_parameters(parameters, all_parameters)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def get_parameters_scale(self, all_parameters=False):
        return self.get_parameters_scale(all_parameters)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def value_parameters_d1(self, e_vectors, n_vectors):
        return self.value_parameters_d1(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def gradient_parameters_d1(self, e_vectors, n_vectors):
        return self.gradient_parameters_d1(e_vectors, n_vectors)

    @nb.njit(nogil=True, parallel=False, cache=True)
    def laplacian_parameters_d1(self, e_vectors, n_vectors):
        return self.laplacian_parameters_d1(e_vectors, n_vectors)


structref.define_boxing(Gjastrow_class_t, Gjastrow)
