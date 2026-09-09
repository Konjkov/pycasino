"""Constraints on the linear parameters of one channel of a generic Jastrow term.

Three things fix some coefficients of a channel in terms of the others: the term
does not change when identical particles are relabelled (Eq. 12); the local
energy must stay finite where two particles meet, which pins the radial
derivative of the term there to the Kato value or to zero (Eqs. 20-25); and a
coefficient the rules have removed is zero. All three are linear in the
coefficients, so one gaussian elimination says which coefficients the rest
determine and what they come to.
CASINO source: gjastrow.f90, init_symm_constraints and init_value_constraints

Framework for constructing generic Jastrow correlation factors
P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
Phys. Rev. E 86, 036703
"""

import collections

import numpy as np

TOLERANCE = 1e-12

# the equations that determine a coefficient, one per coefficient they determine:
# the two parts a cutoff length scales, the cutoff length of every equation, its
# right hand side, and the coefficient it determines
Reduced = collections.namedtuple('Reduced', 'value slope channel rhs pivot')


def power_products(order_a, order_b):
    """Index of the product of two functions of a power basis, whose functions are
    the powers of one core function: f_i f_j = f_(i+j-1).
    """
    table = np.zeros(shape=(order_a + 1, order_b + 1), dtype=int)
    for i in range(1, order_a + 1):
        table[i, 1:] = i + np.arange(1, order_b + 1) - 1
    table[0, 0] = 1
    table[1:, 0] = np.arange(1, order_a + 1)
    table[0, 1:] = np.arange(1, order_b + 1)
    return table


def independent_products(order_a, order_b, unity_a, unity_b):
    """Index of the product of two functions of unrelated bases, where no product
    is any other, the unit function of either basis aside.
    """
    table = np.zeros(shape=(order_a + 1, order_b + 1), dtype=int)
    count = 0
    for j in range(1, order_b + 1):
        for i in range(1, order_a + 1):
            count += 1
            table[i, j] = count
    if 0 < unity_a <= order_a and 0 < unity_b <= order_b:
        table[0, 0] = table[unity_a, unity_b]
    else:
        count += 1
        table[0, 0] = count
    if 0 < unity_a <= order_a:
        table[0, 1:] = table[unity_a, 1:]
    else:
        table[0, 1:] = count + np.arange(1, order_b + 1)
        count += order_b
    if 0 < unity_b <= order_b:
        table[1:, 0] = table[1:, unity_b]
    else:
        table[1:, 0] = count + np.arange(1, order_a + 1)
    return table


def matrices(vector, rank):
    """The e-e and e-n pair matrices of an index vector or of a signature."""
    ee = np.zeros(shape=(rank[0], rank[0]), dtype=int)
    en = np.zeros(shape=(rank[0], rank[1]), dtype=int)
    k = 0
    for i in range(rank[0]):
        for j in range(i + 1, rank[0]):
            ee[i, j] = ee[j, i] = vector[k]
            k += 1
    for i in range(rank[0]):
        for j in range(rank[1]):
            en[i, j] = vector[k]
            k += 1
    return ee, en


def pair(position, rank):
    """The two particles of a pair, as the place of the electron among the
    electrons and that of the other particle among the electrons or the nuclei.
    """
    size_ee = rank[0] * (rank[0] - 1) // 2
    if position < size_ee:
        for i in range(rank[0]):
            for j in range(i + 1, rank[0]):
                if position == 0:
                    return i, j
                position -= 1
    return divmod(position - size_ee, rank[1])


def permute(index, rank, electrons, nuclei):
    """The index vector a relabelling of the particles carries an index vector to."""
    ee, en = matrices(index, rank)
    out = [ee[electrons[i], electrons[j]] for i in range(rank[0]) for j in range(i + 1, rank[0])]
    out += [en[electrons[i], nuclei[j]] for i in range(rank[0]) for j in range(rank[1])]
    return tuple(out)


class Constraints:
    """The linear system that constrains the coefficients of one channel, built one
    group of equations at a time and solved once they are all in.

    :param rank: numbers of electrons and of nuclei the term is a function of
    :param shape: expansion orders, one axis per pair of particles
    :param signature: group of every pair, in the same order as the axes
    :param permutations: relabellings of the particles that leave the channel as
        it is, as pairs of electron and nucleus permutations
    """

    def __init__(self, rank, shape, signature, permutations):
        self.rank = rank
        self.shape = shape
        self.signature = signature
        self.permutations = permutations
        self.size = int(np.prod(shape))
        self.rows = []
        self.values = []
        self.slopes = []
        self.channels = []
        self.rhs = []
        # the affine map from the coefficients no equation determines to all of
        # them, which is the identity until an equation determines one
        self.jacobian = np.eye(self.size)
        self.offset = np.zeros(shape=self.size)
        self.pivots = []
        self.pivot_rows = []
        # how many equations each kind of constraint contributed, as casino reports
        self.counts = {'removal': 0, 'symmetry': 0, 'coalescence': 0}

    def add(self, kind, value, slope=None, channel=-1, factor=(0.0, 1.0), rhs=0.0):
        """One equation, kept in the two parts the cutoff of a pair scales: the values
        of the basis functions where the two particles meet, which the radial
        derivative of the cutoff there multiplies, and their radial derivatives, which
        its value multiplies. An equation no cutoff length enters is kept in the first
        part alone, so that changing a cutoff length rebuilds the system it belongs to.
        :param factor: value and radial derivative of the cutoff of the pair at zero
        :param channel: the cutoff length of the pair, -1 for an equation without one
        """
        if slope is None:
            slope = np.zeros(self.size)
        row = factor[1] * value + factor[0] * slope
        self.rows.append(row)
        if channel < 0:
            # a pair with no cutoff of its own scales nothing, so the equation is
            # kept as it comes out, and the first part is the whole of it
            value, slope = row, np.zeros(self.size)
        self.values.append(value)
        self.slopes.append(slope)
        self.channels.append(channel)
        self.rhs.append(rhs)
        self.counts[kind] += 1

    def index(self, index):
        """Position of a coefficient in the flat list the equations are written in."""
        return int(np.ravel_multi_index(index, self.shape))

    def removal(self, removed):
        """A coefficient the rules have removed is zero."""
        for index in np.argwhere(removed):
            row = np.zeros(self.size)
            row[self.index(tuple(index))] = 1.0
            self.add('removal', row)

    def symmetry(self):
        """Coefficients that a relabelling of identical particles carries onto one
        another are equal, which is one equation per member of an orbit but the
        first. The sign of the relabelling is one throughout, every basis
        supported here being a function of a distance alone.
        """
        seen = set()
        for index in np.ndindex(self.shape):
            if index in seen:
                continue
            seen.add(index)
            for electrons, nuclei in self.permutations:
                other = permute(index, self.rank, electrons, nuclei)
                if other in seen:
                    continue
                seen.add(other)
                row = np.zeros(self.size)
                row[self.index(index)] = 1.0
                row[self.index(other)] = -1.0
                self.add('symmetry', row)

    def coalescence(self, position, basis, factor, channel, cusp, products, unity):
        """Where the two particles of a pair meet, the radial derivative of the term
        is pinned: to the Kato value of the pair for the part of the term that is a
        function of that pair alone, and to zero for every other part of it, or the
        local energy would diverge.

        The two particles become one there, so the pairs each of them makes with a
        third particle collapse into a single distance, and the coefficients whose
        basis functions then multiply out to the same function of that distance go
        into one equation. PRODUCTS says which products those are.

        :param position: the pair whose particles meet, as an axis of the shape
        :param basis: values and radial derivatives of the basis functions of that
            pair at zero, one per expansion index
        :param factor: value and radial derivative of the cutoff of that pair at zero
        :param channel: the cutoff length of that pair, -1 where it has no cutoff
        :param cusp: the Kato value of the pair, zero where the term is not to
            carry the cusp
        :param products: index of the product of two basis functions, one table per
            pair of axes, keyed by the pair of axes
        :param unity: index of the basis function that is one, per kind of pair, and
            zero where a cutoff makes the pair contribute more than its basis
        """
        target = factor[1] * basis[0] + factor[0] * basis[1]
        equations = {}
        for index in np.ndindex(self.shape):
            signature, coalescing = self.equation(index, position, products)
            carries_cusp = bool(cusp) and self.is_pair_only(index, position, unity)
            if target[coalescing - 1] == 0.0 and not carries_cusp:
                continue
            value, slope, rhs = equations.setdefault(signature, (np.zeros(self.size), np.zeros(self.size), [0.0]))
            value[self.index(index)] = basis[0][coalescing - 1]
            slope[self.index(index)] = basis[1][coalescing - 1]
            if carries_cusp:
                rhs[0] = cusp
        for value, slope, rhs in equations.values():
            self.add('coalescence', value, slope, channel, factor, rhs[0])

    def is_pair_only(self, index, position, unity):
        """Whether a coefficient belongs to the part of the term that is a function
        of one pair alone, which is the part the Kato cusp of that pair applies to.
        """
        size_ee = self.rank[0] * (self.rank[0] - 1) // 2
        for other, value in enumerate(index):
            if other == position:
                continue
            if value + 1 != unity[0 if other < size_ee else 1]:
                return False
        return True

    def equation(self, index, position, products):
        """Which equation a coefficient goes into for the coalescence of one pair,
        as the vector of the indices left once the two particles are one particle.
        """
        size_ee = self.rank[0] * (self.rank[0] - 1) // 2
        # relabel so that the two particles that meet come first
        electrons = list(range(self.rank[0]))
        nuclei = list(range(self.rank[1]))
        first, second = pair(position, self.rank)
        electrons[0], electrons[first] = electrons[first], electrons[0]
        if position < size_ee:
            electrons[1], electrons[second] = electrons[second], electrons[1]
        else:
            nuclei[0], nuclei[second] = nuclei[second], nuclei[0]
        ee, en = matrices(np.array(index) + 1, self.rank)
        sig_ee, sig_en = matrices(self.signature, self.rank)
        if position < size_ee:
            coalescing = ee[electrons[0], electrons[1]]
            for i in range(2, self.rank[0]):
                same = sig_ee[electrons[0], electrons[i]] == sig_ee[electrons[1], electrons[i]]
                table = products['e-e', 'e-e', same]
                ee[electrons[1], electrons[i]] = ee[electrons[i], electrons[1]] = table[
                    ee[electrons[0], electrons[i]], ee[electrons[1], electrons[i]]
                ]
            for j in range(self.rank[1]):
                same = sig_en[electrons[0], nuclei[j]] == sig_en[electrons[1], nuclei[j]]
                table = products['e-n', 'e-n', same]
                en[electrons[1], nuclei[j]] = table[en[electrons[0], nuclei[j]], en[electrons[1], nuclei[j]]]
            signature = [ee[electrons[j], electrons[i]] for i in range(1, self.rank[0]) for j in range(i + 1, self.rank[0])]
            signature += [en[electrons[i], nuclei[j]] for i in range(1, self.rank[0]) for j in range(self.rank[1])]
        else:
            coalescing = en[electrons[0], nuclei[0]]
            table = products['e-e', 'e-n', False]
            for i in range(1, self.rank[0]):
                ee[electrons[0], electrons[i]] = ee[electrons[i], electrons[0]] = table[ee[electrons[0], electrons[i]], en[electrons[i], nuclei[0]]]
            signature = [en[electrons[i], nuclei[j]] for i in range(self.rank[0]) for j in range(1, self.rank[1])]
            signature += [ee[electrons[j], electrons[i]] for i in range(self.rank[0]) for j in range(i + 1, self.rank[0])]
        return tuple(signature), coalescing

    def solve(self, parameters):
        """Reduce the equations and read off which coefficients they determine.

        Elimination goes by coefficient rather than by equation, so a coefficient
        is determined by the first equation that still involves it once the earlier
        ones are eliminated, as it is in casino.
        :return: which coefficients are determined, and their values
        """
        values = parameters.ravel().copy()
        determined = np.zeros(shape=self.size, dtype=bool)
        if not self.rows:
            return determined.reshape(self.shape), values.reshape(self.shape)
        matrix = np.array(self.rows)
        rhs = np.array(self.rhs)
        order = list(range(len(matrix)))
        row = 0
        for column in range(self.size):
            if row == len(matrix):
                break
            candidate = row + int(np.argmax(np.abs(matrix[row:, column])))
            if abs(matrix[candidate, column]) <= TOLERANCE:
                continue
            matrix[[row, candidate]] = matrix[[candidate, row]]
            rhs[[row, candidate]] = rhs[[candidate, row]]
            order[row], order[candidate] = order[candidate], order[row]
            rhs[row] /= matrix[row, column]
            matrix[row] /= matrix[row, column]
            factor = matrix[:, column].copy()
            factor[row] = 0.0
            matrix -= np.outer(factor, matrix[row])
            rhs -= factor * rhs[row]
            self.pivots.append(column)
            self.pivot_rows.append(order[row])
            row += 1
        pivots = self.pivots
        for i in range(row, len(matrix)):
            if abs(rhs[i]) > TOLERANCE:
                raise ValueError('the constraints on the channel have no solution')
        # elimination leaves every equation with one coefficient of its own, so a
        # determined coefficient is an affine function of the coefficients that are
        # left, and the map is what the optimizer differentiates through
        for i, column in enumerate(pivots):
            determined[column] = True
            self.jacobian[column] = -matrix[i]
            self.jacobian[column, column] = 0.0
            self.offset[column] = rhs[i]
        values[determined] = 0.0
        values = self.jacobian @ values + self.offset
        return determined.reshape(self.shape), values.reshape(self.shape)

    def reduced(self):
        """The equations elimination kept, one per coefficient they determine. Which
        equations those are, and which coefficient each of them determines, is what
        elimination decides, and a cutoff length only scales the equations it is in,
        so the system can be rebuilt and solved again at another cutoff length
        without eliminating anything twice.
        """
        rows = np.array(self.pivot_rows, dtype=int)
        return Reduced(
            np.array(self.values).reshape(-1, self.size)[rows],
            np.array(self.slopes).reshape(-1, self.size)[rows],
            np.array(self.channels, dtype=int)[rows],
            np.array(self.rhs)[rows],
            np.array(self.pivots, dtype=int),
        )
