import numpy as np

from casino.readers import casl, channels, constraints

# functional bases and cutoff functions, numbered as in CASINO's gbasis.f90
BASIS_CODE = {
    'none': 0,
    'natural power': 1,
    'cosine': 2,
    'cosine with k-cutoff': 3,
    'r/(r+a) power': 4,
    'r/(r^b+a) power': 5,
    '1/(r+a) power': 6,
    'natural power vectorial': 7,
    'natural polynomial': 8,
    'natural polynomial vectorial': 9,
    'RPA': 10,
    'logarithmic cusp': 11,
    'dipole cusp': 12,
    'half-integer power': 13,
    'tilted dipole cusp': 14,
    'nu': 15,
    'exp power': 16,
    'CHAMP 2-body': 17,
    'CHAMP 2-body cutoff': 18,
    'CHAMP 3-body': 19,
    'CHAMP 3-body cutoff': 20,
    'CHAMP 3-body fix': 21,
    'CHAMP 3-body cutoff fix': 22,
}
CUTOFF_CODE = {
    'none': 0,
    'polynomial': 23,
    'alt polynomial': 24,
    'gaussian': 25,
    'anisotropic polynomial': 26,
    'quasicusp': 27,
    'spline': 28,
    'orbital cusp': 29,
}
FUNCTION_NAME = {code: name for name, code in {**BASIS_CODE, **CUTOFF_CODE}.items()}

# name, default value and default limits of the parameters of every function
# supported here, which are the isotropic ones an aperiodic system can use
PARAMETERS = {
    0: (),
    1: (),
    4: (('a', 3.0, 1.1e-8, np.inf),),
    5: (('a', 3.0, 1.1e-8, np.inf), ('b', 1.3, 1.0, np.inf)),
    6: (('a', 3.0, 1.1e-8, np.inf),),
    16: (('a', 1.0, 1.1e-8, np.inf),),
    23: (('L', 3.0, 1e-3, np.inf),),
    24: (('L', 3.0, 1e-3, np.inf),),
    25: (('L', 3.0, 1e-3, np.inf),),
    27: (('L', 1.0, 1e-3, np.inf),),
    28: (('L', 1.0, 1e-3, np.inf), ('x', 0.5, 0.05, 0.95)),
}
# name and default value of the constants of every such function, the integer
# ones being those whose default is an int
CONSTANTS = {
    4: (('L', np.inf),),
    23: (('C', 3),),
    24: (('C', 3),),
    25: (('L_hard', 1e3),),
    28: (('C', 2),),
}

# index of the function of a set that is identically one, zero where it has none
UNITY = {0: 1, 1: 1, 4: 1, 5: 1, 6: 1, 16: 1}
# the bases whose functions are the powers of one core function, so that the
# product of two of them is another function of the set
POWER = (1, 4, 5, 6, 16)

DETERMINED, FIXED, OPTIMIZABLE, UNSET = -1, 0, 1, 2
FLAG_CODE = {'determined': DETERMINED, '-1': DETERMINED, 'fixed': FIXED, '0': FIXED, 'opt': OPTIMIZABLE, 'optimizable': OPTIMIZABLE, '1': OPTIMIZABLE}
FLAG_NAME = {DETERMINED: 'determined', FIXED: 'fixed', OPTIMIZABLE: 'optimizable'}
INFINITE = ('none', '-inf', 'inf', '+inf')


def scalar(item):
    """The value of an item and the flags that follow it, an item being either
    the bare value or the inline block that carries the flags.
    """
    if isinstance(item, casl.Block):
        return item.implicit()
    return [item]


def flag(values):
    """The optimizability an item declares, a value on its own being optimizable."""
    if len(values) < 2:
        return OPTIMIZABLE
    name = casl.unique(values[1])
    if name not in FLAG_CODE:
        raise ValueError(f'unknown optimizability flag {values[1]}')
    return FLAG_CODE[name]


def unit_index(function):
    """Index of the function of a set that is identically one, zero if it has none."""
    return UNITY.get(function.code if function else 0, 0)


def kato_ee(spin, other):
    """Gamma of the electron-electron Kato cusp in three dimensions: the strength of
    the coulomb divergence over the dimensionality plus one for a pair of identical
    particles and minus one for a distinguishable pair.
    """
    return 0.25 if spin == other else 0.5


def kato_en(charge):
    """Gamma of the electron-nucleus Kato cusp in three dimensions."""
    return -charge


def products(code, other, order, other_order, same):
    """Index of the product of a function of one set and one of another, so that
    coefficients whose functions multiply out to the same function of one distance
    can be put in one equation.
    CASINO source: gbasis.f90, eval_eqprod_gbasis
    """
    if code != other:
        return constraints.independent_products(order, other_order, 0, 0)
    if code == 0:
        return np.ones(shape=(order + 1, other_order + 1), dtype=int)
    # a power basis has no parameters of its own only for the natural powers, so
    # every other one gives independent products across two different channels
    if code in POWER and (same or code == 1):
        return constraints.power_products(order, other_order)
    return constraints.independent_products(order, other_order, 0, 0)


def model(name):
    """The set of particles a 'Channel <model>' item is named after."""
    unique = casl.unique(name)
    if not unique.startswith('channel'):
        raise ValueError(f'expected a "Channel <model>" item, found {name}')
    return unique[len('channel') :]


class Function:
    """A functional basis or a cutoff function of one term, with the constants it
    is defined by and the parameters of every channel it is given.
    CASINO source: gbasis.f90
    """

    def __init__(self, code, order=1):
        self.code = code
        self.order = order
        self.names = [name for name, _, _, _ in PARAMETERS[code]]
        self.constants = {name: value for name, value in CONSTANTS.get(code, ())}
        self.channels = []
        self.parameters = np.zeros(shape=(0, len(self.names)))
        self.flags = np.zeros(shape=(0, len(self.names)), dtype=int)
        self.limits = np.zeros(shape=(0, len(self.names), 2))

    @classmethod
    def read(cls, block, table, order, label, groups, kind):
        """The function a block declares, None if the block is not there at all."""
        if block is None:
            return None
        name = block.item('Type')
        if name is None:
            raise ValueError(f'{label} declares no Type')
        code = next((value for key, value in table.items() if casl.unique(key) == casl.unique(name)), None)
        if code is None:
            raise ValueError(f'unknown {label} type {name}')
        if code not in PARAMETERS:
            raise NotImplementedError(f'{label} of type {name} is not supported')
        function = cls(code, order)
        function.read_constants(block.item('Constants'))
        function.read_parameters(block.item('Parameters'), groups, kind, label)
        return function

    def read_constants(self, block):
        for name, default in CONSTANTS.get(self.code, ()):
            value = block and block.item(name)
            if value is not None:
                self.constants[name] = int(value) if isinstance(default, int) else casl.to_float(value)

    def read_parameters(self, block, groups, kind, label):
        """A basis or a cutoff has one set of parameters per group of pairs, and
        one the file leaves out keeps the defaults casino gives it.
        """
        self.channels = groups.pair_names(kind)
        shape = (len(self.channels), len(self.names))
        self.parameters = np.zeros(shape=shape)
        self.flags = np.full(shape=shape, fill_value=UNSET)
        self.limits = np.zeros(shape=shape + (2,))
        for j, (_, default, low, high) in enumerate(PARAMETERS[self.code]):
            self.parameters[:, j] = default
            self.limits[:, j] = low, high
        given = {}
        for name, channel in (block or {}).items():
            i = groups.pair_index(model(name), kind)
            if i in given:
                raise ValueError(f'{label}: channels {given[i]} and {name} are one and the same')
            given[i] = name
            self.read_channel(i, channel)

    def read_channel(self, i, block):
        for j, name in enumerate(self.names):
            item = block.item(name)
            if item is None:
                continue
            values = scalar(item)
            if '%' in values[0]:
                raise NotImplementedError(f'percent initialization of parameter {name}')
            if casl.unique(values[0]) != 'default':
                self.parameters[i, j] = casl.to_float(values[0])
            self.flags[i, j] = flag(values)
            limits = isinstance(item, casl.Block) and item.item('limits')
            if not limits:
                continue
            # a declared limit only ever narrows the range the parameter is defined in
            for k, (limit, narrow) in enumerate(zip(limits.implicit(), (max, min))):
                if '%' in limit:
                    raise NotImplementedError(f'percent initialization of the limits of parameter {name}')
                if casl.unique(limit) == 'default':
                    continue
                if casl.unique(limit) in INFINITE:
                    self.limits[i, j, k] = -np.inf if k == 0 else np.inf
                else:
                    self.limits[i, j, k] = narrow(self.limits[i, j, k], casl.to_float(limit))

    def at_zero(self, i):
        """Value and radial derivative of every function of the set at zero, at the
        parameters of channel i.
        CASINO source: gbasis.f90, eval_cusp_gbasis
        """
        f0 = np.ones(shape=self.order)
        dfdr0 = np.zeros(shape=self.order)
        if self.code in (1, 16):  # natural power, exp power
            f0[1:] = 0.0
            dfdr0[1:2] = 1.0
        elif self.code in (4, 5):  # r/(r+a) power, r/(r^b+a) power
            f0[1:] = 0.0
            dfdr0[1:2] = 1 / self.parameters[i, 0]
        elif self.code == 6:  # 1/(r+a) power
            k = np.arange(self.order)
            f0 = self.parameters[i, 0] ** -k
            dfdr0 = -k * self.parameters[i, 0] ** -(k + 1.0)
        elif self.code == 23:  # polynomial cutoff
            dfdr0[0] = -self.constants['C'] / self.parameters[i, 0]
        elif self.code == 24:  # alt polynomial cutoff
            f0[0] = (-self.parameters[i, 0]) ** self.constants['C']
            dfdr0[0] = self.constants['C'] * (-self.parameters[i, 0]) ** (self.constants['C'] - 1)
        elif self.code == 27:  # quasicusp cutoff
            raise NotImplementedError('constraints of a quasicusp cutoff, which is not a plain function of the distance at zero')
        # 'none', the gaussian and the spline are one at zero with no slope there
        return f0, dfdr0

    def write(self, order=False):
        block = casl.Block()
        block.add('Type', FUNCTION_NAME[self.code])
        if order:
            block.add('Order', str(self.order))
        if self.constants:
            constants = casl.Block()
            for name, value in self.constants.items():
                constants.add(name, str(value) if isinstance(value, int) else casl.to_casl(value))
            block.add('Constants', constants)
        parameters = casl.Block()
        for i, channel in enumerate(self.channels):
            item = self.write_channel(i)
            if len(item):
                parameters.add(f'Channel {channel}', item)
        if len(parameters):
            block.add('Parameters', parameters)
        return block

    def write_channel(self, i):
        channel = casl.Block()
        for j, name in enumerate(self.names):
            if self.flags[i, j] == UNSET:
                continue
            item = casl.inline(casl.to_casl(self.parameters[i, j]), FLAG_NAME[self.flags[i, j]])
            low, high = self.limits[i, j]
            if np.isfinite(low) or np.isfinite(high):
                item.add(
                    'limits', casl.inline(casl.to_casl(low) if np.isfinite(low) else '-Inf', casl.to_casl(high) if np.isfinite(high) else '+Inf')
                )
            channel.add(name, item)
        return channel


class Term:
    """One term of the generic Jastrow factor: a linear expansion in the products
    of the e-e and e-n basis functions of rank_e electrons and rank_n nuclei, cut
    off so that it and its derivatives vanish at the cutoff length.
    CASINO source: gjastrow.f90
    """

    def __init__(self, name):
        self.name = name
        self.rank = (0, 0)
        self.rules = []
        self.only = []
        self.cusp = (False, False)
        self.waive = ([], [])
        self.ee_basis = None
        self.en_basis = None
        self.ee_cutoff = None
        self.en_cutoff = None
        self.groups = None
        self.orders = (0, 0)
        self.has_cusp = (False, False)
        self.equations = []
        self.channels = []
        self.channel_cusp = np.zeros(shape=(0, 2), dtype=bool)
        self.channel_cusp_given = np.zeros(shape=(0, 2), dtype=bool)
        self.parameters = np.zeros(shape=(0,))
        self.flags = np.zeros(shape=(0,), dtype=int)
        self.constraints = []

    @property
    def size_ee(self):
        """Number of electron pairs the term is a function of."""
        return self.rank[0] * (self.rank[0] - 1) // 2

    @property
    def size_en(self):
        """Number of electron-nucleus pairs the term is a function of."""
        return self.rank[0] * self.rank[1]

    def read(self, block, nele, atom_numbers):
        rank = block.item('Rank')
        if rank is None:
            raise ValueError(f'{self.name} declares no Rank')
        self.rank = tuple(int(value) for value in rank.implicit())
        if self.rank[0] < 1 or self.rank[1] < 0 or sum(self.rank) < 2:
            raise ValueError(f'{self.name} has a rank no term can be generated for: {self.rank}')
        for name in ('Indexing', 'e-e indexing', 'e-n indexing'):
            if block.item(name) is not None:
                raise NotImplementedError(f'{name} of {self.name}, whose expansion is not the standard one')
        self.read_rules(block.item('Rules'))
        if self.only:
            raise NotImplementedError(f'Rules:Only of {self.name}')
        self.groups = channels.Channels(self.rank, self.rules, nele, atom_numbers)
        self.orders = (self.order(block, 'e-e', self.size_ee), self.order(block, 'e-n', self.size_en))
        # a term that does not say whether it carries the cusp is decided later,
        # by whether it can carry one and whether an earlier term already does
        self.cusp = tuple(block.item(name) and casl.to_bool(block.item(name)) for name in ('e-e cusp', 'e-n cusp'))
        self.has_cusp = tuple(bool(value) for value in self.cusp)
        self.waive = tuple((block.item(name) or casl.Block()).implicit() for name in ('Waive e-e cusp', 'Waive e-n cusp'))
        for i, (kind, size) in enumerate((('e-e', self.size_ee), ('e-n', self.size_en))):
            basis = Function.read(block.item(f'{kind} basis'), BASIS_CODE, self.orders[i], f'{kind} basis of {self.name}', self.groups, kind)
            cutoff = Function.read(block.item(f'{kind} cutoff'), CUTOFF_CODE, 1, f'{kind} cutoff of {self.name}', self.groups, kind)
            if kind == 'e-e':
                self.ee_basis, self.ee_cutoff = basis, cutoff
            else:
                self.en_basis, self.en_cutoff = basis, cutoff
        self.read_linear(block.item('Linear parameters'))

    def read_rules(self, block):
        if block is None:
            return
        self.rules = block.implicit()
        only = block.item('Only')
        if only is not None:
            self.only = only.implicit()

    def order(self, block, kind, size):
        """Expansion order of a basis, one if it is not given so that 'Type: none'
        works, and zero if the term has no pair of that kind at all.
        """
        basis = block.item(f'{kind} basis')
        value = basis and basis.item('Order')
        if value is None:
            return 1 if size else 0
        if not size:
            raise ValueError(f'{self.name} has no {kind} pair, but an {kind} expansion order was found')
        return int(value)

    def read_linear(self, block):
        """One set of linear parameters per channel, in the order the rules put the
        channels in rather than the order the file happens to list them in.
        """
        self.channels = self.groups.names
        shape = (len(self.channels),) + (self.orders[0],) * self.size_ee + (self.orders[1],) * self.size_en
        self.channel_cusp = np.zeros(shape=(len(self.channels), 2), dtype=bool)
        self.channel_cusp_given = np.zeros(shape=(len(self.channels), 2), dtype=bool)
        self.parameters = np.zeros(shape=shape)
        self.flags = np.full(shape=shape, fill_value=UNSET)
        given = {}
        for channel, items in (block or {}).items():
            i = self.groups.index(model(channel))
            if i in given:
                raise ValueError(f'{self.name}: channels {given[i]} and {channel} are one and the same')
            given[i] = channel
            self.read_channel(i, channel, items)

    def read_channel(self, i, channel, block):
        for name, item in block.items():
            if casl.unique(name) in ('e-ecusp', 'e-ncusp'):
                k = int(casl.unique(name) == 'e-ncusp')
                self.channel_cusp[i, k] = casl.to_bool(item)
                self.channel_cusp_given[i, k] = True
            elif casl.unique(name).startswith('c_'):
                index = (i,) + tuple(int(number) - 1 for number in name[2:].split(','))
                values = scalar(item)
                self.parameters[index] = casl.to_float(values[0])
                self.flags[index] = flag(values)
            else:
                raise NotImplementedError(f'{name} of channel {channel} of {self.name}')

    @property
    def unity(self):
        """Index of the basis function that is one, per kind of pair, and zero where
        a cutoff makes the pair contribute more than its basis alone.
        """
        return (
            unit_index(self.ee_basis) if unit_index(self.ee_cutoff) == 1 else 0,
            unit_index(self.en_basis) if unit_index(self.en_cutoff) == 1 else 0,
        )

    def cusp_applicable(self, k):
        """Whether the term has a part that is a function of one pair of the kind
        alone, which is the only kind of part a Kato cusp can be imposed on.
        """
        unity = self.unity
        for index in np.ndindex(self.parameters.shape[1:]):
            counts = [0, 0]
            for position, value in enumerate(index):
                kind = 0 if position < self.size_ee else 1
                if value + 1 != unity[kind]:
                    counts[kind] += 1
            if counts == [1, 0] if k == 0 else counts == [0, 1]:
                return True
        return False

    def pairs(self, group, k):
        """The particle pairs one group of pairs holds."""
        if k == 0:
            return {(min(i, j), max(i, j)) for i, j in np.argwhere(self.groups.ee == group)}
        return {(spin, ion) for ion, spin in np.argwhere(self.groups.en == group)}

    def waived(self):
        """The groups of pairs the term asks to leave uncusped, whose particles it
        takes never to meet.
        """
        return tuple({self.groups.pair_index(name, kind) + 1 for name in self.waive[k]} for k, kind in enumerate(('e-e', 'e-n')))

    def coalescences(self, i, atom_charges, applied, pending):
        """Every pair of a channel whose particles can meet, with the derivative of
        the term where they do and the value that derivative is pinned to.
        """
        signature = self.groups.signatures[i]
        spins, ions = channels.parse_model(self.channels[i], self.rank)
        waived = self.waived()
        for position in range(self.size_ee + self.size_en):
            k = 0 if position < self.size_ee else 1
            group = signature[position]
            if group in waived[k]:
                continue
            basis, cutoff = (self.ee_basis, self.ee_cutoff) if k == 0 else (self.en_basis, self.en_cutoff)
            f0, dfdr0 = basis.at_zero(group - 1) if basis else (np.ones(self.orders[k]), np.zeros(self.orders[k]))
            f0_cut, dfdr0_cut = cutoff.at_zero(group - 1) if cutoff else (np.ones(1), np.zeros(1))
            factor = (f0_cut[0], dfdr0_cut[0])
            target = factor[1] * f0 + factor[0] * dfdr0
            cusp = 0.0
            pairs = self.pairs(group, k)
            if self.channel_cusp[i, k] and not pairs & applied[k]:
                first, second = constraints.pair(position, self.rank)
                if k == 0:
                    cusp = kato_ee(spins[first], spins[second])
                else:
                    cusp = kato_en(atom_charges[ions[second]])
                if abs(cusp) > 1e-11:
                    pending[k].update(pairs)
            if not target.any() and not cusp:
                continue
            # a pair with no cutoff of its own contributes no cutoff length to the
            # equations, so the cutoff lengths are the ones of the truncated pairs
            yield position, (f0, dfdr0), factor, group - 1 if cutoff and cutoff.code else -1, cusp

    def constrain(self, atom_charges, applied):
        """Impose the symmetry and the cusp constraints of every channel: flag the
        coefficients they determine and set them to what they determine.
        :param applied: the particle pairs an earlier term already cusps, added to
            with the ones this term takes over
        """
        tables = {}
        for k, kind in enumerate(('e-e', 'e-n')):
            basis = self.ee_basis if k == 0 else self.en_basis
            for same in (True, False):
                tables[kind, kind, same] = products(basis.code if basis else 0, basis.code if basis else 0, self.orders[k], self.orders[k], same)
        tables['e-e', 'e-n', False] = products(
            self.ee_basis.code if self.ee_basis else 0, self.en_basis.code if self.en_basis else 0, self.orders[0], self.orders[1], False
        )
        unity = self.unity
        pending = (set(), set())
        self.equations = []
        for i in range(len(self.channels)):
            system = constraints.Constraints(self.rank, self.parameters.shape[1:], self.groups.signatures[i], self.groups.permutations[i])
            system.symmetry()
            for position, basis, factor, channel, cusp in self.coalescences(i, atom_charges, applied, pending):
                system.coalescence(position, basis, factor, channel, cusp, tables, unity)
            determined, values = system.solve(self.parameters[i])
            self.equations.append(system.counts)
            self.constraints.append(system.reduced())
            self.parameters[i] = values
            # a coefficient the file does not mention is optimizable, as it is in casino
            self.flags[i] = np.where(self.flags[i] == UNSET, OPTIMIZABLE, self.flags[i])
            self.flags[i] = np.where(self.flags[i] == DETERMINED, OPTIMIZABLE, self.flags[i])
            self.flags[i] = np.where(determined, DETERMINED, self.flags[i])
        for k in (0, 1):
            applied[k].update(pending[k])

    def write(self, print_determined=False):
        block = casl.Block()
        block.add('Rank', casl.inline(*(str(rank) for rank in self.rank)))
        block.add('Rules', self.write_rules())
        for kind, basis, cutoff, cusp in (('e-e', self.ee_basis, self.ee_cutoff, 0), ('e-n', self.en_basis, self.en_cutoff, 1)):
            if basis is not None:
                block.add(f'{kind} basis', basis.write(order=True))
            if self.cusp[cusp] is not None:
                block.add(f'{kind} cusp', 'T' if self.cusp[cusp] else 'F')
            if self.waive[cusp]:
                block.add(f'Waive {kind} cusp', casl.inline(*self.waive[cusp]))
            if cutoff is not None:
                block.add(f'{kind} cutoff', cutoff.write())
        linear = casl.Block()
        for i, channel in enumerate(self.channels):
            item = self.write_channel(i, print_determined)
            if len(item):
                linear.add(f'Channel {channel}', item)
        if len(linear):
            block.add('Linear parameters', linear)
        return block

    def write_rules(self):
        if not self.only:
            return casl.inline(*self.rules)
        rules = casl.Block()
        for rule in self.rules:
            rules.add(None, rule)
        rules.add('Only', casl.inline(*self.only))
        return rules

    def write_channel(self, i, print_determined):
        channel = casl.Block()
        for kind, cusp in (('e-e', 0), ('e-n', 1)):
            if self.channel_cusp_given[i, cusp]:
                channel.add(f'{kind} cusp', 'T' if self.channel_cusp[i, cusp] else 'F')
        for index in np.ndindex(self.parameters.shape[1:]):
            if self.flags[(i,) + index] == UNSET:
                continue
            if self.flags[(i,) + index] == DETERMINED and not print_determined:
                continue
            name = 'c_' + ','.join(str(number + 1) for number in index)
            channel.add(name, casl.inline(casl.to_casl(self.parameters[(i,) + index]), FLAG_NAME[self.flags[(i,) + index]]))
        return channel


class Gjastrow:
    """Generic Jastrow factor.
    CASINO manual: 7.8 Wave function parameter file: parameters.casl

    Framework for constructing generic Jastrow correlation factors
    P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs
    Phys. Rev. E 86, 036703
    """

    def __init__(self, neu, ned):
        self.title = 'no title given'
        self.print_determined = False
        self.nele = (neu, ned)
        self.atom_numbers = np.zeros(shape=0, dtype=int)
        self.atom_charges = np.zeros(shape=0)
        self.terms = []

    def set_atoms(self, atom_numbers, atom_charges):
        """The nuclei, which the rules of a term group by atomic number and whose
        charge sets the electron-nucleus cusp.
        """
        self.atom_numbers = atom_numbers
        self.atom_charges = atom_charges

    def read(self, base_path):
        """Read the JASTROW block of parameters.casl."""
        tree = casl.read(base_path)
        block = tree and tree.item('JASTROW')
        if not block:
            return
        self.title = block.item('Title') or self.title
        self.print_determined = casl.to_bool(block.item('Print determined') or 'F')
        for name, item in block.items():
            if casl.unique(name).startswith('term'):
                term = Term(name)
                term.read(item, self.nele, self.atom_numbers)
                self.terms.append(term)
        self.constrain()

    def constrain(self):
        """Decide which term carries which cusp, then constrain every term. A term
        that says nothing about the electron-electron cusp carries it if it can and
        no earlier term does; one that says nothing about the electron-nucleus cusp
        does not carry it.
        """
        applied = (set(), set())
        carried = [False, False]
        for term in self.terms:
            has_cusp = []
            for k, kind in enumerate(('e-e', 'e-n')):
                applicable = term.cusp_applicable(k)
                if term.cusp[k] and not applicable:
                    raise ValueError(f'{term.name} is asked to carry the {kind} cusp, but no part of it is a function of one {kind} pair alone')
                if term.cusp[k] is None:
                    has_cusp.append(k == 0 and applicable and not carried[k])
                else:
                    has_cusp.append(term.cusp[k])
                carried[k] = carried[k] or has_cusp[k]
            term.has_cusp = tuple(has_cusp)
            term.channel_cusp[~term.channel_cusp_given] = np.broadcast_to(term.has_cusp, term.channel_cusp.shape)[~term.channel_cusp_given]
            term.constrain(self.atom_charges, applied)

    def write(self, base_path, version):
        """Write the JASTROW block back in the layout casino writes it in."""
        block = casl.Block()
        block.add('Title', self.title)
        if self.print_determined:
            block.add('Print determined', 'T')
        for term in self.terms:
            block.add(term.name, term.write(self.print_determined))
        casl.write(casl.Block([('JASTROW', block)]), base_path, f'parameters.{version}.casl')
