import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pytest

from casino import gjastrow
from casino.jastrow import Jastrow
from casino.readers import CasinoConfig, casl
from casino.readers.gjastrow import DETERMINED, OPTIMIZABLE, UNSET, Gjastrow, Term
from casino.slater import Slater
from casino.wfn import Wfn

DOCUMENT = """\
JASTROW:
  Title: 'Standard JASTROW'
  Print determined: T
  TERM 1:
    Rank: [ 2, 0 ]
    Rules: [ 1-1=2-2 ]
    e-e basis:
      Type: natural power
      Order: 3
    e-e cusp: T
    e-e cutoff:
      Type: polynomial
      Constants:
        C: 3
      Parameters:
        Channel 1-1:
          L: [ 5.6411717219379245, optimizable, limits: [ 1.0000000000000000E-003, +Inf ] ]
        Channel 1-2:
          L: [ 8.9665169384808134, optimizable, limits: [ 1.0000000000000000E-003, +Inf ] ]
    Linear parameters:
      Channel 1-1:
        c_2: [ -0.37365483842411884, optimizable ]
        c_3: [ -6.4314909625944189E-002, optimizable ]
      Channel 1-2:
        c_2: [ -3.9305519401293017E-002, optimizable ]
        c_3: [ 7.4883517199534511E-002, optimizable ]
  TERM 2:
    Rank: [ 2, 1 ]
    Rules: [ Z, 1=2 ]
    e-e basis:
      Type: natural power
      Order: 2
    e-n basis:
      Type: r/(r+a) power
      Order: 2
      Constants:
        L: +Inf
      Parameters:
        Channel 1-n1:
          a: [ 2.5000000000000000, fixed, limits: [ 1.0999999999999999E-008, +Inf ] ]
    e-n cutoff:
      Type: alt polynomial
      Constants:
        C: 3
      Parameters:
        Channel 1-n1:
          L: [ 5.5913465706263148, optimizable, limits: [ 1.0000000000000000E-003, +Inf ] ]
    Linear parameters:
      Channel 1-1-n1:
        c_1,1,1: [ 0.11891703142144745, determined ]
        c_1,2,2: [ -0.95727520152348855, optimizable ]
        c_2,2,1: [ 0.99414825112800731, optimizable ]
"""


# two up and two down electrons around one nucleus of atomic number 4
NELE = (2, 2)
ATOM_NUMBERS = np.array([4])
ATOM_CHARGES = np.array([4.0])


def parse(text=DOCUMENT):
    return casl.parse(text.splitlines(keepends=True))['JASTROW']


def read(name, text=DOCUMENT, nele=NELE, atom_numbers=ATOM_NUMBERS):
    term = Term(name)
    term.read(parse(text)[name], nele, atom_numbers)
    return term


class TestGjastrow(unittest.TestCase):
    """The generic Jastrow schema: what a TERM of parameters.casl declares.
    CASINO source: gjastrow.f90, gbasis.f90
    """

    def test_term(self):
        """Ranks, rules, cusp and expansion orders of a term."""
        term = read('TERM 1')
        assert term.rank == (2, 0)
        assert term.rules == ['1-1=2-2']
        assert term.cusp == (True, None)
        assert (term.size_ee, term.size_en) == (1, 0)
        assert term.ee_basis.order == 3
        assert term.en_basis is None
        assert term.en_cutoff is None

    def test_linear_parameters(self):
        """Linear parameters go into a dense array of one index per pair, the
        ones the file leaves out staying unset for a constraint to determine.
        """
        term = read('TERM 1')
        assert term.channels == ['1-1', '1-2']
        assert term.parameters.shape == (2, 3)
        assert term.parameters[0, 1] == pytest.approx(-0.37365483842411884)
        assert term.flags[0, 0] == UNSET
        assert term.flags[0, 1] == OPTIMIZABLE
        term = read('TERM 2')
        assert term.parameters.shape == (1, 2, 2, 2)
        assert term.parameters[0, 0, 1, 1] == pytest.approx(-0.95727520152348855)
        assert term.flags[0, 0, 0, 0] == DETERMINED

    def test_cutoff(self):
        """A cutoff carries its truncation order as a constant and its length as
        a parameter of every channel it is defined in.
        """
        cutoff = read('TERM 1').ee_cutoff
        assert cutoff.constants == {'C': 3}
        assert cutoff.channels == ['1-1', '1-2']
        assert cutoff.parameters[1, 0] == pytest.approx(8.9665169384808134)
        assert cutoff.flags[1, 0] == OPTIMIZABLE
        assert cutoff.limits[1, 0].tolist() == [1e-3, np.inf]

    def test_default_constants(self):
        """A constant the file leaves out takes the default casino gives it."""
        text = DOCUMENT.replace('      Constants:\n        C: 3\n', '')
        assert read('TERM 1', text).ee_cutoff.constants == {'C': 3}
        assert read('TERM 2', text).en_cutoff.constants == {'C': 3}

    def test_default_parameters(self):
        """A basis parameter that is not given takes its default value, and one
        given without a flag is optimizable.
        """
        basis = read('TERM 2').en_basis
        assert basis.names == ['a']
        assert basis.parameters[0, 0] == pytest.approx(2.5)
        text = DOCUMENT.replace('a: [ 2.5000000000000000, fixed,', 'a: [ default, optimizable,')
        basis = read('TERM 2', text).en_basis
        assert basis.parameters[0, 0] == pytest.approx(3.0)
        assert basis.flags[0, 0] == OPTIMIZABLE

    def test_limits(self):
        """A declared limit only narrows the range a parameter is defined in, and
        an infinite one widens it to the whole line.
        """
        text = DOCUMENT.replace('limits: [ 1.0000000000000000E-003, +Inf ]', 'limits: [ 1.0000000000000000E-004, 8.0 ]')
        assert read('TERM 1', text).ee_cutoff.limits[0, 0].tolist() == [1e-3, 8.0]
        text = DOCUMENT.replace('limits: [ 1.0000000000000000E-003, +Inf ]', 'limits: [ none, +Inf ]')
        assert read('TERM 1', text).ee_cutoff.limits[0, 0].tolist() == [-np.inf, np.inf]

    def test_block_form(self):
        """Rank and Rules read the same whether they are written inline or as a
        block of implicit items.
        """
        text = DOCUMENT.replace('    Rank: [ 2, 0 ]\n    Rules: [ 1-1=2-2 ]\n', '    Rank:\n      2\n      0\n    Rules:\n      1-1=2-2\n')
        term = read('TERM 1', text)
        assert term.rank == (2, 0)
        assert term.rules == ['1-1=2-2']

    def test_unsupported(self):
        """A function casino knows but this reader does not is refused by name,
        and so is an expansion that is not indexed the standard way.
        """
        with pytest.raises(NotImplementedError):
            read('TERM 1', DOCUMENT.replace('Type: natural power', 'Type: cosine'))
        with pytest.raises(ValueError):
            read('TERM 1', DOCUMENT.replace('Type: natural power', 'Type: chebyshev'))
        with pytest.raises(NotImplementedError):
            read('TERM 1', DOCUMENT.replace('    Rules: [ 1-1=2-2 ]', '    Indexing:\n      Maximum sum: 3'))

    def test_bad_rank(self):
        """A term of a rank no term can be generated for is an error."""
        with pytest.raises(ValueError):
            read('TERM 1', DOCUMENT.replace('Rank: [ 2, 0 ]', 'Rank: [ 1, 0 ]'))

    @staticmethod
    def round_trip(text):
        """The text a file comes back as once it has been through the schema."""
        with tempfile.TemporaryDirectory() as base_path:
            with open(f'{base_path}/parameters.casl', 'w') as f:
                f.write(text)
            jastrow = Gjastrow(*NELE)
            jastrow.set_atoms(ATOM_NUMBERS, ATOM_CHARGES)
            jastrow.read(base_path)
            assert [term.name for term in jastrow.terms] == ['TERM 1', 'TERM 2']
            jastrow.write(base_path, 0)
            with open(f'{base_path}/parameters.0.casl') as f:
                return f.read()

    def test_write(self):
        """Nothing a file declares is lost on the way through the schema: what is
        written back reads as the same thing and writes out the same way again.
        """
        written = self.round_trip(DOCUMENT)
        assert self.round_trip(written) == written
        assert '-0.37365483842411884, optimizable' in written
        assert '5.6411717219379245, optimizable, limits: [' in written
        assert '2.5000000000000000, fixed' in written

    def test_determined_parameters(self):
        """A determined parameter is regenerated from its constraint on every read,
        so it is only written back when the file asks to see it.
        """
        assert 'determined' in self.round_trip(DOCUMENT)
        assert 'determined' not in self.round_trip(DOCUMENT.replace('  Print determined: T\n', ''))


TERMS = """\
JASTROW:
  TERM 1:
    Rank: [ 2, 0 ]
    Rules: [ 1-1=2-2 ]
    e-e basis:
      Type: natural power
      Order: 3
    e-e cutoff:
      Type: polynomial
      Parameters:
        Channel 1-1:
          L: [ 4.0, optimizable ]
        Channel 1-2:
          L: [ 5.0, optimizable ]
  TERM 2:
    Rank: [ 1, 1 ]
    Rules: [ Z, 1=2 ]
    e-n basis:
      Type: natural power
      Order: 3
    e-n cutoff:
      Type: alt polynomial
      Parameters:
        Channel 1-n1:
          L: [ 6.0, optimizable ]
  TERM 3:
    Rank: [ 2, 1 ]
    Rules: [ Z, 1=2 ]
    e-e basis:
      Type: natural power
      Order: 2
    e-n basis:
      Type: natural power
      Order: 3
    e-n cutoff:
      Type: alt polynomial
      Parameters:
        Channel 1-n1:
          L: [ 6.0, optimizable ]
"""


def configure(text=TERMS, nele=NELE, atom_numbers=ATOM_NUMBERS, atom_charges=ATOM_CHARGES):
    """A config of a jastrow read from a document, as the numba side is given one."""
    with tempfile.TemporaryDirectory() as base_path:
        with open(f'{base_path}/parameters.casl', 'w') as f:
            f.write(text)
        jastrow = Gjastrow(*nele)
        jastrow.set_atoms(atom_numbers, atom_charges)
        jastrow.read(base_path)
    return types.SimpleNamespace(input=types.SimpleNamespace(neu=nele[0], ned=nele[1]), jastrow=jastrow)


class TestGjastrowTerms(unittest.TestCase):
    """The arrays the three computed ranks are evaluated from."""

    def test_channels(self):
        """Every set of particles is looked up by the spin of its electrons and the
        number of its nuclei, a set the term is not a function of giving -1.
        """
        u, chi, f = gjastrow.term_by_rank(configure())
        assert u.channel.tolist() == [[0, 1], [1, 0]]
        assert chi.channel.tolist() == [[0], [0]]
        assert f.channel.tolist() == [[[0], [0]], [[0], [0]]]
        assert f.parameters.shape == (1, 2, 3, 3)

    def test_cutoff_channels(self):
        """The cutoff of a term is defined per group of pairs, so it is looked up by
        one pair of particles rather than by the whole set.
        """
        u, chi, f = gjastrow.term_by_rank(configure())
        assert u.cutoff.tolist() == [4.0, 5.0]
        assert u.cutoff_channel.tolist() == [[0, 1], [1, 0]]
        assert (u.trunc, u.alt) == (3, False)
        assert chi.cutoff_channel.tolist() == [[0], [0]]
        assert (chi.trunc, chi.alt) == (3, True)
        assert f.cutoff_channel.tolist() == [[0], [0]]

    def test_missing_term(self):
        """A rank the file carries no term of contributes no channel at all."""
        _, chi, _ = gjastrow.term_by_rank(configure(TERMS[: TERMS.index('  TERM 2:')] + TERMS[TERMS.index('  TERM 3:') :]))
        assert chi.channel.tolist() == [[-1], [-1]]
        assert chi.parameters.shape == (0, 0)
        assert chi.cutoff.shape == (0,)

    def test_removed_channel(self):
        """A pair the rules take out of a term takes every set that holds it out."""
        text = TERMS.replace('Rules: [ 1-1=2-2 ]', 'Rules: [ !1-1 ]').replace('Channel 1-1:', 'Channel 2-2:', 1)
        u, _, _ = gjastrow.term_by_rank(configure(text))
        assert u.channel.tolist() == [[-1, 0], [0, 1]]
        assert u.cutoff.tolist() == [5.0, 4.0]

    def test_unsupported(self):
        """A term casino computes but this module does not is refused by what makes
        it more than the standard u, chi and f terms.
        """
        with pytest.raises(NotImplementedError):
            gjastrow.check(read('TERM 1', TERMS.replace('Rank: [ 2, 0 ]', 'Rank: [ 3, 0 ]')))
        with pytest.raises(NotImplementedError):
            gjastrow.check(read('TERM 2'))
        with pytest.raises(NotImplementedError):
            gjastrow.check(read('TERM 1', DOCUMENT.replace('Type: polynomial', 'Type: gaussian')))
        assert gjastrow.check(read('TERM 1'))[0] == 'e-e'

    def test_cutoff(self):
        """The two forms of the cutoff and their derivatives, the alternative form
        being the other one times (-L)^C.
        """
        r, L, C, delta = 1.3, 4.0, 3, 1e-4
        for alt in (False, True):
            value, d1, d2 = gjastrow.cutoff(r, L, C, alt)
            plus, minus = gjastrow.cutoff(r + delta, L, C, alt)[0], gjastrow.cutoff(r - delta, L, C, alt)[0]
            assert d1 == pytest.approx((plus - minus) / delta / 2)
            assert d2 == pytest.approx((plus - 2 * value + minus) / delta**2)
            numerical = (gjastrow.cutoff(r, L + delta, C, alt)[0] - gjastrow.cutoff(r, L - delta, C, alt)[0]) / delta / 2
            assert gjastrow.cutoff_d1(r, L, C, alt) == pytest.approx(numerical)
        assert gjastrow.cutoff(r, L, C, True)[0] == pytest.approx((-L) ** C * gjastrow.cutoff(r, L, C, False)[0])

    def test_object(self):
        """The arrays go into the structure with the types it declares."""
        jastrow = gjastrow.Gjastrow(configure())
        assert (jastrow.max_ee_order, jastrow.max_en_order) == (3, 3)


CHANNELS = """\
JASTROW:
  TERM 1:
    Rank: [ 2, 0 ]
    e-e basis:
      Type: natural power
      Order: 3
    e-e cusp: F
    e-e cutoff:
      Type: polynomial
      Parameters:
        Channel 1-1:
          L: [ 4.0, fixed ]
        Channel 1-2:
          L: [ 5.0, fixed ]
    Linear parameters:
      Channel 1-1:
        c_1: [ 0.1, fixed ]
        c_2: [ 0.2, fixed ]
        c_3: [ 0.3, fixed ]
      Channel 1-2:
        c_1: [ 0.4, fixed ]
        c_2: [ 0.5, fixed ]
        c_3: [ 0.6, fixed ]
"""


def vectors(r_e, atom_positions):
    """The relative coordinates a jastrow is a function of."""
    return r_e[:, None, :] - r_e[None, :, :], r_e[None, :, :] - atom_positions[:, None, :]


def pair(parameters, r, L, C=3):
    """One pair of a term of rank (2, 0) cut off by the polynomial."""
    return sum(parameters[k] * r**k for k in range(len(parameters))) * (1 - r / L) ** C


def cutoff_block(kind, C, channel, L):
    return [
        f'    {kind} cutoff:',
        '      Type: alt polynomial',
        '      Constants:',
        f'        C: {C}',
        '      Parameters:',
        f'        Channel {channel}:',
        f'          L: [ {float(L)!r}, fixed ]',
    ]


def coefficient(index, value):
    return '        c_' + ','.join(str(i + 1) for i in index) + f': [ {float(value)!r}, fixed ]'


def standard_document(jastrow):
    """The u, chi and f terms of a correlation.data as a CASL document. They are
    the same functions of the same distances, cut off by the alternative
    polynomial, so the linear parameters carry over as they are. The system has
    one electron of each spin, so only the antiparallel set of every term is used.
    """
    u, chi, f = jastrow.u_parameters, jastrow.chi_parameters[0], jastrow.f_parameters[0]
    lines = ['JASTROW:']
    lines += [
        '  TERM 1:',
        '    Rank: [ 2, 0 ]',
        '    Rules: [ 1-1=2-2 ]',
        '    e-e basis:',
        '      Type: natural power',
        f'      Order: {u.shape[1]}',
    ]
    lines += ['    e-e cusp: T'] + cutoff_block('e-e', jastrow.trunc, '1-2', jastrow.u_cutoff[0]['value'])
    lines += ['    Linear parameters:', '      Channel 1-2:']
    lines += [coefficient((k,), u[-1, k]) for k in range(u.shape[1])]
    lines += [
        '  TERM 2:',
        '    Rank: [ 1, 1 ]',
        '    Rules: [ Z, 1=2 ]',
        '    e-n basis:',
        '      Type: natural power',
        f'      Order: {chi.shape[1]}',
    ]
    lines += ['    e-n cusp: F'] + cutoff_block('e-n', jastrow.trunc, '1-n1', jastrow.chi_cutoff[0]['value'])
    lines += ['    Linear parameters:', '      Channel 1-n1:']
    lines += [coefficient((k,), chi[0, k]) for k in range(chi.shape[1])]
    lines += ['  TERM 3:', '    Rank: [ 2, 1 ]', '    Rules: [ Z, 1=2 ]']
    lines += ['    e-e basis:', '      Type: natural power', f'      Order: {f.shape[1]}']
    lines += ['    e-n basis:', '      Type: natural power', f'      Order: {f.shape[2]}']
    lines += ['    e-e cusp: F', '    e-n cusp: F'] + cutoff_block('e-n', jastrow.trunc, '1-n1', jastrow.f_cutoff[0]['value'])
    lines += ['    Linear parameters:', '      Channel 1-2-n1:']
    lines += [coefficient(index, f[-1][index]) for index in np.ndindex(f.shape[1:])]
    return '\n'.join(lines) + '\n'


DERIVATIVES = """\
JASTROW:
  TERM 1:
    Rank: [ 2, 0 ]
    e-e basis:
      Type: natural power
      Order: 4
    e-e cusp: T
    e-e cutoff:
      Type: polynomial
      Parameters:
        Channel 1-1:
          L: [ 4.0, fixed ]
        Channel 1-2:
          L: [ 5.0, fixed ]
    Linear parameters:
      Channel 1-1:
        c_2: [ 0.2, fixed ]
        c_3: [ -0.1, fixed ]
        c_4: [ 0.05, fixed ]
      Channel 1-2:
        c_2: [ 0.3, fixed ]
        c_3: [ 0.15, fixed ]
        c_4: [ -0.02, fixed ]
  TERM 2:
    Rank: [ 1, 1 ]
    Rules: [ Z ]
    e-n basis:
      Type: natural power
      Order: 4
    e-n cusp: F
    e-n cutoff:
      Type: alt polynomial
      Parameters:
        Channel 1-n1:
          L: [ 4.5, fixed ]
        Channel 2-n1:
          L: [ 5.5, fixed ]
    Linear parameters:
      Channel 1-n1:
        c_2: [ 0.4, fixed ]
        c_3: [ -0.2, fixed ]
        c_4: [ 0.1, fixed ]
      Channel 2-n1:
        c_2: [ -0.3, fixed ]
        c_3: [ 0.25, fixed ]
        c_4: [ 0.05, fixed ]
  TERM 3:
    Rank: [ 2, 1 ]
    Rules: [ Z ]
    e-e basis:
      Type: natural power
      Order: 2
    e-n basis:
      Type: natural power
      Order: 3
    e-e cusp: F
    e-n cusp: F
    e-n cutoff:
      Type: polynomial
      Parameters:
        Channel 1-n1:
          L: [ 4.0, fixed ]
        Channel 2-n1:
          L: [ 4.5, fixed ]
    Linear parameters:
      Channel 1-1-n1:
        c_2,2,2: [ 0.3, fixed ]
        c_2,3,3: [ -0.1, fixed ]
      Channel 1-2-n1:
        c_2,2,2: [ 0.2, fixed ]
        c_2,3,2: [ -0.15, fixed ]
        c_2,2,3: [ 0.25, fixed ]
"""


class TestGjastrowValue(unittest.TestCase):
    """What the three computed ranks evaluate to: value, gradient and laplacian."""

    def test_channel_value(self):
        """Every pair is cut off at the length of its own channel: two electrons
        of one spin are inside theirs while the third is outside every one it is
        in, and then all three are inside.
        """
        config = configure(CHANNELS, nele=(2, 1))
        parameters = config.jastrow.terms[0].parameters
        generic = gjastrow.Gjastrow(config)
        r_e = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 10.0]])
        e_vectors, n_vectors = vectors(r_e, np.zeros(shape=(1, 3)))
        assert generic.value(e_vectors, n_vectors) == pytest.approx(pair(parameters[0], 1.0, 4.0))
        r_e[2, 2] = 2.0
        e_vectors, n_vectors = vectors(r_e, np.zeros(shape=(1, 3)))
        expected = pair(parameters[0], 1.0, 4.0) + pair(parameters[1], 2.0, 5.0) + pair(parameters[1], 1.0, 5.0)
        assert generic.value(e_vectors, n_vectors) == pytest.approx(expected)

    def test_removed_channel_value(self):
        """A pair the rules take out of a term contributes nothing to it."""
        text = CHANNELS.replace('  TERM 1:\n', '  TERM 1:\n    Rules: [ !1-1 ]\n')
        text = text.replace('        Channel 1-1:\n          L: [ 4.0, fixed ]\n', '')
        text = text.replace('      Channel 1-1:\n        c_1: [ 0.1, fixed ]\n        c_2: [ 0.2, fixed ]\n        c_3: [ 0.3, fixed ]\n', '')
        config = configure(text, nele=(2, 1))
        parameters = config.jastrow.terms[0].parameters
        generic = gjastrow.Gjastrow(config)
        r_e = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        e_vectors, n_vectors = vectors(r_e, np.zeros(shape=(1, 3)))
        expected = pair(parameters[0], 2.0, 5.0) + pair(parameters[0], 1.0, 5.0)
        assert generic.value(e_vectors, n_vectors) == pytest.approx(expected)

    def test_against_standard_jastrow(self):
        """A generic jastrow written out of the u, chi and f terms of a
        correlation.data is the same function of the electron positions, term by
        term, the constraints of the terms determining the same coefficients.
        """
        config = CasinoConfig(Path(__file__).resolve().parent / 'inputs/Jastrow/He')
        config.read()
        standard = Jastrow(config)
        generic_config = configure(
            standard_document(config.jastrow), (config.input.neu, config.input.ned), config.wfn.atom_numbers, config.wfn.atom_charges
        )
        u, chi, f = (term.parameters[0] for term in generic_config.jastrow.terms)
        assert u == pytest.approx(config.jastrow.u_parameters[-1])
        assert chi == pytest.approx(config.jastrow.chi_parameters[0][0])
        assert f == pytest.approx(config.jastrow.f_parameters[0][-1])
        generic = gjastrow.Gjastrow(generic_config)
        np.random.seed(1)
        for _ in range(5):
            r_e = np.random.uniform(-2, 2, (config.input.neu + config.input.ned, 3))
            e_vectors, n_vectors = vectors(r_e, config.wfn.atom_positions)
            assert generic.value(e_vectors, n_vectors) == pytest.approx(standard.value(e_vectors, n_vectors))
            assert generic.gradient(e_vectors, n_vectors) == pytest.approx(standard.gradient(e_vectors, n_vectors))
            assert generic.laplacian(e_vectors, n_vectors)[0] == pytest.approx(standard.laplacian(e_vectors, n_vectors)[0])


class TestGjastrowDerivatives(unittest.TestCase):
    """The gradient and the laplacian of a jastrow of all three ranks, whose
    channels differ in their cutoff lengths and whose rank (2, 1) term is not
    symmetric in its two electron-nucleus indices.
    """

    def setUp(self):
        self.atom_positions = np.array([[0.0, 0.0, -1.2], [0.0, 0.0, 1.2]])
        config = configure(DERIVATIVES, nele=(2, 1), atom_numbers=np.array([4, 4]), atom_charges=np.array([4.0, 4.0]))
        self.jastrow = gjastrow.Gjastrow(config)
        np.random.seed(1)
        self.r_e = self.atom_positions[np.random.choice(2, 3)] + np.random.uniform(-1, 1, (3, 3))
        self.e_vectors, self.n_vectors = vectors(self.r_e, self.atom_positions)

    def test_gradient(self):
        analytical = self.jastrow.gradient(self.e_vectors, self.n_vectors)
        assert np.abs(analytical).max() > 0
        assert analytical == pytest.approx(self.jastrow.numerical_gradient(self.e_vectors, self.n_vectors))

    def test_laplacian(self):
        analytical = self.jastrow.laplacian(self.e_vectors, self.n_vectors)[0]
        assert analytical != 0
        assert analytical == pytest.approx(self.jastrow.numerical_laplacian(self.e_vectors, self.n_vectors), rel=1e-5)

    def test_value_1e(self):
        """A single-electron move changes the jastrow by exactly what the part of
        it that contains that electron changes by.
        """
        n_powers = self.jastrow.en_powers(self.n_vectors)
        value = self.jastrow.value(self.e_vectors, self.n_vectors)
        for e in range(self.r_e.shape[0]):
            moved = self.r_e.copy()
            moved[e] += np.array([0.1, -0.2, 0.15])
            before = self.jastrow.value_1e(self.r_e[e] - self.r_e, n_powers, e)
            self.jastrow.update_en_powers_1e(n_powers, moved[e] - self.atom_positions, e)
            after = self.jastrow.value_1e(moved[e] - moved, n_powers, e)
            self.jastrow.update_en_powers_1e(n_powers, self.r_e[e] - self.atom_positions, e)
            assert after - before == pytest.approx(self.jastrow.value(*vectors(moved, self.atom_positions)) - value)

    def test_gradient_1e(self):
        """The gradient w.r.t. one electron is the row of the whole gradient."""
        n_powers = self.jastrow.en_powers(self.n_vectors)
        gradient = self.jastrow.gradient(self.e_vectors, self.n_vectors).reshape(-1, 3)
        for e in range(self.r_e.shape[0]):
            analytical = self.jastrow.gradient_1e(self.r_e[e] - self.r_e, self.n_vectors[:, e], n_powers, e)
            assert analytical == pytest.approx(gradient[e])


# the same terms with the coefficients and the cutoff lengths of the first two
# declared optimizable, so that every rank has parameters to move and one of them
# keeps its own fixed
HEAD, TAIL = DERIVATIVES.split('  TERM 3:')
PARAMETERS = ''.join(
    line.replace('fixed', 'optimizable') if line.strip().startswith(('c_', 'L:')) else line for line in HEAD.splitlines(keepends=True)
)
PARAMETERS += '  TERM 3:' + TAIL


class TestGjastrowParameters(unittest.TestCase):
    """Optimization of the linear parameters: the coefficients the file declares
    optimizable move, the ones it fixes stay, and the ones the constraints
    determine follow the first through the map the constraints leave.
    """

    def setUp(self):
        self.atom_positions = np.array([[0.0, 0.0, -1.2], [0.0, 0.0, 1.2]])
        self.config = configure(PARAMETERS, nele=(2, 1), atom_numbers=np.array([4, 4]), atom_charges=np.array([4.0, 4.0]))
        self.jastrow = gjastrow.Gjastrow(self.config)
        np.random.seed(1)
        self.r_e = self.atom_positions[np.random.choice(2, 3)] + np.random.uniform(-1, 1, (3, 3))
        self.e_vectors, self.n_vectors = vectors(self.r_e, self.atom_positions)

    def test_parameters(self):
        """A parameter the file gives a value to is fixed, the rest are optimizable
        but for the coefficients the constraints determine.
        """
        parameters = self.jastrow.get_parameters(False)
        cutoffs = sum(gjastrow.cutoff_function(term)[1].flags[:, 0].tolist().count(OPTIMIZABLE) for term in self.config.jastrow.terms)
        assert parameters.size == cutoffs + sum((term.flags == OPTIMIZABLE).sum() for term in self.config.jastrow.terms)
        assert parameters.size < self.jastrow.get_parameters(True).size

    def test_set_parameters(self):
        """Setting the parameters back leaves every parameter as it was."""
        u, chi, f = (self.jastrow.u_parameters.copy(), self.jastrow.chi_parameters.copy(), self.jastrow.f_parameters.copy())
        self.jastrow.set_parameters(self.jastrow.get_parameters(False), False)
        assert self.jastrow.u_parameters == pytest.approx(u)
        assert self.jastrow.chi_parameters == pytest.approx(chi)
        assert self.jastrow.f_parameters == pytest.approx(f)

    def test_cusp_is_kept(self):
        """Moving the free parameters of the term of rank (2, 0) leaves it with the
        Kato cusp of every channel, which is what its determined coefficient carries,
        and the cutoff length it is moved to is the one that coefficient is solved at.
        """
        term = self.config.jastrow.terms[0]
        parameters = self.jastrow.get_parameters(False)
        self.jastrow.set_parameters(parameters + np.random.uniform(-0.1, 0.1, parameters.size), False)
        assert term.ee_cutoff.parameters[:, 0].tolist() != [4.0, 5.0]
        for channel, gamma in enumerate((0.25, 0.5)):
            L = term.ee_cutoff.parameters[channel, 0]
            assert term.parameters[channel, 1] - term.ee_cutoff.constants['C'] / L * term.parameters[channel, 0] == pytest.approx(gamma)

    def test_value_parameters_d1(self):
        analytical = self.jastrow.value_parameters_d1(self.e_vectors, self.n_vectors)
        assert np.abs(analytical).max() > 0
        assert analytical == pytest.approx(self.jastrow.value_parameters_numerical_d1(self.e_vectors, self.n_vectors, False))

    def test_gradient_parameters_d1(self):
        analytical = self.jastrow.gradient_parameters_d1(self.e_vectors, self.n_vectors)
        assert np.abs(analytical).max() > 0
        assert analytical == pytest.approx(self.jastrow.gradient_parameters_numerical_d1(self.e_vectors, self.n_vectors, False))

    def test_laplacian_parameters_d1(self):
        analytical = self.jastrow.laplacian_parameters_d1(self.e_vectors, self.n_vectors)
        assert np.abs(analytical).max() > 0
        assert analytical == pytest.approx(self.jastrow.laplacian_parameters_numerical_d1(self.e_vectors, self.n_vectors, False), rel=1e-5)


class TestGjastrowWfn(unittest.TestCase):
    """A wave function whose jastrow is the generic one, which reaches the optimizer
    through the interface the standard jastrow reaches it by.
    """

    def setUp(self):
        np.random.seed(1)
        config = CasinoConfig(Path(__file__).resolve().parent / 'inputs/Jastrow/He')
        config.read()
        document = ''.join(
            line.replace('fixed', 'optimizable') if line.strip().startswith('c_') else line
            for line in standard_document(config.jastrow).splitlines(keepends=True)
        )
        generic = configure(document, (config.input.neu, config.input.ned), config.wfn.atom_numbers, config.wfn.atom_charges)
        self.wfn = Wfn(config, Slater(config, cusp=None), jastrow=gjastrow.Gjastrow(generic))
        self.wfn.opt_jastrow = True
        self.wfn.set_parameters_projector()
        self.r_e = np.random.uniform(-1, 1, (config.input.neu + config.input.ned, 3))

    def test_value_parameters_d1(self):
        analytical = self.wfn.value_parameters_d1(self.r_e)
        assert np.abs(analytical).max() > 0
        assert analytical == pytest.approx(self.wfn.value_parameters_numerical_d1(self.r_e), rel=1e-4)

    def test_energy_parameters_d1(self):
        analytical = self.wfn.energy_parameters_d1(self.r_e)
        assert np.abs(analytical).max() > 0
        assert analytical == pytest.approx(self.wfn.energy_parameters_numerical_d1(self.r_e), rel=1e-4)


if __name__ == '__main__':
    unittest.main()
