import tempfile
import unittest

import numpy as np
import pytest

from casino.readers import casl
from casino.readers.gjastrow import DETERMINED, OPTIMIZABLE, UNSET, Gjastrow, Term

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


if __name__ == '__main__':
    unittest.main()
