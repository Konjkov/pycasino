import unittest

import numpy as np
import pytest

from casino.readers import casl
from casino.readers.gjastrow import DETERMINED, OPTIMIZABLE, Gjastrow, Term

# two up and two down electrons around one nucleus of atomic number and charge 4
NELE = (2, 2)
ATOM_NUMBERS = np.array([4])
ATOM_CHARGES = np.array([4.0])

L_EE = 5.6411717219379245
L_EN = 5.8072488660115456
C_2 = -0.37365483842411884

TERMS = f"""\
JASTROW:
  TERM 1:
    Rank: [ 2, 0 ]
    Rules: [ 1-1=2-2 ]
    e-e basis:
      Type: natural power
      Order: 4
    e-e cusp: T
    e-e cutoff:
      Type: polynomial
      Constants:
        C: 3
      Parameters:
        Channel 1-1:
          L: [ {L_EE}, optimizable ]
        Channel 1-2:
          L: [ {L_EE}, optimizable ]
    Linear parameters:
      Channel 1-1:
        c_2: [ {C_2}, optimizable ]
      Channel 1-2:
        c_2: [ {C_2}, optimizable ]
  TERM 2:
    Rank: [ 1, 1 ]
    Rules: [ Z, 1=2 ]
    e-n basis:
      Type: natural power
      Order: 4
    e-n cutoff:
      Type: alt polynomial
      Constants:
        C: 3
      Parameters:
        Channel 1-n1:
          L: [ {L_EN}, optimizable ]
    Linear parameters:
      Channel 1-n1:
        c_2: [ 1.0442936019193270, optimizable ]
  TERM 3:
    Rank: [ 2, 1 ]
    Rules: [ Z, 1=2 ]
    e-e basis:
      Type: natural power
      Order: 5
    e-n basis:
      Type: natural power
      Order: 5
    e-n cutoff:
      Type: alt polynomial
      Constants:
        C: 3
      Parameters:
        Channel 1-n1:
          L: [ {L_EN}, optimizable ]
    Linear parameters:
      Channel 1-1-n1:
        c_1,2,2: [ 0.11891703142144745, optimizable ]
"""


def read(text=TERMS, nele=NELE):
    """The terms of a document, constrained the way the reader constrains them."""
    jastrow = Gjastrow(*nele)
    jastrow.set_atoms(ATOM_NUMBERS, ATOM_CHARGES)
    block = casl.parse(text.splitlines(keepends=True))['JASTROW']
    for name, item in block.items():
        if casl.unique(name).startswith('term'):
            term = Term(name)
            term.read(item, nele, ATOM_NUMBERS)
            jastrow.terms.append(term)
    jastrow.constrain()
    return jastrow.terms


class TestConstraints(unittest.TestCase):
    """What the symmetry and the cusp conditions determine.
    CASINO source: gjastrow.f90, init_symm_constraints and init_value_constraints
    """

    def test_ee_cusp(self):
        """The first coefficient of an e-e term is what the Kato cusp of the pair
        makes it: with the polynomial cutoff, -C/L c_1 + c_2 = Gamma.
        """
        term = read()[0]
        for i, gamma in enumerate((0.25, 0.5)):
            assert term.flags[i, 0] == DETERMINED
            assert term.flags[i, 1] == OPTIMIZABLE
            assert term.parameters[i, 0] == pytest.approx((C_2 - gamma) * L_EE / 3)

    def test_no_cusp(self):
        """A term that does not carry the cusp of a pair must not introduce one, so
        the same equation holds with a right-hand side of zero.
        """
        term = read(TERMS.replace('    e-e cusp: T\n', '    e-e cusp: F\n'))[0]
        assert term.parameters[0, 0] == pytest.approx(C_2 * L_EE / 3)

    def test_en_no_cusp(self):
        """The electron-nucleus cusp is left to the orbitals unless the term asks
        for it, so an e-n term is pinned to no cusp at the nucleus: with the alt
        polynomial cutoff, C L^(C-1) c_1 - L^C c_2 = 0.
        """
        term = read()[1]
        assert term.has_cusp == (False, False)
        assert term.parameters[0, 0] == pytest.approx(1.0442936019193270 * L_EN / 3)

    def test_en_cusp(self):
        """Asked for, the electron-nucleus cusp is minus the charge of the nucleus."""
        term = read(TERMS.replace('    Rules: [ Z, 1=2 ]\n    e-n basis', '    Rules: [ Z, 1=2 ]\n    e-n cusp: T\n    e-n basis', 1))[1]
        assert term.has_cusp == (False, True)
        value = (1.0442936019193270 * L_EN**3 - ATOM_CHARGES[0]) / (3 * L_EN**2)
        assert term.parameters[0, 0] == pytest.approx(value)

    def test_cusp_goes_to_the_first_term_that_can_carry_it(self):
        """A term that says nothing about the e-e cusp carries it if it can and no
        earlier term does, and a term that cannot carry one may not ask for it.
        """
        terms = read(TERMS.replace('    e-e cusp: T\n', ''))
        assert [term.has_cusp for term in terms] == [(True, False), (False, False), (False, False)]
        with pytest.raises(ValueError):
            read(TERMS.replace('    e-e basis:\n      Type: natural power\n      Order: 5\n', '    e-e cusp: T\n', 1))

    def test_symmetry(self):
        """Coefficients a relabelling of identical particles carries onto one
        another are equal, which for a rank (2, 1) term of one channel is the swap
        of the two electron-nucleus indices.
        """
        term = read()[2]
        assert term.groups.permutations[0] == [((0, 1), (0,)), ((1, 0), (0,))]
        assert np.allclose(term.parameters[0], np.swapaxes(term.parameters[0], 1, 2))
        assert (term.equations[0]['symmetry'], term.equations[0]['coalescence']) == (50, 27)
        assert int((term.flags[0] == DETERMINED).sum()) == 68

    def test_open_shell(self):
        """Where the two spins are not equated the channels are separate, and the
        one of two unlike electrons has no symmetry of its own.
        """
        terms = read(TERMS.replace('Rules: [ Z, 1=2 ]', 'Rules: [ Z ]'), nele=(5, 2))
        term = terms[2]
        assert term.channels == ['1-1-n1', '1-2-n1', '2-2-n1']
        counts = [(equations['symmetry'], equations['coalescence']) for equations in term.equations]
        assert counts == [(50, 27), (0, 27), (50, 27)]

    def test_products(self):
        """Coefficients whose basis functions multiply out to the same function of
        one distance go into one equation, which for a power basis makes the number
        of e-e equations of a rank (2, 1) term the number of sums of two indices.
        """
        term = read()[2]
        assert term.orders == (5, 5)
        assert term.equations[0]['coalescence'] == 9 + 9 + 9


if __name__ == '__main__':
    unittest.main()
