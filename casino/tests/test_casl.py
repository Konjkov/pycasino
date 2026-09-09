import unittest

import pytest

from casino.readers import casl

TERM = """\
JASTROW:
  Title: 'Standard JASTROW'
  TERM 1:
    Rank: [ 2, 0 ]
    Rules: [ 1-1=2-2 ]
    e-e basis:
      Type: natural power
      Order: 9
    e-e cusp: T
    e-e cutoff:
      Type: polynomial
      Constants:
        C: 3
      Parameters:
        Channel 1-1:
"""

# the line CASINO folds at column 79 keeps the blank it was cut after
DOCUMENT = (
    TERM + '          L: [ 3.4475444445082695, optimizable, limits: [ \n'
    '               1.0000000000000000E-003, +Inf ] ]\n'
    '    Linear parameters:\n'
    '      Channel 1-1:\n'
    '        c_2: [ -0.13990307283061440, optimizable ]\n'
)


class TestCasl(unittest.TestCase):
    """The CASL language itself: what a CASL file means and what it looks like
    when written back. CASINO source: casl.f90
    """

    def test_round_trip(self):
        """A file in the layout CASINO writes comes back byte for byte, the fold
        of the over-long line at column 79 included.
        """
        tree = casl.parse(DOCUMENT.splitlines(keepends=True))
        assert casl.dumps(tree) == DOCUMENT

    def test_scalars_stay_strings(self):
        """CASL has no types of its own: 'T' is not a bool and '+Inf' is not a
        float until something asks for one.
        """
        term = casl.parse(DOCUMENT.splitlines(keepends=True))['JASTROW']['TERM 1']
        assert term['e-e cusp'] == 'T'
        assert casl.to_bool(term['e-e cusp'])
        cutoff = term['e-e cutoff']['Parameters']['Channel 1-1']['L']
        assert casl.to_float(cutoff['%u1']) == pytest.approx(3.4475444445082695)
        assert casl.to_float(cutoff['limits']['%u2']) == float('inf')
        assert casl.to_float('1.0D-3') == pytest.approx(1e-3)

    def test_implicit_items(self):
        """A line that carries no name is an item of its block, and is what YAML
        would instead fold into one multi-line scalar.
        """
        text = 'Constraints:\n  2^g_4,4=2^g_5,5\n  2^g_2,2=3^g_2,2\n'
        block = casl.parse(text.splitlines(keepends=True))['Constraints']
        assert block.implicit() == ['2^g_4,4=2^g_5,5', '2^g_2,2=3^g_2,2']
        assert casl.dumps(casl.parse(text.splitlines(keepends=True))) == text

    def test_inline_separator(self):
        """Items of an inline block are separated by a comma and a space, so a
        comma inside a name does not split it.
        """
        block = casl.parse(['g_3,12: [ 0.5, fixed ]\n'])
        assert list(block) == ['g_3,12']
        assert block['g_3,12'].implicit() == ['0.5', 'fixed']

    def test_continuation_line(self):
        """A deeper-indented line continues the one before it unless that one
        ended at its ':'.
        """
        block = casl.parse(['Rules: [ Z,\n', '   1=2 ]\n'])
        assert block['Rules'].implicit() == ['Z', '1=2']

    def test_comment(self):
        """'#' starts a comment anywhere in a line."""
        block = casl.parse(['# a title\n', 'Title: none # and a comment\n'])
        assert list(block) == ['Title']
        assert block['Title'] == 'none'

    def test_directive(self):
        """'%!' comments out an item and everything under it."""
        block = casl.parse(['TERM 1:\n', '  Rank: [ 2, 0 ]\n', '%! TERM 2:\n', '  Rank: [ 1, 1 ]\n'])
        assert list(block) == ['TERM 1']

    def test_names(self):
        """Item names are matched in lowercase and without spaces, and no two
        items of a block may share one.
        """
        block = casl.parse(['TERM 1:\n', '  Rank: [ 2, 0 ]\n'])
        assert block.item('term1') is block['TERM 1']
        assert block.item('TERM 2') is None
        with pytest.raises(ValueError):
            casl.parse(['Rank: [ 2, 0 ]\n', 'rank: [ 1, 1 ]\n'])

    def test_bad_indentation(self):
        """An item that lines up with no sibling is an error, as it is in CASINO."""
        with pytest.raises(ValueError):
            casl.parse(['TERM 1:\n', '    Rank: [ 2, 0 ]\n', '  Rules: [ ]\n'])


if __name__ == '__main__':
    unittest.main()
