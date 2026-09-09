import unittest

import numpy as np
import pytest

from casino.readers.channels import Channels

BE = np.array([4])
O3 = np.array([8, 8, 8])
CO2 = np.array([6, 8, 8])


class TestChannels(unittest.TestCase):
    """The channels of a term: the classes the rules put the particles into.
    CASINO source: gbasis.f90
    """

    def test_no_rules(self):
        """Without rules every pair of spins is its own channel, and a spin with
        one electron has no pair of its own.
        """
        assert Channels((2, 0), [], (2, 2), BE).names == ['1-1', '1-2', '2-2']
        assert Channels((2, 0), [], (5, 2), BE).names == ['1-1', '1-2', '2-2']
        assert Channels((2, 0), [], (2, 1), BE).names == ['1-1', '1-2']
        assert Channels((2, 1), [], (2, 2), BE).names == ['1-1-n1', '1-2-n1', '2-2-n1']

    def test_pair_equality(self):
        """'1-1=2-2' makes the parallel-spin pairs one channel, the antiparallel
        one staying on its own.
        """
        channels = Channels((2, 0), ['1-1=2-2'], (2, 2), BE)
        assert channels.names == ['1-1', '1-2']
        assert channels.ee.tolist() == [[1, 2], [2, 1]]

    def test_particle_equality(self):
        """'1=2' makes the two spins the same particle, which collapses the e-e
        and the e-n pairs at once.
        """
        channels = Channels((2, 1), ['1=2'], (2, 2), BE)
        assert channels.names == ['1-1-n1']
        assert channels.ee.tolist() == [[1, 1], [1, 1]]
        assert channels.en.tolist() == [[1, 1]]
        assert Channels((1, 1), ['1=2'], (2, 2), BE).names == ['1-n1']

    def test_atomic_number(self):
        """'Z' makes every nucleus of one atomic number the same nucleus."""
        assert Channels((1, 1), [], (12, 12), O3).names == ['1-n1', '1-n2', '1-n3', '2-n1', '2-n2', '2-n3']
        assert Channels((1, 1), ['Z'], (12, 12), O3).names == ['1-n1', '2-n1']
        assert Channels((1, 1), ['Z', '1=2'], (12, 12), O3).names == ['1-n1']
        # a nucleus of its own atomic number stays on its own
        assert Channels((1, 1), ['Z', '1=2'], (11, 11), CO2).names == ['1-n1', '1-n2']

    def test_nucleus_equality(self):
        """'n2=n3' makes two named nuclei the same nucleus, whatever their charge."""
        channels = Channels((1, 1), ['n2=n3', '1=2'], (11, 11), CO2)
        assert channels.names == ['1-n1', '1-n2']
        assert channels.en.tolist() == [[1, 1], [2, 2], [2, 2]]

    def test_removal(self):
        """'!1-1' takes a pair out of the term, and every channel it is in with it."""
        channels = Channels((2, 0), ['!1-1'], (2, 2), BE)
        assert channels.names == ['1-2', '2-2']
        assert channels.ee.tolist() == [[-1, 1], [1, 2]]

    def test_removal_spreads(self):
        """Equating a pair with one the rules have removed removes it as well: the
        removed pairs are the group -1, and an equality merges into the lower group.
        """
        assert Channels((2, 0), ['!1-1', '1-1=2-2'], (2, 2), BE).names == ['1-2']
        assert Channels((2, 0), ['1-1=2-2', '!1-1'], (2, 2), BE).names == ['1-2']

    def test_rules_are_ignored_off_target(self):
        """A rule about a pair the term is not a function of is ignored, which is
        what lets one ruleset be written on every term.
        """
        assert Channels((1, 1), ['1-1=2-2', 'Z'], (2, 2), BE).names == ['1-n1', '2-n1']
        assert Channels((2, 0), ['Z', '1=2'], (2, 2), BE).names == ['1-1']

    def test_permutations(self):
        """The permutations of a channel are those that leave its signature as it
        is, and they are what the symmetry of the parameters follows from.
        """
        channels = Channels((2, 1), ['1=2'], (2, 2), BE)
        assert channels.permutations[0] == [((0, 1), (0,)), ((1, 0), (0,))]
        channels = Channels((2, 1), [], (2, 2), BE)
        # the pair of an up and a down electron is not symmetric under their swap
        assert channels.permutations[channels.names.index('1-2-n1')] == [((0, 1), (0,))]

    def test_pair_channels(self):
        """A basis or a cutoff is defined over the pairs, not over the channels."""
        channels = Channels((2, 1), ['Z', '1=2'], (12, 12), O3)
        assert channels.names == ['1-1-n1']
        assert channels.pair_names('e-e') == ['1-1']
        assert channels.pair_names('e-n') == ['1-n1']
        assert channels.index('2-1-n3') == 0
        channels = Channels((2, 1), [], (2, 2), BE)
        assert channels.pair_names('e-e') == ['1-1', '1-2', '2-2']
        assert channels.pair_names('e-n') == ['1-n1', '2-n1']
        assert channels.pair_index('2-n1', 'e-n') == 1

    def test_channel_index(self):
        """A channel name is a set of particles, and any set of the class names it."""
        channels = Channels((2, 0), ['1-1=2-2'], (2, 2), BE)
        assert channels.index('2-2') == channels.index('1-1') == 0
        assert channels.index('2-1') == channels.index('1-2') == 1
        with pytest.raises(ValueError):
            Channels((2, 0), ['!1-1'], (2, 2), BE).index('1-1')
        with pytest.raises(ValueError):
            Channels((2, 0), [], (1, 1), BE).index('1-1')

    def test_bad_rules(self):
        """A rule casino would refuse is refused here too."""
        with pytest.raises(ValueError):
            Channels((1, 1), ['Z1=Z4'], (2, 2), BE)
        with pytest.raises(ValueError):
            Channels((2, 0), ['1-1=2'], (2, 2), BE)
        with pytest.raises(ValueError):
            Channels((1, 1), ['n1'], (2, 2), BE)


if __name__ == '__main__':
    unittest.main()
