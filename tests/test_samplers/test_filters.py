from copy import deepcopy
from unittest import TestCase

from prefsampling.approval import impartial
from prefsampling.core import rename_candidates, coin_flip_ties
from prefsampling.ordinal import single_crossing, mallows


class TestFilters(TestCase):
    def test_rename_candidates(self):
        assert rename_candidates([]) == []

        with self.assertRaises(ValueError):
            rename_candidates({(1, 4): 4, (4, 2): 24})

        votes = single_crossing(10, 10)
        copied_votes = deepcopy(votes)
        rename_candidates(copied_votes)
        assert votes == copied_votes

        votes = impartial(10, 10, 0.4)
        copied_votes = deepcopy(votes)
        rename_candidates(copied_votes, num_candidates=10)
        assert votes == copied_votes

    def test_coin_flip_ties(self):
        num_candidates = 5
        ordinal_votes = mallows(10, num_candidates, 0.3)

        weak_votes = coin_flip_ties(ordinal_votes, 0.4)

        assert all([sum(len(c) for c in v) == num_candidates] for v in weak_votes)

        with self.assertRaises(ValueError):
            coin_flip_ties(ordinal_votes, -0.4)
        with self.assertRaises(ValueError):
            coin_flip_ties(ordinal_votes, 1.4)
