from copy import deepcopy
from unittest import TestCase

from prefsampling.approval import impartial
from prefsampling.core import rename_candidates
from prefsampling.ordinal import single_crossing


class TestFilters(TestCase):
    def test_rename_candidates(self):
        assert rename_candidates([]) == []

        with self.assertRaises(ValueError):
            rename_candidates({(1, 4): 4, (4, 2): 24})

        votes = single_crossing(10, 10)
        copied_votes = deepcopy(votes)
        rename_candidates(copied_votes)
        assert (votes == copied_votes).all()

        votes = impartial(10, 10, 0.4)
        copied_votes = deepcopy(votes)
        rename_candidates(copied_votes, num_candidates=10)
        assert votes == copied_votes
