from unittest import TestCase

from prefsampling.ordinal import urn
from prefsampling.ordinal.impartial import impartial_anonymous_culture, impartial_culture
from prefsampling.ordinal.singlecrossing import single_crossing
from prefsampling.ordinal.singlepeaked import single_peaked_Walsh, single_peaked_Conitzer, single_peaked_circle_Conitzer
from prefsampling.approval.resampling import resampling


ALL_SAMPLERS = [
    impartial_culture,
    impartial_anonymous_culture,
    urn,
    single_peaked_Conitzer,
    single_peaked_circle_Conitzer,
    single_peaked_Walsh,
    single_crossing,
    resampling
]


class TestSamplers(TestCase):

    def helper_test_all_samplers(self, sampler, num_voters, num_candidates):
        # Test that all the arguments are there
        sampler(num_voters, num_candidates)
        sampler(num_voters, num_candidates, seed=23)
        sampler(num_voters=num_voters, num_candidates=num_candidates, seed=23)

        # Test that the samplers are decorated to exclude bad number of voters and/or candidates arguments
        with self.assertRaises(ValueError):
            sampler(1, -2)
        with self.assertRaises(ValueError):
            sampler(-2, 1)
        with self.assertRaises(ValueError):
            sampler(-2, -2)
        with self.assertRaises(TypeError):
            sampler(1.5, 2)
        with self.assertRaises(TypeError):
            sampler(1, 2.5)
        with self.assertRaises(TypeError):
            sampler(1.5, 2.5)

    def test_all_samplers(self):
        num_voters = 200
        num_candidates = 5

        for sampler in ALL_SAMPLERS:
            with self.subTest(sampler=sampler):
                self.helper_test_all_samplers(sampler, num_voters, num_candidates)
