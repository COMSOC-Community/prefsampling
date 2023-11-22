import numpy as np

from unittest import TestCase

from prefsampling.ordinal import urn
from prefsampling.ordinal.impartial import impartial_anonymous_culture, impartial_culture
from prefsampling.ordinal.singlecrossing import single_crossing
from prefsampling.ordinal.singlepeaked import single_peaked_Walsh, single_peaked_Conitzer, single_peaked_circle_Conitzer


ALL_ORDINAL_SAMPLERS = [
    impartial_culture,
    impartial_anonymous_culture,
    urn,
    single_peaked_Conitzer,
    single_peaked_circle_Conitzer,
    single_peaked_Walsh,
    single_crossing
]


class TestOrdinalSamplers(TestCase):

    def helper_test_all_ordinal_samplers(self, sampler, num_agents, num_candidates):
        result = sampler(num_agents, num_candidates)

        # Test if the function returns a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Test if the shape of the returned array is correct
        self.assertEqual(result.shape, (num_agents, num_candidates))

        # Test if the values are within the range of candidates
        for vote in result:
            self.assertTrue(set(vote) == set(range(num_candidates)))

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

    def test_all_ordinal_samplers(self):
        num_agents = 200
        num_candidates = 5

        for sampler in ALL_ORDINAL_SAMPLERS:
            with self.subTest(sampler=sampler):
                self.helper_test_all_ordinal_samplers(sampler, num_agents, num_candidates)
