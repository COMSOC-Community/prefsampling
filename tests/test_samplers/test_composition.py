from unittest import TestCase

from prefsampling.core import mixture, concatenation
from prefsampling.ordinal import identity, single_crossing, single_peaked_walsh


class TestSamplerComposition(TestCase):
    def test_concatenation_with_shared_arguments(self):
        shared = {"num_voters": 99, "num_candidates": 99}
        votes = concatenation([2, 3], 4, [identity, identity], [shared, shared])
        self.assertEqual(len(votes), 5)
        self.assertTrue(all(len(vote) == 4 for vote in votes))
        self.assertEqual(shared, {"num_voters": 99, "num_candidates": 99})

    def test_sampler_mixture(self):
        with self.assertRaises(ValueError):
            mixture(
                10,
                10,
                [single_crossing],
                [0.5, 0.2],
                [{}, {}],
            )
        with self.assertRaises(ValueError):
            mixture(
                10,
                10,
                [single_crossing, single_peaked_walsh],
                [0.5, 0.2],
                [{}, {}, {}],
            )
        with self.assertRaises(ValueError):
            mixture(
                10,
                10,
                [single_crossing, single_peaked_walsh],
                [0.5, -0.2],
                [{}, {}],
            )
        with self.assertRaises(ValueError):
            mixture(10, 10, [single_crossing, single_peaked_walsh], [0, 0], [{}, {}])

    def test_sampler_concatenation(self):
        with self.assertRaises(ValueError):
            concatenation([10, 13], 10, [single_crossing], [{}])
        with self.assertRaises(ValueError):
            concatenation([10, 13], 10, [single_crossing, single_peaked_walsh], [{}])
        with self.assertRaises(ValueError):
            concatenation(
                [10, 13],
                10,
                [lambda num_voters, num_candidates: 1, single_peaked_walsh],
                [{}, {}],
            )
