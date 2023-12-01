import numpy as np

from unittest import TestCase

from prefsampling.core.euclidean import EuclideanSpace
from prefsampling.ordinal import plackett_luce
from prefsampling.ordinal.urn import urn
from prefsampling.ordinal.impartial import (
    impartial_anonymous,
    impartial,
    stratification,
)
from prefsampling.ordinal.singlecrossing import single_crossing
from prefsampling.ordinal.singlepeaked import (
    single_peaked_walsh,
    single_peaked_conitzer,
    single_peaked_circle,
)
from prefsampling.ordinal.mallows import mallows, norm_mallows
from prefsampling.ordinal.euclidean import euclidean

ALL_ORDINAL_SAMPLERS = [
    impartial,
    impartial_anonymous,
    stratification,
    urn,
    single_peaked_conitzer,
    single_peaked_circle,
    single_peaked_walsh,
    single_crossing,
    mallows,
    norm_mallows,
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, space=EuclideanSpace.UNIFORM, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, space=EuclideanSpace.GAUSSIAN, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: euclidean(
        num_voters, num_candidates, space=EuclideanSpace.SPHERE, seed=seed
    ),
    lambda num_voters, num_candidates, seed=None: plackett_luce(
        num_voters, num_candidates, [1] * num_candidates, seed
    ),
]


class TestOrdinalSamplers(TestCase):
    def helper_test_all_ordinal_samplers(self, sampler, num_voters, num_candidates):
        result = sampler(num_voters, num_candidates)

        # Test if the function returns a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Test if the shape of the returned array is correct
        self.assertEqual(result.shape, (num_voters, num_candidates))

        # Test if the values are within the range of candidates
        for vote in result:
            self.assertTrue(set(vote) == set(range(num_candidates)))

        # Test if the value are int
        for vote in result:
            for candidate in vote:
                self.assertTrue(int(candidate) == candidate)

    def test_all_ordinal_samplers(self):
        num_voters = 200
        num_candidates = 5

        for sampler in ALL_ORDINAL_SAMPLERS:
            for test_sampler in [
                sampler,
                lambda x, y: sampler(num_voters=x, num_candidates=y),
                lambda x, y: sampler(x, y, seed=363),
            ]:
                with self.subTest(sampler=test_sampler):
                    self.helper_test_all_ordinal_samplers(
                        test_sampler, num_voters, num_candidates
                    )
