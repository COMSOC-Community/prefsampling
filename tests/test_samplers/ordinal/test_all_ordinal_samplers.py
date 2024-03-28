import numpy as np

from unittest import TestCase

from prefsampling.core import (
    resample_as_central_vote,
    rename_candidates,
    permute_voters,
    mixture,
)

from prefsampling.ordinal import (
    norm_mallows,
)
from tests.test_samplers.ordinal.test_ordinal_didi import random_ord_didi_samplers
from tests.test_samplers.ordinal.test_ordinal_euclidean import random_ord_euclidean_samplers
from tests.test_samplers.ordinal.test_ordinal_group_separable import \
    random_ord_group_separable_samplers
from tests.test_samplers.ordinal.test_ordinal_impartial import random_ord_impartial_samplers
from tests.test_samplers.ordinal.test_ordinal_mallows import random_ord_mallows_samplers
from tests.test_samplers.ordinal.test_ordinal_plackettluce import random_ord_plackett_luce_samplers
from tests.test_samplers.ordinal.test_ordinal_single_crossing import \
    random_ord_single_crossing_samplers
from tests.test_samplers.ordinal.test_ordinal_single_peaked import random_ord_single_peaked_samplers
from tests.test_samplers.ordinal.test_ordinal_urn import random_ord_urn_samplers


def random_ord_samplers():
    samplers = random_ord_didi_samplers()
    samplers += random_ord_euclidean_samplers()
    samplers += random_ord_group_separable_samplers()
    samplers += random_ord_impartial_samplers()
    samplers += random_ord_mallows_samplers()
    samplers += random_ord_plackett_luce_samplers()
    samplers += random_ord_single_crossing_samplers()
    samplers += random_ord_single_peaked_samplers()
    samplers += random_ord_urn_samplers()

    samplers_permute = [
        lambda num_voters, num_candidates, seed=None: permute_voters(
            sampler(num_voters, num_candidates, seed)
        )
        for sampler in samplers
    ]
    samplers_rename_candidates = [
        lambda num_voters, num_candidates, seed=None: rename_candidates(
            sampler(num_voters, num_candidates, seed)
        )
        for sampler in samplers
    ]
    sampler_resample_as_central_vote = [
        lambda num_voters, num_candidates, seed=None: resample_as_central_vote(
            sampler(num_voters, num_candidates, seed),
            norm_mallows,
            {"norm_phi": 0.4, "seed": seed, "num_candidates": num_candidates},
        )
        for sampler in samplers
    ]
    samplers_mixture = [
        lambda num_voters, num_candidates, seed=None: mixture(
            num_voters,
            num_candidates,
            [sampler1, sampler2, sampler3],
            [0.5, 0.2, 0.3],
            [{}, {}, {}],
        )
        for sampler1, sampler2, sampler3 in np.random.choice(
            samplers, size=(1000, 3)
        )
    ]

    samplers += samplers_permute
    samplers += samplers_rename_candidates
    samplers += sampler_resample_as_central_vote
    samplers += samplers_mixture
    return samplers


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

        for sampler in random_ord_samplers():
            for test_sampler in [
                sampler,
                lambda x, y: sampler(num_voters=x, num_candidates=y),
                lambda x, y: sampler(x, y, seed=363),
            ]:
                with self.subTest(sampler=test_sampler):
                    self.helper_test_all_ordinal_samplers(
                        test_sampler, num_voters, num_candidates
                    )
