from unittest import TestCase

import numpy as np

from prefsampling.approval import resampling
from prefsampling.core import (
    resample_as_central_vote,
    rename_candidates,
    permute_voters,
    mixture,
)

from tests.test_samplers.approval.test_approval_euclidean import (
    random_app_euclidean_samplers,
)
from tests.test_samplers.approval.test_approval_identity import (
    random_app_identity_samplers,
)
from tests.test_samplers.approval.test_approval_impartial import (
    random_app_impartial_samplers,
)
from tests.test_samplers.approval.test_approval_noise import random_app_noise_samplers
from tests.test_samplers.approval.test_approval_resampling import (
    random_app_resampling_samplers,
)
from tests.test_samplers.approval.test_approval_truncated_ordinal import (
    random_app_truncated_ordinal_samplers,
)
from tests.test_samplers.approval.test_approval_urn import random_app_urn_samplers


def random_app_samplers():
    samplers = []
    samplers += random_app_euclidean_samplers()
    samplers += random_app_identity_samplers()
    samplers += random_app_impartial_samplers()
    samplers += random_app_noise_samplers()
    samplers += random_app_resampling_samplers()
    samplers += random_app_urn_samplers()
    samplers += random_app_truncated_ordinal_samplers()

    samplers_permute = [
        lambda num_voters, num_candidates, seed=None: permute_voters(
            sampler(num_voters, num_candidates, seed)
        )
        for sampler in np.random.choice(samplers, size=200)
    ]
    samplers_rename_candidates = [
        lambda num_voters, num_candidates, seed=None: rename_candidates(
            sampler(num_voters, num_candidates, seed)
        )
        for sampler in np.random.choice(samplers, size=200)
    ]
    sampler_resample_as_central_vote = [
        lambda num_voters, num_candidates, seed=None: resample_as_central_vote(
            sampler(num_voters, num_candidates, seed),
            resampling,
            {"phi": 0.4, "p": 0.523, "seed": seed, "num_candidates": num_candidates},
        )
        for sampler in np.random.choice(samplers, size=200)
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
            samplers, size=(200, 3)
        )
    ]

    samplers += samplers_permute
    samplers += samplers_rename_candidates
    samplers += sampler_resample_as_central_vote
    samplers += samplers_mixture
    return samplers


class TestApprovalSamplers(TestCase):
    def helper_test_all_approval_samplers(self, sampler, num_voters, num_candidates):
        result = sampler(num_voters, num_candidates)

        # Test whether the function returns a list of the correct size
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) == num_voters)

        # Test whether we have sets
        for vote in result:
            self.assertIsInstance(vote, set)

        # Test whether the values are within the range of candidates
        for vote in result:
            self.assertTrue(vote <= set(range(num_candidates)))

        # Test whether the value are int
        for vote in result:
            for candidate in vote:
                self.assertTrue(int(candidate) == candidate)

    def test_all_approval_samplers(self):
        num_voters = 200
        num_candidates = 5

        for sampler in random_app_samplers():
            for test_sampler in [
                sampler,
                lambda x, y: sampler(num_voters=x, num_candidates=y),
                lambda x, y: sampler(x, y, seed=363),
            ]:
                with self.subTest(sampler=test_sampler):
                    self.helper_test_all_approval_samplers(
                        test_sampler, num_voters, num_candidates
                    )
