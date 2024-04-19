from unittest import TestCase

import numpy as np

from prefsampling.approval import resampling
from tests.test_samplers.approval.test_approval_euclidean import (
    all_test_samplers_approval_euclidean,
)
from tests.test_samplers.approval.test_approval_identity import (
    all_test_samplers_approval_identity,
)
from tests.test_samplers.approval.test_approval_impartial import (
    all_test_samplers_approval_impartial,
)
from tests.test_samplers.approval.test_approval_noise import (
    all_test_samplers_approval_noise,
)
from tests.test_samplers.approval.test_approval_resampling import (
    all_test_samplers_approval_resampling,
)
from tests.test_samplers.approval.test_approval_truncated_ordinal import (
    all_test_samplers_approval_truncated_ordinal,
)
from tests.test_samplers.approval.test_approval_urn import (
    all_test_samplers_approval_urn,
)
from tests.utils import (
    TestSampler,
    sample_then_permute,
    sample_then_rename,
    sample_then_resample_as_central_vote,
    sample_mixture,
)


def all_test_samplers_approval():
    test_samplers = all_test_samplers_approval_euclidean()
    test_samplers += all_test_samplers_approval_identity()
    test_samplers += all_test_samplers_approval_impartial()
    test_samplers += all_test_samplers_approval_noise()
    test_samplers += all_test_samplers_approval_resampling()
    test_samplers += all_test_samplers_approval_truncated_ordinal()
    test_samplers += all_test_samplers_approval_urn()

    permute_test_samplers = [
        TestSampler(sample_then_permute, {"main_test_sampler": test_sampler})
        for test_sampler in np.random.choice(test_samplers, size=20)
    ]
    rename_test_samplers = [
        TestSampler(sample_then_rename, {"main_test_sampler": test_sampler})
        for test_sampler in np.random.choice(test_samplers, size=20)
    ]
    resample_as_central_vote_test_samplers = [
        TestSampler(
            sample_then_resample_as_central_vote,
            {
                "main_test_sampler": test_sampler,
                "resampler": resampling,
                "resampler_params": {"phi": 0.9684, "rel_size_central_vote": 0.1345},
            },
        )
        for test_sampler in np.random.choice(test_samplers, size=20)
    ]
    mixture_test_samplers = [
        TestSampler(
            sample_mixture,
            {
                "test_sampler_1": sampler1,
                "test_sampler_2": sampler2,
                "test_sampler_3": sampler3,
            },
        )
        for sampler1, sampler2, sampler3 in np.random.choice(
            test_samplers, size=(40, 3)
        )
    ]

    test_samplers += permute_test_samplers
    test_samplers += rename_test_samplers
    test_samplers += resample_as_central_vote_test_samplers
    test_samplers += mixture_test_samplers

    return test_samplers


class TestApprovalSamplers(TestCase):
    def helper_test_approval_sampler(
        self, test_sampler, test_sampler_method, num_voters, num_candidates
    ):
        result = test_sampler.test_sample(
            test_sampler_method, num_voters, num_candidates
        )

        # Test whether the function returns a list of the correct size
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), num_voters)

        # Test whether we have sets
        for vote in result:
            self.assertIsInstance(vote, set)

        # Test whether the values are within the range of candidates
        for vote in result:
            self.assertGreaterEqual(set(range(num_candidates)), vote)

        # Test whether the value are int
        for vote in result:
            for candidate in vote:
                self.assertEqual(int(candidate), candidate)

    def test_all_approval_samplers(self):
        num_voters = 200
        num_candidates = 5

        for test_sampler in all_test_samplers_approval():
            for test_sampler_method in ["positional", "kwargs", "seed"]:
                with self.subTest(
                    sampler=test_sampler, test_sampler_method=test_sampler_method
                ):
                    self.helper_test_approval_sampler(
                        test_sampler, test_sampler_method, num_voters, num_candidates
                    )
