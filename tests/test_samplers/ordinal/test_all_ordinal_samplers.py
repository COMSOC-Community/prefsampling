from unittest import TestCase

import numpy as np

from prefsampling.ordinal import mallows
from tests.test_samplers.ordinal.test_ordinal_didi import (
    all_test_samplers_ordinal_didi,
)
from tests.test_samplers.ordinal.test_ordinal_euclidean import (
    all_test_samplers_euclidean,
)
from tests.test_samplers.ordinal.test_ordinal_group_separable import (
    all_test_samplers_ordinal_group_separable,
)
from tests.test_samplers.ordinal.test_ordinal_identity import (
    all_test_samplers_ordinal_identity,
)
from tests.test_samplers.ordinal.test_ordinal_impartial import (
    all_test_samplers_ordinal_impartial,
)
from tests.test_samplers.ordinal.test_ordinal_mallows import (
    all_test_samplers_ordinal_mallows,
)
from tests.test_samplers.ordinal.test_ordinal_plackettluce import (
    all_test_samplers_ordinal_plackett_luce,
)
from tests.test_samplers.ordinal.test_ordinal_single_crossing import (
    all_test_samplers_ordinal_single_crossing,
)
from tests.test_samplers.ordinal.test_ordinal_single_peaked import (
    all_test_samplers_ordinal_single_peaked,
)
from tests.test_samplers.ordinal.test_ordinal_urn import all_test_samplers_ordinal_urn
from tests.utils import (
    TestSampler,
    sample_then_permute,
    sample_then_rename,
    sample_then_resample_as_central_vote,
    sample_mixture,
)


def all_test_samplers_ordinal():
    test_samplers = all_test_samplers_ordinal_didi()
    test_samplers += all_test_samplers_euclidean()
    test_samplers += all_test_samplers_ordinal_group_separable()
    test_samplers += all_test_samplers_ordinal_impartial()
    test_samplers += all_test_samplers_ordinal_mallows()
    test_samplers += all_test_samplers_ordinal_plackett_luce()
    test_samplers += all_test_samplers_ordinal_single_crossing()
    test_samplers += all_test_samplers_ordinal_single_peaked()
    test_samplers += all_test_samplers_ordinal_urn()
    test_samplers += all_test_samplers_ordinal_identity()

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
                "resampler": mallows,
                "resampler_params": {"phi": 0.5},
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


class TestOrdinalSamplers(TestCase):
    def helper_test_ordinal_sampler(
        self, test_sampler, test_sampler_method, num_voters, num_candidates
    ):
        result = test_sampler.test_sample(
            test_sampler_method, num_voters, num_candidates
        )

        # Test if the function returns a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Test if the shape of the returned array is correct
        self.assertEqual(result.shape, (num_voters, num_candidates))

        # Test if the values are within the range of candidates
        for vote in result:
            self.assertEqual(set(vote), set(range(num_candidates)))

        # Test if the value are int
        for vote in result:
            for candidate in vote:
                self.assertEqual(int(candidate), candidate)

    def test_all_ordinal_samplers(self):
        num_voters = 200
        num_candidates = 5

        for test_sampler in all_test_samplers_ordinal():
            for test_sampler_method in ["positional", "kwargs", "seed"]:
                with self.subTest(
                    sampler=test_sampler, test_sampler_method=test_sampler_method
                ):
                    self.helper_test_ordinal_sampler(
                        test_sampler, test_sampler_method, num_voters, num_candidates
                    )
