from unittest import TestCase

import numpy as np

from prefsampling.approval.truncated_ordinal import truncated_ordinal
from prefsampling.ordinal import mallows, urn
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_approval_truncated_ordinal():
    samplers = [
        TestSampler(
            truncated_ordinal,
            {
                "rel_num_approvals": random_rel_num_approvals,
                "ordinal_sampler": urn,
                "ordinal_sampler_parameters": {"alpha": random_alpha},
            },
        )
        for random_rel_num_approvals in float_parameter_test_values(0, 1, 2)
        for random_alpha in float_parameter_test_values(0, 10, 2)
    ]

    def truncated_ordinal_several_p(num_voters, num_candidates, seed=None):
        rng = np.random.default_rng(seed)
        return truncated_ordinal(
            num_voters,
            num_candidates,
            rng.random(size=num_voters),
            ordinal_sampler=urn,
            ordinal_sampler_parameters={"alpha": 6},
            seed=seed,
        )

    samplers += [TestSampler(truncated_ordinal_several_p, {}) for _ in range(3)]
    return samplers


class TestApprovalTruncatedOrdinal(TestCase):
    def test_approval_truncated_ordinal(self):
        with self.assertRaises(ValueError):
            truncated_ordinal(4, 5, -0.5, mallows, {"phi": 0.4})
        with self.assertRaises(ValueError):
            truncated_ordinal(4, 5, 1.5, mallows, {"phi": 0.4})
        with self.assertRaises(ValueError):
            truncated_ordinal(4, 5, [0.5, 0.7, 0.8], mallows, {"phi": 0.4})
        with self.assertRaises(ValueError):
            truncated_ordinal(4, 5, [0.5, 0.7, 0.8, 2], mallows, {"phi": 0.4})
        with self.assertRaises(ValueError):
            truncated_ordinal(4, 5, [0.5, 0.7, 0.8, -0.5], mallows, {"phi": 0.4})
