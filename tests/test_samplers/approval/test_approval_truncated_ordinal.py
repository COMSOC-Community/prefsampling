from unittest import TestCase


from prefsampling.approval.truncated_ordinal import truncated_ordinal
from prefsampling.ordinal import mallows, urn
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_approval_truncated_ordinal():
    return [
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


class TestApprovalTruncatedOrdinal(TestCase):
    def test_approval_truncated_ordinal(self):
        with self.assertRaises(ValueError):
            truncated_ordinal(4, 5, -0.5, mallows, {"phi": 0.4})
        with self.assertRaises(ValueError):
            truncated_ordinal(4, 5, 1.5, mallows, {"phi": 0.4})
