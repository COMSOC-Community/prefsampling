from unittest import TestCase

from prefsampling.approval.urn import urn, urn_constant_size, urn_partylist
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_approval_urn():
    samplers = [
        TestSampler(urn, {"p": random_p, "alpha": random_alpha})
        for random_p in float_parameter_test_values(0, 1, 2)
        for random_alpha in float_parameter_test_values(0, 10, 2)
    ]
    samplers += [
        TestSampler(
            urn_constant_size,
            {"rel_num_approvals": random_rel_num_approvals, "alpha": random_alpha},
        )
        for random_rel_num_approvals in float_parameter_test_values(0, 1, 2)
        for random_alpha in float_parameter_test_values(0, 10, 2)
    ]
    samplers += [
        TestSampler(
            urn_partylist, {"parties": random_num_parties, "alpha": random_alpha}
        )
        for random_num_parties in range(1, 6)
        for random_alpha in float_parameter_test_values(0, 10, 2)
    ]
    return samplers


class TestApprovalUrn(TestCase):
    def test_approval_urn(self):
        with self.assertRaises(ValueError):
            urn(4, 5, p=0.5, alpha=-1)
        with self.assertRaises(ValueError):
            urn(4, 5, p=-0.5, alpha=4)
        with self.assertRaises(ValueError):
            urn(4, 5, p=2, alpha=4)
