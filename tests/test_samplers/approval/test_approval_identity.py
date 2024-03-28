from unittest import TestCase

from prefsampling.approval.identity import identity, empty, full
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_approval_identity():
    samplers = [
        TestSampler(identity, {"rel_num_approvals": random_rel_num_approvals})
        for random_rel_num_approvals in float_parameter_test_values(0, 1, 2)
    ]
    samplers.append(TestSampler(empty, {}))
    samplers.append(TestSampler(full, {}))
    return samplers


class TestApprovalIdentity(TestCase):
    def test_approval_identity(self):
        with self.assertRaises(ValueError):
            identity(4, 5, rel_num_approvals=-0.5)
        with self.assertRaises(ValueError):
            identity(4, 5, rel_num_approvals=1.5)
