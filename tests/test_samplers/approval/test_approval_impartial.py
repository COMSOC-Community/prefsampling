from unittest import TestCase

from prefsampling.approval.impartial import impartial, impartial_constant_size
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_approval_impartial():
    samplers = [
        TestSampler(impartial, {"p": random_p})
        for random_p in float_parameter_test_values(0, 1, 2)
    ]
    samplers += [
        TestSampler(
            impartial_constant_size, {"rel_num_approvals": random_rel_num_approvals}
        )
        for random_rel_num_approvals in float_parameter_test_values(0, 1, 2)
    ]
    return samplers


class TestApprovalImpartial(TestCase):
    def test_approval_impartial(self):
        with self.assertRaises(ValueError):
            impartial(4, 5, p=-0.5)
        with self.assertRaises(ValueError):
            impartial(4, 5, p=1.5)

    def test_approval_impartial_constant_size(self):
        with self.assertRaises(ValueError):
            impartial_constant_size(4, 5, rel_num_approvals=-0.5)
        with self.assertRaises(ValueError):
            impartial_constant_size(4, 5, rel_num_approvals=1.3)

        for _ in range(100):
            votes = impartial_constant_size(50, 50, rel_num_approvals=0.5)
            for vote in votes:
                assert len(vote) == 25
