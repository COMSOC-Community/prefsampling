from unittest import TestCase

import numpy as np

from prefsampling.approval.impartial import impartial, impartial_constant_size
from tests.utils import float_parameter_test_values


def random_app_impartial_samplers():
    samplers = [
        lambda num_voters, num_candidates, seed=None: impartial(
            num_voters, num_candidates, random_p, seed=seed
        )
        for random_p in float_parameter_test_values(0, 1, 4)
    ]
    samplers += [
        lambda num_voters, num_candidates, seed=None: impartial_constant_size(
            num_voters, num_candidates, random_rel_num_approvals, seed=seed
        )
        for random_rel_num_approvals in float_parameter_test_values(0, 1, 4)
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
