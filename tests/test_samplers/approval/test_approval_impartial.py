from unittest import TestCase

import numpy as np

from prefsampling.approval.impartial import impartial, impartial_constant_size
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_approval_impartial():
    def impartial_several_p(num_voters, num_candidates, seed=None):
        rng = np.random.default_rng(seed)
        return impartial(num_voters, num_candidates, rng.random(size=num_voters), seed=seed)

    samplers = [TestSampler(impartial_several_p, {}) for _ in range(3)]

    for random_p in float_parameter_test_values(0, 1, 2):
        samplers.append(TestSampler(impartial, {"p": random_p}))

    def impartial_cst_size_several_p(num_voters, num_candidates, seed=None):
        rng = np.random.default_rng(seed)
        return impartial_constant_size(num_voters, num_candidates, rng.random(size=num_voters), seed=seed)

    samplers = [TestSampler(impartial_cst_size_several_p, {}) for _ in range(3)]

    for random_p in float_parameter_test_values(0, 1, 2):
        samplers.append(TestSampler(impartial, {"p": random_p}))

    for random_rel_num_approvals in float_parameter_test_values(0, 1, 2):
        samplers.append(
            TestSampler(
                impartial_constant_size, {"rel_num_approvals": random_rel_num_approvals}
            ))
    return samplers


class TestApprovalImpartial(TestCase):
    def test_approval_impartial(self):
        with self.assertRaises(ValueError):
            impartial(4, 5, p=-0.5)
        with self.assertRaises(ValueError):
            impartial(4, 5, p=1.5)
        with self.assertRaises(ValueError):
            impartial(4, 5, p=[1, 0.5])
        with self.assertRaises(ValueError):
            impartial(4, 5, p=[1, 0.8, 0.7, 2])

    def test_approval_impartial_constant_size(self):
        with self.assertRaises(ValueError):
            impartial_constant_size(4, 5, rel_num_approvals=-0.5)
        with self.assertRaises(ValueError):
            impartial_constant_size(4, 5, rel_num_approvals=1.3)
        with self.assertRaises(ValueError):
            impartial_constant_size(4, 5, rel_num_approvals=[1, 0.5])
        with self.assertRaises(ValueError):
            impartial_constant_size(4, 5, rel_num_approvals=[1, 0.8, 0.7, 2])

        for _ in range(100):
            votes = impartial_constant_size(50, 50, rel_num_approvals=0.5)
            for vote in votes:
                assert len(vote) == 25


