from unittest import TestCase

from prefsampling.approval.identity import identity, empty, full
from tests.utils import float_parameter_test_values


def random_app_identity_samplers():
    samplers = [
        lambda num_voters, num_candidates, seed=None: identity(
            num_voters, num_candidates, random_p, seed=seed
        )
        for random_p in float_parameter_test_values(0, 1, 2)
    ]
    samplers += [empty, full]
    return samplers


class TestApprovalIdentity(TestCase):
    def test_approval_identity(self):
        with self.assertRaises(ValueError):
            identity(4, 5, rel_num_approvals=-0.5)
        with self.assertRaises(ValueError):
            identity(4, 5, rel_num_approvals=1.5)
