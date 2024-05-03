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

    def urn_partylist_party_votes(num_voters, num_candidates, seed=None):
        party_votes = [set(), set()]
        for k in range(num_candidates):
            if k % 2 == 0:
                party_votes[0].add(k)
            else:
                party_votes[1].add(k)
        return urn_partylist(num_voters, num_candidates, 0.5, party_votes=party_votes, seed=seed)
    samplers.append(TestSampler(urn_partylist_party_votes, {}))

    return samplers


class TestApprovalUrn(TestCase):
    def test_approval_urn(self):
        with self.assertRaises(ValueError):
            urn(4, 5, p=0.5, alpha=-1)
        with self.assertRaises(ValueError):
            urn(4, 5, p=-0.5, alpha=4)
        with self.assertRaises(ValueError):
            urn(4, 5, p=2, alpha=4)

    def test_approval_urn_constant_size(self):
        with self.assertRaises(ValueError):
            urn_constant_size(4, 5, rel_num_approvals=0.5, alpha=-1)
        with self.assertRaises(ValueError):
            urn_constant_size(4, 5, rel_num_approvals=-0.5, alpha=4)
        with self.assertRaises(ValueError):
            urn_constant_size(4, 5, rel_num_approvals=2, alpha=4)

    def test_approval_urn_partylist(self):
        with self.assertRaises(ValueError):
            urn_partylist(4, 5, alpha=-1)
        with self.assertRaises(ValueError):
            urn_partylist(4, 5, alpha=5, parties=10)
        with self.assertRaises(ValueError):
            urn_partylist(4, 5, alpha=5, party_votes=[{0, 1, 2, 3, 4, 5}, {6, 7, 8}])
        with self.assertRaises(ValueError):
            urn_partylist(4, 5, alpha=5, party_votes=[{0, 3, 4, 5}, {2}])
        with self.assertRaises(ValueError):
            urn_partylist(4, 5, alpha=5, party_votes=[{0, 1, 2}, {2, 3, 4}])

