from unittest import TestCase

from tests.test_samplers.approval.test_all_approval_samplers import random_app_samplers
from tests.test_samplers.ordinal.test_all_ordinal_samplers import random_ord_samplers


def random_samplers():
    samplers = random_app_samplers()
    samplers += random_ord_samplers()
    return samplers


class TestSamplers(TestCase):
    def helper_test_all_samplers(self, sampler, num_voters, num_candidates):
        # All the necessary arguments are there
        sampler(num_voters, num_candidates)
        sampler(num_voters, num_candidates, seed=23)
        sampler(num_voters=num_voters, num_candidates=num_candidates, seed=23)

        # The samplers are decorated to exclude bad number of voters and/or candidates arguments
        with self.assertRaises(ValueError):
            sampler(1, -2)
        with self.assertRaises(ValueError):
            sampler(-2, 1)
        with self.assertRaises(ValueError):
            sampler(-2, -2)
        with self.assertRaises(TypeError):
            sampler(1.5, 2)
        with self.assertRaises(TypeError):
            sampler(1, 2.5)
        with self.assertRaises(TypeError):
            sampler(1.5, 2.5)

    def test_all_samplers(self):
        num_voters = 200
        num_candidates = 5

        for sampler in random_samplers():
            with self.subTest(sampler=sampler):
                self.helper_test_all_samplers(sampler, num_voters, num_candidates)
