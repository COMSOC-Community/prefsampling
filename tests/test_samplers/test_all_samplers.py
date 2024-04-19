from unittest import TestCase

from tests.test_samplers.approval.test_all_approval_samplers import (
    all_test_samplers_approval,
)
from tests.test_samplers.ordinal.test_all_ordinal_samplers import (
    all_test_samplers_ordinal,
)


def all_random_samplers():
    samplers = all_test_samplers_ordinal()
    samplers += all_test_samplers_approval()
    return samplers


class TestSamplers(TestCase):
    def helper_test_all_samplers(self, sampler, num_voters, num_candidates):
        # All the necessary arguments are there
        for test_sampler_method in ["positional", "kwargs", "seed"]:
            sampler.test_sample(test_sampler_method, num_voters, num_candidates)

        # The samplers are decorated to exclude bad number of voters and/or candidates arguments
        with self.assertRaises(ValueError):
            sampler.test_sample_positional(1, -2)
        with self.assertRaises(ValueError):
            sampler.test_sample_positional(-2, 1)
        with self.assertRaises(ValueError):
            sampler.test_sample_positional(-2, -2)
        with self.assertRaises(TypeError):
            sampler.test_sample_positional(1.5, 2)
        with self.assertRaises(TypeError):
            sampler.test_sample_positional(1, 2.5)
        with self.assertRaises(TypeError):
            sampler.test_sample_positional(1.5, 2.5)

    def test_all_samplers(self):
        num_voters = 200
        num_candidates = 5

        for test_sampler in all_random_samplers():
            with self.subTest(sampler=test_sampler):
                self.helper_test_all_samplers(test_sampler, num_voters, num_candidates)
