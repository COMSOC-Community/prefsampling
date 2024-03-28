from unittest import TestCase

import numpy as np

from prefsampling.ordinal.plackettluce import plackett_luce
from tests.utils import TestSampler


def all_test_samplers_ordinal_plackett_luce():
    def random_plackett_luce(num_voters, num_candidates, all_alphas, seed=None):
        alphas = []
        for i in range(num_candidates):
            alphas.append(all_alphas[i % len(all_alphas)])
        return plackett_luce(num_voters, num_candidates, alphas, seed=seed)

    return [
        TestSampler(random_plackett_luce, {"all_alphas": random_all_alphas * 4 + 0.1})
        for random_all_alphas in np.random.random((2, 10))
    ]


class TestOrdinalPlackettLuce(TestCase):
    def test_ordinal_plackett_luce(self):
        with self.assertRaises(ValueError):
            plackett_luce(4, 5, alphas=[0.4, 0.4])
        with self.assertRaises(ValueError):
            plackett_luce(4, 5, alphas=[0.4, 0.4, 0.4, 0.4, -0.4])
        with self.assertRaises(ValueError):
            plackett_luce(4, 5, alphas=[0, 0, 0, 0, 0])
