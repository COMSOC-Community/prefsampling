from unittest import TestCase

import numpy as np

from prefsampling.ordinal import didi
from tests.utils import TestSampler


def all_test_samplers_ordinal_didi():
    def random_didi(num_voters, num_candidates, all_alphas, seed=None):
        alphas = []
        for i in range(num_candidates):
            alphas.append(all_alphas[i % len(all_alphas)])
        return didi(num_voters, num_candidates, alphas, seed=seed)

    return [
        TestSampler(random_didi, {"all_alphas": random_all_alphas * 4 + 0.1})
        for random_all_alphas in np.random.random((2, 10))
    ]


class TestOrdinalDidi(TestCase):
    def test_ordinal_did(self):
        with self.assertRaises(ValueError):
            didi(4, 5, alphas=[0.4, 0.4])
        with self.assertRaises(ValueError):
            didi(4, 5, alphas=[0.4, 0.4, 0.4, 0.4, -0.4])
        with self.assertRaises(ValueError):
            didi(4, 5, alphas=[0, 0, 0, 0, 0])
        didi(4, 5, alphas=[0.1, 0.4, 0.8, 0.9, 1])
