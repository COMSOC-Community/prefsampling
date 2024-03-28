from unittest import TestCase

import numpy as np

from prefsampling.ordinal import didi


def random_ord_didi_samplers():
    samplers = [
        lambda num_voters, num_candidates, seed=None: didi(
            num_voters, num_candidates, [1] * num_candidates, seed=seed
        )
    ]
    for _ in range(10):
        samplers.append(
            lambda num_voters, num_candidates, seed=None: didi(
                num_voters, num_candidates, np.random.random(size=num_candidates) * 3 + 0.1, seed=seed
            )
        )
    return samplers


class TestOrdinalDidi(TestCase):
    def test_ordinal_did(self):
        with self.assertRaises(ValueError):
            didi(4, 5, alphas=[0.4, 0.4])
        with self.assertRaises(ValueError):
            didi(4, 5, alphas=[0.4, 0.4, 0.4, 0.4, -0.4])
        with self.assertRaises(ValueError):
            didi(4, 5, alphas=[0, 0, 0, 0, 0])
        didi(4, 5, alphas=[0.1, 0.4, 0.8, 0.9, 1])
