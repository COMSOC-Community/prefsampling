from unittest import TestCase

import numpy as np

from prefsampling.ordinal.plackettluce import plackett_luce


def random_ord_plackett_luce_samplers():
    samplers = [
        lambda num_voters, num_candidates, seed=None: plackett_luce(
            num_voters, num_candidates, [1] * num_candidates, seed=seed
        )
    ]
    for _ in range(10):
        samplers.append(
            lambda num_voters, num_candidates, seed=None: plackett_luce(
                num_voters, num_candidates, np.random.random(size=num_candidates) * 3 + 0.1, seed=seed
            )
        )
    return samplers


class TestOrdinalPlackettLuce(TestCase):
    def test_ordinal_plackett_luce(self):
        with self.assertRaises(ValueError):
            plackett_luce(4, 5, alphas=[0.4, 0.4])
        with self.assertRaises(ValueError):
            plackett_luce(4, 5, alphas=[0.4, 0.4, 0.4, 0.4, -0.4])
        with self.assertRaises(ValueError):
            plackett_luce(4, 5, alphas=[0, 0, 0, 0, 0])
