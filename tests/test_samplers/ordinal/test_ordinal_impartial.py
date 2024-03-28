from unittest import TestCase

from prefsampling.ordinal import stratification, impartial_anonymous, impartial
from tests.utils import float_parameter_test_values


def random_ord_impartial_samplers():
    samplers = [
        lambda num_voters, num_candidates, seed=None: stratification(
            num_voters, num_candidates, random_weight, seed
        )
        for random_weight in float_parameter_test_values(0, 1, 4)
    ]
    samplers.append(impartial)
    samplers.append(impartial_anonymous)
    return samplers


class TestOrdinalImpartial(TestCase):
    def test_ordinal_stratification(self):
        with self.assertRaises(ValueError):
            stratification(4, 5, 1.1)
        with self.assertRaises(ValueError):
            stratification(4, 5, -0.5)
