from unittest import TestCase

from prefsampling.ordinal import stratification, impartial_anonymous, impartial
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_ordinal_impartial():
    samplers = [
        TestSampler(stratification, {"weight": random_weight})
        for random_weight in float_parameter_test_values(0, 1, 2)
    ]
    samplers.append(TestSampler(impartial, {}))
    samplers.append(TestSampler(impartial_anonymous, {}))
    return samplers


class TestOrdinalImpartial(TestCase):
    def test_ordinal_stratification(self):
        with self.assertRaises(ValueError):
            stratification(4, 5, 1.1)
        with self.assertRaises(ValueError):
            stratification(4, 5, -0.5)
