from unittest import TestCase

from prefsampling.ordinal.urn import urn
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_ordinal_urn():
    return [
        TestSampler(urn, {"alpha": random_alpha})
        for random_alpha in float_parameter_test_values(0, 10, 2)
    ]


class TestOrdinalUrn(TestCase):
    def test_ordinal_urn(self):
        with self.assertRaises(ValueError):
            urn(4, 5, alpha=-0.4)
