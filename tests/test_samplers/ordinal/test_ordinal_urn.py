from unittest import TestCase

from prefsampling.ordinal.urn import urn
from tests.utils import float_parameter_test_values


def random_ord_urn_samplers():
    return [
        lambda num_voters, num_candidates, seed=None: urn(
            num_voters, num_candidates, random_alpha, seed=seed
        )
        for random_alpha in float_parameter_test_values(0, 10, 2)
    ]


class TestOrdinalUrn(TestCase):
    def test_ordinal_urn(self):
        with self.assertRaises(ValueError):
            urn(4, 5, alpha=-0.4)
