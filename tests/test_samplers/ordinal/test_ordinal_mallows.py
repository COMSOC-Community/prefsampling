from unittest import TestCase

from prefsampling.ordinal.mallows import mallows, norm_mallows, phi_from_norm_phi
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_ordinal_mallows():
    samplers = [
        TestSampler(
            mallows,
            {
                "phi": random_phi,
                "normalise_phi": random_normalise_phi,
                "impartial_central_vote": impartial_central_vote,
            },
        )
        for random_phi in float_parameter_test_values(0, 1, 2)
        for random_normalise_phi in [True, False]
        for impartial_central_vote in [True, False]
    ]
    samplers += [
        TestSampler(
            norm_mallows,
            {"norm_phi": random_phi, "impartial_central_vote": impartial_central_vote},
        )
        for random_phi in float_parameter_test_values(0, 1, 2)
        for impartial_central_vote in [True, False]
    ]
    return samplers


class TestOrdinalMawllos(TestCase):
    def test_ordinal_mallows(self):
        with self.assertRaises(ValueError):
            mallows(4, 5, phi=-0.4)
        with self.assertRaises(ValueError):
            mallows(4, 5, phi=1.4)
        with self.assertRaises(ValueError):
            norm_mallows(4, 5, norm_phi=-0.4)
        with self.assertRaises(ValueError):
            norm_mallows(4, 5, norm_phi=1.4)

    def test_phi_from_norm_phi(self):
        self.assertTrue(phi_from_norm_phi(5, 1) == 1)
        self.assertTrue(phi_from_norm_phi(5, 1.5) == 0.5)
        with self.assertRaises(ValueError):
            phi_from_norm_phi(5, -0.5)
        with self.assertRaises(ValueError):
            phi_from_norm_phi(5, 2.1)

    def test_impartial_central_vote(self):
        mallows(4, 5, phi=0.4, impartial_central_vote=True)
