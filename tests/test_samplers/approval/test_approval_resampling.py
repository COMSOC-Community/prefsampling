from unittest import TestCase

from prefsampling.approval.resampling import (
    resampling,
    disjoint_resampling,
    moving_resampling,
)
from tests.utils import (
    float_parameter_test_values,
    int_parameter_test_values,
    TestSampler,
)


def all_test_samplers_approval_resampling():
    samplers = [
        TestSampler(
            resampling,
            {
                "rel_size_central_vote": random_p,
                "phi": random_phi,
                "impartial_central_vote": imp_central,
            },
        )
        for random_p in float_parameter_test_values(0, 1, 2)
        for random_phi in float_parameter_test_values(0, 1, 2)
        for imp_central in [True, False]
    ]
    samplers += [
        TestSampler(
            disjoint_resampling,
            {
                "rel_size_central_vote": random_p,
                "phi": random_phi,
                "num_central_votes": random_g,
                "impartial_central_votes": imp_central,
            },
        )
        for random_g in int_parameter_test_values(1, 10, 2)
        for random_p in float_parameter_test_values(0, 1, 2)
        for random_phi in float_parameter_test_values(0, 1, 2)
        if random_g * random_p <= 1
        for imp_central in [True, False]
    ]
    samplers += [
        TestSampler(
            moving_resampling,
            {
                "rel_size_central_vote": random_p,
                "phi": random_phi,
                "num_legs": 1,
                "impartial_central_vote": imp_central,
            },
        )
        for random_p in float_parameter_test_values(0, 1, 2)
        for random_phi in float_parameter_test_values(0, 1, 2)
        for imp_central in [True, False]
    ]
    return samplers


class TestApprovalResampling(TestCase):
    def test_approval_resampling(self):
        with self.assertRaises(ValueError):
            resampling(4, 5, rel_size_central_vote=0.5, phi=-0.4)
        with self.assertRaises(ValueError):
            resampling(4, 5, rel_size_central_vote=0.5, phi=4)
        with self.assertRaises(ValueError):
            resampling(4, 5, rel_size_central_vote=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            resampling(4, 5, rel_size_central_vote=-0.4, phi=0.5)

        with self.assertRaises(ValueError):
            resampling(4, 5, rel_size_central_vote=0.5, phi=0.4, central_vote="1234")
        with self.assertRaises(ValueError):
            resampling(
                4,
                5,
                rel_size_central_vote=0.5,
                phi=0.4,
                central_vote={1, 2, 3, 4, 5, 6, 7},
            )

        resampling(
            4, 5, rel_size_central_vote=0.5, phi=0.4, impartial_central_vote=True
        )

    def test_approval_disjoint_resampling(self):
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, rel_size_central_vote=0.5, phi=-0.4)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, rel_size_central_vote=0.5, phi=4)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, rel_size_central_vote=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, rel_size_central_vote=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            disjoint_resampling(
                4, 5, rel_size_central_vote=0.4, phi=0.5, num_central_votes=10
            )
