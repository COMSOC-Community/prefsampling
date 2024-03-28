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
        TestSampler(resampling, {"p": random_p, "phi": random_phi})
        for random_p in float_parameter_test_values(0, 1, 2)
        for random_phi in float_parameter_test_values(0, 1, 2)
    ]
    samplers += [
        TestSampler(
            disjoint_resampling, {"p": random_p, "phi": random_phi, "g": random_g}
        )
        for random_g in int_parameter_test_values(1, 10, 2)
        for random_p in float_parameter_test_values(0, 1, 2)
        for random_phi in float_parameter_test_values(0, 1, 2)
        if random_g * random_p <= 1
    ]
    samplers += [
        TestSampler(moving_resampling, {"p": random_p, "phi": random_phi, "num_legs": random_num_legs})
        for random_p in float_parameter_test_values(0, 1, 2)
        for random_phi in float_parameter_test_values(0, 1, 2)
        for random_num_legs in int_parameter_test_values(1, 4, 1)
    ]
    return samplers


class TestApprovalResampling(TestCase):
    def test_approval_resampling(self):
        with self.assertRaises(ValueError):
            resampling(4, 5, p=0.5, phi=-0.4)
        with self.assertRaises(ValueError):
            resampling(4, 5, p=0.5, phi=4)
        with self.assertRaises(ValueError):
            resampling(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            resampling(4, 5, p=-0.4, phi=0.5)

        with self.assertRaises(ValueError):
            resampling(4, 5, p=0.5, phi=0.4, central_vote="1234")
        with self.assertRaises(ValueError):
            resampling(4, 5, p=0.5, phi=0.4, central_vote={1, 2, 3, 4, 5, 6, 7})

    def test_approval_disjoint_resampling(self):
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, p=0.5, phi=-0.4)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, p=0.5, phi=4)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            disjoint_resampling(4, 5, p=0.4, phi=0.5, g=10)

    def test_impartial_central_vote(self):
        resampling(4, 5, p=0.5, phi=0.4, impartial_central_vote=True)
