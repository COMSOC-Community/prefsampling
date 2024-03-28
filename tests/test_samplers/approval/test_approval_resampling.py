from unittest import TestCase

import numpy as np

from prefsampling.approval.resampling import (
    resampling,
    disjoint_resampling,
    moving_resampling,
)
from tests.utils import float_parameter_test_values, int_parameter_test_values


def random_app_resampling_samplers():
    samplers = [
        lambda num_voters, num_candidates, seed=None: resampling(
            num_voters, num_candidates, random_phi, random_p, seed=seed
        )
        for random_p in float_parameter_test_values(0, 1, 4)
        for random_phi in float_parameter_test_values(0, 1, 4)
    ]
    samplers += [
        lambda num_voters, num_candidates, seed=None: disjoint_resampling(
            num_voters, num_candidates, random_phi, random_p, random_g, seed=seed
        )
        for random_g in int_parameter_test_values(1, 10, 4)
        for random_p in float_parameter_test_values(0, 1 / random_g, 4)
        for random_phi in float_parameter_test_values(0, 1, 4)
    ]
    samplers += [
        lambda num_voters, num_candidates, seed=None: moving_resampling(
            num_voters, num_candidates, random_phi, random_p, seed=seed
        )
        for random_p in float_parameter_test_values(0, 1, 4)
        for random_phi in float_parameter_test_values(0, 1, 4)
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
