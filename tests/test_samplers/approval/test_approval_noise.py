from unittest import TestCase


from prefsampling.approval.noise import noise, NoiseType
from tests.utils import float_parameter_test_values


def random_app_noise_samplers():
    return [
        lambda num_voters, num_candidates, seed=None: noise(
            num_voters,
            num_candidates,
            random_p,
            random_phi,
            noise_type=noise_type,
            seed=seed,
        )
        for noise_type in NoiseType
        for random_p in float_parameter_test_values(0, 1, 4)
        for random_phi in float_parameter_test_values(0, 1, 4)
    ]


class TestApprovalNoise(TestCase):
    def test_approval_noise(self):
        with self.assertRaises(ValueError):
            noise(4, 5, p=0.5, phi=-0.4)
        with self.assertRaises(ValueError):
            noise(4, 5, p=0.5, phi=4)
        with self.assertRaises(ValueError):
            noise(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            noise(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            noise(4, 5, p=0.4, phi=0.5, noise_type="aze")
