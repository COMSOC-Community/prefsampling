from unittest import TestCase


from prefsampling.approval.noise import noise, NoiseType
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_approval_noise():
    return [
        TestSampler(noise, {"p": random_p, "phi": random_phi, "noise_type": noise_type})
        for noise_type in NoiseType
        for random_p in float_parameter_test_values(0, 1, 2)
        for random_phi in float_parameter_test_values(0, 1, 2)
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

        # Test when len(A) = 0
        noise(4, 5, p=0, phi=0, noise_type=NoiseType.HAMMING)
        noise(4, 5, p=0, phi=0, noise_type=NoiseType.ZELINKA)
        noise(4, 5, p=0, phi=0, noise_type=NoiseType.JACCARD)
        noise(4, 5, p=0, phi=0.3, noise_type=NoiseType.JACCARD)
        noise(4, 5, p=0, phi=0.3, noise_type=NoiseType.BUNKE_SHEARER)
