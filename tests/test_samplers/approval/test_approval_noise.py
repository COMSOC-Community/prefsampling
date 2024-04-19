from enum import Enum
from unittest import TestCase


from prefsampling.approval.noise import noise, SetDistance
from tests.utils import float_parameter_test_values, TestSampler


def all_test_samplers_approval_noise():
    return [
        TestSampler(
            noise,
            {
                "rel_size_central_vote": random_p,
                "phi": random_phi,
                "distance": distance,
            },
        )
        for distance in SetDistance
        for random_p in float_parameter_test_values(0, 1, 2)
        for random_phi in float_parameter_test_values(0, 1, 2)
    ]


class TestApprovalNoise(TestCase):
    def test_approval_noise(self):
        with self.assertRaises(ValueError):
            noise(4, 5, rel_size_central_vote=0.5, phi=-0.4)
        with self.assertRaises(ValueError):
            noise(4, 5, rel_size_central_vote=0.5, phi=4)
        with self.assertRaises(ValueError):
            noise(4, 5, rel_size_central_vote=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            noise(4, 5, rel_size_central_vote=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            noise(4, 5, rel_size_central_vote=0.4, phi=0.5, distance="aze")
        with self.assertRaises(ValueError):

            class TestEnum(Enum):
                a = "1"

            noise(4, 5, rel_size_central_vote=0.4, phi=0.5, distance=TestEnum.a)

        # Test when len(A) = 0
        noise(4, 5, rel_size_central_vote=0, phi=0, distance=SetDistance.HAMMING)
        noise(4, 5, rel_size_central_vote=0, phi=0, distance=SetDistance.ZELINKA)
        noise(4, 5, rel_size_central_vote=0, phi=0, distance=SetDistance.JACCARD)
        noise(4, 5, rel_size_central_vote=0, phi=0.3, distance=SetDistance.JACCARD)
        noise(
            4, 5, rel_size_central_vote=0, phi=0.3, distance=SetDistance.BUNKE_SHEARER
        )
