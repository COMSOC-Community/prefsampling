from unittest import TestCase

from prefsampling import CONSTANTS, EuclideanSpace, TreeSampler, NoiseType


class TestConstants(TestCase):
    def test_constants(self):
        assert CONSTANTS.UNIFORM == EuclideanSpace.UNIFORM
        assert CONSTANTS.GAUSSIAN == EuclideanSpace.GAUSSIAN
        assert CONSTANTS.SPHERE == EuclideanSpace.SPHERE
        assert CONSTANTS.BALL == EuclideanSpace.BALL
        assert CONSTANTS.HAMMING == NoiseType.HAMMING
        assert CONSTANTS.JACCARD == NoiseType.JACCARD
        assert CONSTANTS.ZELINKA == NoiseType.ZELINKA
        assert CONSTANTS.BUNKE_SHEARER == NoiseType.BUNKE_SHEARER
        assert CONSTANTS.SCHROEDER == TreeSampler.SCHROEDER
        assert CONSTANTS.SCHROEDER_UNIFORM == TreeSampler.SCHROEDER_UNIFORM
        assert CONSTANTS.SCHROEDER_LESCANNE == TreeSampler.SCHROEDER_LESCANNE
        assert CONSTANTS.CATERPILLAR == TreeSampler.CATERPILLAR
        assert CONSTANTS.BALANCED == TreeSampler.BALANCED
