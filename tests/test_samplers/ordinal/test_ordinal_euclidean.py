from unittest import TestCase

from prefsampling.ordinal.euclidean import euclidean


class TestEuclideanSamplers(TestCase):
    def test_ordinal_euclidean(self):
        with self.assertRaises(ValueError):
            euclidean(5, 20, space=-1)
