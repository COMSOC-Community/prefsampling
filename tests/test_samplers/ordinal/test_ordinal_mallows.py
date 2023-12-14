from unittest import TestCase

from prefsampling.ordinal.mallows import mallows, norm_mallows


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
