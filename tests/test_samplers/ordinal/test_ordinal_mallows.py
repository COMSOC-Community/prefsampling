from unittest import TestCase

from prefsampling.ordinal.mallows import mallows, norm_mallows, phi_from_norm_phi


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
        