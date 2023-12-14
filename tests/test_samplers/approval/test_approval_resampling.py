from unittest import TestCase

from prefsampling.approval.resampling import resampling


class TestOrdinalSamplers(TestCase):
    def test_ordinal_urn(self):
        with self.assertRaises(ValueError):
            resampling(4, 5, p=0.5, phi=-0.4)
        with self.assertRaises(ValueError):
            resampling(4, 5, p=0.5, phi=4)
        with self.assertRaises(ValueError):
            resampling(4, 5, p=-0.4, phi=0.5)
        with self.assertRaises(ValueError):
            resampling(4, 5, p=-0.4, phi=0.5)
