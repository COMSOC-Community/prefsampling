from unittest import TestCase

from prefsampling.ordinal import didi


class TestOrdinalDidi(TestCase):
    def test_ordinal_did(self):
        with self.assertRaises(ValueError):
            didi(4, 5, alphas=[0.4, 0.4])
        with self.assertRaises(ValueError):
            didi(4, 5, alphas=[0.4, 0.4, 0.4, 0.4, -0.4])
        didi(4, 5, alphas=[0, 0, 0, 0, 0])
