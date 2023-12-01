from unittest import TestCase

from prefsampling.ordinal.plackettluce import plackett_luce


class TestOrdinalPlackettLuce(TestCase):
    def test_ordinal_plackett_luce(self):
        with self.assertRaises(ValueError):
            plackett_luce(4, 5, alphas=[0.4, 0.4])
        with self.assertRaises(ValueError):
            plackett_luce(4, 5, alphas=[0.4, 0.4, 0.4, -0.4])
        with self.assertRaises(ValueError):
            plackett_luce(4, 5, alphas=[0, 0, 0, 0])
