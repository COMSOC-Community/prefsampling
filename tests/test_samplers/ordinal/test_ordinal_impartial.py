from unittest import TestCase

from prefsampling.ordinal import stratification


class TestOrdinalImpartial(TestCase):
    def test_ordinal_stratification(self):
        with self.assertRaises(ValueError):
            stratification(4, 5, 1.1)
        with self.assertRaises(ValueError):
            stratification(4, 5, -0.5)
