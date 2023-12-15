from unittest import TestCase

from prefsampling.ordinal import group_separable


class TestOrdinalGroupSeparable(TestCase):
    def test_ordinal_graoup_separable(self):
        with self.assertRaises(ValueError):
            group_separable(4, 5, "caterpillar")
