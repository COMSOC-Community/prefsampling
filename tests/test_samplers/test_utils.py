from unittest import TestCase

from prefsampling.utils import comb, _comb


class TestUtils(TestCase):
    def test_comb(self):
        for c in [comb, _comb]:
            self.assertEqual(c(0, 0), 1)
            self.assertEqual(c(0, 1), 0)
            self.assertEqual(c(0, 2), 0)
            self.assertEqual(c(0, 3), 0)
            self.assertEqual(c(0, 4), 0)
            self.assertEqual(c(1, 0), 1)
            self.assertEqual(c(1, 1), 1)
            self.assertEqual(c(1, 2), 0)
            self.assertEqual(c(1, 3), 0)
            self.assertEqual(c(1, 4), 0)
            self.assertEqual(c(2, 0), 1)
            self.assertEqual(c(2, 1), 2)
            self.assertEqual(c(2, 2), 1)
            self.assertEqual(c(2, 3), 0)
            self.assertEqual(c(2, 4), 0)
            self.assertEqual(c(3, 0), 1)
            self.assertEqual(c(3, 1), 3)
            self.assertEqual(c(3, 2), 3)
            self.assertEqual(c(3, 3), 1)
            self.assertEqual(c(3, 4), 0)
            self.assertEqual(c(4, 0), 1)
            self.assertEqual(c(4, 1), 4)
            self.assertEqual(c(4, 2), 6)
            self.assertEqual(c(4, 3), 4)
            self.assertEqual(c(4, 4), 1)
