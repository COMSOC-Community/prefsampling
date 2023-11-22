from unittest import TestCase

from prefsampling.ordinal import urn


class TestOrdinalSamplers(TestCase):

    def test_ordinal_urn(self):

        with self.assertRaises(ValueError):
            urn(4, 5, alpha=-0.4)
