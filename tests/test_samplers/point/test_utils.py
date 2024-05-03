from unittest import TestCase

from prefsampling.point.utils import validate_width


class TestPointUtils(TestCase):

    def test_validate_width(self):
        with self.assertRaises(ValueError):
            validate_width([1, 2, 3], 2)
        with self.assertRaises(TypeError):
            validate_width("aze", 2)
