from unittest import TestCase

from prefsampling.inputvalidators import validate_num_voters_candidates, validate_int


class TestInputValidators(TestCase):
    def test_num_voters_candidates_validator(self):
        decorated_function = validate_num_voters_candidates(lambda x, y: x + y)
        with self.assertRaises(ValueError):
            decorated_function(-2, -3)
        with self.assertRaises(ValueError):
            decorated_function(2, -3)
        with self.assertRaises(ValueError):
            decorated_function(-2, 3)
        with self.assertRaises(TypeError):
            decorated_function(1.5, 2.5)
        with self.assertRaises(TypeError):
            decorated_function(0.1, 5)
        with self.assertRaises(TypeError):
            decorated_function(1, 2.5)

    def test_validate_int(self):
        with self.assertRaises(TypeError):
            validate_int("aze")
        with self.assertRaises(TypeError):
            validate_int(0.123)
        with self.assertRaises(ValueError):
            validate_int(65, lower_bound=100)
        with self.assertRaises(ValueError):
            validate_int(105, upper_bound=100)
