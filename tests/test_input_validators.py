from unittest import TestCase

from prefsampling.inputvalidators import validate_num_voters_candidates


class TestInputValidators(TestCase):
    def test_num_voters_candidates_decorator(self):
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
