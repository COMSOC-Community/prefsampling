from unittest import TestCase

from prefsampling.point import ball_uniform


class TestPointBall(TestCase):

    def test_ball(self):
        with self.assertRaises(TypeError):
            ball_uniform(5, 5, 0, 10)
        with self.assertRaises(ValueError):
            ball_uniform(5, 5, [3, 4, 2, 3], 10)
        with self.assertRaises(ValueError):
            ball_uniform(5, 5, ["a", "b", "c", "d", "e"], 10)
