from unittest import TestCase

import numpy as np

from tests.test_samplers.point.ball import random_ball_samplers
from tests.test_samplers.point.cube import random_cube_samplers
from tests.test_samplers.point.gaussian import random_gaussian_samplers


def random_point_samplers(num_dimensions=3):
    samplers = random_cube_samplers(num_dimensions)
    samplers += random_ball_samplers(num_dimensions)
    samplers += random_gaussian_samplers(num_dimensions)
    return samplers


class TestAllPointSamplers(TestCase):

    def helper_test_all_point_samplers(self, sampler, num_points, num_dimensions):
        result = sampler(num_points, num_dimensions)

        # Test if the function returns a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Test if the shape of the returned array is correct
        self.assertEqual(result.shape, (num_points, num_dimensions))

        # Test if the value are float
        for point in result:
            for coordinate in point:
                self.assertTrue(float(coordinate) == coordinate)

    def test_all_point_samplers(self):
        num_points = 200

        for num_dimensions in [1, 2, 3, 4, 5]:
            all_samplers = random_point_samplers(num_dimensions)
            for sampler in all_samplers:
                for test_sampler in [
                    sampler,
                    lambda x, y: sampler(num_points=x, num_dimensions=y),
                    lambda x, y: sampler(x, y, seed=363),
                ]:
                    with self.subTest(
                        sampler=test_sampler, num_dimensions=num_dimensions
                    ):
                        self.helper_test_all_point_samplers(
                            test_sampler, num_points, num_dimensions
                        )
