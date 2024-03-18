from unittest import TestCase

import numpy as np

from prefsampling.point import ball_uniform, gaussian, ball_resampling, uniform


def random_ball_samplers(num_dim):
    samplers = []
    for center_point in [None] + 3 * [np.random.random(num_dim)]:
        for widths in 3 * [np.random.random(1)] + 3 * [np.random.random(num_dim)]:
            for only_envelope in [True, False]:
                samplers.append(
                    lambda num_points, num_dimensions, seed=None: ball_uniform(
                        num_points,
                        num_dimensions,
                        center_point=center_point,
                        widths=widths,
                        only_envelope=only_envelope,
                        seed=seed,
                    )
                )
    return samplers


def random_ball_resampling_samplers(num_dim):
    samplers = []
    for center_point in [None] + 3 * [np.random.random(num_dim)]:
        for width in range(1, 5):
            for inner_sampler in [np.random.normal, np.random.random]:
                samplers.append(
                    lambda num_points, num_dimensions, seed=None: ball_resampling(
                        num_points,
                        num_dimensions,
                        inner_sampler,
                        {"size": num_dimensions},
                        center_point=center_point,
                        width=width,
                    )
                )
    return samplers


def random_gaussian_samplers(num_dim):
    samplers = []
    for center_point in [None] + 3 * [np.random.random(num_dim)]:
        for widths in 3 * [np.random.random(1)] + 3 * [np.random.random(num_dim)]:
            for bounds in [None] + [widths + 3 * np.random.random(num_dim)]:
                samplers.append(
                    lambda num_points, num_dimensions, seed=None: gaussian(
                        num_points,
                        num_dimensions,
                        center_point=center_point,
                        widths=widths,
                        bounds=bounds,
                        seed=seed,
                    )
                )
    return samplers


def random_uniform_samplers(num_dim):
    samplers = []
    for center_point in [None] + 3 * [np.random.random(num_dim)]:
        for widths in 3 * [np.random.random(1)] + 3 * [np.random.random(num_dim)]:
            samplers.append(
                lambda num_points, num_dimensions, seed=None: uniform(
                    num_points,
                    num_dimensions,
                    center_point=center_point,
                    widths=widths,
                    seed=seed,
                )
            )
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
            all_samplers = random_gaussian_samplers(num_dimensions)
            all_samplers += random_ball_samplers(num_dimensions)
            all_samplers += random_ball_resampling_samplers(num_dimensions)
            all_samplers += random_uniform_samplers(num_dimensions)
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
