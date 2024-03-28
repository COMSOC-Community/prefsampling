from unittest import TestCase

import numpy as np

from prefsampling.point import ball_uniform, ball_resampling


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


class TestPointBall(TestCase):

    def test_ball(self):
        with self.assertRaises(TypeError):
            ball_uniform(5, 5, 0, 10)
        with self.assertRaises(ValueError):
            ball_uniform(5, 5, [3, 4, 2, 3], 10)
        with self.assertRaises(ValueError):
            ball_uniform(5, 5, ["a", "b", "c", "d", "e"], 10)
