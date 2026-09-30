from unittest import TestCase

import numpy as np

from prefsampling.point import gaussian


def random_gaussian_samplers(num_dim):
    samplers = []
    for center_point in [None] + 2 * [np.random.random(num_dim)]:
        for widths in 2 * [np.random.random(1)] + 2 * [np.random.random(num_dim)]:
            for bounds in [None] + [widths + 2 * np.random.random(num_dim)]:
                samplers.append(
                    lambda num_points, num_dimensions, seed=None: gaussian(
                        num_points,
                        num_dimensions,
                        center_point=center_point,
                        sigmas=widths,
                        widths=bounds,
                        seed=seed,
                    )
                )
    return samplers


class TestPointGaussian(TestCase):
    def test_invalid_truncation_widths(self):
        for widths in [[0, 1], [-1, 1], [np.nan, 1], [np.inf, 1], [[1], [1]]]:
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                gaussian(3, 2, widths=widths, seed=42)

    def test_invalid_sigmas(self):
        for sigmas in [[-1, 1], [np.nan, 1], [np.inf, 1], [[1], [1]]]:
            with self.subTest(sigmas=sigmas), self.assertRaises(ValueError):
                gaussian(3, 2, sigmas=sigmas, seed=42)

    def test_gaussian(self):
        with self.assertRaises(TypeError):
            gaussian(3, 2, widths=1)
        with self.assertRaises(ValueError):
            gaussian(3, 2, widths=[1, 4, 2])
