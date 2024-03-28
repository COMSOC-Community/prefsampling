import numpy as np

from prefsampling.point import gaussian


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
                        sigmas=widths,
                        widths=bounds,
                        seed=seed,
                    )
                )
    return samplers
