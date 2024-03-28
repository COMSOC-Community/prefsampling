import numpy as np

from prefsampling.point import cube


def random_cube_samplers(num_dim):
    samplers = []
    for center_point in [None] + 2 * [np.random.random(num_dim)]:
        for widths in 2 * [np.random.random(1)] + 2 * [np.random.random(num_dim)]:
            samplers.append(
                lambda num_points, num_dimensions, seed=None: cube(
                    num_points,
                    num_dimensions,
                    center_point=center_point,
                    widths=widths,
                    seed=seed,
                )
            )
    return samplers
