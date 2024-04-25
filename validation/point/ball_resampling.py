import numpy as np

from prefsampling.point import ball_resampling, gaussian, cube

import matplotlib.pyplot as plt

from validation.point.utils import add_scatter_points


def uniform_cube(num_dimensions):
    return cube(1, num_dimensions, widths=[4] * num_dimensions)[0]


def gamma_square(num_dimensions):
    return np.random.gamma(2, size=num_dimensions) * np.random.choice(
        (-1, 1), size=num_dimensions
    )


def unbounded_gaussian(num_dimensions):
    return gaussian(1, num_dimensions)[0]


def plot_ball_resampling(path):

    fig = plt.figure(figsize=(16, 12))

    ax_limits = (-4.5, 4.5)
    num_points = 2000
    for num_dimensions in [1, 2, 3]:
        all_params = (
            {
                "inner_sampler": uniform_cube,
                "inner_sampler_args": {"num_dimensions": num_dimensions},
            },
            {
                "inner_sampler": unbounded_gaussian,
                "inner_sampler_args": {"num_dimensions": num_dimensions},
            },
            {
                "inner_sampler": gamma_square,
                "inner_sampler_args": {"num_dimensions": num_dimensions},
            },
        )
        for i, params in enumerate(all_params):
            points = ball_resampling(
                num_points,
                num_dimensions,
                inner_sampler=params["inner_sampler"],
                inner_sampler_args=params["inner_sampler_args"],
                width=4,
            )
            ax = fig.add_subplot(
                3,
                4,
                (num_dimensions - 1) * 4 + i + 1,
                projection="3d" if num_dimensions == 3 else None,
            )
            add_scatter_points(
                points,
                ax,
                [ax_limits] * num_dimensions,
                title=f"Sampler: {params['inner_sampler'].__name__}",
            )
    plt.savefig(path, dpi=300, bbox_inches="tight")
