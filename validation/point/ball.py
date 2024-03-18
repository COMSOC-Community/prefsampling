import numpy as np

from prefsampling.point import ball_uniform, ball_resampling, uniform

import matplotlib.pyplot as plt


def plot_scatter_points(points, ax, limits, path):
    args = [points[:, 0]]
    ax.set_xlim(limits[0][0], limits[0][1])
    if len(points[0]) > 1:
        args.append(points[:, 1])
        ax.set_ylim(limits[1][0], limits[1][1])
    else:
        args.append([0 for _ in range(len(points))])
    if len(points[0]) > 2:
        args.append(points[:, 2])
        ax.set_zlim(limits[2][0], limits[2][1])
    ax.scatter(*args, s=1)
    plt.savefig(path, dpi=300, bbox_inches="tight")


def plot_ball_uniform(path):

    fig = plt.figure(figsize=(16, 12))

    all_params = (
        {"widths": (4, (4, 4), (4, 4, 4)), "only_envelope": False},
        {"widths": (4, (4, 4), (4, 4, 4)), "only_envelope": True},
        {"widths": (4, (4, 1), (4, 1, 3)), "only_envelope": False},
        {"widths": (4, (4, 1), (4, 1, 3)), "only_envelope": True},
    )
    ax_limits = (-4.5, 4.5)
    num_points = 2000
    for num_dimensions in [1, 2, 3]:
        for i, params in enumerate(all_params):
            points = ball_uniform(num_points, num_dimensions, widths=params["widths"][num_dimensions - 1], only_envelope=params["only_envelope"])
            ax = fig.add_subplot(3, 4, (num_dimensions - 1) * 4 + i + 1, projection="3d" if num_dimensions == 3 else None)
            plot_scatter_points(points, ax, [ax_limits] * num_dimensions, path)


def uniform_square(num_dimensions):
    return np.random.uniform(size=num_dimensions) * 8 - 4


def gamma_square(num_dimensions):
    return np.random.gamma(2, size=num_dimensions) * np.random.choice((-1, 1), size=num_dimensions)


def plot_ball_resampling(path):

    fig = plt.figure(figsize=(16, 12))

    ax_limits = (-4.5, 4.5)
    num_points = 2000
    for num_dimensions in [1, 2, 3]:
        all_params = (
            {"inner_sampler": uniform_square, "inner_sampler_args": {"num_dimensions": num_dimensions}},
            {"inner_sampler": np.random.normal, "inner_sampler_args": {"size": num_dimensions}},
            {"inner_sampler": gamma_square, "inner_sampler_args": {"num_dimensions": num_dimensions}},
        )
        for i, params in enumerate(all_params):
            points = ball_resampling(num_points, num_dimensions, inner_sampler=params["inner_sampler"], inner_sampler_args=params["inner_sampler_args"], width=4)
            ax = fig.add_subplot(3, 4, (num_dimensions - 1) * 4 + i + 1, projection="3d" if num_dimensions == 3 else None)
            plot_scatter_points(points, ax, [ax_limits] * num_dimensions, path)
