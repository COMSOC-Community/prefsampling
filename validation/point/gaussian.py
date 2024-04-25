from prefsampling.point import gaussian

import matplotlib.pyplot as plt

from validation.point.utils import add_scatter_points


def plot_gaussian(path):

    fig = plt.figure(figsize=(16, 12))

    all_params = (
        {"widths": (None, None, None), "sigmas": (1, (1, 1), (1, 1, 1))},
        {"widths": (None, None, None), "sigmas": (0.5, (0.5, 1.5), (0.5, 1.5, 3))},
        {"widths": ((4,), (1, 4), (1, 2, 4)), "sigmas": (1, (1, 1), (1, 1, 1))},
        {"widths": ((4,), (1, 4), (1, 2, 4)), "sigmas": (0.2, (0.2, 0.5), (0.2, 0.5, 1))},
    )
    ax_limits = (-4.5, 4.5)
    num_points = 2000
    for num_dimensions in [1, 2, 3]:
        for i, params in enumerate(all_params):
            points = gaussian(
                num_points,
                num_dimensions,
                widths=params["widths"][num_dimensions - 1],
                sigmas=params["sigmas"][num_dimensions - 1],
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
                title=f"Sigmas: {params['sigmas'][num_dimensions - 1]},"
                      f" Widths: {params['widths'][num_dimensions - 1]}, "
            )

    plt.savefig(path, dpi=300, bbox_inches="tight")
