from prefsampling.point import ball_uniform

import matplotlib.pyplot as plt

from validation.point.utils import add_scatter_points


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
            points = ball_uniform(
                num_points,
                num_dimensions,
                widths=params["widths"][num_dimensions - 1],
                only_envelope=params["only_envelope"],
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
                title=f"Widths: {params['widths'][num_dimensions - 1]}, "
                f"Envelope: {params['only_envelope']}",
            )

    plt.savefig(path, dpi=300, bbox_inches="tight")
