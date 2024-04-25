from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from prefsampling.inputvalidators import validate_int
from prefsampling.point.utils import validate_center_point, validate_width


def cube(
    num_points: int,
    num_dimensions: int,
    center_point: Iterable[float] = None,
    widths: float | Iterable[float] = 1,
    seed: int = None,
) -> np.ndarray:
    """
    Samples points uniformly at random in a cube.

    Parameters
    ----------
        num_points: int
            The number of points to sample.
        num_dimensions: int
            The number of dimensions for the cube.
        center_point: Iterable[float]
            The coordinates of the center point of the cube. It needs to have one coordinate per
            dimension.
        widths: float | Iterable[float], default: :code:`1`
            The width of the space distribution. If a single value is given, the width is applied
            to all dimensions. In case multiple values are given, they are applied to each dimension
            independently.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            The coordinates of the :code:`num_points` points that have been sampled.

    Validation
    ----------

        .. image:: ../validation_plots/point/cube.png
            :width: 800
            :alt: Observed frequencies for a uniform cube model
    """
    validate_int(num_points, "num_points", 0)
    validate_int(num_dimensions, "num_dimensions", 1)
    center_point = validate_center_point(center_point, num_dimensions)
    widths = validate_width(widths, num_dimensions)
    rng = np.random.default_rng(seed)
    return rng.random((num_points, num_dimensions)) * widths + (
        center_point - widths / 2
    )
