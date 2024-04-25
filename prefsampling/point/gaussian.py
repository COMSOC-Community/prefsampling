from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from prefsampling.inputvalidators import validate_int
from prefsampling.point.utils import validate_center_point, validate_width


def gaussian(
    num_points: int,
    num_dimensions: int,
    center_point: Iterable[float] = None,
    sigmas: float | Iterable[float] = 1,
    widths: Iterable[float] = None,
    seed: int = None,
) -> np.ndarray:
    """
    Samples points uniformly at random in a gaussian space.

    Parameters
    ----------
        num_points: int
            The number of points to sample.
        num_dimensions: int
            The number of dimensions for the Gaussian space.
        center_point: Iterable[float]
            The coordinates of the center point of the gaussian distribution. It needs to have one
            coordinate per dimension.
        sigmas: float | Iterable[float], default: :code:`1`
            The standard deviation of the gaussian distribution. If a single value
            is given, the width is applied to all dimensions. In case multiple values are given,
            they are applied to each dimension independently.
        widths: Iterable[float], default: :code:`None`
            Maximal widths for the Gaussian distributions. One width per dimension needs to be
            provided. When sampling points, if the distance between the point and the center point
            is larger than half the width of a dimension, the point is resampled.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            The coordinates of the :code:`num_points` points that have been sampled.

    Validation
    ----------

        .. image:: ../validation_plots/point/gaussian.png
            :width: 800
            :alt: Observed frequencies for a Gaussian model
    """
    validate_int(num_points, "num_points", 0)
    validate_int(num_dimensions, "num_dimensions", 1)
    if widths is not None:
        if isinstance(widths, Iterable):
            widths = np.array(widths, dtype=float)
            if len(widths) != num_dimensions:
                raise ValueError(
                    f"The number of widths needs to be equal to the number of dimensions "
                    f"({len(widths)} given for {num_dimensions} dimensions)."
                )
        else:
            raise TypeError(
                "The 'widths' parameter needs to be an iterable with one value per dimension."
            )
    center_point = validate_center_point(center_point, num_dimensions)
    sigmas = validate_width(sigmas, num_dimensions)

    rng = np.random.default_rng(seed)
    if widths is None:
        return rng.normal(
            loc=center_point, scale=sigmas, size=(num_points, num_dimensions)
        )
    else:
        points = []
        for _ in range(num_points):
            point = rng.normal(loc=center_point, scale=sigmas, size=num_dimensions)
            while not (np.abs(point - center_point) <= (widths / 2)).all():
                point = rng.normal(loc=center_point, scale=sigmas, size=num_dimensions)
            points.append(point)

        return np.array(points, dtype=float)
