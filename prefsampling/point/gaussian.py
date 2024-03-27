from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from prefsampling.inputvalidators import validate_int
from prefsampling.point.utils import validate_center_point, validate_width


def gaussian(
    num_points: int,
    num_dimensions: int,
    center_point: Iterable[float] = None,
    widths: float | Iterable[float] = 1,
    bounds: Iterable[float] = None,
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
        widths: float | Iterable[float], default: :code:`1`
            The width of the gaussian distribution, i.e., the standard deviation. If a single value
            is given, the width is applied to all dimensions. In case multiple values are given,
            they are applied to each dimension independently.
        bounds: Iterable[float], default: :code:`None`
            Bounds for the Gaussian distribution. One bound per dimension needs to be provided.
            When sampling points, the ones that fall outside the bounds are resampled.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            The coordinates of the :code:`num_points` points that have been sampled.

    """
    validate_int(num_points, "num_points", 0)
    validate_int(num_dimensions, "num_dimensions", 1)
    if bounds is not None:
        if isinstance(bounds, Iterable):
            bounds = np.array(bounds, dtype=float)
            if len(bounds) != num_dimensions:
                raise ValueError(
                    f"The number of bounds needs to be equal to the number of dimensions "
                    f"({len(bounds)} given for {num_dimensions} dimensions)."
                )
        else:
            raise TypeError(
                "The 'bounds' parameter needs to be an iterable with one value per dimension."
            )
    center_point = validate_center_point(center_point, num_dimensions)
    widths = validate_width(widths, num_dimensions)

    rng = np.random.default_rng(seed)
    if bounds is None:
        return rng.normal(
            loc=center_point, scale=widths, size=(num_points, num_dimensions)
        )
    else:
        points = []
        for _ in range(num_points):
            point = rng.normal(loc=center_point, scale=widths, size=num_dimensions)
            while not (np.abs(point - center_point) <= bounds).all():
                point = rng.normal(loc=center_point, scale=widths, size=num_dimensions)
            points.append(point)

        return np.array(points)
