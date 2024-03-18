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
    validate_int(num_points, "num_points", 0)
    validate_int(num_dimensions, "num_dimensions", 1)
    if bounds is not None:
        if isinstance(bounds, Iterable):
            bounds = np.array(bounds, dtype=float)
            if len(bounds) != num_dimensions:
                raise ValueError(
                    f"The number of bounds needs to be equal to the number of dimensions ({len(bounds)} given "
                    f"for {num_dimensions} dimensions)."
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
