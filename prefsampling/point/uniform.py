from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from prefsampling.inputvalidators import validate_int
from prefsampling.point.utils import validate_center_point, validate_width


def uniform(
    num_points: int,
    num_dimensions: int,
    center_point: Iterable[float] = None,
    widths: float | Iterable[float] = 1,
    seed: int = None,
) -> np.ndarray:
    validate_int(num_points, "num_points", 0)
    validate_int(num_dimensions, "num_dimensions", 1)
    center_point = validate_center_point(center_point, num_dimensions)
    widths = validate_width(widths, num_dimensions)
    rng = np.random.default_rng(seed)
    return rng.random((num_points, num_dimensions)) * widths + center_point
