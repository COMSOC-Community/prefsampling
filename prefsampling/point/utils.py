from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def validate_center_point(
    center_point: Iterable[float], num_dimensions: int
) -> np.ndarray[float]:
    if center_point is None:
        center_point = np.array([0 for _ in range(num_dimensions)])
    else:
        if not isinstance(center_point, Iterable):
            raise TypeError(
                "The 'center_point' parameter needs to be an iterable with one value per dimension."
            )
        center_point = np.array(center_point, dtype=float)
        if len(center_point) != num_dimensions:
            raise ValueError(
                f"The center point needs to have one coordinate per dimensions ({len(center_point)} given "
                f"for {num_dimensions} dimensions)."
            )
    return center_point


def validate_width(
    widths: float | Iterable[float], num_dimensions: int
) -> np.ndarray(float):
    if isinstance(widths, Iterable):
        widths = np.array(widths, dtype=float)
        if len(widths) != num_dimensions:
            raise ValueError(
                f"If the width is an iterable, it needs to have as many elements as there are dimensions "
                f"({len(widths)} elements given for {num_dimensions} dimensions)."
            )
    else:
        try:
            widths = float(widths)
        except TypeError:
            raise TypeError("If the width is not an iterable, it needs to be a float.")
        widths = np.array([widths for _ in range(num_dimensions)])
    return widths
