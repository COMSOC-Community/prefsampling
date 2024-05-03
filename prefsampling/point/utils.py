from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def validate_center_point(
    center_point: Iterable[float], num_dimensions: int
) -> np.ndarray[float]:
    """
    Validates the center_point parameter for the point samplers and return a suitable object to be
    used by the samplers.

    Parameters
    ----------
        center_point: Iterable[float]
            The center_point parameter passed to the samplers.
        num_dimensions: int
            The number of dimensions

    Returns
    -------
        np.ndarray[float]
            The coordinates of the center point to be used by the point samplers.
    """
    if center_point is None:
        center_point = np.array([0 for _ in range(num_dimensions)], dtype=float)
    else:
        if not isinstance(center_point, Iterable):
            raise TypeError(
                "The 'center_point' parameter needs to be an iterable with one value per dimension."
            )
        center_point = np.array(center_point, dtype=float)
        if len(center_point) != num_dimensions:
            raise ValueError(
                f"The center point needs to have one coordinate per dimensions "
                f"({len(center_point)} given for {num_dimensions} dimensions)."
            )
    return center_point


def validate_width(
    widths: float | Iterable[float], num_dimensions: int
) -> np.ndarray(float):
    """
    Validates the width parameter for the point samplers and return a suitable object to be
    used by the samplers.

    Parameters
    ----------
        widths: float | Iterable[float]
            The width parameter passed to the samplers.
        num_dimensions: int
            The number of dimensions

    Returns
    -------
        np.ndarray[float]
            The widths to be used by the point samplers, one per dimension.
    """
    if isinstance(widths, Iterable) and not isinstance(widths, str):
        widths = np.array(widths, dtype=float)
        if len(widths) != num_dimensions:
            raise ValueError(
                f"If the width is an iterable, it needs to have as many elements as there are "
                f"dimensions ({len(widths)} elements given for {num_dimensions} dimensions)."
            )
    else:
        try:
            widths = float(widths)
        except ValueError:
            raise TypeError("If the width is not an iterable, it needs to be a float.")
        widths = np.array([widths for _ in range(num_dimensions)], dtype=float)
    return widths
