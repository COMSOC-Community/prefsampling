from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from prefsampling.inputvalidators import validate_int


def ball(
    num_points: int,
    dimension: int,
    center_point: Iterable[float] = None,
    width: float | Iterable[float] = 1,
    only_enveloppe=False,
    seed: int = None,
) -> np.ndarray:
    """
    Samples points uniformly at random in a ball. This function can also handle spheres and spheres of different
    width per dimensions (discus).

    Parameters
    ----------
        num_points: int
            The number of points to sample.
        dimension: int
            The number of dimensions for the ball.
        center_point: Iterable[float]
            The coordinates of the center point of the ball.
        width: float | Iterable[float], defaults to :code:`1`
            The width of the ball. If a single value is given, the width is applied to all dimensions. In case multiple
            values are given, they are applied to each dimension independently.
        only_enveloppe
        seed

    Returns
    -------
        np.ndarray
            The coordinates of the :code:`num_points` points that have been sampled.

    """
    validate_int(num_points, "num_points", 0)
    validate_int(dimension, "dimension", 1)
    if center_point is None:
        center_point = np.array([0 for _ in range(dimension)])
    else:
        center_point = np.array(center_point)
        if len(center_point) != dimension:
            raise ValueError(
                f"The center point needs to have one coordinate per dimensions ({len(center_point)} given "
                f"for {dimension} dimensions)."
            )
    if isinstance(width, Iterable):
        width = np.array(width)
        if len(width) != dimension:
            raise ValueError(
                f"If the width is an iterable, it needs to have as many elements as there are dimensions "
                f"({len(width)} elements given for {dimension} dimensions)."
            )
    else:
        try:
            width = float(width)
        except TypeError:
            raise TypeError("If the width is not an iterable, it needs to be a float.")
        width = np.array([width for _ in range(dimension)])

    rng = np.random.default_rng(seed)

    random_directions = rng.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    if only_enveloppe:
        random_radii = 1.0
    else:
        random_radii = rng.random(num_points) ** (1 / dimension)
    return center_point + width * (random_directions * random_radii).T


def sphere(
    num_points: int,
    dimension: int,
    center_point: Iterable[float] = None,
    width: float | Iterable[float] = 1,
    seed: int = None,
):
    return ball(
        num_points, dimension, center_point, width, only_enveloppe=True, seed=seed
    )
