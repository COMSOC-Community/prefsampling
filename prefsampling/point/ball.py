from __future__ import annotations

import warnings
from collections.abc import Iterable, Callable

import numpy as np

from prefsampling.inputvalidators import validate_int
from prefsampling.point.utils import validate_center_point, validate_width


def ball_uniform(
    num_points: int,
    num_dimensions: int,
    center_point: Iterable[float] = None,
    widths: float | Iterable[float] = 1,
    only_envelope: bool = False,
    seed: int = None,
) -> np.ndarray:
    """
    Samples points uniformly at random in a ball. This function can also handle spheres and spheres
    of different width per dimensions, leading to distributions that are not really balls. For
    instance, you can obtain an ellipse by providing two different widths in the 2D case.

    Parameters
    ----------
        num_points: int
            The number of points to sample.
        num_dimensions: int
            The number of dimensions for the ball.
        center_point: Iterable[float]
            The coordinates of the center point of the ball. It needs to have one coordinate per
            dimension.
        widths: float | Iterable[float], default: :code:`1`
            The width of the ball. If a single value is given, the width is applied to all
            dimensions. In case multiple values are given, they are applied to each dimension
            independently.
        only_envelope: bool, default: :code:`False`
            If set to :code:`True` only points on the envelope of the ball (the corresponding
            sphere) are sampled, and not in the inside.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            The coordinates of the :code:`num_points` points that have been sampled.

    Validation
    ----------

        .. image:: ../validation_plots/point/ball-uniform.png
            :width: 800
            :alt: Observed frequencies for a uniform ball model
    """
    validate_int(num_points, "num_points", 0)
    validate_int(num_dimensions, "num_dimensions", 1)
    center_point = validate_center_point(center_point, num_dimensions)
    widths = validate_width(widths, num_dimensions)

    rng = np.random.default_rng(seed)

    random_directions = rng.normal(size=(num_dimensions, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    if only_envelope:
        random_radii = 1.0
    else:
        random_radii = rng.random(num_points) ** (1 / num_dimensions)
    return center_point + (widths / 2) * (random_directions * random_radii).T


def sphere_uniform(
    num_points: int,
    num_dimensions: int,
    center_point: Iterable[float] = None,
    widths: float | Iterable[float] = 1,
    seed: int = None,
) -> np.ndarray:
    """
    Samples points uniformly at random in a sphere, that is, in the envelope of a ball.

    This is simply a shortcut of the :py:func:`~prefsampling.point.ball_uniform` with
    :code:`only_envelope = True`.

    Parameters
    ----------
        num_points: int
            The number of points to sample.
        num_dimensions: int
            The number of dimensions for the ball.
        center_point: Iterable[float]
            The coordinates of the center point of the ball. It needs to have one coordinate per
            dimension.
        widths: float | Iterable[float], default: :code:`1`
            The width of the ball. If a single value is given, the width is applied to all
            dimensions. In case multiple values are given, they are applied to each dimension
            independently.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            The coordinates of the :code:`num_points` points that have been sampled.

    Validation
    ----------

        .. image:: ../validation_plots/point/sphere-uniform.png
            :width: 800
            :alt: Observed frequencies for a uniform sphere model
    """
    return ball_uniform(
        num_points, num_dimensions, center_point, widths, only_envelope=True, seed=seed
    )


def ball_resampling(
    num_points: int,
    num_dimensions: int,
    inner_sampler: Callable,
    inner_sampler_args: dict,
    center_point: Iterable[float] = None,
    width: float = 1,
    max_numer_resampling: int = 1000,
    seed: int = None,
) -> np.ndarray:
    """
    Uses another point sampler and reject all points that do not fall inside the ball described
    as parameter.

    Be careful, if not set properly, this can run until the end of time (if always resampling for
    instance).

    Parameters
    ----------
        num_points: int
            The number of points to sample.
        num_dimensions: int
            The number of dimensions for the ball.
        inner_sampler: Callable
            The function used to sample points before resampling. This function should accept an
            optional :code:`seed` parameter. Other parameters should be passed by name via the
            :code:`inner_sampler_args` argument.
        inner_sampler_args: dict
            The arguments passed to the `inner_sampler`.
        center_point: Iterable[float]
            The coordinates of the center point of the ball. It needs to have one coordinate per
            dimension.
        width: float, default: :code:`1`
            The width of the ball. Can only be a single value (as opposed to
            :py:func:`~prefsampling.point.ball.ball_uniform`).
        max_numer_resampling : int, default: :code:`1000`
            The maximum number of resampling. If exceeded, the center point is used.
        seed : int, default: :code:`None`
            Seed for numpy random number generator. We increase the seed by one each time we
            resample (to avoid always resampling the same point).

    Returns
    -------
        np.ndarray
            The coordinates of the :code:`num_points` points that have been sampled.

    Validation
    ----------

        .. image:: ../validation_plots/point/ball-resampling.png
            :width: 800
            :alt: Observed frequencies for a uniform ball model
    """
    validate_int(num_points, "num_points", 0)
    validate_int(num_dimensions, "num_dimensions", 1)
    center_point = validate_center_point(center_point, num_dimensions)

    if seed is not None:
        inner_sampler_args["seed"] = seed

    points = []
    for _ in range(num_points):
        point = inner_sampler(**inner_sampler_args)
        if isinstance(point, Iterable):
            point = np.array(point, dtype=float)
        else:
            point = np.array([point], dtype=float)
        if len(point) != num_dimensions:
            raise ValueError(
                f"The inner sampler did not return a point with the suitable number "
                f"of dimensions ({num_dimensions} needed but {len(point)} returned)."
            )
        num_resampling = 0
        while np.linalg.norm(point - center_point) > (width / 2):
            if seed is not None:
                seed += 1
                inner_sampler_args["seed"] = seed
            point = inner_sampler(**inner_sampler_args)
            num_resampling += 1
            if num_resampling == max_numer_resampling:
                warnings.warn(
                    "Too many resampling attempt for the resampling_ball. We used the center point "
                    "instead.",
                    RuntimeWarning,
                )
                point = center_point
                break
        points.append(point)
    return np.array(points, dtype=float)
