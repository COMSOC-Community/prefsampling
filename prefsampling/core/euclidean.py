from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates, validate_int
from prefsampling.point import ball_uniform, cube, ball_resampling, gaussian


class EuclideanSpace(Enum):
    """
    Constants for some pre-defined Euclidean distributions.
    """

    UNIFORM_BALL = "uniform_ball"
    UNIFORM_SPHERE = "uniform_sphere"
    UNIFORM_CUBE = "uniform_cube"
    GAUSSIAN_BALL = "gaussian_ball"
    GAUSSIAN_CUBE = "gaussian_cube"
    UNBOUNDED_GAUSSIAN = "unbounded_gaussian"


def euclidean_space_to_sampler(space: EuclideanSpace, num_dimensions: int) -> (Callable, dict):
    """
    Returns the point sampler together with its arguments corresponding to the EuclideanSpace
    passed as argument.
    """
    if space == EuclideanSpace.UNIFORM_BALL:
        return ball_uniform, {"num_dimensions": num_dimensions}
    if space == EuclideanSpace.UNIFORM_SPHERE:
        return ball_uniform, {"num_dimensions": num_dimensions, "only_envelope": True}
    if space == EuclideanSpace.UNIFORM_CUBE:
        return cube, {"num_dimensions": num_dimensions}
    if space == EuclideanSpace.GAUSSIAN_BALL:
        return ball_resampling, {
            "num_dimensions": num_dimensions,
            "inner_sampler": lambda **kwargs: gaussian(**kwargs)[0],
            "inner_sampler_args": {"num_dimensions": num_dimensions, "num_points": 1},
        }
    if space == EuclideanSpace.GAUSSIAN_CUBE:
        return gaussian, {
            "num_dimensions": num_dimensions,
            "widths": np.array([1 for _ in range(num_dimensions)]),
        }
    if space == EuclideanSpace.UNBOUNDED_GAUSSIAN:
        return gaussian, {"num_dimensions": num_dimensions}
    raise ValueError(
        "The 'euclidean_space' and/or the 'candidate_euclidean_space' arguments need to be one of "
        "the constant defined in the core.euclidean.EuclideanSpace enumeration. Choices are: "
        + ", ".join(str(s) for s in EuclideanSpace)
        + "."
    )


def _sample_points(
    num_points: int,
    num_dimensions: int,
    positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    positions_args: dict,
    sampled_object_name: str,
) -> np.ndarray:
    """
    Samples the points (if necessary) based on the input of the Euclidean function.
    """
    if isinstance(positions, Iterable):
        try:
            positions = np.array(positions, dtype=float)
        except Exception as e:
            msg = (
                "When trying to cast the provided positions to a numpy array, the above "
                "exception occurred..."
            )
            raise Exception(msg) from e

        if num_dimensions == 1:
            expected_shape = (num_points,)
        else:
            expected_shape = (num_points, num_dimensions)
        if positions.shape != expected_shape:
            raise ValueError(
                f"The provided positions do not match the expected shape. Shape is "
                f"{positions.shape} while {expected_shape} was expected "
                f"(num_{sampled_object_name}, num_dimensions)."
            )
        return positions

    if not isinstance(positions, Callable):
        try:
            if isinstance(positions, Enum):
                space = EuclideanSpace(positions.value)
            else:
                space = EuclideanSpace(positions)
        except Exception as e:
            msg = (
                f"If the positions for the {sampled_object_name} is not an Iterable (already, "
                f"given positions) or a Callable (a sampler), then it should be a "
                f"EuclideanSpace element. Casting the input to EuclideanSpace failed with the "
                f"above exception."
            )
            raise Exception(msg) from e

        positions, new_positions_args = euclidean_space_to_sampler(space, num_dimensions)
        new_positions_args.update(positions_args)
        positions_args = new_positions_args
    positions_args["num_points"] = num_points
    positions = np.array(positions(**positions_args))

    if positions.shape != (num_points, num_dimensions):
        raise ValueError(
            "After sampling the position, the obtained shape is not as expected. "
            f"Shape is {positions.shape} while {(num_points, num_dimensions)} was "
            f"expected (num_{sampled_object_name}, num_dimensions)."
        )

    return positions


@validate_num_voters_candidates
def sample_election_positions(
    num_voters: int,
    num_candidates: int,
    num_dimensions: int,
    voters_positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    candidates_positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    voters_positions_args: dict = None,
    candidates_positions_args: dict = None,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        num_dimensions: int
            The number of dimensions to use. Using this argument is mandatory when passing a space
            as argument. If you pass samplers as arguments and use the num_dimensions, then, the
            value of num_dimensions is passed as a kwarg to the samplers.
        voters_positions: py:class:`~prefsampling.core.euclidean.EuclideanSpace` | Callable | Iterable[Iterable[float]]
            The positions of the voters, or a way to determine them. If an Iterable is passed,
            then it is assumed to be the positions themselves. Otherwise, it is assumed that a
            sampler for the positions is passed. It can be either the nickname of a sampler---when
            passing a py:class:`~prefsampling.core.euclidean.EuclideanSpace`; or a sampler.
            A sampler is a function that takes as keywords arguments: 'num_points',
            'num_dimensions', and 'seed'. Additional arguments can be provided with by using the
            :code:`voters_positions_args` argument.
        candidates_positions: py:class:`~prefsampling.core.euclidean.EuclideanSpace` | Callable | Iterable[Iterable[float]]
            The positions of the candidates, or a way to determine them. If an Iterable is passed,
            then it is assumed to be the positions themselves. Otherwise, it is assumed that a
            sampler for the positions is passed. It can be either the nickname of a sampler---when
            passing a py:class:`~prefsampling.core.euclidean.EuclideanSpace`; or a sampler.
            A sampler is a function that takes as keywords arguments: 'num_points',
            'num_dimensions', and 'seed'. Additional arguments can be provided with by using the
            :code:`candidates_positions_args` argument.
        voters_positions_args: dict, default: :code:`dict()`
            Additional keyword arguments passed to the :code:`voters_positions` sampler when the
            latter is a Callable.
        candidates_positions_args: dict, default: :code:`dict()`
            Additional keyword arguments passed to the :code:`candidates_positions` sampler when the
            latter is a Callable.
        seed : int, default: :code:`None`
            Seed for numpy random number generator. Also passed to the point samplers if
            a value is provided.

    Returns
    -------
        tuple[np.ndarray, np.ndarray]
            The positions of the voters and of the candidates.

    """
    validate_int(num_dimensions, lower_bound=0, value_descr="number of dimensions")
    if voters_positions_args is None:
        voters_positions_args = dict()
    if candidates_positions_args is None:
        candidates_positions_args = dict()
    if seed is not None:
        voters_positions_args["seed"] = seed
        candidates_positions_args["seed"] = seed

    voters_positions_args["num_dimensions"] = num_dimensions
    candidates_positions_args["num_dimensions"] = num_dimensions

    voters_pos = _sample_points(
        num_voters, num_dimensions, voters_positions, voters_positions_args, "voters"
    )
    cand_pos = _sample_points(
        num_candidates,
        num_dimensions,
        candidates_positions,
        candidates_positions_args,
        "candidates",
    )
    return voters_pos, cand_pos
