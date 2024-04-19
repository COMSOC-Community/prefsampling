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


def euclidean_space_to_sampler(space: EuclideanSpace, num_dimensions: int):
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
    sampler: Callable,
    sampler_args: dict,
    positions: Iterable[float],
    sampled_object_name: str,
) -> np.ndarray:
    if positions is None:
        if sampler is None:
            raise ValueError(
                f"You need to either provide a sampler for the {sampled_object_name} "
                f"or their positions."
            )
        if sampler_args is None:
            sampler_args = dict()
        sampler_args["num_points"] = num_points
        positions = sampler(**sampler_args)
    else:
        positions = np.array(positions, dtype=float)
        if len(positions) != num_points:
            raise ValueError(
                f"The provided number of points does not match the number of "
                f"{sampled_object_name} required ({len(positions)} points provided for"
                f"{num_points} {sampled_object_name}."
            )
    return positions


@validate_num_voters_candidates
def sample_election_positions(
    num_voters: int,
    num_candidates: int,
    euclidean_space: EuclideanSpace = None,
    candidate_euclidean_space: EuclideanSpace = None,
    num_dimensions: int = None,
    point_sampler: Callable = None,
    point_sampler_args: dict = None,
    candidate_point_sampler: Callable = None,
    candidate_point_sampler_args: dict = None,
    voters_positions: Iterable[Iterable[float]] = None,
    candidates_positions: Iterable[Iterable[float]] = None,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        euclidean_space: EuclideanSpace, default: :code:`None`
            Use a pre-defined Euclidean space for sampling the position of the voters. If no
            `candidate_euclidean_space` is provided, the value of 'euclidean_space' is used for the
            candidates as well. A number of dimension needs to be provided.
        candidate_euclidean_space: EuclideanSpace, default: :code:`None`
            Use a pre-defined Euclidean space for sampling the position of the candidates. If no
            value is provided, the value of 'euclidean_space' is used. A number of dimension needs
            to be provided.
        num_dimensions: int, default: :code:`None`
            The number of dimensions to use. Using this argument is mandatory when passing a space
            as argument. If you pass samplers as arguments and use the num_dimensions, then, the
            value of num_dimensions is passed as a kwarg to the samplers.
        point_sampler : Callable, default: :code:`None`
            The sampler used to sample point in the space. It should be a function accepting
            arguments 'num_points' and 'seed'. Used for both voters and candidates unless a
            `candidate_space` is provided.
        point_sampler_args : dict, default: :code:`None`
            The arguments passed to the `point_sampler`. The argument `num_points` is ignored
            and replaced by the number of voters or candidates.
        candidate_point_sampler : Callable, default: :code:`None`
            The sampler used to sample the points of the candidates. It should be a function
            accepting  arguments 'num_points' and 'seed'. If a value is provided, then the
            `point_sampler_args` argument is only used for voters.
        candidate_point_sampler_args : dict
            The arguments passed to the `candidate_point_sampler`. The argument `num_points`
            is ignored and replaced by the number of candidates.
        voters_positions : Iterable[Iterable[float]]
            Position of the voters.
        candidates_positions : Iterable[Iterable[float]]
            Position of the candidates.
        seed : int, default: :code:`None`
            Seed for numpy random number generator. Also passed to the point samplers if
            a value is provided.

    Returns
    -------
        tuple[np.ndarray, np.ndarray]
            The positions of the voters and of the candidates.

    """
    if euclidean_space:
        if num_dimensions is None:
            raise ValueError(
                "If you are using the 'euclidean_space' argument, you need to also "
                "provide a number of dimensions."
            )
        validate_int(num_dimensions, "number of dimensions", 1)
        if isinstance(euclidean_space, Enum):
            euclidean_space = EuclideanSpace(euclidean_space.value)
        else:
            euclidean_space = EuclideanSpace(euclidean_space)

        point_sampler, point_sampler_args = euclidean_space_to_sampler(
            euclidean_space, num_dimensions
        )
    if candidate_euclidean_space:
        if num_dimensions is None:
            raise ValueError(
                "If you are using the 'candidate_euclidean_space' argument, you need "
                "to also provide a number of dimensions."
            )
        if isinstance(candidate_euclidean_space, Enum):
            candidate_euclidean_space = EuclideanSpace(candidate_euclidean_space.value)
        else:
            candidate_euclidean_space = EuclideanSpace(candidate_euclidean_space)

        candidate_point_sampler, candidate_point_sampler_args = (
            euclidean_space_to_sampler(candidate_euclidean_space, num_dimensions)
        )

    if point_sampler_args is None:
        point_sampler_args = dict()
    if seed is not None:
        point_sampler_args["seed"] = seed
        if candidate_point_sampler is not None:
            candidate_point_sampler_args["seed"] = seed
    if num_dimensions is not None:
        point_sampler_args["num_dimensions"] = num_dimensions
        if candidate_point_sampler is not None:
            candidate_point_sampler_args["num_dimensions"] = num_dimensions

    voters_pos = _sample_points(
        num_voters, point_sampler, point_sampler_args, voters_positions, "voters"
    )
    dimension = len(voters_pos[0])
    if candidate_point_sampler:
        point_sampler = candidate_point_sampler
        point_sampler_args = candidate_point_sampler_args
    cand_pos = _sample_points(
        num_candidates,
        point_sampler,
        point_sampler_args,
        candidates_positions,
        "candidates",
    )

    if len(cand_pos[0]) != dimension:
        raise ValueError(
            "The position of the voters and of the candidates do not have the same dimension ("
            f"{dimension} for the voters and {len(cand_pos[0])} for the candidates)."
        )

    return voters_pos, cand_pos
