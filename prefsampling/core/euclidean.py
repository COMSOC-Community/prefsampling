from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


def _sample_points(
    num_points: int,
    sampler: Callable,
    sampler_args: dict,
    positions: Iterable[float],
    sampled_object_name: str,
) -> np.ndarray:
    if positions is None:
        if sampler is None:
            raise ValueError(f"You need to either provide a sampler for the {sampled_object_name} "
                             f"or their positions.")
        if sampler_args is None:
            sampler_args = dict()
        sampler_args["num_points"] = num_points
        positions = sampler(**sampler_args)
    else:
        positions = np.array(positions)
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
    point_sampler: Callable = None,
    point_sampler_args: dict = None,
    candidate_point_sampler: Callable = None,
    candidate_point_sampler_args: dict = None,
    voters_positions: Iterable[float] = None,
    candidates_positions: Iterable[float] = None,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        point_sampler : Callable
            The sampler used to sample point in the space. It should be a function accepting
            arguments 'num_points' and 'seed'. Used for both voters and candidates unless a
            `candidate_space` is provided.
        point_sampler_args : dict
            The arguments passed to the `point_sampler`. The argument `num_points` is ignored
            and replaced by the number of voters or candidates.
        candidate_point_sampler : Callable, default: :code:`None`
            The sampler used to sample the points of the candidates. It should be a function
            accepting  arguments 'num_points' and 'seed'. If a value is provided, then the
            `point_sampler_args` argument is only used for voters.
        candidate_point_sampler_args : dict
            The arguments passed to the `candidate_point_sampler`. The argument `num_points`
            is ignored and replaced by the number of candidates.
        voters_positions : Iterable[float]
            Position of the voters.
        candidates_positions : Iterable[float]
            Position of the candidates.
        seed : int, default: :code:`None`
            Seed for numpy random number generator. Also passed to the point samplers if
            a value is provided.

    Returns
    -------
        tuple[np.ndarray, np.ndarray]
            The positions of the voters and of the candidates.

    """
    if point_sampler_args is None:
        point_sampler_args = dict()
    if seed is not None:
        point_sampler_args["seed"] = seed
        if candidate_point_sampler is not None:
            candidate_point_sampler_args["seed"] = seed

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
