from __future__ import annotations

from collections.abc import Callable

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def sample_election_positions(
    num_voters: int,
    num_candidates: int,
    point_sampler: Callable,
    point_sampler_args: dict,
    candidate_point_sampler: Callable = None,
    candidate_point_sampler_args: dict = None,
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
            The sampler used to sample point in the space. Used for both voters and candidates
            unless a `candidate_space` is provided.
        point_sampler_args : dict
            The arguments passed to the `point_sampler`. The argument `num_points` is ignored
            and replaced by the number of voters or candidates.
        candidate_point_sampler : Callable, default: :code:`None`
            The sampler used to sample the points of the candidates. If a value is provided,
            then the `space` argument is only used for voters.
        candidate_point_sampler_args : dict
            The arguments passed to the `candidate_point_sampler`. The argument `num_points`
            is ignored and replaced by the number of candidates.
        seed : int, default: :code:`None`
            Seed for numpy random number generator. Also passed to the point samplers if
            a value is provided.

    Returns
    -------
        tuple[np.ndarray, np.ndarray]
            The positions of the voters and of the candidates.

    """
    if candidate_point_sampler is not None and candidate_point_sampler_args is None:
        raise ValueError(
            "If candidate_point_sampler is not None, a value needs to be "
            "passed to candidate_point_sampler_args (even if it's just "
            "an empty dictionary)."
        )

    if seed is not None:
        point_sampler_args["seed"] = seed
        if candidate_point_sampler is not None:
            candidate_point_sampler_args["seed"] = seed

    point_sampler_args["num_points"] = num_voters
    voters_pos = point_sampler(**point_sampler_args)
    dimension = len(voters_pos[0])
    if candidate_point_sampler is None:
        point_sampler_args["num_points"] = num_candidates
        candidates_pos = point_sampler(**point_sampler_args)
    else:
        candidate_point_sampler_args["num_points"] = num_candidates
        candidates_pos = candidate_point_sampler(**candidate_point_sampler_args)
        if len(candidates_pos[0]) != dimension:
            raise ValueError(
                "The position of the voters and of the candidates do not have the "
                "same dimension. Use different point samplers to solve this "
                "problem."
            )
    return voters_pos, candidates_pos
