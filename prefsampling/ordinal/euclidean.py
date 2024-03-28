from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy import linalg

from prefsampling.core.euclidean import sample_election_positions
from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def euclidean(
    num_voters: int,
    num_candidates: int,
    point_sampler: Callable = None,
    point_sampler_args: dict = None,
    candidate_point_sampler: Callable = None,
    candidate_point_sampler_args: dict = None,
    voters_positions: Iterable[float] = None,
    candidates_positions: Iterable[float] = None,
    seed: int = None,
) -> np.ndarray:
    """
    Generates approval votes according to the Euclidean model.

    In this model voters and candidates are assigned random positions in a Euclidean space
    (positions can also be provided as argument to the function).
    A voter then ranks the candidates in increasing order of distance: their most preferred
    candidate is the closest one to them, etc.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above (as long as the point distribution is independent and identical).
    Generates ordinal votes according to the Euclidean model.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
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
        voters_positions : Iterable[float]
            Position of the voters.
        candidates_positions : Iterable[float]
            Position of the candidates.
        seed : int, default: :code:`None`
            Seed for numpy random number generator. Also passed to the point samplers if
            a value is provided.

    Returns
    -------
        np.ndarray
            Ordinal votes.

    """

    voters_pos, candidates_pos = sample_election_positions(
        num_voters,
        num_candidates,
        point_sampler,
        point_sampler_args,
        candidate_point_sampler,
        candidate_point_sampler_args,
        voters_positions,
        candidates_positions,
        seed,
    )

    dimension = len(voters_pos[0])
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)
    for i in range(num_voters):
        for j in range(num_candidates):
            distances[i][j] = np.linalg.norm(
                voters_pos[i] - candidates_pos[j], ord=dimension
            )
        votes[i] = np.argsort(distances[i])

    return votes
