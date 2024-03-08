from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy import linalg

from prefsampling.core.euclidean import sample_election_positions
from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def euclidean(
    num_voters: int,
    num_candidates: int,
    point_sampler: Callable,
    point_sampler_args: dict,
    candidate_point_sampler: Callable = None,
    candidate_point_sampler_args: dict = None,
    seed: int = None,
) -> np.ndarray:
    """
    Generates ordinal votes according to the Euclidean model.

    In the Euclidean model voters and candidates are assigned random positions in a Euclidean space.
    A voter then ranks the candidates in increasing order of distance: their most preferred
    candidate is the closest one to them, etc.

    Several Euclidean spaces can be considered. The possibilities are defined in the
    :py:class:`~prefsampling.core.euclidean.EuclideanSpace` enumeration. You can also change the
    dimension with the parameter :code:`dimension`. Note that you can specify different spaces for
    the voters and for the candidates.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

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
        seed,
    )

    dimension = len(voters_pos[0])
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)
    for i in range(num_voters):
        for j in range(num_candidates):
            distances[i][j] = np.linalg.norm(voters_pos[i] - candidates_pos[j], ord=dimension)
        votes[i] = np.argsort(distances[i])

    return votes
