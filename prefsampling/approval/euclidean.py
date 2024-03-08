from __future__ import annotations

from collections.abc import Callable

import numpy as np

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
    radius: float = 0.5,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes according to the Euclidean model.

    In the Euclidean model voters and candidates are assigned random positions in a Euclidean space.
    A voter then approves of the candidates that are within a certain radius. In other words, a
    voter approves of all the candidates that are with distance :code:`radius` of their position.

    Several Euclidean spaces can be considered. The possibilities are defined in the
    :py:class:`~prefsampling.core.euclidean.EuclideanSpace` enumeration. You can also change the
    dimension with the parameter :code:`dimension`.

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
        radius : float, default: 0.5
            Radius of approval.
        seed : int, default: :code:`None`
            Seed for numpy random number generator. Also passed to the point samplers if
            a value is provided.

    Returns
    -------
        list[set[int]]
            Approval votes.
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

    votes = [set() for _ in range(num_voters)]
    for v in range(num_voters):
        for c in range(num_candidates):
            if radius >= np.linalg.norm(voters_pos[v] - candidates_pos[c]):
                votes[v].add(c)

    return votes
