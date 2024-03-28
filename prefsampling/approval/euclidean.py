from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from prefsampling.core.euclidean import sample_election_positions
from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def euclidean_threshold(
    num_voters: int,
    num_candidates: int,
    threshold: float,
    point_sampler: Callable = None,
    point_sampler_args: dict = None,
    candidate_point_sampler: Callable = None,
    candidate_point_sampler_args: dict = None,
    voters_positions: Iterable[float] = None,
    candidates_positions: Iterable[float] = None,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes according to the threshold Euclidean model.

    In this model voters and candidates are assigned random positions in a Euclidean space
    (positions can also be provided as argument to the function).
    A voter then approves of the candidates that are at a distance no greater tha
    `min_d * threshold` where `min_d` is the minimum distance between the voter and any candidates.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above (as long as the point distribution is independent and identical).

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        threshold : float
            Threshold of approval. Voters approve all candidates that are at distance threshold
            times minimum distance between the voter and any candidates. This value should be 1 or
            more.
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
        list[set[int]]
            Approval votes.
    """
    if threshold < 1:
        raise ValueError(
            f"Threshold cannot be lower than 1 (current value: {threshold})."
        )

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

    votes = []
    for voter_pos in voters_pos:
        distances = [
            np.linalg.norm(voter_pos - candidates_pos[c]) for c in range(num_candidates)
        ]
        min_dist = min(distances)
        votes.append(
            {c for c, dist in enumerate(distances) if dist <= min_dist * threshold}
        )
    return votes


@validate_num_voters_candidates
def euclidean_vcr(
    num_voters: int,
    num_candidates: int,
    voters_radius: float | Iterable[float],
    candidates_radius: float | Iterable[float],
    point_sampler: Callable = None,
    point_sampler_args: dict = None,
    candidate_point_sampler: Callable = None,
    candidate_point_sampler_args: dict = None,
    voters_positions: Iterable[float] = None,
    candidates_positions: Iterable[float] = None,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes according to the voters and candidates range Euclidean model.

    In this model voters and candidates are assigned random positions in a Euclidean space
    (positions can also be provided as argument to the function).
    The voters and the candidates have a radius (can be the set agent per agent, or globally).
    A voter approves of all the candidates that are at distance no more than
    `voter_radius + candidate_radius`, where these two values can be agent-specific. It models the
    idea that a voter approves of a candidate if and only if their respective influence spheres
    overlap.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above (as long as the point distribution is independent and identical).

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        voters_radius : float | Iterable[float]
            Radius of approval. Voters approve all candidates for which the two balls centered in
            the position of the voter and the candidate of radius voter_radius and candidate_radius
            overlap. If a single value is given, it applies to all voters. Otherwise, it is assumed
            that one value per voter is provided.
        candidates_radius : float | Iterable[float]
            Radius of approval. Voters approve all candidates for which the two balls centered in
            the position of the voter and the candidate of radius voter_radius and candidate_radius
            overlap. If a single value is given, it applies to all voters. Otherwise, it is assumed
            that one value per voter is provided.
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
        list[set[int]]
            Approval votes.
    """

    if isinstance(voters_radius, Iterable):
        voters_radius = np.array(voters_radius, dtype=float)
        if len(voters_radius) != num_voters:
            raise ValueError(
                "If the 'voter_radius' parameter is an iterable, it needs to have one "
                f"element per voter ({len(voters_radius)} provided for num_voters="
                f"{num_voters}"
            )
    else:
        voters_radius = np.array(
            [voters_radius for _ in range(num_voters)], dtype=float
        )
    if isinstance(candidates_radius, Iterable):
        candidates_radius = np.array(candidates_radius, dtype=float)
        if len(candidates_radius) != num_candidates:
            raise ValueError(
                "If the 'candidates_radius' parameter is an iterable, it needs to "
                f"have one element per candidate ({len(candidates_radius)} provided "
                f"for num_candidates={num_candidates}"
            )
    else:
        candidates_radius = np.array(
            [candidates_radius for _ in range(num_candidates)], dtype=float
        )

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
    votes = []
    for v, voter_pos in enumerate(voters_pos):
        ballot = set()
        radius = voters_radius[v]
        for c in range(num_candidates):
            distance = np.linalg.norm(voter_pos - candidates_pos[c])
            if distance <= radius + candidates_radius[c]:
                ballot.add(c)
        votes.append(ballot)
    return votes


@validate_num_voters_candidates
def euclidean_constant_size(
    num_voters: int,
    num_candidates: int,
    rel_num_approvals: float,
    point_sampler: Callable = None,
    point_sampler_args: dict = None,
    candidate_point_sampler: Callable = None,
    candidate_point_sampler_args: dict = None,
    voters_positions: Iterable[float] = None,
    candidates_positions: Iterable[float] = None,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes according to the constant size Euclidean model.

    In this model voters and candidates are assigned random positions in a Euclidean space
    (positions can also be provided as argument to the function).
    A voter then approves of the `rel_num_approvals * num_candidates` the closest candidates to
    their position. This ensures that all approval ballots have length `⌊rel_num_approvals *
    num_candidates⌋`.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above (as long as the point distribution is independent and identical).

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        rel_num_approvals : float
            Proportion of approved candidates in a ballot.
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
        voters_positions,
        candidates_positions,
        seed,
    )

    num_approvals = int(rel_num_approvals * num_candidates)
    votes = []
    for voter_pos in voters_pos:
        distances = np.array(
            [
                np.linalg.norm(voter_pos - candidates_pos[c])
                for c in range(num_candidates)
            ]
        )
        arg_sort_distances = distances.argsort()
        votes.append(set(arg_sort_distances[:num_approvals]))
    return votes
