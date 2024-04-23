from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from prefsampling.core.euclidean import sample_election_positions, EuclideanSpace
from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def euclidean_threshold(
    num_voters: int,
    num_candidates: int,
    threshold: float,
    num_dimensions: int,
    voters_positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    candidates_positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    voters_positions_args: dict = None,
    candidates_positions_args: dict = None,
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
        num_dimensions,
        voters_positions,
        candidates_positions,
        voters_positions_args,
        candidates_positions_args,
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
    num_dimensions: int,
    voters_positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    candidates_positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    voters_positions_args: dict = None,
    candidates_positions_args: dict = None,
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
        num_dimensions,
        voters_positions,
        candidates_positions,
        voters_positions_args,
        candidates_positions_args,
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
    num_dimensions: int,
    voters_positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    candidates_positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    voters_positions_args: dict = None,
    candidates_positions_args: dict = None,
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
        list[set[int]]
            Approval votes.
    """

    if rel_num_approvals < 0 or 1 < rel_num_approvals:
        raise ValueError(
            f"Incorrect value of rel_num_approvals: {rel_num_approvals}. Value should "
            f"be in [0, 1]"
        )

    voters_pos, candidates_pos = sample_election_positions(
        num_voters,
        num_candidates,
        num_dimensions,
        voters_positions,
        candidates_positions,
        voters_positions_args,
        candidates_positions_args,
        seed,
    )

    num_approvals = int(rel_num_approvals * num_candidates)
    votes = []
    for voter_pos in voters_pos:
        distances = np.array(
            [
                np.linalg.norm(voter_pos - candidates_pos[c])
                for c in range(num_candidates)
            ],
            dtype=float,
        )
        arg_sort_distances = distances.argsort()
        votes.append(set(arg_sort_distances[:num_approvals]))
    return votes
