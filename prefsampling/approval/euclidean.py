"""
In Euclidean models, the voters and the candidates are assigned random positions in a given space.
The preferences of a voter are then defined based on the distance between the voter and the
candidates.
"""

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
        voters_positions: :py:class:`~prefsampling.core.euclidean.EuclideanSpace` | Callable | Iterable[Iterable[float]]
            The positions of the voters, or a way to determine them. If an Iterable is passed,
            then it is assumed to be the positions themselves. Otherwise, it is assumed that a
            sampler for the positions is passed. It can be either the nickname of a sampler---when
            passing a :py:class:`~prefsampling.core.euclidean.EuclideanSpace`; or a sampler.
            A sampler is a function that takes as keywords arguments: 'num_points',
            'num_dimensions', and 'seed'. Additional arguments can be provided with by using the
            :code:`voters_positions_args` argument.
        candidates_positions: :py:class:`~prefsampling.core.euclidean.EuclideanSpace` | Callable | Iterable[Iterable[float]]
            The positions of the candidates, or a way to determine them. If an Iterable is passed,
            then it is assumed to be the positions themselves. Otherwise, it is assumed that a
            sampler for the positions is passed. It can be either the nickname of a sampler---when
            passing a :py:class:`~prefsampling.core.euclidean.EuclideanSpace`; or a sampler.
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

    Examples
    --------

        **Using** :py:class:`~prefsampling.core.euclidean.EuclideanSpace`

        The easiest is to use one of the Euclidean spaces defined in
        :py:class:`~prefsampling.core.euclidean.EuclideanSpace`.

        .. testcode::

            from prefsampling.approval import euclidean_threshold
            from prefsampling.core.euclidean import EuclideanSpace

            # Here for 2 voters and 3 candidates with 5D uniform ball for both voters and candidates
            # with threshold 2.5
            euclidean_threshold(
                2,  # Number of voters
                3,  # Number of candidates
                2.5,  # Threshold value
                5,  # Number of dimensions
                EuclideanSpace.UNIFORM_BALL,  # For the voters
                EuclideanSpace.UNIFORM_BALL  # For the candidates
            )

            # You can use different spaces for the voters and the candidates
            euclidean_threshold(
                2,
                3,
                2.5,
                5,
                EuclideanSpace.UNIFORM_SPHERE,
                EuclideanSpace.GAUSSIAN_CUBE,
                )

        **Using** :py:mod:`prefsampling.point`

        If you need more flexibility, you can also pass the point samplers directly.

        .. testcode::

            from prefsampling.approval import euclidean_threshold
            from prefsampling.point import ball_uniform

            # Here for 2 voters and 3 candidates with 5D uniform ball for both voters and candidates
            # with threshold value 2.5
            euclidean_threshold(2, 3, 2.5, 5, ball_uniform, ball_uniform)

            # You can specify additional arguments to the point sampler
            euclidean_threshold(
                2,
                3,
                2.5,
                5,
                ball_uniform,
                ball_uniform,
                voters_positions_args = {'widths': (1, 3, 2, 4, 2)}
            )

            # You can also specify different point samplers for voters and candidates
            from prefsampling.point import cube

            euclidean_threshold(
                2,
                3,
                2.5,
                5,
                ball_uniform,
                ball_uniform,
                voters_positions_args = {'widths': (4, 7, 3, 3, 1), 'only_envelope': True},
                candidates_positions_args = {'center_point': (0.5, 1, 0, 0, 0)}
            )

        **Using already known-positions**

        If you already have positions for the voters or the candidates, you can also pass them to
        the sampler.

        .. testcode::

            from prefsampling.approval import euclidean_threshold
            from prefsampling.point import gaussian
            from prefsampling.core.euclidean import EuclideanSpace

            # First sampler positions of the 3 candidates in 2 dimensions
            candidates_positions = gaussian(3, 2, sigmas=(0.4, 0.8), widths=(5, 1))

            # Then sample preferences for 2 voters based on the candidates positions
            euclidean_threshold(
                2,
                3,
                2.5,
                2,
                EuclideanSpace.GAUSSIAN_BALL,
                candidates_positions
            )

    References
    ----------

        `An Analysis of Approval-Based Committee Rules for 2D-Euclidean Elections
        <https://ojs.aaai.org/index.php/AAAI/article/view/16686>`_,
        *Michał T. Godziszewski, Paweł Batko, Piotr Skowron and Piotr Faliszewski*,
        Proceedings of the AAAI Conference on Artificial Intelligence, 2021.

        `Price of Fairness in Budget Division and Probabilistic Social Choice
        <https://ojs.aaai.org/index.php/AAAI/article/view/5594>`_,
        *Marcin Michorzewski, Dominik Peters and Piotr Skowron*,
        Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

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
        voters_positions: :py:class:`~prefsampling.core.euclidean.EuclideanSpace` | Callable | Iterable[Iterable[float]]
            The positions of the voters, or a way to determine them. If an Iterable is passed,
            then it is assumed to be the positions themselves. Otherwise, it is assumed that a
            sampler for the positions is passed. It can be either the nickname of a sampler---when
            passing a :py:class:`~prefsampling.core.euclidean.EuclideanSpace`; or a sampler.
            A sampler is a function that takes as keywords arguments: 'num_points',
            'num_dimensions', and 'seed'. Additional arguments can be provided with by using the
            :code:`voters_positions_args` argument.
        candidates_positions: :py:class:`~prefsampling.core.euclidean.EuclideanSpace` | Callable | Iterable[Iterable[float]]
            The positions of the candidates, or a way to determine them. If an Iterable is passed,
            then it is assumed to be the positions themselves. Otherwise, it is assumed that a
            sampler for the positions is passed. It can be either the nickname of a sampler---when
            passing a :py:class:`~prefsampling.core.euclidean.EuclideanSpace`; or a sampler.
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

    Examples
    --------

        **Using** :py:class:`~prefsampling.core.euclidean.EuclideanSpace`

        The easiest is to use one of the Euclidean spaces defined in
        :py:class:`~prefsampling.core.euclidean.EuclideanSpace`.

        .. testcode::

            from prefsampling.approval import euclidean_vcr
            from prefsampling.core.euclidean import EuclideanSpace

            # Here for 2 voters and 3 candidates with 5D uniform ball for both voters and candidates
            # with radius 2.5 for the voters and 5.3 for the candidates
            euclidean_vcr(
                2,  # Number of voters
                3,  # Number of candidates
                2.5,  # Voters radius
                5.3,  # Candidates radius
                5,  # Number of dimensions
                EuclideanSpace.UNIFORM_BALL,  # For the voters
                EuclideanSpace.UNIFORM_BALL  # For the candidates
            )

            # You can use different spaces for the voters and the candidates
            euclidean_vcr(
                2,
                3,
                2.5,
                5.3,
                5,
                EuclideanSpace.UNIFORM_SPHERE,
                EuclideanSpace.GAUSSIAN_CUBE,
                )

            # You can provide individual radius
            euclidean_vcr(
                2,
                3,
                (2.5, 2.8),
                (5.3, 5.1, 5.9),
                5,
                EuclideanSpace.UNIFORM_SPHERE,
                EuclideanSpace.UNIFORM_CUBE,
                )

        **Using** :py:mod:`prefsampling.point`

        If you need more flexibility, you can also pass the point samplers directly.

        .. testcode::

            from prefsampling.approval import euclidean_vcr
            from prefsampling.point import ball_uniform

            # Here for 2 voters and 3 candidates with 5D uniform ball for both voters and candidates
            # with radius 2.5 for the voters and 5.3 for the candidates
            euclidean_vcr(2, 3, 2.5, 5.3, 5, ball_uniform, ball_uniform)

            # You can specify additional arguments to the point sampler
            euclidean_vcr(
                2,
                3,
                2.5,
                5.3,
                5,
                ball_uniform,
                ball_uniform,
                voters_positions_args = {'widths': (1, 3, 2, 4, 2)}
            )

            # You can also specify different point samplers for voters and candidates
            from prefsampling.point import cube

            euclidean_vcr(
                2,
                3,
                2.5,
                5.3,
                5,
                ball_uniform,
                ball_uniform,
                voters_positions_args = {'widths': (4, 7, 3, 3, 1), 'only_envelope': True},
                candidates_positions_args = {'center_point': (0.5, 1, 0, 0, 0)}
            )

        **Using already known-positions**

        If you already have positions for the voters or the candidates, you can also pass them to
        the sampler.

        .. testcode::

            from prefsampling.approval import euclidean_vcr
            from prefsampling.point import gaussian
            from prefsampling.core.euclidean import EuclideanSpace

            # First sampler positions of the 3 candidates in 2 dimensions
            candidates_positions = gaussian(3, 2, sigmas=(0.4, 0.8), widths=(5, 1))

            # Then sample preferences for 2 voters based on the candidates positions
            euclidean_vcr(
                2,
                3,
                2.5,
                5.3,
                2,
                EuclideanSpace.GAUSSIAN_BALL,
                candidates_positions
            )

    References
    ----------

        `An Experimental View on Committees Providing Justified Representation
        <https://www.ijcai.org/proceedings/2019/16>`_,
        *Robert Bredereck, Piotr Faliszewski, Andrzej Kaczmarczyk and Rolf Niedermeier*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2019.

        `How to Sample Approval Elections?
        <https://www.ijcai.org/proceedings/2022/71>`_,
        *Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
        Krzysztof Sornat and Nimrod Talmon*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2022.

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
        voters_positions: :py:class:`~prefsampling.core.euclidean.EuclideanSpace` | Callable | Iterable[Iterable[float]]
            The positions of the voters, or a way to determine them. If an Iterable is passed,
            then it is assumed to be the positions themselves. Otherwise, it is assumed that a
            sampler for the positions is passed. It can be either the nickname of a sampler---when
            passing a :py:class:`~prefsampling.core.euclidean.EuclideanSpace`; or a sampler.
            A sampler is a function that takes as keywords arguments: 'num_points',
            'num_dimensions', and 'seed'. Additional arguments can be provided with by using the
            :code:`voters_positions_args` argument.
        candidates_positions: :py:class:`~prefsampling.core.euclidean.EuclideanSpace` | Callable | Iterable[Iterable[float]]
            The positions of the candidates, or a way to determine them. If an Iterable is passed,
            then it is assumed to be the positions themselves. Otherwise, it is assumed that a
            sampler for the positions is passed. It can be either the nickname of a sampler---when
            passing a :py:class:`~prefsampling.core.euclidean.EuclideanSpace`; or a sampler.
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

    Examples
    --------

        **Using** :py:class:`~prefsampling.core.euclidean.EuclideanSpace`

        The easiest is to use one of the Euclidean spaces defined in
        :py:class:`~prefsampling.core.euclidean.EuclideanSpace`.

        .. testcode::

            from prefsampling.approval import euclidean_constant_size
            from prefsampling.core.euclidean import EuclideanSpace

            # Here for 2 voters and 3 candidates with 5D uniform ball for both voters and candidates
            # with relative size of the approval ballots 0.5
            euclidean_constant_size(
                2,  # Number of voters
                3,  # Number of candidates
                0.5,  # Relative size of the approval ballots
                5,  # Number of dimensions
                EuclideanSpace.UNIFORM_BALL,  # For the voters
                EuclideanSpace.UNIFORM_BALL  # For the candidates
            )

            # You can use different spaces for the voters and the candidates
            euclidean_constant_size(
                2,
                3,
                0.5,
                5,
                EuclideanSpace.UNIFORM_SPHERE,
                EuclideanSpace.GAUSSIAN_CUBE,
            )

            # The relative size of the approval ballots need to be in [0, 1]
            try:
                euclidean_constant_size(
                    2,
                    3,
                    1.5,
                    5,
                    EuclideanSpace.UNIFORM_SPHERE,
                    EuclideanSpace.GAUSSIAN_CUBE,
                )
            except ValueError:
                pass
            try:
                euclidean_constant_size(
                    2,
                    3,
                    -0.5,
                    5,
                    EuclideanSpace.UNIFORM_SPHERE,
                    EuclideanSpace.GAUSSIAN_CUBE,
                )
            except ValueError:
                pass

        **Using** :py:mod:`prefsampling.point`

        If you need more flexibility, you can also pass the point samplers directly.

        .. testcode::

            from prefsampling.approval import euclidean_constant_size
            from prefsampling.point import ball_uniform

            # Here for 2 voters and 3 candidates with 5D uniform ball for both voters and candidates
            # with relative size of the approval ballots 2.5
            euclidean_constant_size(2, 3, 0.5, 5, ball_uniform, ball_uniform)

            # You can specify additional arguments to the point sampler
            euclidean_constant_size(
                2,
                3,
                0.5,
                5,
                ball_uniform,
                ball_uniform,
                voters_positions_args = {'widths': (1, 3, 2, 4, 2)}
            )

            # You can also specify different point samplers for voters and candidates
            from prefsampling.point import cube

            euclidean_constant_size(
                2,
                3,
                0.5,
                5,
                ball_uniform,
                ball_uniform,
                voters_positions_args = {'widths': (4, 7, 3, 3, 1), 'only_envelope': True},
                candidates_positions_args = {'center_point': (0.5, 1, 0, 0, 0)}
            )

        **Using already known-positions**

        If you already have positions for the voters or the candidates, you can also pass them to
        the sampler.

        .. testcode::

            from prefsampling.approval import euclidean_threshold
            from prefsampling.point import gaussian
            from prefsampling.core.euclidean import EuclideanSpace

            # First sampler positions of the 3 candidates in 2 dimensions
            candidates_positions = gaussian(3, 2, sigmas=(0.4, 0.8), widths=(5, 1))

            # Then sample preferences for 2 voters based on the candidates positions
            euclidean_constant_size(
                2,
                3,
                0.5,
                2,
                EuclideanSpace.GAUSSIAN_BALL,
                candidates_positions
            )

    References
    ----------

        `Perpetual Voting: Fairness in Long-Term Decision Making
        <https://ojs.aaai.org/index.php/AAAI/article/view/5584>`_,
        *Martin Lackner*,
        Proceedings  of the AAAI Conference on Artificial Intelligence, 2020.

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
