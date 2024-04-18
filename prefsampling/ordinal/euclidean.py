from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy import linalg

from prefsampling.core.euclidean import sample_election_positions, EuclideanSpace
from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def euclidean(
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
        euclidean_space: :py:class:`~prefsampling.core.euclidean.EuclideanSpace`, default: :code:`None`
            Use a pre-defined Euclidean space for sampling the position of the voters. If no
            `candidate_euclidean_space` is provided, the value of 'euclidean_space' is used for the
            candidates as well. A number of dimension needs to be provided.
        candidate_euclidean_space: :py:class:`~prefsampling.core.euclidean.EuclideanSpace`, default: :code:`None`
            Use a pre-defined Euclidean space for sampling the position of the candidates. If no
            value is provided, the value of 'euclidean_space' is used. A number of dimension needs
            to be provided.
        num_dimensions: int, default: :code:`None`
            The number of dimensions to use. Using this argument is mandatory when passing a space
            as argument.
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
        np.ndarray
            Ordinal votes.

    Examples
    --------

        The easiest is to use one of the Euclidean spaces defined in
        :py:class:`~prefsampling.core.euclidean.EuclideanSpace`.

        .. testcode::

            from prefsampling.ordinal import euclidean
            from prefsampling.core.euclidean import EuclideanSpace

            # Here for 2 voters and 3 candidates with uniform ball
            euclidean(2, 3, euclidean_space = EuclideanSpace.UNIFORM_BALL, num_dimensions=3)

            # You can use different spaces for the voters and the candidates
            euclidean(
                2,
                3,
                num_dimensions=3
                euclidean_space = EuclideanSpace.UNIFORM_SPHERE,
                euclidean_space = EuclideanSpace.GAUSSIAN_CUBE,
                )

            # Don't forget to pass the number of dimensions
            try:
                euclidean(2, 3, euclidean_space = EuclideanSpace.UNBOUNDED_GAUSSIAN)
            except ValueError:
                pass

        If you need more flexibility, you can also pass the point samplers directly.

        .. testcode::

            from prefsampling.ordinal import euclidean
            from prefsampling.point import ball_uniform

            # Here for 2 voters and 3 candidates with uniform ball
            euclidean(2, 3, num_dimensions=5, point_sampler = ball_uniform)

            # You can specify additional arguments to the point sampler
            euclidean(
                2,
                3,
                num_dimensions=5,  # can be here or in the point_sampler_args
                point_sampler = ball_uniform,
                point_sampler_args = {'widths': (1, 3, 2, 4, 2)}
            )

            # You can also specify different point samplers for voters and candidates
            from prefsampling.point import cube

            euclidean(
                2,
                3,
                num_dimensions=2,
                point_sampler = ball_uniform,
                point_sampler_args = {'widths': (3, 1), 'only_envelope': True},
                point_sampler_candidates = cube,
                point_sampler_candidates_args = {'center_point': (0.5, 1)}
            )

            # You can also mix the two methods.
            from prefsampling.core.euclidean import EuclideanSpace

            euclidean(
                2,
                3,
                num_dimensions=2,
                point_sampler = ball_uniform,
                point_sampler_args = {'widths': (3, 1), 'only_envelope': True},
                euclidean_space_candidates = EuclideanSpace.UNIFORM_CUBE,
            )

        If you already have positions for the voters or the candidates, you can also pass them to
        the sampler.

        .. testcode::

            from prefsampling.ordinal import euclidean
            from prefsampling.point import gaussian
            from prefsampling.core.euclidean import EuclideanSpace

            # First sampler positions of the 3 candidates in 2 dimensions
            candidates_positions = gaussian(3, 2, sigmas=(0.4, 0.8), widths=(5, 1))

            # Then sample preferences for 2 voters based on the candidates positions
            euclidean(
                2,
                3,
                num_dimensions=2,
                euclidean_space=EuclideanSpace.GAUSSIAN_BALL,
                candidates_positions=candidates_positions  # use voters_positions for voters
            )

    Validation
    ----------

        There is no known expression for the probability distribution governing Euclidean models.

    References
    ----------
        `The spatial theory of voting: An introduction
        <https://www.cambridge.org/nl/universitypress/subjects/politics-international-relations/political-theory/spatial-theory-voting-introduction>`_,
        *Enelow, James M., and Melvin J. Hinich*,
        Cambridge University Press, 1984.
    """

    voters_pos, candidates_pos = sample_election_positions(
        num_voters,
        num_candidates,
        euclidean_space,
        candidate_euclidean_space,
        num_dimensions,
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
