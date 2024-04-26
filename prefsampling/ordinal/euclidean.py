"""
In Euclidean models, the voters and the candidates are assigned random positions in a given space.
The preferences of a voter are then defined based on the distance between the voter and the
candidates.
"""

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
    num_dimensions: int,
    voters_positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    candidates_positions: EuclideanSpace | Callable | Iterable[Iterable[float]],
    voters_positions_args: dict = None,
    candidates_positions_args: dict = None,
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
        np.ndarray
            Ordinal votes.

    Examples
    --------

        **Using** :py:class:`~prefsampling.core.euclidean.EuclideanSpace`

        The easiest is to use one of the Euclidean spaces defined in
        :py:class:`~prefsampling.core.euclidean.EuclideanSpace`.

        .. testcode::

            from prefsampling.ordinal import euclidean
            from prefsampling.core.euclidean import EuclideanSpace

            # Here for 2 voters and 3 candidates with 5D uniform ball for both voters and candidates
            euclidean(2, 3, 5, EuclideanSpace.UNIFORM_BALL, EuclideanSpace.UNIFORM_BALL)

            # You can use different spaces for the voters and the candidates
            euclidean(
                2,
                3,
                5,
                EuclideanSpace.UNIFORM_SPHERE,
                EuclideanSpace.GAUSSIAN_CUBE,
                )

        **Using** :py:mod:`prefsampling.point`

        If you need more flexibility, you can also pass the point samplers directly.

        .. testcode::

            from prefsampling.ordinal import euclidean
            from prefsampling.point import ball_uniform

            # Here for 2 voters and 3 candidates with 5D uniform ball for both voters and candidates
            euclidean(2, 3, 5, ball_uniform, ball_uniform)

            # You can specify additional arguments to the point sampler
            euclidean(
                2,
                3,
                5,
                ball_uniform,
                ball_uniform,
                voters_positions_args = {'widths': (1, 3, 2, 4, 2)}
            )

            # You can also specify different point samplers for voters and candidates
            from prefsampling.point import cube

            euclidean(
                2,
                3,
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

            from prefsampling.ordinal import euclidean
            from prefsampling.point import gaussian
            from prefsampling.core.euclidean import EuclideanSpace

            # First sampler positions of the 3 candidates in 2 dimensions
            candidates_positions = gaussian(3, 2, sigmas=(0.4, 0.8), widths=(5, 1))

            # Then sample preferences for 2 voters based on the candidates positions
            euclidean(
                2,
                3,
                2,
                EuclideanSpace.GAUSSIAN_BALL,
                candidates_positions
            )

    Validation
    ----------

        There is no known expression for the probability distribution governing Euclidean models.

        **With a Single Voter**

        Still, if there is a single voter, we know that we should obtain a uniform distribution
        over all rankings.

        .. image:: ../validation_plots/ordinal/euclidean_uniform_UNIFORM_BALL.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a ball Euclidean model with n=1

        .. image:: ../validation_plots/ordinal/euclidean_uniform_UNIFORM_SPHERE.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a sphere Euclidean model with n=1

        .. image:: ../validation_plots/ordinal/euclidean_uniform_UNIFORM_CUBE.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a cube Euclidean model with n=1

        .. image:: ../validation_plots/ordinal/euclidean_uniform_GAUSSIAN_BALL.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Gaussian ball Euclidean model with n=1

        .. image:: ../validation_plots/ordinal/euclidean_uniform_GAUSSIAN_CUBE.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Gaussian cube Euclidean model with n=1

        .. image:: ../validation_plots/ordinal/euclidean_uniform_UNBOUNDED_GAUSSIAN.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Gaussian Euclidean model with n=1

        **With Fixed Candidates Positions on the Line**

        If the positions of the candidates are fixed, the probability distribution can be computed
        by considering the size of the hyperplanes separating two candidates. We apply this method
        in one dimension to validate the sampler.

        .. image:: ../validation_plots/ordinal/euclidean_line_UNIFORM_BALL.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a ball Euclidean model with fixed candidates positions

        .. image:: ../validation_plots/ordinal/euclidean_line_UNIFORM_CUBE.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a cub Euclidean model with fixed candidates positions

        .. image:: ../validation_plots/ordinal/euclidean_line_UNBOUNDED_GAUSSIAN.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Gaussian Euclidean model with fixed candidates positions

        **In General**

        In the general case, we obtain the following distribution of frequencies.

        .. image:: ../validation_plots/ordinal/euclidean_UNIFORM_BALL.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a ball Euclidean model with n=3

        .. image:: ../validation_plots/ordinal/euclidean_UNIFORM_SPHERE.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a sphere Euclidean model with n=3

        .. image:: ../validation_plots/ordinal/euclidean_UNIFORM_CUBE.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a cube Euclidean model with n=3

        .. image:: ../validation_plots/ordinal/euclidean_GAUSSIAN_BALL.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Gaussian ball Euclidean model with n=3

        .. image:: ../validation_plots/ordinal/euclidean_GAUSSIAN_CUBE.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Gaussian cube Euclidean model with n=3

        .. image:: ../validation_plots/ordinal/euclidean_UNBOUNDED_GAUSSIAN.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Gaussian Euclidean model with n=3

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
        num_dimensions,
        voters_positions,
        candidates_positions,
        voters_positions_args,
        candidates_positions_args,
        seed=seed,
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
