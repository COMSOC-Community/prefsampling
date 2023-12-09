import numpy as np
from numpy import linalg

from prefsampling.core.euclidean import election_positions, EuclideanSpace
from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def euclidean(
    num_voters: int,
    num_candidates: int,
    space: EuclideanSpace = EuclideanSpace.UNIFORM,
    dimension: int = 2,
    seed: int = None,
) -> np.ndarray:
    """
    Generates ordinal votes according to the Euclidean model.

    In the Euclidean model voters and candidates are assigned random positions in a Euclidean space.
    A voter then ranks the candidates in increasing order of distance: their most preferred
    candidate is the closest one to them, etc.

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
        space : EuclideanSpace, default: :py:class:`~prefsampling.core.euclidean.EuclideanSpace.UNIFORM`
            Type of space considered. Should be a constant defined in the
            :py:class:`~prefsampling.core.euclidean.EuclideanSpace` enumeration.
        dimension : int, default: `2`
            Number of dimensions for the space considered
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.

    """
    rng = np.random.default_rng(seed)
    votes = np.zeros([num_voters, num_candidates], dtype=int)

    voters, candidates = election_positions(
        num_voters, num_candidates, space, dimension, rng
    )

    distances = np.zeros([num_voters, num_candidates], dtype=float)
    for i in range(num_voters):
        for j in range(num_candidates):
            distances[i][j] = np.linalg.norm(voters[i] - candidates[j], ord=dimension)
        votes[i] = np.argsort(distances[i])

    return votes
