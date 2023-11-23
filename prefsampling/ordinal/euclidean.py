import numpy as np
from numpy import linalg

from prefsampling.core.euclidean import EUCLIDEAN_SPACE_UNIFORM, election_positions


def euclidean(
    num_voters: int,
    num_candidates: int,
    space: int = EUCLIDEAN_SPACE_UNIFORM,
    dimension: int = 2,
    seed: int = None,
) -> np.ndarray:
    """
    Generates ordinal votes according to the Euclidean model.

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    space : int
        Type of space considered. Should be a constant such as
        :py:const:`~prefsampling.core.euclidean.EUCLIDEAN_SPACE_UNIFORM`.
    dimension : int
        Number of dimensions for the sapce considered
    seed : int
        Seed for numpy random number generator.

    Returns
    -------
    np.ndarray
        The votes.

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
