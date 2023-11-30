import numpy as np
from numpy import linalg

from prefsampling.core.euclidean import EUCLIDEAN_SPACE_UNIFORM, election_positions
from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def euclidean(
    num_voters: int,
    num_candidates: int,
    space: int = EUCLIDEAN_SPACE_UNIFORM,
    dimension: int = 2,
    seed: int = None,
) -> np.ndarray:
    """
    Generates ordinal votes according to the Euclidean model.

    In the Euclidean model voters and candidates are assigned random positions in a Euclidean space. A voter then
    ranks the candidates in increasing order of distance: their most preferred candidate is the closest one to them,
    etc.

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
