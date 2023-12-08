import numpy as np

from prefsampling.core.euclidean import election_positions, EuclideanSpace
from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def euclidean(
    num_voters: int,
    num_candidates: int,
    space: EuclideanSpace = EuclideanSpace.UNIFORM,
    dimension: int = 2,
    radius: float = 0,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes from Euclidean model.

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    space : EuclideanSpace
        Type of space considered. Should be a constant defined in the
        :py:class:`~prefsampling.core.euclidean.EuclideanSpace`.
    dimension : int
        Number of Dimensions.
    radius : float
        The radius.
    seed : int
        Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.
    """
    rng = np.random.default_rng(seed)
    votes = [set() for _ in range(num_voters)]
    voters, candidates = election_positions(
        num_voters, num_candidates, space, dimension, rng
    )
    for v in range(num_voters):
        for c in range(num_candidates):
            if radius >= np.linalg.norm(voters[v] - candidates[c]):
                votes[v].add(c)

    return votes
