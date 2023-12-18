import numpy as np

from prefsampling.core.euclidean import election_positions, EuclideanSpace
from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def euclidean(
    num_voters: int,
    num_candidates: int,
    radius: float = 0.5,
    space: EuclideanSpace = EuclideanSpace.UNIFORM,
    dimension: int = 2,
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
        radius : float, default: 0.5
            Radius of approval.
        space : EuclideanSpace, default: :py:const:`~prefsampling.core.euclidean.EuclideanSpace.UNIFORM`
            Type of space considered. Should be a constant defined in the
            :py:class:`~prefsampling.core.euclidean.EuclideanSpace`.
        dimension : int, default: 2
            Number of Dimensions.
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
