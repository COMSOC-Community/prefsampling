from __future__ import annotations

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def identity(
    num_voters: int, num_candidates: int, rel_num_approvals: float, seed: int = None
) -> list[set[int]]:
    """
    Generates approval votes from the identity culture. These votes are simples: all voters
    approves of the candidates `0, 1, 2, ...,  ⌊rel_num_approvals * num_candidates⌋` and only these
    ones.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        rel_num_approvals : float
            Proportion of approved candidates in a ballot.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    Raises
    ------
        ValueError
            When `rel_num_approvals` is not in the [0, 1] interval.
    """

    if rel_num_approvals < 0 or 1 < rel_num_approvals:
        raise ValueError(
            f"Incorrect value of rel_num_approvals: {rel_num_approvals}. Value should "
            f"be in [0, 1]"
        )

    k = int(rel_num_approvals * num_candidates)
    return [set(range(k)) for _ in range(num_voters)]


@validate_num_voters_candidates
def full(num_voters: int, num_candidates: int, seed: int = None) -> list[set[int]]:
    """
    Generates approval votes where all voters approve of all the candidates.

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    seed : int
        Seed for numpy random number generator.

    Returns
    -------
    list[set[int]]
        Approval votes.
    """
    return [set(range(num_candidates)) for _ in range(num_voters)]


@validate_num_voters_candidates
def empty(num_voters: int, num_candidates: int, seed: int = None) -> list[set[int]]:
    """
    Generates approval votes where each vote is empty.

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    seed : int
        Seed for numpy random number generator.

    Returns
    -------
    list[set[int]]
        Approval votes.
    """
    return [set() for _ in range(num_voters)]
