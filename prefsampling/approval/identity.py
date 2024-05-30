"""
Identity samplers are not fascinating per se as all voters have the same preferences. There are
useful tools however, for instance when using them in mixtures.
"""

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
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    Examples
    --------

        .. testcode::

            from prefsampling.approval import identity

            # Sample a unanimous profile with 2 voters and 3 candidates.
            # Voters approve 60% of the candidates (1 in this case).
            identity(2, 3, 0.6)

            # The seed will not change anything here, but you can still set it.
            identity(2, 3, 0.6, seed=1002)

            # Parameter rel_num_approvals needs to be in [0, 1]
            try:
                identity(2, 3, 1.6)
            except ValueError:
                pass
            try:
                identity(2, 3, -0.6)
            except ValueError:
                pass

    Validation
    ----------
        Validation is trivial here, we thus omit it.

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
    seed : int, default: :code:`None`
        Seed for numpy random number generator.

    Returns
    -------
    list[set[int]]
        Approval votes.

    Examples
    --------

        .. testcode::

            from prefsampling.approval import full

            # "Sample" a profile of 2 voters approving all candidates.
            full(2, 3)

            # The seed will not change anything here, but you can still set it.
            full(2, 3, seed=1002)

    Validation
    ----------
        Validation is trivial here, we thus omit it.
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
    seed : int, default: :code:`None`
        Seed for numpy random number generator.

    Returns
    -------
    list[set[int]]
        Approval votes.

    Examples
    --------

        .. testcode::

            from prefsampling.approval import empty

            # "Sample" a profile of 2 voters approving of no candidates.
            # The number of candidates is not used but must be given to keep
            # the samplers' signatures consistent.
            empty(2, 3)

            # The seed will not change anything here, but you can still set it.
            empty(2, 3, seed=1002)

    Validation
    ----------
        Validation is trivial here, we thus omit it.
    """
    return [set() for _ in range(num_voters)]
