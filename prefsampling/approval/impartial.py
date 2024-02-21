from __future__ import annotations

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates, validate_int


@validate_num_voters_candidates
def impartial(
    num_voters: int, num_candidates: int, p: float, seed: int = None
) -> list[set]:
    """
    Generates approval votes from impartial culture.

    Under the approval culture, when generating a single vote, each candidate has the same
    probability :code:`p` of being approved. This models ensures that the average number of
    approved candidate per voter is `p * num_candidate`.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        p : float
            Probability of approving of any given candidates.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set]
            Approval votes.

    Raises
    ------
        ValueError
            When `p` not in [0,1] interval.
    """

    if p < 0 or 1 < p:
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0, 1]")

    rng = np.random.default_rng(seed)

    votes = [
        set(j for j in range(num_candidates) if rng.random() <= p)
        for _ in range(num_voters)
    ]

    return votes


@validate_num_voters_candidates
def impartial_constant_size(
    num_voters: int, num_candidates: int, num_approvals: int, seed: int = None
) -> list[set]:
    """
    Generates approval votes from impartial culture with constant size.

    Under this culture, all ballots are of size :code:`num_approvals`. The ballot is selected
    uniformly at random over all ballots of size :code:`num_approvals`.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        num_approvals : int
            Number of approvals per ballot, i.e., size of the approval ballot.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set]
            Approval votes.

    Raises
    ------
        TypeError
            When `num_approvals` is not an int.
        ValueError
            When `num_approvals` is not in [0, num_candidates] interval.
    """

    validate_int(num_approvals, "number of approvals", lower_bound=0)
    if num_approvals > num_candidates:
        raise ValueError("The number of approval is higher than the number of candidates:"
                         f" {num_approvals} > {num_candidates}.")

    rng = np.random.default_rng(seed)
    candidate_range = range(num_candidates)
    votes = [
        set(rng.choice(candidate_range, size=num_approvals, replace=False))
        for _ in range(num_voters)
    ]

    return votes
