from __future__ import annotations

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


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

    See :py:func:`prefsampling.approval.impartial.impartial_constant_size` for a version in which
    all voters approve of the same number of candidates.

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

    Examples
    --------

        .. testcode::

            from prefsampling.approval import impartial

            # Sample from an impartial culture with 2 voters and 3 candidates where
            # a candidate is approved with 60% probability.
            impartial(2, 3, 0.6)

            # For reproducibility, you can set the seed.
            impartial(2, 3, 0.6, seed=1002)
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
    num_voters: int, num_candidates: int, rel_num_approvals: float, seed: int = None
) -> list[set]:
    """
    Generates approval votes from impartial culture with constant size.

    Under this culture, all ballots are of size `⌊rel_num_approvals * num_candidates⌋`. The ballot
    is selected uniformly at random over all ballots of size `⌊rel_num_approvals * num_candidates⌋`.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    See :py:func:`prefsampling.approval.impartial.impartial` for a version in the probability of
    approving any candidate is constant and independent.

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
        list[set]
            Approval votes.

    Examples
    --------

        .. testcode::

            from prefsampling.approval import impartial_constant_size

            # Sample from an impartial culture with 2 voters and 3 candidates where
            # all voters approve of 60% of the candidates (in this case 1).
            impartial_constant_size(2, 3, 0.6)

            # For reproducibility, you can set the seed.
            impartial_constant_size(2, 3, 0.6, seed=1002)
    """

    if rel_num_approvals < 0 or 1 < rel_num_approvals:
        raise ValueError(
            f"Incorrect value of rel_num_approvals: {rel_num_approvals}. Value should"
            f" be in [0,1]"
        )

    num_approvals = int(rel_num_approvals * num_candidates)
    rng = np.random.default_rng(seed)
    candidate_range = range(num_candidates)
    votes = [
        set(rng.choice(candidate_range, size=num_approvals, replace=False))
        for _ in range(num_voters)
    ]

    return votes
