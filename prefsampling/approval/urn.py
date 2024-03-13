from __future__ import annotations

import numpy as np

from prefsampling.core.urn import urn_scheme
from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def urn(
    num_voters: int, num_candidates: int, p: float, alpha: float, seed: int = None
) -> list[set]:
    """

    Generates votes following the Pólya-Eggenberger urn culture. The process is as follows. The urn
    is initially empty and votes are generated one after the other, in turns. When generating a
    vote, the following happens. With a probability of 1/(urn_size + 1),an approval ballots in
    which all candidates have the same probability `p` of being approved is selected at random
    (following an impartial culture as in the
    :py:func:`~prefsampling.approval.impartial` function). With probability
    `1/urn_size` a vote from the urn is selected uniformly at random. In both cases, the vote is
    put back in the urn together with `alpha * (2**m)` copies of the vote (where `m` is the number
    of candidates).

    Note that for a given number of voters, votes are not sampled independently.

    Parameters
    ----------
        num_voters: int
            Number of voters
        num_candidates: int
            Number of candidates
        p : float
            Proportion of approved candidates in a ballot.
        alpha: float
            The dispersion coefficient (`alpha * m!` copies of a vote are put back in the urn after
            a draw). Must be non-negative.
        seed: int, default: :code:`None`
            The seed for the random number generator.

    Returns
    -------
        list[set]
            The votes
    """
    if p < 0 or 1 < p:
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0,1]")

    rng = np.random.default_rng(seed)
    return urn_scheme(
        num_voters,
        alpha,
        lambda x: set(j for j in range(num_candidates) if rng.random() <= p),
        rng,
    )


@validate_num_voters_candidates
def urn_constant_size(
    num_voters: int,
    num_candidates: int,
    rel_num_approvals: float,
    alpha: float,
    seed: int = None,
) -> list[set]:
    """
    Generates votes following the Pólya-Eggenberger urn culture. The process is as follows. The urn
    is initially empty and votes are generated one after the other, in turns. When generating a
    vote, the following happens. With a probability of 1/(urn_size + 1), an approval ballots of
    size `⌊rel_num_approvals * num_candidates⌋` is selected uniformly at random
    (following an impartial culture as in the
    :py:func:`~prefsampling.approval.impartial_constant_size` function). With probability
    `1/urn_size` a vote from the urn is selected uniformly at random. In both cases, the vote is
    put back in the urn together with `alpha * (m choose ⌊rel_num_approvals * num_candidates⌋)`
    copies of the vote (where `m` is the number of candidates).

    Note that for a given number of voters, votes are not sampled independently.

    Parameters
    ----------
        num_voters: int
            Number of voters
        num_candidates: int
            Number of candidates
        rel_num_approvals : float
            Proportion of approved candidates in a ballot.
        alpha: float
            The dispersion coefficient (`alpha * m!` copies of a vote are put back in the urn after
            a draw). Must be non-negative.
        seed: int, default: :code:`None`
            The seed for the random number generator.

    Returns
    -------
        list[set]
            The votes
    """
    if rel_num_approvals < 0 or 1 < rel_num_approvals:
        raise ValueError(
            f"Incorrect value of rel_num_approvals: {rel_num_approvals}. Value should"
            f" be in [0,1]"
        )

    num_approvals = int(rel_num_approvals * num_candidates)
    rng = np.random.default_rng(seed)
    candidate_range = range(num_candidates)
    return urn_scheme(
        num_voters,
        alpha,
        lambda x: set(rng.choice(candidate_range, size=num_approvals, replace=False)),
        rng,
    )


@validate_num_voters_candidates
def urn_partylist(
    num_voters: int,
    num_candidates: int,
    alpha: float,
    parties: int | list[float],
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes partylist model.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        alpha: float
            Parameter for Urn model.
        parties : int | list[float]
            Fractional sizes of the parties.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.
    """
    rng = np.random.default_rng(seed)

    if type(parties) is int:
        # If not specified, parties are of equal size
        parties = [1 / parties for _ in range(parties)]

    num_parties = len(parties)

    # Generate votes for parties
    party_votes = np.zeros([num_voters])

    urn_size = 1.0
    for j in range(num_voters):
        rho = np.random.uniform(0, urn_size)
        if rho <= 1.0:
            # party_votes[j] = rng.randint(0, num_parties)
            party_votes[j] = rng.integers(0, num_parties)
        else:
            party_votes[j] = party_votes[np.random.randint(0, j)]
        urn_size += alpha

    # Convert parties to candidates
    votes = []
    cumv = np.cumsum(parties)
    cumv = np.insert(cumv, 0, 0)

    for i in range(num_voters):
        party_id = int(party_votes[i])
        shift = cumv[party_id] * num_candidates
        vote = set(
            [int(c + shift) for c in range(int(parties[party_id] * num_candidates))]
        )

        votes.append(vote)

    return votes
