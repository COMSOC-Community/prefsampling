from __future__ import annotations

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


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
