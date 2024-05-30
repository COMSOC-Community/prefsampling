from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from prefsampling.core.urn import urn_scheme
from prefsampling.inputvalidators import validate_num_voters_candidates, validate_int


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

    Examples
    --------

        .. testcode::

            from prefsampling.approval import urn

            # Sample from an urn model with 2 voters and 3 candidates, alpha parameter is 0.5,
            # parameter p is 0.7
            urn(2, 3, 0.7, 0.5)

            # For reproducibility, you can set the seed.
            urn(2, 3, 0.7, 0.5, seed=1002)

            # Passing a negative alpha will fail
            try:
                urn(2, 3, 0.7, -0.5)
            except ValueError:
                pass

            # Parameter p needs to be in [0, 1]
            try:
                urn(2, 3, 1.7, 0.5)
            except ValueError:
                pass
            try:
                urn(2, 3, -0.7, 0.5)
            except ValueError:
                pass

    References
    ----------
        None.
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

    Examples
    --------

        .. testcode::

            from prefsampling.approval import urn_constant_size

            # Sample from an urn model with 2 voters and 3 candidates, alpha parameter is 0.5,
            # parameter rel_num_approvals is 0.7
            urn_constant_size(2, 3, 0.7, 0.5)

            # For reproducibility, you can set the seed.
            urn_constant_size(2, 3, 0.7, 0.5, seed=1002)

            # Passing a negative alpha will fail
            try:
                urn_constant_size(2, 3, 0.7, -0.5)
            except ValueError:
                pass

            # Parameter rel_num_approvals needs to be in [0, 1]
            try:
                urn_constant_size(2, 3, 1.7, 0.5)
            except ValueError:
                pass
            try:
                urn_constant_size(2, 3, -0.7, 0.5)
            except ValueError:
                pass

    References
    ----------
        None.
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
    parties: int | Iterable[float] = None,
    party_votes: Iterable[set[int]] = None,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes partylist model. In this model, the candidates are partitioned into
    parties. Voters are assigned a party using an urn model with parameter :code:`alpha` where the
    balls represent the parties. A voter then approves of all the candidates belonging to their
    party.

    If the votes of the parties are not provided (argument :code:`party_votes`), they are determined
    by the relative size of the parties relative size (argument :code:`parties`). The vote of the
    first party (of size s1) will always be `{0, 1, ..., s1 - 1}`, the vote of the second party
    (of size s2) will always be `{s1, s1 + 1, ..., s2}`, etc... In particular, it can be that some
    candidates are not assigned any party and will thus never be approved (depending on the relative
    size of the parties). If only the number of parties is given (when :code:`parties` is a
    integer), then they are assumed to be of equal size.

    Note that for a given number of voters, votes are not sampled independently.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        alpha: float
            Parameter for Urn model.
        parties : int | Iterable[float], defaults: :code:`None`
            Fractional sizes of the parties. If an integer is given, then it is assumed that it
            represents the number of parties that have equal size.  Needed if the argument
            :code:`party_votes` is not provided.
        party_votes : Iterable[set[int]], defaults: :code:`None`
            The votes of the parties. Needed if the argument :code:`parties` is not provided.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    Examples
    --------

        .. testcode::

            from prefsampling.approval import urn_partylist

            # Sample from an urn party-list model with 2 voters and 3 candidates,
            # alpha parameter is 0.5, and there are 2 equal-sized parties
            urn_partylist(2, 3, 0.5, parties=2)

            # For reproducibility, you can set the seed.
            urn_partylist(2, 3, 0.5, parties=2, seed=1002)

            # You can use parties of different sizes (sizes are normalised)
            urn_partylist(2, 5, 0.5, parties=[1, 5, 0.8])

            # You can also just provide the votes of the parties
            urn_partylist(2, 5, 0.5, party_votes=[{0, 2}, {1, 3, 4}])

            # The votes of the parties have to be disjoint
            try:
                urn_partylist(2, 5, 0.5, party_votes=[{0, 2}, {0, 3, 4}])
            except ValueError:
                pass

            # You need to pass a value to either 'parties' or 'party_votes'
            try:
                urn_partylist(2, 3, 0.5)
            except ValueError:
                pass

            # Passing a negative alpha will fail
            try:
                urn_partylist(2, 3, -0.5, parties=2)
            except ValueError:
                pass

    References
    ----------
        None.
    """
    rng = np.random.default_rng(seed)

    # Votes of the parties
    if party_votes is None:
        if parties is None:
            raise ValueError(
                "If you do not provide the votes of the parties, then you need to "
                "provide their number or their relative size."
            )
        if not isinstance(parties, Iterable):
            validate_int(parties, "number of parties", lower_bound=1)
            if parties > num_candidates:
                raise ValueError(
                    "In the urn party-list model, the number of parties cannot exceed "
                    "the number of candidates."
                )
            # If not specified, parties are of equal size
            parties = [1 / parties for _ in range(parties)]

        party_votes = []
        cum_size = 0
        for party_id, party_rel_size in enumerate(parties):
            size = int(party_rel_size * num_candidates)
            party_votes.append(set(range(cum_size, cum_size + size)))
            cum_size += size
    else:
        seen_candidates = set()
        for vote in party_votes:
            for cand in vote:
                if cand in seen_candidates:
                    raise ValueError(
                        "In the urn partylist model the votes of the parties need to be disjoint. "
                        f"Currently, candidate {cand} appears in at least 2 parties."
                    )
                seen_candidates.add(cand)
        if len(seen_candidates) > num_candidates:
            raise ValueError(
                "There are more candidates appearing in the party votes than the number of "
                "candidates provided as an argument."
            )
        if max(seen_candidates) >= num_candidates:
            raise ValueError(
                "The candidates need to be called 0, 1, ..., num_candidates, this is not the case "
                "in the provided party votes."
            )

    num_parties = len(party_votes)
    # Map voters to parties
    voters_to_party = urn_scheme(
        num_voters, alpha, lambda x: x.integers(0, num_parties), rng
    )

    # Find the votes
    votes = []
    for party_id in voters_to_party:
        votes.append(party_votes[party_id])
    return votes
