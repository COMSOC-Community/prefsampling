import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates

from prefsampling.core.urn import urn_votes


@validate_num_voters_candidates
def truncated_urn(
    num_voters: int, num_candidates: int, alpha: float, p: float, seed: int = None
) -> list[set[int]]:
    """
    Generates approval votes from a truncated variant of Polya-Eggenberger urn culture.

    Note that for a given number of voters, votes are not sampled independently.

    Parameters
    ----------
        num_voters: int
            Number of voters
        num_candidates: int
            Number of candidates
        alpha: float,
            The dispersion coefficient (`alpha * m!` copies of a vote are put back in the urn after
            a draw). Must be non-negative.
        p: float,
            Ratio of approved candidates.
        seed: int, default: :code:`None`
            The seed for the random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes

    Raises
    ------
        ValueError
            When `alpha` is a negative number.
            When `p` not in [0,1] interval.
    """

    if alpha < 0:
        raise ValueError("Alpha needs to be non-negative for an urn model.")

    if p < 0 or 1 < p:
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0,1]")

    rng = np.random.default_rng(seed)

    ordinal_votes = urn_votes(num_voters, num_candidates, alpha, rng)

    votes = []
    vote_length = int(p * num_candidates)
    for v in range(num_voters):
        set_ = set(ordinal_votes[v][0:vote_length])
        set_ = {int(x) for x in set_}
        votes.append(set_)

    return votes
