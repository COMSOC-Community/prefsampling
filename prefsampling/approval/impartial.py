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
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0,1]")

    rng = np.random.default_rng(seed)

    votes = [
        set(j for j in range(num_candidates) if rng.random() <= p)
        for _ in range(num_voters)
    ]

    return votes
