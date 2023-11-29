import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def impartial(
    num_voters: int, num_candidates: int, p: float = 0.5, seed: int = None
) -> list[set]:
    """
    Generates approval votes from impartial culture.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        p : float, default: 0.5
            Resampling model parameter, denoting the average vote length.
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
