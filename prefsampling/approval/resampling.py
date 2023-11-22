import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def resampling(
    num_voters: int,
    num_candidates: int,
    phi: float = 0.5,
    p: float = 0.5,
    seed: int = None,
) -> list[set]:
    """
    Generates approval votes from the resampling model.

    Parameters
    ----------
        num_voters : int
            Number of voters.
        num_candidates : int
            Number of candidates.
        phi : float, default: 0.5
            Resampling model parameter, denoting the noise.
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
            When `phi` not in [0,1] interval.
            When `p` not in [0,1] interval.
    """

    if phi < 0 or 1 < phi:
        raise ValueError("Resampling model is not well defined for `phi` not in [0,1] interval")

    if p < 0 or 1 < p:
        raise ValueError("Resampling model is not well defined for `p` not in [0,1] interval")

    rng = np.random.default_rng(seed)

    k = int(p * num_candidates)
    central_vote = {i for i in range(k)}

    votes = [set() for _ in range(num_voters)]
    for v in range(num_voters):
        vote = set()
        for c in range(num_candidates):
            if rng.random() <= phi:
                if rng.random() <= p:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes[v] = vote

    return votes
