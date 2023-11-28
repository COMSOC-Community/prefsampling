import numpy as np
from prefsampling.ordinal.mallows import mallows


def mallowsify_votes(votes: np.ndarray, phi: float) -> np.ndarray:
    """
    Returns votes with added Mallows noise.

    Parameters
    ----------
    votes : np.ndarray
        The votes.
    phi : float
        Noise parameter.

    Returns
    -------
    np.ndarray
        The votes.

    """
    return np.array([_mallowsify_vote(votes[i], phi) for i in range(len(votes))])


def _mallowsify_vote(vote, phi: float):
    num_candidates = len(vote)
    raw_vote = mallows(1, num_candidates, phi)[0]
    new_vote = [0] * len(vote)
    for i in range(num_candidates):
        new_vote[raw_vote[i]] = vote[i]
    return new_vote



