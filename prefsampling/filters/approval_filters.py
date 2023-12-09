import numpy as np
from prefsampling.approval.resampling import resampling


def resampling_filter(
    votes: np.ndarray, phi: float, seed: int = None
) -> list[set[int]]:
    """
    Returns votes with added Resampling filter.

    Parameters
    ----------
        votes : list[set[int]]
            The votes.
        phi : float
            Noise parameter.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.
    """

    return [_resampling_filter_vote(votes[i], phi, seed) for i in range(len(votes))]


def _resampling_filter_vote(vote, phi: float, seed: int = None):
    return resampling(1, len(vote), phi, seed, central_vote=vote)[0]
