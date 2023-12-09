import numpy as np
from prefsampling.ordinal.mallows import mallows


def mallows_filter(votes: np.ndarray, phi: float, seed: int = None) -> np.ndarray:
    """
    Returns votes with added Mallows filter.

    Parameters
    ----------
        votes : np.ndarray
            The votes.
        phi : float
            Noise parameter.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """
    return np.array(
        (_mallows_filter_vote(votes[i], phi, seed) for i in range(len(votes)))
    )


def _mallows_filter_vote(vote, phi: float, seed: int = None):
    return mallows(1, len(vote), phi, seed=seed, central_vote=vote)[0]
