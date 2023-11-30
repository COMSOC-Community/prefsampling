import numpy as np
from prefsampling.ordinal.mallows import mallows


def resampling_filter(votes: np.ndarray, phi: float) -> list[set[int]]:
    """
    Returns votes with added Resampling filter.

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
    # TODO: implement
    return [{1}]

