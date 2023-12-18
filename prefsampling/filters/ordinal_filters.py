import numpy as np
from prefsampling.ordinal.mallows import mallows


def permute_ordinal_voters(votes: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Permutes the voters in ordinal votes.

    Parameters
    ----------
        votes : np.ndarray
            Ordinal votes.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """
    rng = np.random.default_rng(seed)
    rng.shuffle(votes)

    return np.array(votes)


def rename_ordinal_candidates(votes: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Renames the candidates in ordinal votes.

    Parameters
    ----------
        votes : np.ndarray
            Ordinal votes.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """
    rng = np.random.default_rng(seed)
    mapping = rng.permutation(votes[0])

    votes = mapping[votes]

    return votes


def mallows_filter(votes: np.ndarray, phi: float, seed: int = None) -> np.ndarray:
    """
    Returns votes with added Mallows filter.

    Parameters
    ----------
        votes : np.ndarray
            Ordinal votes.
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
