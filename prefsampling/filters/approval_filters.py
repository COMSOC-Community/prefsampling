import numpy as np
from prefsampling.approval.resampling import resampling


def permute_approval_voters(votes: list[set[int]], seed: int = None) -> list[set[int]]:
    """
    Permutes the voters in approval votes.

    Parameters
    ----------
        votes : list[set[int]]
            Approval votes.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.
    """
    rng = np.random.default_rng(seed)
    rng.shuffle(votes)

    return votes


def rename_approval_candidates(
    votes: list[set[int]], seed: int = None
) -> list[set[int]]:
    """
    Renames the candidates in approval votes.

    Parameters
    ----------
        votes : list[set[int]]
            Approval votes.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.
    """
    rng = np.random.default_rng(seed)
    max_id = max([max(vote) for vote in votes if len(vote) > 0])
    mapping = rng.permutation(max_id + 1)

    votes = [{mapping[c] for c in vote} for vote in votes]

    return votes


def resampling_filter(
    votes: list[set[int]], phi: float, seed: int = None
) -> list[set[int]]:
    """
    Returns votes with added resampling filter.

    Parameters
    ----------
        votes : list[set[int]]
            Approval votes.
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
