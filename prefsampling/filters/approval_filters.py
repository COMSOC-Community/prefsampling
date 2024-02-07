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
        votes: list[set[int]],
        seed: int = None,
        num_candidates: int = None
) -> list[set[int]]:
    """
    Renames the candidates in approval votes.

    Parameters
    ----------
        votes : list[set[int]]
            Approval votes.
        seed : int
            Seed for numpy random number generator.
        num_candidates : int
            Number of Candidates.

    Returns
    -------
        list[set[int]]
            Approval votes.
    """

    rng = np.random.default_rng(seed)
    if num_candidates is None:
        num_candidates = max([max(vote) for vote in votes if len(vote) > 0]) + 1
    mapping = rng.permutation(num_candidates)
    votes = [{mapping[c] for c in vote} for vote in votes]
    return votes


def resampling_filter(
    votes: list[set[int]], num_candidates, phi: float, p, seed: int = None
) -> list[set[int]]:
    """
    Returns votes with added resampling filter.

    Parameters
    ----------
        votes : list[set[int]]
            Approval votes.
        num_candidates : int
            Number of Candidates.
        phi : float
            Noise parameter.
        p : float
            Resampling model parameter, denoting the average vote length.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.
    """

    return [_resampling_filter_vote(votes[i], num_candidates, phi, p, seed) for i in range(len(votes))]


def _resampling_filter_vote(vote, num_candidates, phi: float, p, seed: int = None):
    return resampling(1, num_candidates, phi, p, seed, central_vote=vote)[0]
