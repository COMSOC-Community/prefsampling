from __future__ import annotations

import math

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates
from prefsampling.ordinal.urn import urn


@validate_num_voters_candidates
def impartial(num_voters: int, num_candidates: int, seed: int = None) -> np.ndarray:
    """
    Generates ordinal votes from impartial culture.

    In an impartial culture, all votes are equally likely to occur. In this function, each vote is
    generated by getting a random permutation of the candidates from the random number generator.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """
    rng = np.random.default_rng(seed)
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    for i in range(num_voters):
        votes[i] = rng.permutation(num_candidates)
    return votes


@validate_num_voters_candidates
def impartial_anonymous(
    num_voters: int, num_candidates: int, seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes from impartial anonymous culture.

    In an impartial anonymous culture, every multi-set of votes is equally likely to occur. For
    instance with 3 voters and 2 candidates, the probability of observing `a > b, a > b, a > b`
    is 1/4. This probability was 1/8 according to the impartial (but not-anonymous) culture.

    Votes are generated by sampling from an urn model (using :py:func:`~prefsampling.ordinal.urn`)
    with parameter `alpha = 1` (see Lepelley, Valognes 2003).

    Note that votes are not generated independently here.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """
    return urn(
        num_voters, num_candidates, alpha=1 / math.factorial(num_candidates), seed=seed
    )


@validate_num_voters_candidates
def stratification(
    num_voters: int, num_candidates: int, weight: float, seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes from stratification model. In the stratification model, candidates are
    split into two classes. Every voters ranks the candidates of the first class above the
    candidates of the second class. Within a class, the ranking is selected uniformly at random.

    The :code:`weight` parameter is used to define the relative size of the first class.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        weight : float
            Size of the upper class.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """
    if weight < 0 or 1 < weight:
        raise ValueError(
            f"Incorrect value of weight: {weight}. Value should be in [0, 1]"
        )
    rng = np.random.default_rng(seed)
    votes = np.zeros((num_voters, num_candidates), dtype=int)
    upper_class_size = int(weight * num_candidates)
    upper_class_candidates = range(upper_class_size, num_candidates)
    for i in range(num_voters):
        votes[i][:upper_class_size] = rng.permutation(upper_class_size)
        votes[i][upper_class_size:] = rng.permutation(upper_class_candidates)
    return votes
