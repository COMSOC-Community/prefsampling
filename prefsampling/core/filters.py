"""
Filters are functions that operate on collections of votes and apply some random operation to them.
"""
from __future__ import annotations

from collections.abc import MutableSequence, Callable
from copy import deepcopy

import numpy as np


def permute_voters(
    votes: np.ndarray | MutableSequence, seed: int = None
) -> np.ndarray | MutableSequence:
    """
    Randomly permutes the voters in an ordered collection of votes.

    Parameters
    ----------
        votes : np.ndarray | MutableSequence
            The votes.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """
    rng = np.random.default_rng(seed)
    rng.shuffle(votes)

    return votes


def rename_candidates(
    votes: list[set[int]] | np.ndarray,
    num_candidates: int = None,
    seed: int = None,
):
    """
    Renames the candidates in approval or ordinal votes.

    Note that if the votes can be incomplete (in the case of approval voting), you need to
    provide the number of candidates as input. If it is not provided, it is assumed to be the
    largest integer appearing in the ballots (candidates are represented as int).

    Parameters
    ----------
        votes : list[set[int]] or np.ndarray
            Approval or ordinal votes.
        num_candidates : int
            Number of Candidates. Needed for incomplete (e.g., approval) votes.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]] or np.ndarray
            Votes with renamed candidates.
    """
    rng = np.random.default_rng(seed)

    if len(votes) == 0:
        return votes

    if num_candidates is None:
        num_candidates = max(max(vote, default=0) for vote in votes) + 1
    renaming = rng.permutation(num_candidates)

    if isinstance(votes, list) and isinstance(votes[0], set):
        renamed_votes = [{renaming[c] for c in vote} for vote in votes]
    elif isinstance(votes, np.ndarray):
        renamed_votes = renaming[votes]
    else:
        raise ValueError(
            "Unsupported input type for renaming. Are you using an unknown ballot format?"
        )

    return renamed_votes


def resample_as_central_vote(
    votes: np.ndarray | list[set[int]], sampler: Callable, sampler_parameters: dict
) -> np.ndarray | list[set[int]]:
    """
    Resamples the votes by using them as the central vote of a given sampler. Only samplers that
    accept a :code:`central_vote` argument can be used.

    Votes are copied before being returned to avoid loss of data.

    Parameters
    ----------
        votes : list[set[int]] or np.ndarray
            Approval or ordinal votes.
        sampler: Callable
            The sampler used to resample the votes.
        sampler_parameters: dict
            Dictionary passed as keyword parameters of the sampler. Number of voters or central vote
             of this dictionary are not taken into account.
    Returns
    -------
        list[set[int]] or np.ndarray
            Votes resampled.
    """
    res = deepcopy(votes)
    sampler_parameters["num_voters"] = 1
    for i, vote in enumerate(votes):
        sampler_parameters["central_vote"] = vote
        res[i] = sampler(**sampler_parameters)[0]
    return res
