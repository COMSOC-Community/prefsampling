"""
Filters are functions that operate on collections of votes and apply some random operation to them.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import numpy as np


def permute_voters(votes: list, seed: int = None) -> list:
    """
    Randomly permutes the voters in an ordered collection of votes.

    Parameters
    ----------
        votes : list
            The votes.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list
            The votes permuted.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import didi
            from prefsampling.core import permute_voters

            # Get some votes
            ordinal_votes = didi(2, 3, (0.5, 0.2, 0.1))

            # Randomly permute the voters
            permute_voters(ordinal_votes)

            # The syntax is the same with approval votes

            from prefsampling.approval import resampling

            approval_votes = resampling(2, 3, 0.5, 0.2)
            permute_voters(approval_votes)

            # You can set the seed for reproducibility

            permute_voters(approval_votes, seed=234)

    """
    rng = np.random.default_rng(seed)
    rng.shuffle(votes)
    return votes


def rename_candidates(
    votes: list[set[int]] | np.ndarray,
    num_candidates: int = None,
    seed: int = None,
) -> list:
    """
    Renames the candidates in approval or ordinal votes.

    Note that if the votes can be incomplete (e.g., in the case of approval voting), you need to
    provide the number of candidates as input. If it is not provided, it is assumed to be the
    largest integer appearing in the ballots (candidates are represented as int).

    Parameters
    ----------
        votes : list[set[int]] or np.ndarray
            Approval or ordinal votes.
        num_candidates : int
            Number of Candidates. Needed for incomplete (e.g., approval) votes.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list
            Votes with renamed candidates.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import didi
            from prefsampling.core import rename_candidates

            # Get some votes
            ordinal_votes = didi(2, 3, (0.5, 0.2, 0.1))

            # Randomly permute the voters
            rename_candidates(ordinal_votes)

            # With approval votes, you need to give the number of candidates

            from prefsampling.approval import resampling

            approval_votes = resampling(2, 3, 0.5, 0.2)
            rename_candidates(approval_votes, num_candidates=3)

            # You can set the seed for reproducibility

            rename_candidates(approval_votes, num_candidates=3, seed=234)
    """
    rng = np.random.default_rng(seed)

    if len(votes) == 0:
        return votes

    if num_candidates is None:
        num_candidates = max(max(vote, default=0) for vote in votes) + 1
    renaming = rng.permutation(num_candidates)

    if isinstance(votes, list) and isinstance(votes[0], set):
        renamed_votes = [{renaming[c] for c in vote} for vote in votes]
    elif isinstance(votes, list) and isinstance(votes[0], list):
        renamed_votes = [[renaming[c] for c in vote] for vote in votes]
    else:
        raise ValueError(
            "Unsupported input type for renaming. Are you using an unknown ballot format?"
        )

    return renamed_votes


def resample_as_central_vote(
    votes: np.ndarray | list[set[int]],
    sampler: Callable,
    sampler_parameters: dict,
    num_candidates: int = None,
) -> list:
    """
    Resamples the votes by using them as the central vote of a given sampler. The outcome is
    obtained as follows: for each input vote, we pass it to the sampler as central vote; a single
    vote is then resampled and added to the outcome.

    Only samplers that accept a :code:`central_vote` argument can be used.

    Note that if the votes can be incomplete (e.g., in the case of approval voting), you need to
    provide the number of candidates as input. If it is not provided, it is assumed to be the
    largest integer appearing in the ballots (candidates are represented as int).

    Votes are copied before being returned to avoid loss of data.

    Parameters
    ----------
        votes : list[set[int]] or np.ndarray
            Approval or ordinal votes.
        sampler : Callable
            The sampler used to resample the votes.
        sampler_parameters : dict
            Dictionary passed as keyword parameters of the sampler. Number of voters or central vote
             of this dictionary are not taken into account.
        num_candidates : int, defaults: :code:`None`
            The number of candidates, useful for incomplete ballots in which this information cannot
            be retrieved from the votes.

    Returns
    -------
        list
            Votes resampled.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import urn, mallows
            from prefsampling.core import resample_as_central_vote

            # Get some votes
            ordinal_votes = urn(2, 3, 0.2)

            # We resample them by passing them as central vote to a Mallows' model
            resample_as_central_vote(ordinal_votes, mallows, {'phi': 0.3})

            # The syntax is the same with approval votes

            from prefsampling.approval import urn, resampling

            approval_votes = urn(2, 3, 0.5, 0.2)
            resample_as_central_vote(
                approval_votes,
                resampling,
                {'phi': 0.4, 'rel_size_central_vote': 0.8}
            )

            # To ensure reproducibility, you need to pass the seed everywhere
            seed = 4234
            approval_votes = urn(2, 3, 0.5, 0.2, seed=seed)
            resample_as_central_vote(
                approval_votes,
                resampling,
                {'phi': 0.4, 'rel_size_central_vote': 0.8, 'seed':seed}
            )
    """
    res = deepcopy(votes)
    sampler_parameters["num_voters"] = 1
    if num_candidates is None:
        num_candidates = max(max(vote, default=0) for vote in votes) + 1
    sampler_parameters["num_candidates"] = 1
    sampler_parameters["num_candidates"] = num_candidates
    for i, vote in enumerate(votes):
        sampler_parameters["central_vote"] = vote
        res[i] = sampler(**sampler_parameters)[0]
    return res


def coin_flip_ties(
    ordinal_votes: list[list[int]], p: float, seed: int = None
) -> list[list[list[int]]]:
    """
    In the coin-flip models, a complete ordered is turned into a weak order by adding ties between
    the candidates as follows: for each pair of consecutively ranked candidates, we add a tie
    between them with probability :code:`p`.

    Parameters
    ----------
        ordinal_votes : list[list[int]]
            The (strict) ordinal votes.
        p : float
            The probability of forming a tie.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[list[int]]]
            The weak orders.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import urn
            from prefsampling.core import coin_flip_ties

            # Get some votes
            strict_ordinal_votes = urn(2, 3, 0.2)

            # We randomly introduce ties
            weak_ordinal_votes = coin_flip_ties(strict_ordinal_votes, 0.34)

            # You can set the seed to ensure reproducibility
            weak_ordinal_votes = coin_flip_ties(strict_ordinal_votes, 0.34, seed=265)

    References
    ----------
        `Generalizing Instant Runoff Voting to Allow Indifferences
        <https://arxiv.org/abs/2404.11407>`_,
        *Théo Delemazure and Dominik Peters*,
        arXiv:2404.11407, 2024.

    """
    if p < 0 or 1 < p:
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0, 1]")

    rng = np.random.default_rng(seed)

    weak_orders = []
    for vote in ordinal_votes:
        weak_order = []
        indif_class = [vote[0]]
        for cand in vote[1:]:
            if rng.random() > p:
                weak_order.append(sorted(indif_class))
                indif_class = [cand]
            else:
                indif_class.append(cand)
        weak_order.append(sorted(indif_class))
        weak_orders.append(weak_order)

    return weak_orders
