"""
Weak-orders represent preferences with ties: some alternatives can be declared to be equally good
(or bad).
"""
from __future__ import annotations

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def coin_flip_ties(
    ordinal_votes: list[list[int]], p: float, seed: int
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
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[list[list[int]]]
            The weak orders.


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
