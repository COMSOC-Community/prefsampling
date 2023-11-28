from collections.abc import Iterable
from copy import deepcopy

import numpy as np

from prefsampling.ordinal import single_peaked_walsh, single_peaked_conitzer
from validation.utils import observed_frequencies


def get_all_single_peaked_ranks(num_candidates: int):
    def recursor(a, b, all_sp_ranks, rank, position):
        if a == b:
            rank[position] = a
            all_sp_ranks.append(tuple(rank))
            return
        rank[position] = a
        recursor(a + 1, b, all_sp_ranks, rank, position - 1)

        rank = deepcopy(rank)
        rank[position] = b
        recursor(a, b - 1, all_sp_ranks, rank, position - 1)

    res = []
    recursor(0, num_candidates - 1, res, [0] * num_candidates, num_candidates - 1)
    return res


def single_peaked_walsh_distribution(all_sp_ranks: list[tuple[int]]):
    return np.full(len(all_sp_ranks), 1 / len(all_sp_ranks))


def single_peaked_walsh_observed_frequencies(
    num_observations: int, all_ranks: list[tuple[int]]
):
    return observed_frequencies(
        num_observations,
        all_ranks,
        lambda: single_peaked_walsh(num_observations, len(all_ranks[0])),
    )


def single_peaked_conitzer_probability(rank: tuple, num_candidates: int):
    res = 1/num_candidates
    for alt in rank:
        if alt == 0 or alt == num_candidates - 1:
            break
        res *= 1/2
    return res


def single_peaked_conitzer_distribution(all_sp_ranks: list[tuple[int]]):
    distribution = np.zeros(len(all_sp_ranks))
    num_candidates = len(all_sp_ranks[0])
    for i, rank in enumerate(all_sp_ranks):
        distribution[i] = single_peaked_conitzer_probability(rank, num_candidates)
    return distribution


def single_peaked_contizer_observed_frequencies(
    num_observations: int, all_ranks: list[tuple[int]]
):
    return observed_frequencies(
        num_observations,
        all_ranks,
        lambda: single_peaked_conitzer(num_observations, len(all_ranks[0])),
    )
