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


def single_peaked_conitzer_distribution(all_sp_ranks: list[tuple[int]]):
    # TODO: THIS IS WRONG!!
    count_per_peak = np.zeros(len(all_sp_ranks[0]))
    for rank in all_sp_ranks:
        count_per_peak[rank[0]] += 1
    distribution = np.zeros(len(all_sp_ranks))
    for i, rank in enumerate(all_sp_ranks):
        distribution[i] = 1 / (count_per_peak[rank[0]] * len(count_per_peak))
    return distribution


def single_peaked_contizer_observed_frequencies(
    num_observations: int, all_ranks: list[tuple[int]]
):
    return observed_frequencies(
        num_observations,
        all_ranks,
        lambda: single_peaked_conitzer(num_observations, len(all_ranks[0])),
    )
