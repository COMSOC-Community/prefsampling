import math

import numpy as np

from prefsampling.ordinal import mallows
from validation.utils import observed_frequencies


def kendall_tau_distance(rank1: tuple, rank2: tuple):
    distance = 0
    for k, alt1 in enumerate(rank1):
        for alt2 in rank1[k + 1 :]:
            if rank2.index(alt2) < rank2.index(alt1):
                distance += 1
    return distance


# assert kendall_tau_distance([0, 1, 2, 3], [3, 2, 1, 0]) == math.comb(4, 2)
# assert kendall_tau_distance([0, 1, 2, 3], [1, 2, 3, 0]) == 3


def mallows_probability(central_rank: tuple, rank: tuple, phi: float):
    return phi ** kendall_tau_distance(central_rank, rank)


def mallows_distribution(central_rank: tuple, phi: float, all_ranks: list[tuple[int]]):
    distribution = np.zeros(len(all_ranks))
    for i, rank in enumerate(all_ranks):
        distribution[i] = mallows_probability(central_rank, rank, phi)
    return distribution / sum(distribution)


def mallows_observed_frequencies(
    central_rank: tuple, phi: float, num_observations: int, all_ranks: list[tuple[int]]
):
    return observed_frequencies(
        num_observations,
        all_ranks,
        lambda: mallows(num_observations, len(central_rank), phi=phi),
    )


def frequencies_by_distance(
    frequencies: np.ndarray, central_rank: tuple, all_ranks: list[tuple[int]]
):
    result = np.zeros(math.comb(len(all_ranks[0]), 2) + 1)
    for i, freq in enumerate(frequencies):
        result[kendall_tau_distance(all_ranks[i], central_rank)] += freq
    return result
