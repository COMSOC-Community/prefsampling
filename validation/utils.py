from collections.abc import Callable
from itertools import permutations

import numpy as np


def get_all_ranks(num_candidates: int) -> list[tuple[int]]:
    return [tuple(rank) for rank in permutations(range(num_candidates))]


def observed_frequencies(
    num_observations: int, all_ranks: list[tuple[int]], sampler: Callable
) -> np.ndarray:
    samples = np.zeros(len(all_ranks))
    for rank in sampler():
        rank = tuple(rank)
        samples[all_ranks.index(rank)] += 1
    return samples / num_observations
