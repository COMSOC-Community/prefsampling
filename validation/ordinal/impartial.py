import numpy as np

from prefsampling.ordinal import impartial
from validation.utils import observed_frequencies


def impartial_distribution(all_ranks: list[tuple[int]]) -> np.ndarray:
    return np.full(len(all_ranks), 1 / len(all_ranks))


def impartial_observed_frequencies(num_observations: int, all_ranks: list[tuple[int]]):
    return observed_frequencies(
        num_observations,
        all_ranks,
        lambda: impartial(num_observations, len(all_ranks[0])),
    )
